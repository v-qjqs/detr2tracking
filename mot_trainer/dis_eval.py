import os
import json 
import pickle
import numpy as np 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
import utils_ as utils
import random
from dataset import build_dataloader, ReIDDataset, worker_init_fn, default_collate
import time
from criterion import CECriterion
from models.model import build_model
from torch.utils.data import DistributedSampler, DataLoader
from functools import partial
import logging
import sys
import tempfile
import torch.distributed as dist
import mmcv
import os.path as osp
import shutil
import pandas as pd 


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)  # batch size per gpu for distributed training
    parser.add_argument('--out_dir', default='./exp_evaldelete_normalized', type=str)
    # distributed
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--workers_per_gpu', default=1, type=int)
    # model
    parser.add_argument('--num_classes', default=32, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    return parser

def eval_(model, data_loader, device):
    model.eval()
    results_ = dict()
    for step, samples in enumerate(data_loader):
        samples = {k: v.to(device) if k not in ['seqname', 'frameid', 'detection_id_pre', 'detection_id_cur'] else v for k,v in samples.items()}
        reid_pos_enc_pre, reid_pos_enc_cur = samples['reid_pos_enc_frame_pre'].to(device), samples['reid_pos_enc_frame_cur'].to(device)
        reid_feat_pre, reid_feat_cur = samples['reid_feat_frame_pre'].to(device), samples['reid_feat_frame_cur'].to(device)
        nbdet_valid_pre, nbdet_valid_cur = samples['nbdet_valid_frame_pre'].to(device), samples['nbdet_valid_frame_cur'].to(device)
        mask_pre, mask_cur = samples['mask_valid_frame_pre'].to(device).to(torch.bool), samples['mask_valid_frame_cur'].to(device).to(torch.bool)
        mask_pre, mask_cur = torch.logical_not(mask_pre), torch.logical_not(mask_cur)
        preds = model(reid_feat_pre, reid_feat_cur, mask_pre, mask_cur, reid_pos_enc_pre, reid_pos_enc_cur)  # preds: [bs,max_nb2, nb_classes]
        labels, pred_probs = model.module.match_label_eval(preds, nbdet_valid_cur, nbdet_valid_pre, max_nb_class=19)
        seqname, frameid = samples['seqname'], samples['frameid']
        assert len(seqname) == len(frameid) == 1  # bs 1 for inference mode
        seqname, frameid = seqname[0], frameid[0]
        frameid = frameid.cpu().item()
        frameid += 1  # NOTE next frame
        detection_id_pre, detection_id_cur = samples['detection_id_pre'].squeeze(0), samples['detection_id_cur'].squeeze(0)
        assert len(set(detection_id_cur.tolist()))==len(detection_id_cur)
        assert detection_id_cur.shape[0] == labels.shape[0], '{} vs {}'.format(detection_id_cur.shape[0], labels.shape[0])
        matched_detid = np.ones((labels.shape[0]),dtype=np.int32)*-1
        mask = labels > -1
        matched_detid[mask] = detection_id_pre[labels[mask]]
        if seqname not in results_:
            results_[seqname]=[]
        results_[seqname].append(dict(frameid_cur=frameid, detection_id_cur=tuple(detection_id_cur.tolist()), 
                    matched_detid_pre=tuple(matched_detid.tolist()), pred_probs=tuple(pred_probs.tolist())))
    return results_

def multi_gpu_test(model, data_loader, device):
    result_part = eval_(model, data_loader, device)
    return collect_results_cpu(result_part, seq_names=result_part.keys(), size=len(data_loader.dataset), device=device)

def collect_results_cpu(result_part, seq_names, size, device, tmpdir=None):
    rank, world_size = utils.get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device=device,
                                )  
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, 
                device=device,
                )
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        ordered_results = {name: list() for name in seq_names}
        for part_dict in part_list:
            for name in seq_names:
                assert isinstance(part_dict[name], list)
                ordered_results[name].extend(part_dict[name])
        assert sum([len(ordered_results[seq_name]) for seq_name in seq_names]) >= size, "{} vs {}".format(
            sum([len(ordered_results[seq_name]) for seq_name in seq_names]), size)
        shutil.rmtree(tmpdir)  # remove tmp dir
        return ordered_results

def get_mot_dataset_info():
    root='/mnt/truenas/scratch/lqf/data/'
    mot17_root=root+'/mot/MOT17/train/'
    eval_mot17_data=['MOT17-11-FRCNN','MOT17-04-FRCNN']
    mot17_datapath_list=[mot17_root+dataname for dataname in eval_mot17_data]
    return mot17_datapath_list, eval_mot17_data

def save_results(results_across_gpus, out_dir, datalen):
    cnt_after_drop = 0
    for seq_name, results in results_across_gpus.items():
        [res.values() for res in results]
        results_list = [tuple(res.values()) for res in results]
        pd_data = pd.DataFrame(results_list)
        pd_data.columns = ['frameid_cur', 'detection_id_cur', 'matched_detid_pre', 'pred_probs']
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        save_path = os.path.join(out_dir, '{}.pkl'.format(seq_name))
        pd_data_droped = pd_data.drop_duplicates(subset=['frameid_cur', 'detection_id_cur'])
        pd_data_droped.to_pickle(save_path)
        # print('LEN: ', len(results_list), len(pd_data), len(pd_data_droped))
        cnt_after_drop += len(pd_data_droped)
    assert cnt_after_drop == datalen, '{} vs {}'.format(cnt_after_drop, datalen)

def func_(frameid_cur, detid_cur, matched_detid_pre, pred_probs):
    assert len(frameid_cur) == len(detid_cur) == len(matched_detid_pre)
    res = []
    for i, frameid in enumerate(frameid_cur):
        assert len(detid_cur[i]) == len(matched_detid_pre[i])
        res.extend([(frameid, detid, matchedid, pred_probsi) for detid,matchedid,pred_probsi in zip(detid_cur[i], matched_detid_pre[i], pred_probs[i])])
    return res

def save_results_format(root, seq_names, out_dir):
    for seq_name in seq_names:
        pd_data = pd.read_pickle(os.path.join(root, '{}.pkl'.format(seq_name)))
        res_per_seq = func_(pd_data['frameid_cur'].values, pd_data['detection_id_cur'].values, 
                            pd_data['matched_detid_pre'].values, pd_data['pred_probs'].values)
        pd_data_new = pd.DataFrame(res_per_seq)
        pd_data_new.columns = pd_data.columns
        assert len(pd_data_new.drop_duplicates(subset=['frameid_cur', 'detection_id_cur'])) == len(pd_data_new)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        save_path = os.path.join(out_dir, '{}_new.pkl'.format(seq_name))
        pd_data_new.to_pickle(save_path)

def substract_detid(seq_name, pd_seq_data_path, out_dir, subv):
    pd_data = pd.read_pickle(pd_seq_data_path)
    detid = pd_data['detection_id_cur'].values
    detid -= subv
    pd_data['detection_id_cur'] = detid
    matched_detid_pre = pd_data['matched_detid_pre'].values
    mask = matched_detid_pre>=0
    pd_data['matched_detid_pre'] = np.where(mask, matched_detid_pre-subv, np.ones_like(matched_detid_pre, dtype=matched_detid_pre.dtype)*-1)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    save_path = os.path.join(out_dir, '{}_new_subv{}.pkl'.format(seq_name, subv))
    pd_data.to_pickle(save_path)

def main(args):
    utils.init_distributed_mode(args)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model
    args.device = 'cuda:{}'.format(args.gpu)
    device = torch.device(args.device)
    model = build_model(args)
    model.to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        raise NotImplementedError

    # load from checkpoint
    # checkpoint_path = os.path.join('./exp_shuffleprob_normalized', 'checkpoint')  # NOTE
    checkpoint_path = os.path.join('/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_shuffleprob_normalized/', 'checkpoint')
    checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_file, map_location={'cuda:0':'{}'.format(args.device)})
    model.load_state_dict(checkpoint['model'])

    # dataset
    seq_paths, seq_names = get_mot_dataset_info()
    dataset = ReIDDataset(seq_paths, seq_names, train=False, feat_path='eval_reid_posenc_feat_merge_dim512_normalized')

    # dataloader
    rank, world_size = utils.get_dist_info()
    assert args.rank == rank
    sampler_train = DistributedSampler(dataset, num_replicas=world_size, rank=args.rank, shuffle=True)
    init_fn = partial(worker_init_fn, num_workers=args.workers_per_gpu, rank=args.rank, seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # images_per_gpu
        sampler=sampler_train,
        num_workers=args.workers_per_gpu,
        collate_fn=partial(default_collate),
        pin_memory=False,
        worker_init_fn=init_fn)
    print('len_data_loader={} len_dataset={}: ', len(data_loader), len(data_loader.dataset))

    # eval
    time_start = time.time()
    results_across_gpus = multi_gpu_test(model, data_loader, device)
    if utils.is_main_process():
        save_results(results_across_gpus, args.out_dir, datalen=len(data_loader.dataset))
        root_ = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_evaldelete_normalized'
        save_results_format(root_, seq_names, out_dir=args.out_dir)
        seq_name = 'MOT17-11-FRCNN'
        substract_detid(seq_name, os.path.join(args.out_dir, '{}_new.pkl'.format(seq_name)), args.out_dir, subv=141)
        total_time = (time.time() - time_start)/3600
        print('Train finished, total_train_time={}h'.format(total_time))


if __name__ == "__main__":
    parser = arg_parse()
    args = parser.parse_args()
    main(args)
    os._exit(0)