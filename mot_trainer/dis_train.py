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


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)  # batch size per gpu for distributed training
    parser.add_argument('--out_dir', default='./exp', type=str)
    parser.add_argument('--report_fq', default=100, type=int)
    # distributed
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--workers_per_gpu', default=1, type=int)
    # lr and scheduler
    parser.add_argument('--lr', default=0.025, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=4e-5, type=float)
    # model
    parser.add_argument('--num_classes', default=32, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--shuffle_in_frame', action='store_true', default=False)
    parser.add_argument('--shuffle_prob', default=0., type=float)
    return parser

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, args, max_norm=0.):
    model.train()
    criterion.train()
    loss_log = utils.AvgrageMeter()
    MAX = -1
    for step, samples in enumerate(data_loader):
        samples = {k: v.to(device) for k,v in samples.items()}
        labels = samples['label'].to(device)  # [bs, max_nb2]
        reid_pos_enc_pre, reid_pos_enc_cur = samples['reid_pos_enc_frame_pre'].to(device), samples['reid_pos_enc_frame_cur'].to(device)
        reid_feat_pre, reid_feat_cur = samples['reid_feat_frame_pre'].to(device), samples['reid_feat_frame_cur'].to(device)
        nbdet_valid_pre, nbdet_valid_cur = samples['nbdet_valid_frame_pre'].to(device), samples['nbdet_valid_frame_cur'].to(device)
        mask_pre, mask_cur = samples['mask_valid_frame_pre'].to(device).to(torch.bool), samples['mask_valid_frame_cur'].to(device).to(torch.bool)
        mask_pre, mask_cur = torch.logical_not(mask_pre), torch.logical_not(mask_cur)
        preds = model(reid_feat_pre, reid_feat_cur, mask_pre, mask_cur, reid_pos_enc_pre, reid_pos_enc_cur)
        loss_dict = criterion(preds, labels, nbdet_valid_cur)
        losses = sum(loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())
        loss_value = losses_reduced_scaled.item()

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # log
        loss_log.update(loss_value, n=1)
        if utils.is_main_process() and step % args.report_fq == 0:
            logging.info('train step={}, loss={:.4f}'.format(step, loss_log.avg))
        MAX = max(MAX, labels.max())
    return loss_log.avg, MAX

def get_mot_dataset_info():
    root='/mnt/truenas/scratch/lqf/data/'
    mot17_root=root+'/mot/MOT17/train/'
    mot15_root=root+'/mot/2DMOT2015/train/'
    mot17_data=['MOT17-02-FRCNN','MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-13-FRCNN']
    mot15_data=['KITTI-17','ETH-Sunnyday','ETH-Bahnhof','PETS09-S2L1','TUD-Stadtmitte']
    mot17_datapath_list=[mot17_root+dataname for dataname in mot17_data]
    mot15_datapath_list=[mot15_root+dataname for dataname in mot15_data]
    datanames = mot17_data + mot15_data
    return mot17_datapath_list+mot15_datapath_list, datanames

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
    criterion = CECriterion(args.num_classes, weights_valid=1., eps=1e-6)
    criterion.to(device)
    model_without_ddp = model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        raise NotImplementedError

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=False)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,245], gamma=0.1, last_epoch=-1)

    # dataset
    seq_paths, seq_names = get_mot_dataset_info()
    print('args.shuffle_in_frame: ', args.shuffle_in_frame)
    dataset = ReIDDataset(seq_paths, seq_names, feat_path='reid_posenc_feat_merge_dim512_normalized', train=True, 
        shuffle_in_frame=args.shuffle_in_frame, shuffle_prob=args.shuffle_prob)

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
    print('len data_loader: ', len(data_loader))

    # logging:
    if utils.is_main_process():
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        fh = logging.FileHandler(os.path.join(args.out_dir, 'log_embdim{}_feedforwarddim{}_nhead{}e{}_lr{}_nbclass{}_shuffleprob{}.txt'.format(
            args.embed_dim, args.dim_feedforward, args.num_heads, args.epochs, args.lr, args.num_classes, args.shuffle_prob)))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    # training
    time_start = time.time()
    for epoch in range(args.epochs):
        sampler_train.set_epoch(epoch)
        train_celoss,MAXV = train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, args)
        if utils.is_main_process():
            logging.info('[INFO] Train: epoch={}, lr={:.4f}, ce_loss={:.4f}'.format(epoch, optimizer.param_groups[0]["lr"], train_celoss))
        lr_scheduler.step()
        if utils.is_main_process():
            print('finished train_one_epoch at epoch={} MAXV={}'.format(epoch, MAXV))
            if epoch == 0 or epoch == args.epochs -1 or epoch % 10 == 0:
                checkpoint_path = os.path.join(args.out_dir, 'checkpoint')
                if not os.path.exists(checkpoint_path):
                    os.mkdir(checkpoint_path)
                save_paths = [os.path.join(checkpoint_path, 'epoch{}.pth'.format(epoch)), os.path.join(checkpoint_path, 'checkpoint.pth')]
                for save_path in save_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                    }, save_path)
                print('finished save checkpoints at epoch={}'.format(epoch))
    total_time = (time.time() - time_start)/3600
    print('Train finished, total_train_time={}h'.format(total_time))


if __name__ == "__main__":
    parser = arg_parse()
    args = parser.parse_args()
    main(args)
    os._exit(0)