import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pickle as pkl 
import json
import pandas as pd 
import numpy as np 
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.distributed as dist
from functools import partial
from torch.utils.data import DistributedSampler
import random
from torch._six import container_abcs, string_classes, int_classes
import re
import argparse
import os 
import utils_


class ReIDDataset(Dataset):
    def __init__(self, seq_paths, seq_names, train=True, feat_path='reid_posenc_feat_merge_dim512', shuffle_in_frame=False, shuffle_prob=0, warmup_epochs=-1):
        super().__init__()
        assert len(seq_paths) == len(seq_names)
        self.seq_paths = seq_paths  # reid, pos_enc
        self.seq_names = seq_names
        self.seqname_frameid = self.create_idxs()
        self.seqname2seq_paths={seqname:os.path.join(seq_path, feat_path)
            for seqname,seq_path in zip(self.seq_names, self.seq_paths)}
        self.train = train
        self.shuffle_in_frame = shuffle_in_frame
        self.shuffle_prob = shuffle_prob

    def create_idxs(self,):
        seqname_frameid = []
        for seq_name, seq_path in zip(self.seq_names, self.seq_paths):
            frames = os.listdir(os.path.join(seq_path, 'img1'))
            frames = [int(frame.split('.')[0]) for frame in frames]
            frames_assert=np.array(list(range(1,len(frames)+1)))
            assert np.all(frames_assert==np.array(frames)), '{} vs {}'.format(frames_assert, frames)
            seqname_frameid.extend([(seq_name, frameid) for frameid in frames[:-1]])  # two adj frame is used for train/test
        return seqname_frameid

    def __len__(self,):
        return len(self.seqname_frameid)

    def match_label(self, detection_id_frames):
        detection_id_pre, detection_id_cur = detection_id_frames  # [n], [m]
        match_mask = detection_id_cur[:,None]==detection_id_pre[None,:]  # [m,n]
        row_matched_idxs, col_matched_idxs = np.nonzero(match_mask)
        assert len(row_matched_idxs) == len(set(row_matched_idxs))
        if len(row_matched_idxs) > 0:
            assert row_matched_idxs.max()< detection_id_cur.shape[0] and col_matched_idxs.max() < detection_id_pre.shape[0]
        labels = {rowid: colid+1 for rowid,colid in zip(row_matched_idxs,col_matched_idxs)}  # NOTE 0 for un-matched (bg)
        nb_det_cur_frame = detection_id_cur.shape[0]
        un_matched_idxs = np.array([idx for idx in range(nb_det_cur_frame) if idx not in row_matched_idxs])
        labels.update({rowid:0 for rowid in un_matched_idxs})
        labels = OrderedDict(sorted(labels.items(), key=lambda x: x[0]))
        labels = labels.values()
        labels = np.array([label for label in labels]).astype(np.int64)
        assert labels.shape[0] == nb_det_cur_frame
        return labels
    
    def __getitem__(self, idx):
        seqname, frameid = self.seqname_frameid[idx]
        feats = dict()
        reid_pos_enc_frames=[]
        detection_id_frames = []
        reid_feat_frams = []
        for id_ in [frameid, frameid+1]:
            reid_posenc_info_path = os.path.join(self.seqname2seq_paths[seqname], '{}.pkl'.format(id_))
            reid_posenc_info = pd.read_pickle(reid_posenc_info_path)  
            reid_posenc_info = reid_posenc_info.drop(columns=['frameid'])  # detection_id  reid_pos_enc
            detection_id, reid_pos_enc = reid_posenc_info['detection_id'].values, reid_posenc_info['reid_pos_enc'].values
            reid_feat = reid_posenc_info['reid_feat'].values
            assert isinstance(detection_id, np.ndarray) and isinstance(reid_pos_enc, np.ndarray)
            assert isinstance(reid_feat, np.ndarray) 
            assert len(detection_id) == len(reid_pos_enc) == len(reid_feat)
            assert len(detection_id.shape)==1 and len(reid_pos_enc.shape)==1 and len(reid_feat.shape)==1
            assert np.all(np.unique(detection_id)==np.array(list(sorted(detection_id)))), '{} vs {}'.format(np.unique(detection_id), detection_id)
            # shuffle bboxes in one frame
            if self.shuffle_in_frame:
                prob = np.random.uniform(0., 1.)
                if prob >= self.shuffle_prob:
                    shuffled_idx = np.random.permutation(len(detection_id))
                    detection_id, reid_pos_enc, reid_feat = detection_id[shuffled_idx], reid_pos_enc[shuffled_idx], reid_feat[shuffled_idx]

            detection_id_frames.append(detection_id)
            reid_pos_enc = np.array([pos_enc for pos_enc in reid_pos_enc]).astype(np.float32)  # [n, num_dim-4+4]
            reid_feat = np.array([feat for feat in reid_feat])  # [n, num_dim]
            reid_pos_enc_frames.append(reid_pos_enc)
            reid_feat_frams.append(reid_feat)
        if self.train:
            feats['label'] = self.match_label(detection_id_frames)
            assert feats['label'].shape[0] == reid_pos_enc_frames[1].shape[0]
        feats['reid_pos_enc_frame_pre'], feats['reid_pos_enc_frame_cur'] = reid_pos_enc_frames[0], reid_pos_enc_frames[1]
        feats['reid_feat_frame_pre'], feats['reid_feat_frame_cur'] = reid_feat_frams[0], reid_feat_frams[1]
        feats['nbdet_valid_frame_pre'] = reid_pos_enc_frames[0].shape[0]
        feats['nbdet_valid_frame_cur'] = reid_pos_enc_frames[1].shape[0]
        feats['mask_valid_frame_pre'] = np.ones((reid_pos_enc_frames[0].shape[0]), dtype=reid_pos_enc_frames[0].dtype)
        feats['mask_valid_frame_cur'] = np.ones((reid_pos_enc_frames[1].shape[0]), dtype=reid_pos_enc_frames[1].dtype)
        if not self.train:
            feats['seqname'] = seqname
            feats['frameid'] = frameid
            feats['detection_id_pre'], feats['detection_id_cur'] = detection_id_frames[0], detection_id_frames[1]
        return feats

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(dataset, imgs_per_gpu, dist=True, seed=None, workers_per_gpu=2, **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        raise NotImplementedError

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(default_collate),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)
    return data_loader

def default_collate(batch):
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    np_str_obj_array_pattern = re.compile(r'[SaUO]')

    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            max_nb_dim0 = np.max([elem.shape[0] for elem in batch])
            batch_pad = [np.concatenate((elem, np.zeros(tuple([max_nb_dim0-elem.shape[0]]+list(elem.shape[1:])), 
                dtype=elem.dtype)), axis=0) for elem in batch]
            return default_collate([torch.as_tensor(b) for b in batch_pad])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
        # raise ValueError
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        raise NotImplementedError
    elif isinstance(elem, container_abcs.Sequence):
        raise ValueError
    raise TypeError(default_collate_err_msg_format.format(elem_type))



def init_distributed_mode():
    args = argparse.ArgumentParser('', add_help=False)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


if __name__ == "__main__":
    root='/mnt/truenas/scratch/lqf/data/'
    mot17_root=root+'/mot/MOT17/train/'
    mot15_root=root+'/mot/2DMOT2015/train/'
    mot17_data=['MOT17-02-FRCNN','MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-13-FRCNN']
    mot15_data=['KITTI-17','ETH-Sunnyday','ETH-Bahnhof','PETS09-S2L1','TUD-Stadtmitte']
    mot17_datapath_list=[mot17_root+dataname for dataname in mot17_data]
    mot15_datapath_list=[mot15_root+dataname for dataname in mot15_data]
    datanames = mot17_data + mot15_data

    init_distributed_mode()
    seed = 0 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # train
    dataset = ReIDDataset(seq_paths=mot17_datapath_list+mot15_datapath_list, seq_names=datanames)

    # # eval
    # seq_names = ["MOT17-11-FRCNN", "MOT17-04-FRCNN"]
    # seq_paths = [mot17_root + name for name in seq_names]
    # dataset = ReIDDataset(seq_paths=seq_paths[:1], seq_names=seq_names[:1], feat_path='eval_reid_posenc_feat_merge_dim512')

    print('len dataset: ', len(dataset))
    dataloader = build_dataloader(dataset, 2, dist=True, seed=None, workers_per_gpu=2)
    epochs = 2
    MAX = -1
    for epoch in range(epochs):
        print('len dataload in epoch: ', len(dataloader), epoch)
        for i,data in enumerate(dataloader):
            label, reid_pos_enc_frame_pre, reid_pos_enc_frame_cur = data['label'], data['reid_pos_enc_frame_pre'], data['reid_pos_enc_frame_cur']
            nb_det_pre, nb_det_cur = data['nbdet_valid_frame_pre'], data['nbdet_valid_frame_cur']
            mask_pre, mask_cur = data['mask_valid_frame_pre'], data['mask_valid_frame_cur']
            MAX = max(max(nb_det_pre.max(), nb_det_cur.max()), MAX)
    print('MAX: ', MAX)   # 18
    os._exit(0)