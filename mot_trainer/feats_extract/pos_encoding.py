import pandas as pd 
import pickle as pkl 
import json
import os
import numpy as np 
import math 
import cv2
from skimage import measure
import time
import configparser
import copy


# # train
root='/mnt/truenas/scratch/lqf/data/'
mot17_root=root+'/mot/MOT17/train/'
mot15_root=root+'/mot/2DMOT2015/train/'
reid2gtbbox_path = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/reid2gtbbox_traindata.pkl'
mot17_data=['MOT17-02-FRCNN','MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-13-FRCNN']  # NOTE order is important
mot15_data=['KITTI-17','ETH-Sunnyday','ETH-Bahnhof','PETS09-S2L1','TUD-Stadtmitte']
datanames = mot17_data + mot15_data
dataname2dataid={name:i+1 for i,name in enumerate(datanames)}
mot17_datapath_list=[mot17_root+dataname for dataname in mot17_data]
mot15_datapath_list=[mot15_root+dataname for dataname in mot15_data]


def pos_encoding(img_size, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
    if scale is None:
        scale = 2 * math.pi
    h,w = img_size
    mask = np.ones((h,w),dtype=np.float32)
    y_embed, x_embed = np.cumsum(mask, axis=0).astype(np.float32), np.cumsum(mask, axis=1).astype(np.float32)  # (h,w)
    if normalize:
        y_embed, x_embed = y_embed / h * scale, x_embed/w*scale
    dim_t = np.arange(num_pos_feats).astype(np.float32)
    dim_t = temperature ** (2*(dim_t//2)/num_pos_feats)
    pos_x, pos_y = x_embed[:,:,None] / dim_t, y_embed[:,:,None]/dim_t   # [h,w,num_pos_feats]
    pos_x = np.stack([np.sin(pos_x[:,:,0::2]), np.cos(pos_x[:,:,1::2])], axis=3).reshape(h,w,num_pos_feats)  # [h,w,num_pos_feats]
    pos_y = np.stack([np.sin(pos_y[:,:,0::2]), np.cos(pos_y[:,:,1::2])], axis=3).reshape(h,w,num_pos_feats)
    pos = np.concatenate((pos_y, pos_x),axis=2).astype(np.float32)  # [h,w,num_pos_feats*2]
    return (pos, img_size[:2])

# per sequence
def get_pos_encoding_per_sequence(dataroot, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
    seqinfo_path = os.path.join(dataroot, 'seqinfo.ini')
    dataroot = os.path.join(dataroot, 'img1')
    filenames = os.listdir(dataroot)
    frames = [int(filename.split('.')[0]) for filename in filenames]
    
    # print('seqinfo_path: ', seqinfo_path)
    if os.path.exists(seqinfo_path):
        config_ = configparser.ConfigParser()
        config_.read(seqinfo_path)
        img_h, img_w = int(config_.get('Sequence', 'imHeight')), int(config_.get('Sequence', 'imWidth'))
    else:
        img = cv2.imread(os.path.join(dataroot, filenames[0]))
        img_h, img_w = img.shape[:2]
        img_h, img_w = int(img_h), int(img_w)
        del img
    pos_enc_one_frame = pos_encoding((img_h,img_w), num_pos_feats, temperature, normalize, scale)
    img2pos_enc = {frameid: copy.copy(pos_enc_one_frame) for frameid in frames}
    return img2pos_enc

# per frame
def dict2dataframe(reid2pos_enc, columns_name=['frameid','detection_id','reid_pos_enc']):
    pd_data = pd.DataFrame.from_dict(reid2pos_enc, 'index')
    pd_data.columns = columns_name
    return pd_data

# per frame
def save_pos_encoding(reid2pos_enc, dataroot, frameid, out_dir='pos_encoding'):
    out_dir = os.path.join(dataroot, out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    det_df = dict2dataframe(reid2pos_enc)
    det_df_path = os.path.join(out_dir, '{}.pkl'.format(frameid))
    det_df.to_pickle(det_df_path)

def reid2pos_encoding_per_sequence(dataroot, dataname, dataname2dataid, reid2gtbbox_info, img2pos_enc, out_dir='pos_encoding_dim512'):
    filenames = os.listdir(os.path.join(dataroot, 'img1'))
    frames = [int(filename.split('.')[0]) for filename in filenames]

    # per frame
    def reid2pos_encoding_per_frame(frameid):
        # datasetid:int, frameid: int
        datasetid = dataname2dataid[dataname]
        pos_enc_framei, (img_h, img_w) = img2pos_enc[frameid]  # [h,w,num_pos_feats*2]
        reid2pos_enc = dict()
        for key, v in reid2gtbbox_info.items():
            datasetid_frame_id = key.split('f')[-1].split('.')[0]
            datasetid_, frameid_ = int(datasetid_frame_id[:3]), int(datasetid_frame_id[3:])
            if datasetid_ != datasetid or frameid_ != frameid:
                continue
            (bbox_l,bbox_t,bbox_w,bbox_h), *_ = v
            bbox_l, bbox_t, bbox_w, bbox_h = int(float(bbox_l)), int(float(bbox_t)), int(float(bbox_w)), int(float(bbox_h))  # NOTE
            bbox_l, bbox_t = bbox_l-1, bbox_t-1  # NOTE 1-based in original coords.
            bbox_b, bbox_r = bbox_t+bbox_h, bbox_l+bbox_w
            if bbox_b > pos_enc_framei.shape[0]:
                bbox_b = pos_enc_framei.shape[0]
            if bbox_r > pos_enc_framei.shape[1]:
                bbox_r = pos_enc_framei.shape[1]
            if bbox_l<0:
                bbox_l=0
            if bbox_t<0:
                bbox_t=0
            # assert bbox_t+bbox_h<=pos_enc_framei.shape[0], '{} vs {}'.format(bbox_t+bbox_h, pos_enc_framei.shape[0])  # TODO check
            # assert bbox_l+bbox_w<=pos_enc_framei.shape[1], '{} vs {}'.format(bbox_l+bbox_w, pos_enc_framei.shape[1])
            reid_pos_enc = pos_enc_framei[bbox_t:bbox_b, bbox_l:bbox_r,:]  # [hi,wi,c]
            reid_pos_enc = measure.block_reduce(reid_pos_enc, (reid_pos_enc.shape[0],reid_pos_enc.shape[1],1), np.max)  # [1,1,c], global avg pool
            reid_pos_enc = reid_pos_enc.squeeze((0,1))  # [c]
            coords_enc = np.array([bbox_l/img_w,bbox_t/img_h,bbox_w/img_w,bbox_h/img_h])  # nomorlize
            reid_pos_enc = np.concatenate((coords_enc,reid_pos_enc),axis=0)  # [pos_enc_dim+4]
            try:
                detection_id = int(key.split('_c')[0])
            except ValueError:
                detection_id = int(key.split('_f')[0])
            key_ = '{}_{}'.format(frameid, detection_id)
            assert key_ not in reid2pos_enc
            reid2pos_enc[key_] = (frameid, detection_id, reid_pos_enc)
        return {frameid: reid2pos_enc}
        
    reid2pos_enc_allframes = map(reid2pos_encoding_per_frame, frames)
    ts = time.time()
    cnt = 0 
    cnt_gt={'MOT17-02-FRCNN':600,'MOT17-05-FRCNN':837,'MOT17-09-FRCNN':525,'MOT17-10-FRCNN':654,'MOT17-13-FRCNN':750}
    cnt_gt.update({'KITTI-17':145,'ETH-Sunnyday':354,'ETH-Bahnhof':1000,'PETS09-S2L1':795,'TUD-Stadtmitte':179})
    cnt_gt.update({'MOT17-11-FRCNN':900, 'MOT17-04-FRCNN':1050})

    for reid2pos_enc_perframe in reid2pos_enc_allframes:
        assert len(reid2pos_enc_perframe) == 1
        for frameid, reid2pos_enc in reid2pos_enc_perframe.items():
            cnt +=1
            save_pos_encoding(reid2pos_enc, dataroot, frameid, out_dir=out_dir)
    print('finished reid2pos_enc_allframes saving for data sequence {}, time={}min, cnt={}'.format(dataname, (time.time()-ts)/60, cnt))
    assert cnt == cnt_gt[dataname], '{} vs {}'.format(cnt, cnt_gt[dataname])


def reid2pos_enc_(dataroot_list, dataname_list, reid2gtbbox_info, num_pos_feats=128, temperature=10000, normalize=True, 
                  scale=None, dataname2dataid=dataname2dataid, out_dir='pos_encoding_dim512'):
    assert len(dataroot_list) == len(dataname_list)
    for dataroot, dataname in zip(dataroot_list, dataname_list):
        img2pos_enc = get_pos_encoding_per_sequence(dataroot, num_pos_feats, temperature=temperature, normalize=normalize, scale=scale)
        reid2pos_encoding_per_sequence(dataroot, dataname, dataname2dataid, reid2gtbbox_info, img2pos_enc, out_dir=out_dir)

def load_reid2gtbbox_info(reid2gtbbox_path):
    with open(reid2gtbbox_path, 'rb') as f:
        info = pkl.load(f)
    return info


# # train
# if __name__ == "__main__":
#     reid2gtbbox_info = load_reid2gtbbox_info(reid2gtbbox_path)
#     reid2pos_enc_(mot17_datapath_list+mot15_datapath_list, datanames, reid2gtbbox_info, num_pos_feats=254)  # 128
#     # reid2pos_enc_(mot15_datapath_list, datanames[5:], reid2gtbbox_info, num_pos_feats=128)


# eval
if __name__ == "__main__":
    reid2gtbbox_path_list = [root+'reid/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/reid2gtbbox.pkl', 
                         root+'reid/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/reid2gtbbox_part2.pkl']
    reid2gtbbox_info = dict()
    cnt = 0
    for reid2gtbbox_path in reid2gtbbox_path_list:
        reid2gtbbox_info_ = load_reid2gtbbox_info(reid2gtbbox_path)
        assert isinstance(reid2gtbbox_info_, dict)
        cnt += len(reid2gtbbox_info_)
        reid2gtbbox_info.update(reid2gtbbox_info_)
    assert len(reid2gtbbox_info) == cnt, "{} vs {}".format(reid2gtbbox_info, cnt)
    eval_mot17_data = ['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
    eval_mot17_datapath_list = [os.path.join(mot17_root+dataname) for dataname in eval_mot17_data]
    eval_dataname2dataid = {name:i+1 for i,name in enumerate(eval_mot17_data)}
    reid2pos_enc_(eval_mot17_datapath_list, eval_mot17_data, reid2gtbbox_info, num_pos_feats=254, 
                  dataname2dataid=eval_dataname2dataid, out_dir='eval_pos_encoding_dim512')  # 128