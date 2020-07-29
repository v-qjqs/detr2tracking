import numpy as np 
import pickle as pkl 
import time
import os 
import json
from pos_encoding import dataname2dataid, datanames, mot17_datapath_list, mot15_datapath_list
import pandas as pd 
from collections import OrderedDict


def posfeat_reidfeat_merge_per_sequence(reidfeat_all_seq, seq_name, seq_path, out_dir, pos_enc_path='pos_encoding_dim512', dataname2dataid=dataname2dataid):
    allkeys = reidfeat_all_seq.keys()
    dataid = dataname2dataid[seq_name]
    seqi_keys = [key for key in allkeys if int(key.split('f')[-1][:3])==dataid]
    frames_seqi = [int(name.split('f')[-1][3:]) for name in seqi_keys]
    frames = os.listdir(os.path.join(seq_path, pos_enc_path))
    frames = [int(name.split('.pkl')[0]) for name in frames if '.pkl' in name and 'save' not in name]
    assert len(set(frames).difference(set(frames_seqi))) == 0, '{} vs {}'.format(frames, frames_seqi)
    assert len(set(frames_seqi).difference(set(frames))) == 0
    out_dir = os.path.join(seq_path, out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    def merge_per_seq_per_frame(frameid):
        framei_keys = [key for key,framei in zip(seqi_keys, frames_seqi) if framei == frameid]
        # pos_enc 
        pos_enc = pd.read_pickle(os.path.join(seq_path, pos_enc_path, '{}.pkl'.format(frameid)))
        assert len(pos_enc) == len(framei_keys), '{} vs {}'.format(len(pos_enc), len(framei_keys))
        detection_id = [int(name.split('_c')[0]) for name in framei_keys]
        detection_id_from_pos_enc = pos_enc['detection_id'].values
        assert len(set(detection_id_from_pos_enc).difference(set(detection_id))) == 0, '{} vs {}'.format(set(detection_id_from_pos_enc), set(detection_id))
        assert len(set(detection_id).difference(set(detection_id_from_pos_enc))) == 0
        reid_feats = {'{}_{}'.format(frameid, det_id):reidfeat_all_seq[k] for k,det_id in zip(framei_keys, detection_id)}
        for feat in reid_feats.values():
            assert isinstance(feat, np.ndarray) and feat.dtype == np.float32, '{}'.format(type(feat))
        reid_feats = OrderedDict({'{}_{}'.format(frameid, k):reid_feats['{}_{}'.format(frameid, k)] for k in detection_id_from_pos_enc})
        assert isinstance(reid_feats, OrderedDict)
        reid_feats = reid_feats.values()
        pos_enc['reid_feat'] = reid_feats  # NOTE merge
        # save
        det_df_path = os.path.join(out_dir, '{}.pkl'.format(frameid))
        pos_enc.to_pickle(det_df_path)
        del detection_id
    print('len frame_seqi: ', len(frames_seqi))
    for frameid in frames_seqi:
        merge_per_seq_per_frame(frameid)

def merge_all_sequences(reidfeat_all_seq, seq_name_list, seq_path_list, out_dir, pos_enc_path='pos_encoding_dim512', dataname2dataid=dataname2dataid):
    assert len(seq_name_list) == len(seq_path_list)
    for seq_name, seq_path in zip(seq_name_list, seq_path_list):
        posfeat_reidfeat_merge_per_sequence(reidfeat_all_seq, seq_name, seq_path, out_dir, pos_enc_path=pos_enc_path, dataname2dataid=dataname2dataid)
        print('merge dataset {} finished.'.format(seq_name))


if __name__ == "__main__":
    root = '/mnt/truenas/scratch/lqf/code/deep-person-reid/log/resnet50-512-619freeze_my/'
    # # train
    # reidfeat_all_seq_path = root + 'reid_feat/reid_feat.pkl'
    # with open(reidfeat_all_seq_path, "rb") as f:
    #     reidfeat_all_seq = pkl.load(f)
    # merge_all_sequences(reidfeat_all_seq, datanames, mot17_datapath_list+mot15_datapath_list, 
    #                     out_dir='reid_posenc_feat_merge_dim512_normalized')

    # eval
    reidfeat_all_seq = dict()
    reidfeat_all_seq_path = [root + 'query_reid_feat/reid_feat.pkl', root + 'gallery_reid_feat/reid_feat.pkl']
    for path in reidfeat_all_seq_path:
        with open(path, 'rb') as f:
            reidfeat_all_seq_ = pkl.load(f)
            assert isinstance(reidfeat_all_seq_, dict)
            reidfeat_all_seq.update(reidfeat_all_seq_)
    datasets = ["MOT17-11-FRCNN", "MOT17-04-FRCNN"]
    tmp_root = '/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
    eval_datapath_list = [tmp_root+data for data in datasets]
    eval_dataname2dataid = {name:i+1 for i,name in enumerate(datasets)}
    merge_all_sequences(reidfeat_all_seq, datasets, eval_datapath_list, 
                        out_dir='eval_reid_posenc_feat_merge_dim512_normalized', pos_enc_path='eval_pos_encoding_dim512', dataname2dataid=eval_dataname2dataid)
