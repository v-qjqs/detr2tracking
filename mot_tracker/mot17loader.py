import os.path as osp 
import pandas as pd 
import pickle as pkl 
import os 
import numpy as np 
import time 
import configparser


DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')


def get_mot17_det_df_from_gt(seq_name, data_root_path):
    detections_file_path = osp.join(data_root_path, seq_name, f"gt/gt.txt")
    det_df = pd.read_csv(detections_file_path, header=None)

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES

    det_df['bb_left'] -= 1 # Coordinates are 1 based
    det_df['bb_top'] -= 1

    det_df = det_df[det_df['label'].isin([1, 2])].copy()
    det_df = det_df[det_df['conf'].eq(1)].copy()
    det_df = det_df[det_df['vis'].ge(0.2)].copy()

    seq_info_dict = _build_scene_info_dict_mot17(seq_name, data_root_path)

    # Correct the detections file name to contain the 'gt' as name
    seq_info_dict['det_file_name'] = 'gt'
    seq_info_dict['is_gt'] = True
    return det_df, seq_info_dict, None

def _build_scene_info_dict_mot17(seq_name, data_root_path):
    info_file_path = osp.join(data_root_path, seq_name, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)
    seq_info_dict = {'seq': seq_name,
                     'seq_path': osp.join(data_root_path, seq_name),
                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),
                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'has_gt': osp.exists(osp.join(data_root_path, seq_name, 'gt'))
                    }
    return seq_info_dict