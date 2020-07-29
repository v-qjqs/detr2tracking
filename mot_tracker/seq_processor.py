"""
modified version from official seq_processor
"""
import pandas as pd
import numpy as np
from lapsolver import solve_dense
from mot17loader import get_mot17_det_df_from_gt
from mot_neural_solver.utils.iou import iou
from mot_neural_solver.utils.rgb import BoundingBoxDataset
import os
import os.path as osp
import shutil
import torch
from torch.utils.data import DataLoader


_ENSURE_BOX_IN_FRAME = {'MOT17': False,
                        'MOT17_gt': False,
                        'MOT15': True,
                        'MOT15_gt': False}


class DataFrameWSeqInfo(pd.DataFrame):
    """
    Class used to store each sequences's processed detections as a DataFrame. We just add a metadata atribute to
    pandas DataFrames it so that sequence metainfo such as fps, etc. can be stored in the attribute 'seq_info_dict'.
    This attribute survives serialization.
    This solution was adopted from:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    """
    _metadata = ['seq_info_dict']
    @property
    def _constructor(self):
        return DataFrameWSeqInfo


class MOTSeqProcessor:
    def __init__(self, dataset_path, seq_name, ):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.det_df_loader = get_mot17_det_df_from_gt

    def _get_det_df(self):
        """
        Loads a pd.DataFrame where each row contains a detections bounding box' coordinates information (self.det_df),
        and, if available, a similarly structured pd.DataFrame with ground truth boxes.
        It also adds seq_info_dict as an attribute to self.det_df, containing sequence metainformation (img size,
        fps, whether it has ground truth annotations, etc.)
        """
        self.det_df, seq_info_dict, self.gt_df = self.det_df_loader(self.seq_name, self.dataset_path,)  # self.dataset_params
        assert self.gt_df is None
        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        # Add some additional box measurements that might be used for graph construction
        self.det_df['bb_bot'] = (self.det_df['bb_top'] + self.det_df['bb_height']).values
        self.det_df['bb_right'] = (self.det_df['bb_left'] + self.det_df['bb_width']).values

        # Just a sanity check. Sometimes there are boxes that lay completely outside the frame
        frame_height, frame_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        conds = (self.det_df['bb_width'] > 0) & (self.det_df['bb_height'] > 0)
        conds = conds & (self.det_df['bb_right'] > 0) & (self.det_df['bb_bot'] > 0)
        conds  =  conds & (self.det_df['bb_left'] < frame_width) & (self.det_df['bb_top'] < frame_height)
        self.det_df = self.det_df[conds].copy()

        self.det_df.sort_values(by = 'frame', inplace = True)
        self.det_df['detection_id'] = np.arange(self.det_df.shape[0]) # This id is used for future tastks
        return self.det_df

    def process_detections(self):
        self._get_det_df()
        return self.det_df

    def load_or_process_detections(self):
        seq_det_df = self.process_detections()
        return seq_det_df


if __name__ == "__main__":
    root='/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
    edge_info_path = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_evaldelete/'
    seq_names = ['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
    for seq_name in seq_names:
        seq_precessor = MOTSeqProcessor(dataset_path=root, seq_name=seq_name)
        det_df = seq_precessor.load_or_process_detections()
        if '11' in seq_name:
            edge_info = pd.read_pickle(os.path.join(edge_info_path, '{}_new_subv141.pkl'.format(seq_name)))
        else:
            edge_info = pd.read_pickle(os.path.join(edge_info_path, '{}_new.pkl'.format(seq_name)))
        edge_info_ = edge_info.rename(columns={'frameid_cur':'frame', 'detection_id_cur':'id'}, inplace=False)
        pd_merged = det_df.merge(edge_info_, how='left', )
        nan_before_len = len(pd_merged)
        pd_merged_remove_nan = pd_merged[~pd_merged['matched_detid_pre'].isnull()].copy() 
        assert len(pd_merged_remove_nan) == len(edge_info_), "{} vs {}".format(len(pd_merged_remove_nan), len(edge_info_))
        

# if __name__ == "__main__":
#     root='/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
#     # edge_info_path = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_evaldelete/'
#     seq_names = ['MOT17-02-FRCNN']
#     for seq_name in seq_names:
#         seq_precessor = MOTSeqProcessor(dataset_path=root, seq_name=seq_name)
#         det_df = seq_precessor.load_or_process_detections()