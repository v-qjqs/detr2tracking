"""
modified version from official MOTGraphDataset
"""
import os.path as osp
import numpy as np
import pandas as pd
from mot_neural_solver.path_cfg import  DATA_PATH
from mot_neural_solver.data.splits import _SPLITS
from mot_graph import MOTGraph
from seq_processor import MOTSeqProcessor
import random
import os 


class MOTGraphDataset:
    """
    Main Dataset Class. It is used to sample graphs from a a set of MOT sequences by instantiating MOTGraph objects.
    It is used both for sampling small graphs for training, as well as for loading entire sequence's graphs
    for testing.
    Its main method is 'get_from_frame_and_seq', where given sequence name and a starting frame position, a graph is
    returned.
    """
    def __init__(self, DATA_PATH, splits, ):
        self.DATA_PATH = DATA_PATH
        seqs_to_retrieve = {self.DATA_PATH: [seq_name for seq_name in splits]}
        # Load all dataframes containing detection information in each sequence of the dataset
        self.seq_det_dfs, self.seq_info_dicts, self.seq_names = self._load_seq_dfs(seqs_to_retrieve)

    def _load_seq_dfs(self, seqs_to_retrieve):
        """
        Loads all the detections dataframes corresponding to the seq_names that constitute the dataset
        Args:
            seqs_to_retrieve: dictionary of pairs (dataset_path: seq_list), where each seq_list is a set of
             sequence names to include in the dataset.
        Returns:
            seq_det_dfs: dictionary of Dataframes of detections corresponding to each sequence in the dataset
            seq_info_dicts: dictionary of dictionarys with metainfo for each sequence
            seq_names: a list of names of all sequences in the dataset
        """
        seq_names = []
        seq_info_dicts = {}
        seq_det_dfs = {}
        for dataset_path, seq_list in seqs_to_retrieve.items():
            for seq_name in seq_list:
                seq_processor = MOTSeqProcessor(dataset_path=dataset_path, seq_name=seq_name,)
                seq_det_df = seq_processor.load_or_process_detections()

                seq_names.append(seq_name)
                seq_info_dicts[seq_name] = seq_det_df.seq_info_dict
                seq_det_dfs[seq_name] = seq_det_df

        return seq_det_dfs, seq_info_dicts, seq_names

    # NOTE merge_for_edge should be called before this func is called.
    def get_from_frame_and_seq(self, seq_name, start_frame = None, end_frame = None, ensure_end_is_in = False,  # inference_mode =False, max_frame_dist,
                               return_full_object = False, ):
        """
        Method behind __getitem__ method. We load a graph object of the given sequence name, starting at 'start_frame'.

        Args:
            seq_name: string indicating which scene to get the graph from
            start_frame: int indicating frame num at which the graph should start
            end_frame: int indicating frame num at which the graph should end (optional)
            ensure_end_is_in: bool indicating whether end_frame needs to be in the graph
            return_full_object: bool indicating whether we need the whole MOTGraph object or only its Graph object
                                (Graph Network's input)

        Returns:
            mot_graph: output MOTGraph object or Graph object, depending on whethter return_full_object == True or not

        """
        seq_det_df = self.seq_det_dfs[seq_name]
        mot_graph = MOTGraph(
                             seq_det_df=seq_det_df,
                             start_frame=start_frame,
                             end_frame=end_frame,
                             ensure_end_is_in=ensure_end_is_in,
                            )
        mot_graph.construct_graph_object()
        if return_full_object:
            return mot_graph
        else:
            return mot_graph.graph_obj
    
    def merge_for_edge(self, edge_info_path = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_evaldelete/', subv=0):
        for seq_name in self.seq_names:
            det_df = self.seq_det_dfs[seq_name]
            if subv > 0:
                edge_info = pd.read_pickle(os.path.join(edge_info_path, '{}_new_subv{}.pkl'.format(seq_name, subv)))
            else:
                edge_info = pd.read_pickle(os.path.join(edge_info_path, '{}_new.pkl'.format(seq_name)))
            
            edge_info_ = edge_info.rename(columns={'frameid_cur':'frame', 'detection_id_cur':'id'}, inplace=False)
            pd_merged = det_df.merge(edge_info_, how='left', )
            pd_merged_remove_nan = pd_merged[~pd_merged['matched_detid_pre'].isnull()].copy() 
            pd_merged_remove_nan['matched_detid_pre'] = pd_merged_remove_nan['matched_detid_pre'].values.astype(np.int32)
            assert len(pd_merged_remove_nan) == len(edge_info_), "{} vs {}".format(len(pd_merged_remove_nan), len(edge_info_))
            self.seq_det_dfs[seq_name] = pd_merged_remove_nan


if __name__ == "__main__":
    DATA_PATH = '/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
    splits = ['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
    dataset = MOTGraphDataset(DATA_PATH, splits)
    dataset.merge_for_edge()
    for seq_name in splits:
        dataset.get_from_frame_and_seq(seq_name, return_full_object=False)  
        print('seq_name={} finished'.format(seq_name))