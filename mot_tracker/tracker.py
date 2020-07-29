import numpy as np 
import os 
import pickle as pkl 
import time 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from projectors import ExactProjector
import pandas as pd 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from mot_graph_dataset import MOTGraphDataset
from utils import compute_mot_metrics


# VIDEO_COLUMNS = ['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right', 'bb_bot']
VIDEO_COLUMNS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']


class Tracker:
    def __init__(self, dataset, eval_params = None,):  # dataset_params=None
        self.dataset = dataset
        self.eval_params = eval_params
        if self.eval_params is None:
            self.eval_params = dict()
            self.eval_params['rounding_method'] = 'exact'
            self.eval_params['solver_backend'] = 'pulp'

    def _load_full_seq_graph_object(self, seq_name):
        """
        Loads a MOTGraph object corresponding to the entire sequence.
        """
        start_frame = self.dataset.seq_det_dfs[seq_name].frame.min()
        end_frame = self.dataset.seq_det_dfs[seq_name].frame.max()

        full_graph = self.dataset.get_from_frame_and_seq(seq_name=seq_name,
                                                         start_frame=start_frame,
                                                         end_frame=end_frame,
                                                         return_full_object=True,
                                                         ensure_end_is_in=True,
                                                        )
        return full_graph

    def _project_graph_model_output(self):
        """
        Rounds MPN predictions either via Linear Programming or a greedy heuristic
        """
        if self.eval_params['rounding_method'] == 'greedy':
            raise NotImplementedError
        elif self.eval_params['rounding_method'] == 'exact':
            projector = ExactProjector(self.full_graph, solver_backend=self.eval_params['solver_backend'])
        else:
            raise RuntimeError("Rounding type for projector not understood")
        projector.project()
        self.full_graph.graph_obj = projector.final_graph.numpy()
        self.full_graph.constr_satisf_rate = projector.constr_satisf_rate

    def _assign_ped_ids(self):
        """
        Assigns pedestrian Ids to each detection in the sequence, by determining all connected components in the graph
        """
        # Only keep the non-zero edges and Express the result as a CSR matrix so that it can be fed to 'connected_components')
        nonzero_mask = self.full_graph.graph_obj.edge_preds == 1
        # nonzero_mask = self.full_graph.graph_obj.edge_preds > 0
        assert nonzero_mask.shape[0] == len(self.full_graph.graph_obj.edge_preds)
        graph_shape = (self.full_graph.graph_obj.num_nodes, self.full_graph.graph_obj.num_nodes)
        csr_graph = csr_matrix((self.full_graph.graph_obj.edge_preds.astype(int), (tuple(self.full_graph.graph_obj.edge_index))), 
            shape=graph_shape)  # 2-D array with graph_shape

        # Get the connected Components:
        n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
        assert len(labels) == self.full_graph.graph_df.shape[0], "Ped Ids Label format is wrong"

        # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
        self.final_projected_output = self.full_graph.graph_df.copy()
        self.final_projected_output['ped_id'] = labels
        self.final_projected_output = self.final_projected_output[VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

    def track(self, seq_name):
        self.full_graph = self._load_full_seq_graph_object(seq_name)
        self._project_graph_model_output()
        self._assign_ped_ids()

        self.tracking_out = self.final_projected_output.copy()
        return self.tracking_out

    def save_results_to_file(self, output_file_path, seq_name):
        """
        Stores the tracking result to a txt file, in MOTChallenge format.
        """
        self.tracking_out['conf'] = 1
        self.tracking_out['x'] = -1
        self.tracking_out['y'] = -1
        self.tracking_out['z'] = -1
        self.tracking_out['bb_left'] += 1 # Indexing is 1-based in the ground truth
        self.tracking_out['bb_top'] += 1
        final_out = self.tracking_out[TRACKING_OUT_COLS].sort_values(by=['frame', 'ped_id'])
        if not os.path.exists(output_file_path):
            os.mkdir(output_file_path)
        final_out.to_csv(os.path.join(output_file_path, '{}.txt'.format(seq_name)), header=False, index=False)

    def track_all_sequences(self, seq_names, out_dir):
        constr_satisf_rate = pd.Series(dtype=float)
        for seq_name in seq_names:
            self.track(seq_name)
            constr_satisf_rate[seq_name] = tracker.full_graph.constr_satisf_rate
            tracker.save_results_to_file(out_dir, seq_name)
            print('track and save for seq_name={} finished.'.format(seq_name))
        constr_satisf_rate['OVERALL'] = constr_satisf_rate.mean()
        return constr_satisf_rate
        
    def metrics_eval(self, constr_satisf_rate, out_files_dir, seq_names):
        mot_metrics_to_log=['mota', 'norm_mota', 'idf1', 'norm_idf1', 'num_switches', 'num_misses', 'num_false_positives', 'num_fragmentations', 'constr_sr']
        mot_metrics_summary = compute_mot_metrics(gt_path=os.path.join('/mnt/truenas/scratch/lqf/data/jiawei/data/', 'MOT_eval_gt'),
                                                out_mot_files_path=out_files_dir,
                                                seqs=seq_names,
                                                print_results = False)
        mot_metrics_summary['constr_sr'] = constr_satisf_rate
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'expand_frame_repr', False):
            cols = [col for col in mot_metrics_summary.columns if col in mot_metrics_to_log]
            print("\n" + str(mot_metrics_summary[cols]))

if __name__ == "__main__":
    DATA_PATH = '/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
    splits = ['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
    splits = splits[:-1]
    dataset = MOTGraphDataset(DATA_PATH, splits)
    dataset.merge_for_edge(edge_info_path = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_evaldelete_normalized/', subv=141)
    # tracker
    tracker = Tracker(dataset)
    constr_satisf_rate = tracker.track_all_sequences(splits, out_dir=DATA_PATH+'res_delete4_normalized/')
    print('START eval ++++++ ')
    ts = time.time()
    tracker.metrics_eval(constr_satisf_rate, out_files_dir=DATA_PATH+'res_delete4_normalized/', seq_names=splits)
    print('finished metrics_eval, time={}min'.format((time.time()-ts)/60))


# # for train debug
# if __name__ == "__main__":
#     DATA_PATH = '/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
#     # splits = ['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
#     # splits = splits[:-1]
#     splits = ['MOT17-09-FRCNN']
#     dataset = MOTGraphDataset(DATA_PATH, splits)
#     dataset.merge_for_edge(edge_info_path = '/mnt/truenas/scratch/lqf/code/deep-person-reid/reid2bbox/exp_evaldelete_train/', subv=239)

#     # tracker
#     tracker = Tracker(dataset)
#     constr_satisf_rate = tracker.track_all_sequences(splits, out_dir=DATA_PATH+'res_delete_train/')
#     print('START eval ++++++ ')
#     ts = time.time()
#     tracker.metrics_eval(constr_satisf_rate, out_files_dir=DATA_PATH+'res_delete_train/', seq_names=splits)
#     print('finished metrics_eval, time={}min'.format((time.time()-ts)/60))
