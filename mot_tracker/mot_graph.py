"""
modified version from official mot_neural_solver.data.mot_graph import MOTGraph
"""
import torch
import  torch.nn.functional as F
import numpy as np
from mot_neural_solver.utils.graph import get_time_valid_conn_ixs
from torch_geometric.data import Data


class Graph(Data):
    """
    This is the class we use to instantiate our graph objects. We inherit from torch_geometric's Data class and add a
    few convenient methods to it, mostly related to changing data types in a single call.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _change_attrs_types(self, attr_change_fn):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        _data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fn: callable function to change a variable's type
        """
        _data_attr_names = [
                           'edge_index',
                           'edge_preds',
                           ]
        for attr_name in _data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def tensor(self):
        self._change_attrs_types(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self._change_attrs_types(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        self._change_attrs_types(attr_change_fn=lambda x: x.cuda())
        return self

    def to(self, device):
        self._change_attrs_types(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.edge_index, torch.Tensor):
            return self.edge_index.device
        return torch.device('cpu')


class MOTGraph(object):
    """
    This the main class we use to create MOT graphs from detection (and possibly ground truth) files. Its main attribute
    is 'graph_obj', which is an instance of the class 'Graph' and serves as input to the tracking model.
    Moreover, each 'MOTGraph' has several additional attributes that provide further information about the detections in
    the subset of frames from which the graph is constructed.
    """
    def __init__(self, seq_det_df, start_frame = None, end_frame = None, ensure_end_is_in = False,
                ):
        self.step_size = None
        self.graph_df, self.frames = self._construct_graph_df(seq_det_df= seq_det_df.copy(),
                                                                  start_frame = start_frame,
                                                                  end_frame = end_frame,
                                                                  ensure_end_is_in=ensure_end_is_in)
        
        print('len matched={} len graph_df={}'.format(len(self.graph_df[self.graph_df['id'].eq(
            self.graph_df['matched_detid_pre'])].copy()), len(self.graph_df)))

    def _construct_graph_df(self, seq_det_df, start_frame, end_frame = None, ensure_end_is_in = False):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.
        Args:
            seq_det_df: DataFrame with scene detections information for a specific data sequence
            start_frame: frame at which the graph starts
            end_frame: (optional) frame at which the graph ends
            ensure_end_is_in: (only if end_frame is given). Bool indicating whether end_frame must be in the graph.

        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
            valid_frames: list of selected frames

        """
        if start_frame is None:
            start_frame = seq_det_df.frame.min()
        if end_frame is not None:
            valid_frames = np.arange(start_frame, end_frame + 1, self.step_size)
            if ensure_end_is_in and (end_frame not in valid_frames):
                valid_frames = valid_frames.tolist() + [end_frame]
        else:
            valid_frames = np.arange(start_frame, seq_det_df.frame.max() + 1, self.step_size)

        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
        graph_df_valid = len(graph_df)
        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)
        assert len(graph_df) == graph_df_valid
        print('len seq_det_df={} len graph_df={} len graph_df_drop={}'.format(len(seq_det_df), graph_df_valid, len(graph_df)))
        return graph_df, sorted(graph_df.frame.unique())

    def _get_edge_ixs(self):
        """
        Constructs graph edges by taking pairs of nodes with valid time connections (not in same frame, not too far
        apart in time) and perhaps taking KNNs according to reid embeddings.
        Args:
            reid_embeddings: torch.tensor with shape (num_nodes, reid_embeds_dim)

        Returns:
            torch.tensor withs shape (2, num_edges)
        """
        index_start = self.graph_df['detection_id'].min()
        edge_ixs, edge_pred_probs = [], []
        frames, matched_detids, pred_probs = self.graph_df['frame'].values, self.graph_df['matched_detid_pre'].values, self.graph_df['pred_probs'].values
        # print(self.graph_df)
        for i,(frameid,matched_detid, pred_prob) in enumerate(zip(frames, matched_detids, pred_probs)):
            if matched_detid < 0:
                continue
            idx = self.graph_df.index[self.graph_df['frame'].eq(frameid-1) & self.graph_df['id'].eq(matched_detid)].tolist()
            if len(idx) > 0:
                assert len(idx) == 1
                idx = idx[0]
                assert idx != i, '{}'.format(i)
                assert self.graph_df['frame'].values[idx] != self.graph_df['frame'].values[i], "{}".format(self.graph_df['frame'].values[idx])
                
                edge_ixs.append((idx, i))
                edge_pred_probs.append(pred_prob)
        
        if len(edge_ixs) > 0:
            edge_ixs = torch.Tensor(edge_ixs).T.to(torch.long)
            edge_pred_probs = torch.Tensor(edge_pred_probs)
        else:
            edge_ixs = torch.empty((2,0)).to(torch.long)
            edge_pred_probs = torch.empty((0))
        return edge_ixs, edge_pred_probs  # [2, num_edges]

    def remove_invalid(self, edge_ixs, edge_preds):
        unique_pdid_pre = edge_ixs[0,:].unique()
        edge_ixs_new, edge_preds_new = [], []
        for pdid in unique_pdid_pre:
            mask = edge_ixs[0,: ]==pdid
            edge_ixs_dump = edge_ixs[:,mask]  # [2, nb_dump]
            edge_preds_dump = edge_preds[mask]  # [nb_dump]
            argmax = edge_preds_dump.argmax(dim=0)
            edge_ixs_new.append(edge_ixs_dump[:,argmax])
            edge_preds_new.append(edge_preds_dump[argmax])
        return torch.stack(edge_ixs_new, dim=1), torch.stack(edge_preds_new, dim=0)  # [2, nb], [nb]

    def construct_graph_object(self):
        edge_ixs, edge_pred_probs = self._get_edge_ixs()
        # edge_ixs, edge_pred_probs = self.remove_invalid(edge_ixs, edge_pred_probs)  # NOTE constrain the in/out degree of each node drops the mota/idf1 performance seriously

        self.graph_obj = Graph(
            edge_index = edge_ixs,
            edge_preds = edge_pred_probs
        )  
        self.graph_obj.to('cpu')
        self.graph_obj.num_nodes = len(self.graph_df)