import numpy as np
import torch
from torch_scatter import scatter_add
from mot_neural_solver.data.mot_graph import Graph
from mot_neural_solver.utils.evaluation import  compute_constr_satisfaction_rate
import pulp as plp
from mot_neural_solver.tracker.projectors import PuLPMinCostFlowSolver


class ExactProjector:
    """
    Constructs a Subgraph with all nodes in a graph that are involved in a violated constraint
    (e.g. their incoming / outgoing flow is >1), and then rounds solutions with a MCF Linear Program.
    the full approach is explained in https://arxiv.org/pdf/1912.07515.pdf, Appending B.2
    """
    def __init__(self, full_graph, solver_backend = 'pulp'):
        self.final_graph = full_graph.graph_obj
        self.solver_backend = solver_backend

    def project(self):
        round_preds = (self.final_graph.edge_preds > 0).float()
        self.constr_satisf_rate, flow_in, flow_out = compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        self.final_graph.edge_preds = round_preds.cpu().numpy()
        return

        # Concat all violated_constraint info
        nodes_mask = (flow_in > 1) | (flow_out >1)
        edges_mask = nodes_mask[self.final_graph.edge_index[0]] | nodes_mask[self.final_graph.edge_index[1]]
        if edges_mask.sum() > 0:
            graph_to_project = Graph()
            graph_to_project.edge_preds = self.final_graph.edge_preds[edges_mask]
            graph_to_project.edge_index = self.final_graph.edge_index.T[edges_mask].T

            if self.solver_backend == 'gurobi':
                raise Exception('Uncomment gurobi code to run gorubi solver')
            else:
                mcf_solver = PuLPMinCostFlowSolver(graph_to_project.numpy())
            mcf_solver.solve()
            # Assign the right values to the original graph's predictions
            self.final_graph.edge_preds = self.final_graph.edge_preds.cpu().numpy()
            edges_mask = edges_mask.cpu().numpy()
            self.final_graph.edge_preds[~edges_mask] = round_preds[~edges_mask].cpu().numpy()
            self.final_graph.edge_preds[edges_mask] = graph_to_project.edge_preds