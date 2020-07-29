import torch
import torch.nn as nn
import torch.nn.functional as F 


class CECriterion(nn.Module):
    def __init__(self, num_classes, weights_valid=1., eps=1e-6):
        super().__init__()
        self.num_classes = num_classes  # NOTE num_classes **include** bg, with bg indexed at 0
        self.weights_valid = weights_valid
        self.eps = eps

    def forward(self, preds, targets, batch_nb_valid):
        # pred: [bs, max_nb_det, num_classes], targets:[bs, max_nb_det], batch_nb_valid:[bs](without padding)
        loss_ce = F.cross_entropy(preds.transpose(1,2), targets, weight=None, reduction='none')  #  [bs, max_nb_det]
        weights_valid = torch.zeros_like(loss_ce, dtype=loss_ce.dtype, device=loss_ce.device)
        bs, max_nb_dim = loss_ce.size()
        count = torch.arange(max_nb_dim).unsqueeze(0).repeat(bs, 1).to(loss_ce.device)  # [bs, max_nb_dim]
        batch_valid = batch_nb_valid.unsqueeze(1).repeat(1, max_nb_dim).to(loss_ce.device)  # [bs, max_nb_dim]
        valid_mask = (count < batch_valid).to(loss_ce.device)  # [bs, max_nb_dim]
        weights_valid[valid_mask] = self.weights_valid
        nb_norm = batch_nb_valid.sum()
        loss_ce = (loss_ce * weights_valid).sum()/(nb_norm+self.eps)  # normalized mean scale value
        return dict(loss_ce=loss_ce)