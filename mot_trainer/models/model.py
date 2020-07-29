import torch
import torch.nn as nn
import torch.nn.functional as F 
from .transformer import build_transformer


class TrackingModel(nn.Module):
    def __init__(self, transformer, num_classes, num_queries=None):
        super().__init__()
        self.transformer = transformer
        self.num_classes = num_classes  # NOTE num_classes *include* bg which is indexed at 0
        hid_dim = self.transformer.embed_dim
        self.match_embed = nn.Linear(hid_dim, num_classes)

    def forward(self, reid_feat_pre, reid_feat_cur, mask_pre, mask_cur, reid_pos_enc_pre, reid_pos_enc_cur, train=True):
        self.train() if train else self.eval()
        hidden_state = self.transformer(reid_feat_pre_frame=reid_feat_pre, src_key_padding_mask=mask_pre, pos_embed=reid_pos_enc_pre, 
            reid_feat_cur_frame=reid_feat_cur, tgt_key_padding_mask=mask_cur, query_pos=reid_pos_enc_cur)  # [nb_step,bs,max_nb2,embed_dim]
        hidden_state = hidden_state[-1]  # last step [bs, max_nb2, embed_dim]
        return self.match_embed(hidden_state)  # [bs, max_nb2, nb_classes],  [bs]

    def match_label(self, preds, nbdet_valid_cur, nbdet_valid_pre):
        # preds: [bs, max_nb2, nb_classes], nbdet_valid_cur:[bs]
        assert preds.size(0) == 1
        nbdet_valid_cur, nbdet_valid_pre = nbdet_valid_cur.squeeze(0), nbdet_valid_pre.squeeze(0)
        preds = preds.squeeze(0)[:nbdet_valid_cur]  # [nbdet_valid_cur, nb_classes]
        labels = preds.argmax(1)  # [nbdet_valid_cur]
        preds_probs = preds.softmax(-1)
        preds_probs = preds_probs[torch.arange(preds_probs.size(0)), labels]  # [nbdet_valid_cur]
        assert preds_probs.size(0) == labels.size(0)
        mask = labels.gt(0) & labels.le(nbdet_valid_pre)  
        labels -= 1
        labels[~mask] = -1  # bg and another
        preds_probs[~mask] = -1
        return labels.cpu().numpy(), preds_probs.cpu().detach().numpy()  # labels: [nbdet_valid_cur], 0-based

    def match_label_eval(self, preds, nbdet_valid_cur, nbdet_valid_pre, max_nb_class=19):
        nbdet_valid_cur, nbdet_valid_pre = nbdet_valid_cur.squeeze(0), nbdet_valid_pre.squeeze(0)
        preds = preds.squeeze(0)[:nbdet_valid_cur][:,:max_nb_class]
        labels = preds.argmax(1)
        preds_probs = preds.softmax(-1)
        preds_probs = preds_probs[torch.arange(preds_probs.size(0)), labels]
        mask = labels.gt(0) & labels.le(nbdet_valid_pre)
        labels -= 1
        labels[~mask] = -1
        preds_probs[~mask] = -1
        return labels.cpu().numpy(), preds_probs.cpu().detach().numpy()

def build_model(args):
    transformer_model = build_transformer(args)
    return TrackingModel(transformer_model, args.num_classes)