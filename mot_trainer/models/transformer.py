import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Optional, List
from torch import Tensor
import copy


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, dim_feedforward, num_heads, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, reid_feats, pos, src_key_padding_mask=None, src_mask=None):
        # reid_feats:[max_nb_reid, bs, embed_dim], pos:[max_nb_reid, bs, embed_dim], src_key_padding_mask:[bs, max_nb_reid]
        src = reid_feats
        q = k = self.with_pos_embed(src, pos)
        attn_output, _ = self.self_attn(q, k, value=reid_feats, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)  # [max_nb_reid, bs, embed_dim]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, dim_feedforward, num_heads, dropout=0.1, activation="relu"):
        super().__init__()
        self.reid_feat_transform = nn.Linear(embed_dim, embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
        
    def forward(self, idx, reid_feat, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None, tgt_mask=None, memory_mask=None):
        # reid_feat:[max_nb_reid2,bs,embed_dim], memory:[max_nb_reid,bs,embed_dim], tgt_key_padding_mask:[bs,max_nb_reid2]
        # memory_key_padding_mask:[bs,max_nb_reid], pos:[max_nb_reid,bs,embed_dim], query_pos:[max_nb_reid2,bs,embed_dim]
        # tgt_reid_feat: transformed from reid_feat
        tgt = self.reid_feat_transform(reid_feat) if idx==0 else reid_feat  # [max_nb_reid2,bs,embed_dim]
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]  # NOTE key_padding_mask
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory,
                                   attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt)
        return self.norm3(tgt)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src, src_key_padding_mask=None, pos=None, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos, src_key_padding_mask=src_key_padding_mask, src_mask=mask,)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None, tgt_mask=None, memory_mask=None):
        output = tgt
        intermediate = []
        for i, layer in enumerate(self.layers):
            output = layer(i, output, memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, tgt_mask=tgt_mask, memory_mask=memory_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class Transformer(nn.Module):
    def __init__(self, embed_dim, dim_feedforward, num_heads, num_encoder_layers=6, num_decoder_layers=6, return_intermediate=False, 
                 dropout=0.1, activation="relu", normalize_before=False,):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(embed_dim, dim_feedforward, num_heads, dropout, activation)
        encoder_norm = nn.LayerNorm(embed_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(embed_dim, dim_feedforward, num_heads, dropout, activation)
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate)
        self._reset_parameters()
        self.embed_dim = embed_dim

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, reid_feat_pre_frame, src_key_padding_mask, pos_embed, reid_feat_cur_frame, tgt_key_padding_mask, query_pos):
        reid_feat_pre_frame, reid_feat_cur_frame = reid_feat_pre_frame.transpose(1,0), reid_feat_cur_frame.transpose(1,0)
        pos_embed, query_pos = pos_embed.transpose(1,0), query_pos.transpose(1,0)
        
        memory = self.encoder(reid_feat_pre_frame, src_key_padding_mask=src_key_padding_mask, pos=pos_embed)  # [max_nb, bs, embed_dim]
        memory_key_padding_mask = src_key_padding_mask
        hs = self.decoder(reid_feat_cur_frame, memory, tgt_key_padding_mask, memory_key_padding_mask, pos=pos_embed, query_pos=query_pos)  # [nb_step,max_nb2,bs,embed_dim]
        return hs.transpose(1,2)  # [nb_step,bs,max_nb2,embed_dim]

def build_transformer(args):
    return Transformer(args.embed_dim, args.dim_feedforward, args.num_heads, num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_decoder_layers, return_intermediate=True, 
                        dropout=args.dropout, activation="relu", normalize_before=False)

if __name__ == "__main__":
    embed_dim=256
    transformer = Transformer(embed_dim=embed_dim, dim_feedforward=512, num_heads=8, return_intermediate=True)
    bs, max_nb, max_nb2 = 1, 300, 100
    reid_feat_pre_frame = torch.rand((max_nb,bs,embed_dim))
    src_key_padding_mask=torch.zeros((bs,max_nb))
    pos_embed = torch.rand((max_nb,bs,embed_dim))
    reid_feat_cur_frame = torch.rand((max_nb2,bs,embed_dim))
    tgt_key_padding_mask = torch.zeros(bs, max_nb2)
    query_pos = torch.rand((max_nb2,bs,embed_dim))
    hs = transformer(reid_feat_pre_frame, src_key_padding_mask, pos_embed, reid_feat_cur_frame, tgt_key_padding_mask, query_pos)
    print('hs size: ', hs.size())