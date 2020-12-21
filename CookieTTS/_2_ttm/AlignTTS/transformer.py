import torch
import torch.nn as nn
import torch.nn.functional as F
from init_layer import *


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.residual_weight = nn.Parameter(torch.ones(1)*0.01)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        assert not (torch.isinf(src) | torch.isnan(src)).any()
        src2, enc_align = self.self_attn(src, src, src,
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask)
        assert not (torch.isinf(src2) | torch.isnan(src2)).any()
        
        src = src + (self.residual_weight*self.dropout(src2))
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src, enc_align
             # [Tar_T, B, Embed] where Tar_T is the target sequence length, B is the batch size, Embed is the embedding dimension.
                  # [B, Tar_T, Sou_T] where B is the batch size, Tar_T is the target sequence length, Sou_T is the source sequence length

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2, dec_align = self.self_attn(tgt,
                                         tgt,
                                         tgt,
                                         attn_mask=tgt_mask,
                                         key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, enc_dec_align = self.multihead_attn(tgt,
                                                  memory,
                                                  memory,
                                                  attn_mask=memory_mask,
                                                  key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, dec_align, enc_dec_align


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self._get_pe_matrix(d_model, max_len))

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)
    
    def _get_pe_matrix(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        
        pe[:, 0::2] = torch.sin(position / div_term)# [5000, d_model]
        pe[:, 1::2] = torch.cos(position / div_term)# [5000, d_model]
        
        return pe
        