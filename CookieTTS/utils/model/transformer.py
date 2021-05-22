import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, *args, w_init_gain='linear', **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

class Conv1d(nn.Conv1d):
    def __init__(self, *args, w_init_gain='linear', **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 ff_kernel_size=1,
                 dropout=0.1,
                 activation="relu",
                 rezero=True,
                 legacy=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff_kernel_size = ff_kernel_size
        
        if self.ff_kernel_size > 1:
            assert ff_kernel_size%2==1, 'ff_kernel_size must be odd integer'
            self.conv1 = Conv1d(d_model, dim_feedforward, ff_kernel_size, padding=(ff_kernel_size-1)//2, w_init_gain=activation)
            self.conv2 = Conv1d(dim_feedforward, d_model, ff_kernel_size, padding=(ff_kernel_size-1)//2)
        else:
            self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
            self.linear2 = Linear(dim_feedforward, d_model)
        
        if (not rezero) or legacy:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        if legacy:
            self.residual_weight = nn.Parameter(torch.ones(1)*0.01)
        elif rezero:
            self.residual_weight1 = nn.Parameter(torch.ones(1)*0.01)
            self.residual_weight2 = nn.Parameter(torch.ones(1)*0.01)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):# [B, in_T, dim]
        if True:# Multi-Head Attention
            src2, enc_align = self.self_attn(src, src, src,
                                             attn_mask=src_mask,
                                             key_padding_mask=src_key_padding_mask)
            src2 = self.dropout(src2)
            if hasattr(self, 'residual_weight1'):
                src2 = self.residual_weight1*src2
            
            if hasattr(self, 'residual_weight'):
                src2 = self.residual_weight*src2
            
            src = src + src2# [B, in_T, dim]
            if hasattr(self, 'norm1'):
                src = self.norm1(src)
        
        if True:# Feed Forward
            if self.ff_kernel_size > 1:
                src2 = self.conv2(self.dropout(F.relu(self.conv1(src.transpose(1, 2))))).transpose(1, 2)# [B, in_T, dim] -> [B, dim, in_T] -> [B, in_T, dim]
            else:
                src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src2 = self.dropout(src2)
            if hasattr(self, 'residual_weight2'):
                src2 = self.residual_weight2*src2
            
            src = src + src2
            if hasattr(self, 'norm2'):
                src = self.norm2(src)
        
        return src, enc_align
             # [Tar_T,     B, Embed] where Tar_T is the target sequence length, B is the batch size, Embed is the embedding dimension.
             # [    B, Tar_T, Sou_T] where B is the batch size, Tar_T is the target sequence length, Sou_T is the source sequence length

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
    
    def forward(self, x):# [L, B, D]
        return x + self.pe[:x.size(0)].unsqueeze(1)
    
    def _get_pe_matrix(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        
        pe[:, 0::2] = torch.sin(position / div_term)# [5000, d_model]
        pe[:, 1::2] = torch.cos(position / div_term)# [5000, d_model]
        
        return pe# [max_len, d_model]
        
