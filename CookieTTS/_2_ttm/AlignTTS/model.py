from math import sqrt, ceil
import random
import numpy as np
from numpy import finfo

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict

from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm, LSTMCellWithZoneout, GMMAttention, DynamicConvolutionAttention
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame, freeze_grads

from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d, LnBatchNorm1d

from init_layer import *
from transformer import *

drop_rate = 0.5

def load_model(hparams):
    model = AlignTTS(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class Prenet(nn.Module):
    def __init__(self, hp):
        super(Prenet, self).__init__()
        self.speaker_embedding = nn.Embedding(hp.n_speakers, hp.speaker_embedding_dim)
        self.speaker_linear = nn.Linear(hp.speaker_embedding_dim, hp.symbols_embedding_dim)
        self.speaker_linear.weight.data *= 0.1
        
        # B, L -> B, L, D
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.dropout = nn.Dropout(0.1)
        self.pos_enc_weight = nn.Parameter(torch.ones(1)*0.5)
        self.norm = nn.LayerNorm(hp.hidden_dim)
    
    def forward(self, text, speaker_ids):
        B, L = text.size(0), text.size(1)# [B, txt_T]
        x = self.Embedding(text).transpose(0,1)# -> [txt_T, B, embed]
        assert not (torch.isinf(x) | torch.isnan(x)).any()
        embed = self.speaker_embedding(speaker_ids).expand(x.shape[0], -1, -1)# [txt_T, B, spkr_embed]
        embed = self.speaker_linear(embed)# [txt_T, B, embed]
        x += self.pos_enc_weight*self.pe[:L].unsqueeze(1)# [txt_T, 1, d_model] +[txt_T, B, embed] -> [txt_T, B, embed]
        x += embed
        x = self.dropout(x)# [txt_T, B, embed]
        x = self.norm(x)
        x = x.transpose(0,1)# -> [B, txt_T, embed]
        assert not (torch.isinf(x) | torch.isnan(x)).any()
        return x# -> [B, txt_T, embed]


class FFT(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, n_layers):
        super(FFT, self).__init__()
        self.FFT_layers = nn.ModuleList(
          [TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim) for _ in range(n_layers)]
        )
    
    def forward(self, x, lengths):# [B, L, D], [B]
        
        alignments = []
        x = x.transpose(0,1)# [B, L, D] -> [L, B, D]
        assert not (torch.isinf(x) | torch.isnan(x)).any()
        mask = ~get_mask_from_lengths(lengths)# -> [B, L]
        for layer in self.FFT_layers:
            x, align = layer(x, src_key_padding_mask=mask)# -> [L, B, D], ???
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)# [L, B, n_layers, L]
        
        assert not (torch.isinf(x) | torch.isnan(x)).any()
        return x.transpose(0,1), alignments# [B, L, D], [L, B, n_layers, L]


class MelEncoder(nn.Module):
    def __init__(self, hparams):
        super(MelEncoder, self).__init__()
        conv_dim = 512
        self.conv = ConvNorm(hparams.n_mel_channels, conv_dim, 
                                kernel_size = 3,
                                     stride = 3,
                                    padding = 0,)
        
        lstm_dim = 32
        self.lstm = nn.LSTM(conv_dim, lstm_dim, num_layers=1, bidirectional=True)
        
        bottleneck_dim = 4
        self.bottleneck = LinearNorm(lstm_dim*2, bottleneck_dim)
        
        self.post = LinearNorm(bottleneck_dim, hparams.symbols_embedding_dim)
    
    def forward(self, gt_mel):
        gt_mel = self.conv(gt_mel)# [B, n_mel, mel_T] -> [B, 512, mel_T//3]
        
        _, states = self.lstm(gt_mel.permute(2, 0, 1))
        state = states[0]# -> [2, B, 32]
        
        state = state.permute(1, 0, 2).reshape(state.shape[1], -1)# [2, B, 32] -> [B, 64]
        out = self.bottleneck(state)# [B, 64] -> [B,   4]
        
        out = self.post(out)        # [B,  4] -> [B, 512]
        return out

class AlignTTS(nn.Module):
    def __init__(self, hparams):
        super(AlignTTS, self).__init__()
        self.MelEnc = MelEncoder(hparams)
        self.Prenet = Prenet(hparams)
        self.FFT_lower = FFT(hparams.hidden_dim, hparams.n_heads, hparams.ff_dim, hparams.n_layers)
        self.MDN = nn.Sequential(Linear(hparams.hidden_dim, hparams.hidden_dim),
                                 nn.LayerNorm(hparams.hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 Linear(hparams.hidden_dim, 2*hparams.n_mel_channels))
        
        self.tm_linear = nn.Linear(hparams.tt_torchMoji_attDim, hparams.tt_torchMoji_crushedDim)
        if hparams.tt_torchMoji_BatchNorm:
            self.tm_bn = MaskedBatchNorm1d(hparams.tt_torchMoji_attDim, eval_only_momentum=False, momentum=0.2)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self, text, text_lengths, gt_mel, mel_lengths, speaker_id, torchmoji_hdn, save_alignments=False, log_viterbi=False, cpu_viterbi=False):
        global_cond = self.MelEnc(gt_mel)
        
        encoder_input = self.Prenet(text, speaker_id)
        B, embed_dim = global_cond.shape
        encoder_input = encoder_input+global_cond.unsqueeze(1).expand(B, encoder_input.shape[1], embed_dim)# [B, txt_T, embed]+[B, 1, embed]
        hidden_states, _ = self.FFT_lower(encoder_input, text_lengths)
        mu_logvar = self.MDN(hidden_states)
        
        # package into dict for output
        outputs = {
                   "mu_logvar": mu_logvar,# [B, txt_T, 2*n_mel]
        }
        return outputs
    
    def update_device(self, **inputs):
        target_device = next(self.parameters()).device
        target_float_dtype = next(self.parameters()).dtype
        outputs = {}
        for key, input in inputs.items():
            if type(input) == Tensor:# move all Tensor types to GPU
                if input.dtype == torch.float32:
                    outputs[key] = input.to(target_device, target_float_dtype)# convert float to half if required.
                else:
                    outputs[key] = input.to(target_device                    )# leave Long / Bool unchanged in datatype
            else:
                outputs[key] = input
        return outputs
    
    def viterbi(self, log_prob_matrix, text_lengths, mel_lengths):
        B, L, T = log_prob_matrix.size()
        log_beta = log_prob_matrix.new_ones(B, L, T)*(-1e15)
        log_beta[:, 0, 0] = log_prob_matrix[:, 0, 0]
        
        for t in range(1, T):
            prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-1e15)], dim=-1).max(dim=-1)[0]
            log_beta[:, :, t] = prev_step+log_prob_matrix[:, :, t]
        
        curr_rows = text_lengths-1
        curr_cols = mel_lengths-1
        path = [curr_rows*1.0]
        for _ in range(T-1):
            is_go = log_beta[torch.arange(B), (curr_rows-1).to(torch.long), (curr_cols-1).to(torch.long)]\
                     > log_beta[torch.arange(B), (curr_rows).to(torch.long), (curr_cols-1).to(torch.long)]
            curr_rows = F.relu(curr_rows-1.0*is_go+1.0)-1.0
            curr_cols = F.relu(curr_cols-1+1.0)-1.0
            path.append(curr_rows*1.0)
        
        path.reverse()
        path = torch.stack(path, -1)
        
        indices = path.new_tensor(torch.arange(path.max()+1).view(1,1,-1)) # 1, 1, L
        align = 1.0*(path.new_tensor(indices==path.unsqueeze(-1))) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-mel_lengths[i]
            align[i] = F.pad(align[i], (0,0,-pad,pad))
        
        return align.transpose(1,2)# [B, txt_T, mel_T]
    
    def fast_viterbi(self, log_prob_matrix, text_lengths, mel_lengths):
        B, L, T = log_prob_matrix.size()
        
        _log_prob_matrix = log_prob_matrix.cpu()

        curr_rows = text_lengths.cpu().to(torch.long)-1
        curr_cols = mel_lengths.cpu().to(torch.long)-1
        
        path = [curr_rows*1]       
        
        for _ in range(T-1):
#             print(curr_rows-1)
#             print(curr_cols-1)
            is_go = _log_prob_matrix[torch.arange(B), curr_rows-1, curr_cols-1]\
                     > _log_prob_matrix[torch.arange(B), curr_rows, curr_cols-1]
#             curr_rows = F.relu(curr_rows-1*is_go+1)-1
#             curr_cols = F.relu(curr_cols)-1
            curr_rows = F.relu(curr_rows-1*is_go+1)-1
            curr_cols = F.relu(curr_cols-1+1)-1
            path.append(curr_rows*1)

        path.reverse()
        path = torch.stack(path, -1)
        
        indices = path.new_tensor(torch.arange(path.max()+1).view(1,1,-1)) # 1, 1, L
        align = 1.0*(path.new_tensor(indices==path.unsqueeze(-1))) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-mel_lengths[i]
            align[i] = F.pad(align[i], (0,0,-pad,pad))
            
        return align.transpose(1,2)
    
    def viterbi_cpu(self, log_prob_matrix, text_lengths, mel_lengths):
        
        original_device = log_prob_matrix.device

        B, L, T = log_prob_matrix.size()
        
        _log_prob_matrix = log_prob_matrix.cpu()
        
        log_beta = _log_prob_matrix.new_ones(B, L, T)*(-1e15)
        log_beta[:, 0, 0] = _log_prob_matrix[:, 0, 0]

        for t in range(1, T):
            prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-1e15)], dim=-1).max(dim=-1)[0]
            log_beta[:, :, t] = prev_step+_log_prob_matrix[:, :, t]

        curr_rows = text_lengths-1
        curr_cols = mel_lengths-1
        path = [curr_rows*1]
        for _ in range(T-1):
            is_go = log_beta[torch.arange(B), curr_rows-1, curr_cols-1]\
                     > log_beta[torch.arange(B), curr_rows, curr_cols-1]
            curr_rows = F.relu(curr_rows - 1 * is_go + 1) - 1
            curr_cols = F.relu(curr_cols) - 1
            path.append(curr_rows*1)

        path.reverse()
        path = torch.stack(path, -1)
        
        indices = path.new_tensor(torch.arange(path.max()+1).view(1,1,-1)) # 1, 1, L
        align = 1.0*(path.new_tensor(indices==path.unsqueeze(-1))) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-mel_lengths[i]
            align[i] = F.pad(align[i], (0,0,-pad,pad))
            
        return align.transpose(1,2).to(original_device)
    
