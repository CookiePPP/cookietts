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

from CookieTTS._2_ttm.ExpFESV_AE.init_layer import *
from CookieTTS._2_ttm.ExpFESV_AE.transformer import *

drop_rate = 0.5

def load_model(hparams):
    model = FESV_AE(hparams)
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


class ResBlock1d(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_dim, kernel_w, bias=True, act_func=nn.LeakyReLU(negative_slope=0.2, inplace=True), dropout=0.0, stride=1, res=False):
        super(ResBlock1d, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else n_dim
            out_dim = output_dim if i+1 == n_layers else n_dim
            pad = (kernel_w-1)//2
            conv = nn.Conv1d(in_dim, out_dim, kernel_w, padding=pad, stride=stride, bias=bias)
            self.layers.append(conv)
        self.act_func = act_func
        self.dropout = dropout
        self.res = res
        if self.res:
            assert input_dim == output_dim, 'residual connection requires input_dim and output_dim to match.'
    
    def forward(self, x): # [B, in_dim, T]
        if len(x.shape) == 4 and x.shape[1] == 1:# if [B, 1, H, W]
            x = x.squeeze(1)# [B, 1, H, W] -> [B, H, W]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)# [B, C] -> [B, C, 1]
        skip = x
        
        for i, layer in enumerate(self.layers):
            is_last_layer = bool( i+1 == len(self.layers) )
            x = layer(x)
            if not is_last_layer:
                x = self.act_func(x)
            if self.dropout > 0.0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
        if self.res:
            x += skip
        return x # [B, out_dim, T]


class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.blocks = []
        self.use_stilt_40  = getattr(hparams, 'use_stilt_40',  0)
        self.use_stilt_80  = getattr(hparams, 'use_stilt_80',  0)
        self.use_stilt_120 = getattr(hparams, 'use_stilt_120', 0)
        
        input_dim = 4
        input_dim+= self.use_stilt_40+self.use_stilt_80+self.use_stilt_120
        for i in range(hparams.encoder_n_blocks):
            output_dim = hparams.encoder_dim
            self.blocks.append(
                ResBlock1d(input_dim, output_dim, hparams.encoder_n_layers, hparams.encoder_dim, hparams.encoder_kernel_size, res=(input_dim==output_dim))
            )
            input_dim = output_dim
        
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.blocks = []
        input_dim = hparams.encoder_dim+hparams.speaker_embedding_dim
        for i in range(hparams.decoder_n_blocks):
            output_dim = hparams.decoder_dim
            if i+1 == hparams.decoder_n_blocks:
                output_dim = hparams.n_mel_channels
            self.blocks.append(
                ResBlock1d(input_dim, output_dim, hparams.decoder_n_layers, hparams.decoder_dim, hparams.decoder_kernel_size, res=(input_dim==output_dim))
            )
            input_dim = output_dim
        
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FESV_AE(nn.Module):
    def __init__(self, hparams):
        super(FESV_AE, self).__init__()
        self.encoder = Encoder(hparams)
        
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(hparams.n_speakers, self.speaker_embedding_dim)
        
        self.decoder = Decoder(hparams)
        
        #self.tm_linear = nn.Linear(hparams.tt_torchMoji_attDim, hparams.tt_torchMoji_crushedDim)
        #if hparams.tt_torchMoji_BatchNorm:
        #    self.tm_bn = MaskedBatchNorm1d(hparams.tt_torchMoji_attDim, eval_only_momentum=False, momentum=0.2)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self, gt_frame_f0,       gt_frame_energy,   gt_frame_voiced,
                      gt_frame_stilt_10, gt_frame_stilt_40, gt_frame_stilt_80, gt_frame_stilt_120,
                      gt_mel, mel_lengths, speaker_id, torchmoji_hdn):
        local_cond = [gt_frame_f0, gt_frame_energy, gt_frame_voiced, gt_frame_stilt_10]
        if self.encoder.use_stilt_40:
            local_cond.append(gt_frame_stilt_40 )
        if self.encoder.use_stilt_80:
            local_cond.append(gt_frame_stilt_80 )
        if self.encoder.use_stilt_120:
            local_cond.append(gt_frame_stilt_120)
        local_cond = torch.stack(local_cond, dim=1)
        B, _, mel_T = local_cond.shape
        
        speaker_embed = self.speaker_embedding(speaker_id)
        speaker_embed = speaker_embed.unsqueeze(-1).repeat(1, 1, mel_T)# [B, embed, mel_T]
        
        encoder_outputs = self.encoder(local_cond)# [B, 4, mel_T] -> [B, C, mel_T]
        
        tokens = torch.cat((encoder_outputs, speaker_embed), dim=1)# -> [B, C+embed, mel_T]
        
        pred_mel = self.decoder(tokens)# -> [B, n_mel, mel_T]
        
        # Variational/GAN Postnet later?
        pred_mel_postnet = pred_mel
        
        # package into dict for output
        outputs = {
           "pred_mel": pred_mel,# [B, txt_T, 2*n_mel]
   "pred_mel_postnet": pred_mel_postnet,# [B, txt_T, 2*n_mel]
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
