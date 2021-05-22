from math import sqrt, log
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

from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm, LSTMCellWithZoneout
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame, freeze_grads, grad_scale, elapsed_timer
from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d
from CookieTTS.utils.model.transformer import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from CookieTTS._4_mtw.hifigan_ct.model import ResBlock1, ResBlock2
from CookieTTS._4_mtw.hifigan.utils import init_weights, get_padding

LRELU_SLOPE = 0.1

def round_up(x, interval):
    return x if x % interval == 0 else x + interval - x % interval

def round_down(x, interval):
    return x if x % interval == 0 else x - x % interval

def load_model(h, device='cuda'):
    model = Model(h)
    if torch.cuda.is_available() or 'cuda' not in device:
        model = model.to(device)
    return model


class ConvStack(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, kernel_size, dropout=0.1, residual=False, act_func=nn.ReLU()):
        super(ConvStack, self).__init__()
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.hidden_dim= hidden_dim
        self.residual = residual
        self.dropout = 0.1
        self.act_func = act_func
        if self.residual:
            assert input_dim == output_dim
        self.conv = []
        for i in range(n_layers):
            input_dim  = self.input_dim  if   i==0        else self.hidden_dim
            output_dim = self.output_dim if 1+i==n_layers else self.hidden_dim
            self.conv.append(ConvNorm(input_dim, output_dim, kernel_size))
        self.conv = nn.ModuleList(self.conv)
    
    def forward(self, x, x_len=None):
        if x_len is not None:
            x_mask = get_mask_from_lengths(x_len).unsqueeze(1)# [B, 1, T]
            x = x*x_mask
        if self.residual:
            x_res = x
        
        for conv in self.conv:
            x = conv(x)
            x = self.act_func(x)
            x = F.dropout(x, self.dropout)
            if x_len is not None: x = x*x_mask
        
        if self.residual:
            x = x + x_res
        return x

class TextEncoder(nn.Module):
    def __init__(self, h):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(h.n_symbols, h.textenc_embed_dim)
        self.conv = ConvStack(h.textenc_embed_dim+h.speaker_embed_dim, h.textenc_conv_dim, h.textenc_conv_dim, n_layers=3, kernel_size=5, act_func=nn.ReLU(), residual=False, dropout=0.1)
        self.lstm = nn.LSTM(h.textenc_conv_dim, h.textenc_lstm_dim, num_layers=1, bidirectional=True)
    
    def forward(self, text_ids, text_lengths, spkrenc_outputs):
        text_embed = self.embedding(text_ids)# [B, txt_T] -> [B, txt_T, text_embed]
        enc_input = torch.cat((text_embed, spkrenc_outputs.unsqueeze(1).expand(-1, text_embed.shape[1], -1)), dim=2)# -> [B, txt_T, text_embed+speaker_embed]
        conv_out = self.conv(enc_input.transpose(1, 2), text_lengths).transpose(1, 2)# -> [B, txt_T, C]
        
        conv_out_masked = nn.utils.rnn.pack_padded_sequence(conv_out, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        lstm_out_masked, (h0, _) = self.lstm(conv_out_masked)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_masked, batch_first=True)# -> [B, txt_T, C]
        
        return lstm_out# [B, txt_T, C]

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.proj = LinearNorm(input_dim, output_dim*2)
    
    #@torch.jit.script
    def jit(self, x):
        x, y = x.chunk(2, dim=-1)
        x = x*y.sigmoid()
        return x
    
    def forward(self, x):
        x = self.proj(x)
        x = self.jit(x)
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self, h):
        super(SpeakerEncoder, self).__init__()
        self.embedding = nn.Embedding(h.n_speakers, h.speaker_embed_dim)
        self.conv = ConvStack(h.speaker_embed_dim+4, h.speaker_embed_dim*2, h.speaker_embed_dim*2, n_layers=3, kernel_size=1, act_func=nn.ReLU(), residual=False, dropout=0.2)
        self.GLU = GLU(h.speaker_embed_dim*2, h.speaker_embed_dim)
    
    def forward(self, speaker_ids, speaker_f0_meanstd, speaker_slyps_meanstd):
        speaker_embed = self.embedding(speaker_ids)
        speaker_embed = self.conv(torch.cat((speaker_embed, speaker_f0_meanstd, speaker_slyps_meanstd), dim=1).unsqueeze(-1)).squeeze(-1)
        speaker_embed = self.GLU(speaker_embed)
        return speaker_embed

class TorchMojiEncoder(nn.Module):
    def __init__(self, h):
        super(TorchMojiEncoder, self).__init__()
        self.dropout = h.torchmoji_dropout
        self.norm = MaskedBatchNorm1d(h.torchmoji_hidden_dim, eval_only_momentum=False, momentum=0.05)
        self.enc = LinearNorm(h.torchmoji_hidden_dim,     h.torchmoji_bottleneck_dim)
        self.dec = LinearNorm(h.torchmoji_bottleneck_dim, h.torchmoji_expanded_dim)
    
    def forward(self, torchmoji_hidden):
        torchmoji_hidden = F.dropout(self.norm(torchmoji_hidden), self.dropout)
        emotion_embed = self.dec(self.enc(torchmoji_hidden))
        return emotion_embed

class FeedForwardBlock(nn.Module):
    def __init__(self, h, output_dim=None):
        super(FeedForwardBlock, self).__init__()
        self.spkrenc = SpeakerEncoder(h)
        self.textenc = TextEncoder(h)
        self.tmenc = TorchMojiEncoder(h)
        
        input_dim = h.speaker_embed_dim + 2*h.textenc_lstm_dim + h.torchmoji_expanded_dim
        self.GLU = GLU(input_dim, output_dim or h.att_value_dim)
    
    def forward(self, text_ids, text_lengths, torchmoji_hidden, speaker_ids, speaker_f0_meanstd, speaker_slyps_meanstd,):
        spkrenc_outputs = self.spkrenc(speaker_ids, speaker_f0_meanstd, speaker_slyps_meanstd)# LongTensor[B], [B, 2], [B, 2] -> [B, speaker_embed]
        encoder_outputs = self.textenc(text_ids, text_lengths, spkrenc_outputs)# LongTensor[B, txt_T], [B, speaker_embed] -> [B, txt_T, text_embed]
        emotion_embed = self.tmenc(torchmoji_hidden)# [B, tm_embed]
        
        encoder_outputs = torch.cat((encoder_outputs,
                                     spkrenc_outputs.unsqueeze(1).expand(-1, encoder_outputs.shape[1], -1),
                                       emotion_embed.unsqueeze(1).expand(-1, encoder_outputs.shape[1], -1)), dim=-1)# -> [B, txt_T, text_embed+speaker_embed+tm_embed]
        encoder_outputs = self.GLU(encoder_outputs)# -> [B, txt_T, memory_dim]
        encoder_outputs = encoder_outputs*get_mask_from_lengths(text_lengths).unsqueeze(-1)
        return encoder_outputs, spkrenc_outputs, emotion_embed

class LearnedNoise(nn.Module):
    def __init__(self, hidden_dim, zero_init=True):
        super(LearnedNoise, self).__init__()
        self.sigma = nn.Parameter(torch.zeros(hidden_dim)+0.001 if zero_init else torch.rand(hidden_dim))
    
    def forward(self, x, x_mask=None, x_len=None, sigma=1.0):# [B, T, C], [B, T, 1], [B]
        x = x + x.new_empty(x.shape).normal_()*(self.sigma[None, None, :]*sigma)# [B, T, C] + ([B, T, C]*[1, 1, C]) -> [B, T, C]
        if x_mask is not None:
            x.masked_fill_(~x_mask, 0.0)
        elif x_len is not None:
            x.masked_fill_(~get_mask_from_lengths(x_len), 0.0)
        return x

class FFT(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, n_layers, ff_kernel_size=1, input_dim=None, output_dim=None,
                    rezero_pos_enc=False, add_position_encoding=False, position_encoding_random_start=False,
                    rezero_transformer=False, learned_noise=False, legacy=False,
                    n_pre_layers=0,   pre_kernel_size=3,   pre_separable=True,
                    n_pre2d_layers=0, pre2d_kernel_size=3, pre2d_separable=True, pre2d_init_channels=32):
        super(FFT, self).__init__()
        self.pre2ds = nn.ModuleList( [ConvNorm2D(1 if i==0 else pre2d_init_channels*(2**(i-1)), pre2d_init_channels*(2**i), (pre2d_kernel_size, 2), stride=(1, 2), padding=((pre2d_kernel_size-1)//2, 0), separable=pre2d_separable) for i in range(n_pre2d_layers)] )
        self.pre2d_output_dim = pre2d_init_channels*(input_dim or hidden_dim)//2 if n_pre2d_layers else 0
        
        if input_dim is not None and (input_dim+self.pre2d_output_dim) != hidden_dim:
            self.pre = LinearNorm(input_dim+self.pre2d_output_dim, hidden_dim)
        
        self.pres = nn.ModuleList( [ConvNorm(hidden_dim, hidden_dim, pre_kernel_size, padding=(pre_kernel_size-1)//2, separable=pre_separable) for _ in range(n_pre_layers)] ) 
        
        self.add_position_encoding = add_position_encoding
        if self.add_position_encoding:
            self.register_buffer('pe', PositionalEncoding(hidden_dim).pe)
            self.position_encoding_random_start = position_encoding_random_start
            self.rezero_pos_enc = rezero_pos_enc
            if self.rezero_pos_enc:
                self.pos_enc_weight = nn.Parameter(torch.ones(1)*1.0)
            if legacy:
                self.norm = nn.LayerNorm(hidden_dim)
        
        self.FFT_layers = nn.ModuleList( [TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim, ff_kernel_size=ff_kernel_size, rezero=rezero_transformer, legacy=legacy) for _ in range(n_layers)] )
        
        self.learned_noise = learned_noise
        if self.learned_noise:
            self.noise_layers = nn.ModuleList( [LearnedNoise(hidden_dim) for _ in range(n_layers)] )
        
        if output_dim is not None and output_dim != hidden_dim:
            self.post = LinearNorm(hidden_dim, output_dim)
    
    def forward(self, x, lengths, return_alignments=False):# [B, L, D], [B]
        padded_mask = ~get_mask_from_lengths(lengths).unsqueeze(-1)# [B, L, 1]
        x = x.masked_fill(padded_mask, 0.0)
        
        if len(self.pre2ds):
            pre_x = x
            x = x.unsqueeze(1)# [B, L, D] -> [B, 1, L, D]
            for conv2d in self.pre2ds:
                x = conv2d(x)# [B, init*(2**i), L, D//(2**i)]    
                x = F.leaky_relu(x, 0.1, inplace=False).masked_fill_(padded_mask.unsqueeze(1), 0.0)# [].fill_([B, 1, L, 1])
            x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)# -> [B, L, init*(2**i), D//(2**i)] -> [B, L, init*D]
            x = torch.cat((pre_x, x), dim=2)
        
        if hasattr(self, 'pre'):
            x = self.pre(x).masked_fill_(padded_mask, 0.0)
        
        pre_x = x
        for pre in self.pres:
            x = F.leaky_relu(pre(x.transpose(1, 2)).transpose(1, 2), 0.1, inplace=False).masked_fill_(padded_mask, 0.0)# [B, L, D] * [B, L, 1]
        if len(self.pres):
            x = (x + pre_x).div_(1.41421356237)
        
        if self.add_position_encoding:
            pos_enc = self.pe# [max_len, D]
            if self.position_encoding_random_start:
                pos_enc = pos_enc.roll(random.randint(0, 4999), 0)# [max_len, D]
            pos_enc = pos_enc[:x.shape[1]]# [max_len, D] -> [L, D]
            if self.rezero_pos_enc:
                pos_enc = pos_enc*self.pos_enc_weight
            x = x + pos_enc.unsqueeze(0)# [B, L, D] + [B, L, D] -> [B, L, D]
            if hasattr(self, 'norm'):
                x = self.norm(x)
        
        x = x.masked_fill_(padded_mask, 0.0)# [B, L, D] * [B, L, 1]
        
        alignments = []
        x = x.transpose(0, 1)# [B, L, D] -> [L, B, D]
        mask = ~get_mask_from_lengths(lengths)# -> [B, L]
        noise_mask = get_mask_from_lengths(lengths).unsqueeze(-1)# -> [B, L, 1]
        for i, layer in enumerate(self.FFT_layers):
            if self.learned_noise:
                x = self.noise_layers[i](x.transpose(0, 1), x_mask=noise_mask).transpose(0, 1)
            x, align = layer(x, src_key_padding_mask=mask)# -> [L, B, D]
            x = x.masked_fill_(padded_mask.transpose(0, 1), 0.0)# [B, L, 1] -> [L, B, 1]
            if return_alignments: alignments.append(align.unsqueeze(1))
        
        x = x.transpose(0, 1)# -> [B, L, D]
        
        if hasattr(self, 'post'):
            x = self.post(x).masked_fill_(padded_mask, 0.0)
        
        if return_alignments:
            return x, torch.cat(alignments, 1)# [B, L, D], [L, B, n_layers, L]
        else:
            return x# [B, L, D]

class Model(nn.Module):
    def __init__(self, h):
        super(Model, self).__init__()
        self.fp16_run = h.fp16_run
        n_classes = h.n_classes
        
        self.conv = ConvStack(h.n_mel_channels, h.conv_hidden_dim, h.fft_hidden_dim, h.conv_n_layers, h.conv_kernel_size, dropout=h.conv_dropout, residual=True, act_func=nn.LeakyReLU(h.conv_relu_slope))
        self.fft = FFT(h.fft_hidden_dim, h.fft_n_heads, h.fft_ff_dim, h.fft_n_layers, ff_kernel_size=h.fft_ff_kernel_size, n_pre_layers=getattr(h, 'fft_n_pre_layers', 0), add_position_encoding=True, output_dim=n_classes)
    
    def parse_batch(self, batch, device='cuda'):
        if self.fp16_run:# convert data to half-precision before giving to GPU (to reduce wasted bandwidth)
            batch = {k: v.half() if type(v) is torch.Tensor and v.dtype is torch.float else v for k,v in batch.items()}
        batch = {k: v.to(device) if type(v) is torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self, gt_mel, mel_lengths):# FloatTensor[B, n_mel, mel_T], FloatTensor[B]
        out = {}
        
        x = self.conv(gt_mel, mel_lengths)# -> [B, H, mel_T]
        class_energies = self.fft(x.transpose(1, 2), mel_lengths)# -> [B, mel_T, n_classes]
        class_probs = F.softmax(class_energies[:, 0, :], dim=2)
        out['class_probs'] = class_probs# [B, n_classes]
        
        return out
