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

class Prenet(nn.Module):
    def __init__(self, h, in_dim, sizes, dropout=0.2):
        super(Prenet, self).__init__()
        self.in_dim = in_dim
        self.p_dropout = dropout
        self.sizes  = sizes
        
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList( [LinearNorm(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, sizes)] )
        self.prenet_batchnorm = getattr(h, 'prenet_batchnorm', False)
        
        self.bn_momentum = h.prenet_bn_momentum
        self.batchnorms = None
        if self.prenet_batchnorm:
            self.batchnorms = nn.ModuleList(
                                   [MaskedBatchNorm1d(size,   eval_only_momentum=False, momentum=self.bn_momentum) for size in sizes] )
            #self.batchnorms.append( MaskedBatchNorm1d(in_dim, eval_only_momentum=False, momentum=self.bn_momentum) )
    
    def forward(self, x, disable_dropout=False):# [B, mel_T, n_mel] / [B, n_mel]
        #if self.batchnorms is not None:
        #    x = self.batchnorms[-1](x.transpose(1, 2)).transpose(1, 2)
        
        for i, linear in enumerate(self.layers):
            x = F.relu(linear(x))
            if self.p_dropout > 0 and (not disable_dropout):
                x = F.dropout(x, p=self.p_dropout, training=True)
            if self.batchnorms is not None:
                x = self.batchnorms[i](x.transpose(1, 2)).transpose(1, 2)
        return x

class LSTMBlock(nn.Module):# LSTM with variable number of layers, zoneout and residual connections
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.0, zoneout=0.1, residual=False):
        super(LSTMBlock, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout  = dropout
        self.zoneout  = zoneout
        self.residual = residual
        
        inp_dim = input_dim
        self.lstm_cell = []
        for i in range(self.n_layers):
            self.lstm_cell.append(LSTMCellWithZoneout(inp_dim, self.hidden_dim, dropout=self.dropout, zoneout=self.zoneout))
            inp_dim = self.hidden_dim
        self.lstm_cell = nn.ModuleList(self.lstm_cell)
    
    def forward(self, x, states=None):# [..., input_dim], TensorList()
        if states is None:
            states = [x[0].new_zeros(B, self.hidden_dim) for i in range(2*self.n_layers)]
        
        xi = x
        for j, lstm_cell in enumerate(self.lstm_cell):
            states[j*2:j*2+2] = lstm_cell(xi, tuple(states[j*2:j*2+2]))
            xi = xi + states[j*2] if self.residual and j>0 else states[j*2]
        
        return xi, states# [..., hidden_dim], TensorList()

class LocationLayer(nn.Module):
    def __init__(self, n_filters, kernel_size,
                 att_dim, out_bias=False):
        super(LocationLayer, self).__init__()
        padding = (kernel_size-1)//2
        self.location_conv = ConvNorm(2, n_filters, kernel_size=kernel_size, padding=padding,
                                         bias=False, stride=1, dilation=1)
        self.location_dense = LinearNorm(n_filters, att_dim, bias=out_bias, w_init_gain='tanh')
    
    def forward(self, attention_weights_cat):# [B, 2, txt_T]
        processed_attention = self.location_conv(attention_weights_cat)# [B, 2, txt_T] -> [B, n_filters, txt_T]
        processed_attention = processed_attention.transpose(1, 2)      # [B, n_filters, txt_T] -> [B, txt_T, n_filters]
        processed_attention = self.location_dense(processed_attention) # [B, txt_T, n_filters] -> [B, txt_T, att_dim]
        return processed_attention# [B, txt_T, att_dim]

class Attention(nn.Module):
    def __init__(self, h, randn_dim=0, output_dim=None):
        super(Attention, self).__init__()
        self.key_encoder    = LinearNorm(h.att_value_dim, h.att_dim, bias=True)
        self.query_encoder  = LinearNorm(h.attlstm_dim, h.att_dim, bias=True)
        self.location_layer = LocationLayer(32, 31, h.att_dim, out_bias=True)
        self.v = LinearNorm(h.att_dim, 1, bias=False)
        
        self.window_offset = h.att_window_offset
        self.window_range  = h.att_window_range
    
    def update_kv(self, value, value_lengths):# [B, txt_T, v_dim], LongTensor[B]
        self.value = value                  # -> [B, txt_T, v_dim]
        self.key   = self.key_encoder(value)# -> [B, txt_T, key_dim]
        self.value_lengths = value_lengths  # -> LongTensor[B]
        self.value_mask = get_mask_from_lengths(value_lengths)# -> BoolTensor[B, txt_T]
    
    def reset_kv(self):
        if hasattr(self, 'key') or hasattr(self, 'value') or hasattr(self, 'value_lengths') or hasattr(self, 'value_mask'):
            del self.key, self.value, self.value_lengths, self.value_mask
    
    def sep_states(self, states):
        attention_weights_cat, current_pos = states
        return attention_weights_cat, current_pos# [B, 3, txt_T], [B]
    
    def init_states(self, x):
        assert hasattr(self, 'value'), '.update_kv() must be called before .init_states()'
        batch_size = x.shape[0]
        attention_weights_cat = torch.zeros(batch_size, 2, self.value_lengths.max().item(), device=x.device, dtype=x.dtype)
        current_pos = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        return attention_weights_cat, current_pos# [B, 3, txt_T], [B]
    
    def col_states(self, attention_weights_cat, current_pos):
        states = [attention_weights_cat, current_pos,]
        return states# TensorTuple()
    
    def forward(self, query, states):
        if states is None:
            attention_weights_cat, current_pos = self.init_states(query)
        else:
            attention_weights_cat, current_pos = self.sep_states(states)
        B, _, txt_T = attention_weights_cat.shape
        
        processed = self.location_layer(attention_weights_cat)# [B, 2, txt_T] -> [B, txt_T, key_dim]
        processed.add_( self.query_encoder(query.unsqueeze(1)).expand_as(self.key) )# [B, attlstm_dim] -> [B, 1, key_dim] -> [B, txt_T, key_dim]
        processed.add_( self.key )# [B, txt_T, key_dim]
        alignment = self.v( torch.tanh( processed ) ).squeeze(-1)# [B, txt_T, key_dim]
        
        mask = ~self.value_mask
        if self.window_range > 0 and current_pos is not None:
            if self.window_offset:
                current_pos = current_pos + self.window_offset
            max_end = self.value_lengths - 1 - self.window_range
            min_start = self.window_range
            current_pos = torch.min(current_pos.clamp(min=min_start), max_end.to(current_pos))
            
            mask_start = (current_pos-self.window_range).clamp(min=0).round() # [B]
            mask_end = (mask_start+(self.window_range*2)).round()             # [B]
            pos_mask = torch.arange(txt_T, device=current_pos.device).unsqueeze(0).expand(B, -1)  # [B, txt_T]
            pos_mask = (pos_mask >= mask_start.unsqueeze(1).expand(-1, txt_T)) & (pos_mask <= mask_end.unsqueeze(1).expand(-1, txt_T))# [B, txt_T]
            
            # attention_weights_cat[pos_mask].view(B, self.window_range*2+1) # for inference masked_select later
            
            mask = mask | ~pos_mask# [B, txt_T] & [B, txt_T] -> [B, txt_T]
        alignment.data.masked_fill_(mask, -float('inf'))# [B, txt_T]
        
        alignment = F.softmax(alignment, dim=1)# [B, txt_T] # softmax along encoder tokens dim
        
        attention_context = (alignment.unsqueeze(1) @ self.value).squeeze(1)
                                #     [B, 1, txt_T] @ [B, txt_T, enc_dim] -> [B, enc_dim]
        
        new_pos = (alignment*torch.arange(txt_T, device=alignment.device).expand(B, -1)).sum(1)
                       # ([B, txt_T] * [B, txt_T]).sum(1) -> [B]
        
        attention_weights_cat = torch.stack((alignment, attention_weights_cat[:, 1]+alignment), dim=1)# cat([B, txt_T], [B, txt_T]) -> [B, 2, txt_T]
        states = self.col_states(attention_weights_cat, new_pos)
        return attention_context, alignment, states

class Projection(nn.Module):
    def __init__(self, h, input_dim=None, output_dim=None):
        super(Projection, self).__init__()
        self.n_frames_per_step = h.n_frames_per_step
        self.linear = LinearNorm(input_dim, (output_dim or h.n_mel_channels+1)*self.n_frames_per_step, bias=True)
    
    def forward(self, x):
        x = self.linear(x) 
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, h, randn_dim=0, output_dim=None):
        super(RecurrentBlock, self).__init__()
        self.n_frames_per_step = h.n_frames_per_step
        self.n_mel = h.n_mel_channels
        
        # inf params
        self.sigma     = None
        self.randn_max = None
        
        # modules
        self.prenet = Prenet(h, h.n_mel_channels*self.n_frames_per_step, [h.prenet_dim,]*h.prenet_n_layers, h.prenet_dropout)
        self.randn_dim = randn_dim
        
        self.attlstm_n_states = h.attlstm_n_layers*2
        self.declstm_n_states = h.declstm_n_layers*2
        self.attention_n_states = 2
        
        self.att_value_dim = h.att_value_dim
        self.attlstm_dim      = h.attlstm_dim
        self.attlstm_n_layers = h.attlstm_n_layers
        self.declstm_dim      = h.declstm_dim
        self.declstm_n_layers = h.declstm_n_layers
        
        input_dim = h.prenet_dim + h.att_value_dim
        if self.randn_dim:
            self.randn_lin = LinearNorm(self.randn_dim, h.prenet_dim)
            if getattr(h, 'randn_rezero', False):
                self.randn_rezero = nn.parameter.Parameter(torch.tensor(0.0))
        self.attLSTM   = LSTMBlock(input_dim, h.attlstm_dim, n_layers=h.attlstm_n_layers, dropout=0.0, zoneout=h.attlstm_zoneout, residual=True)
        self.attention = Attention(h)
        input_dim = h.attlstm_dim + h.att_value_dim
        self.decLSTM   = LSTMBlock(input_dim, h.declstm_dim, n_layers=h.declstm_n_layers, dropout=0.0, zoneout=h.declstm_zoneout, residual=True)
        
        self.projnet = Projection(h, h.att_value_dim+h.declstm_dim, output_dim)
    
    def update_kv(self, encoder_outputs, text_lengths):
        self.attention.update_kv(encoder_outputs, text_lengths)
    
    def reset_kv(self):
        self.attention.reset_kv()
    
    def pre(self, x):# [B, T, in_dim] / [B, in_dim]
        x = self.prenet(x)
        if self.randn_dim:
            rx = torch.randn(*x.shape[:-1], self.randn_dim, device=x.device, dtype=x.dtype)
            if self.sigma is not None:
                rx *= self.sigma
            if self.randn_max is not None:
                rx = rx.clamp(min=-self.randn_max, max=self.randn_max)
            rx = self.randn_lin(rx)
            if hasattr(self, 'randn_rezero'):
                rx = self.randn_rezero*rx
            x = x + rx
        return x# [B, T, H] / [B, H]
    
    def post(self, x, v, reset_kv=False):# [B, T, H] / [B, H]
        if type(x) in (list, tuple):
            x = torch.stack(x, dim=1)# [[B, H],]*mel_T -> [B, mel_T, H]
        if type(v) in (list, tuple):
            v = torch.stack(v, dim=1)# [[B, H],]*mel_T -> [B, mel_T, H]
        
        x = torch.cat((x, v), dim=-1)
        x = self.projnet(x)
        if reset_kv:
            self.reset_kv()
        return x# [B, T, outdim] / [B, outdim]
    
    def sep_states(self, states, batch_size=None, init_states=True):
        if states is not None:
            attlstm_states = states[:self.attlstm_n_states]
            att_states     = states[self.attlstm_n_states:self.attlstm_n_states+self.attention_n_states]
            declstm_states = states[self.attlstm_n_states+self.attention_n_states:]
        elif init_states:
            assert batch_size is not None
            device = next(self.parameters()).device
            dtype  = next(self.parameters()).dtype
            attlstm_states = [torch.zeros(batch_size, self.attlstm_dim, device=device, dtype=dtype),]*self.attlstm_n_states
            att_states     = None
            declstm_states = [torch.zeros(batch_size, self.declstm_dim, device=device, dtype=dtype),]*self.declstm_n_states
        else:
            states[0]# raise Exception
        return attlstm_states, att_states, declstm_states
    
    def col_states(self, attlstm_states, att_states, declstm_states):
        states = [*attlstm_states, *att_states, *declstm_states,]
        return states
    
    def get_attlstm_input(self, x, v):
        if v is None:
            x = torch.cat((x, torch.zeros(x.shape[0], self.attention.value_dim, device=x.device, dtype=x.dtype)), dim=1)
        else:
            x = torch.cat((x, v), dim=1)
        return x
    
    def main(self, x, v, states):# [B, C], TensorList(), [B, txt_T, C]
        attlstm_states, att_states, declstm_states = self.sep_states(states, batch_size=x.shape[0])
        if v is None:
            v = torch.zeros(x.shape[0], self.att_value_dim, device=x.device, dtype=x.dtype)
        
        x = self.get_attlstm_input(x, v)
        x, attlstm_states = self.attLSTM(x, attlstm_states)
        v, alignment, att_states = self.attention(x, att_states)
        x, declstm_states = self.decLSTM(torch.cat((x, v), dim=1), declstm_states)
        states = self.col_states(attlstm_states, att_states, declstm_states)
        return x, v, alignment, states# [B, C], TensorList()
    
    def forward(self, x, v, states):# ff_forward # [B, H], TensorList(), [B, txt_T, C]
        x, v, alignment, states = self.main(x, v, states)
        return x, v, alignment, states# [B, H], [B, H], TensorList()
    
    def ar_forward(self, x, v, states):# [B, in_dim], TensorList(), [B, txt_T, C]
        x = self.pre(x.unsqueeze(0)).squeeze(0)
        x, v, alignment, states = self.main(x, v, states)
        x = self.post(x, v)
        return x, v, alignment, states# [B, outdim], TensorList()


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

class Aligner(nn.Module):
    def __init__(self, h):
        super(Aligner, self).__init__()
        self.encoder = FeedForwardBlock(h)
        self.decoder = RecurrentBlock(h)
        self.n_frames_per_step = h.n_frames_per_step
        self.n_mel = h.n_mel_channels
    
    def reshape_outputs(self, pr_melgate, alignments):
        pr_melgate = pr_melgate.view(pr_melgate.shape[0], -1, pr_melgate.shape[2]//self.n_frames_per_step)# [B, mel_T, (n_mel+1)*n_frames_per_step] -> [B, mel_T, n_mel+1]
        pr_mel  = pr_melgate[:, :, 1:].transpose(1, 2)# -> [B, n_mel, mel_T]
        pr_gate = pr_melgate[:, :, 0]                 # -> [B, mel_T]
        alignments = torch.stack(alignments, dim=1)       # -> [B, mel_T, txt_T]
        return pr_mel, pr_gate, alignments
    
    def forward(self,# gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd,# FloatTensor[B, 2]
                             gt_sylps,# FloatTensor[B]
                        torchmoji_hdn,# FloatTensor[B, embed]
                     freq_grad_scalar,# FloatTensor[B]
                 inference_mode=False,
             teacher_force_prob=1.0):
        out = {}
        
        encoder_outputs, spkrenc_outputs, emotion_embed = self.encoder(text, text_lengths, torchmoji_hdn, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
        self.decoder.update_kv(encoder_outputs, text_lengths)
        out['encoder_outputs'] = encoder_outputs
        out['spkrenc_outputs'] = spkrenc_outputs
        out['emotion_embed']   = emotion_embed
        
        gt_mel_shifted = F.pad(gt_mel[..., :(gt_mel.shape[-1]//self.n_frames_per_step)*self.n_frames_per_step], (self.n_frames_per_step, -self.n_frames_per_step))
        gt_mel_shifted = gt_mel_shifted.transpose(1, 2).reshape(gt_mel.shape[0], -1, self.n_mel*self.n_frames_per_step)# [B, mel_T, n_mel] -> [B, mel_T//n_frames_per_step, n_frames_per_step*n_mel]
        prenet_outputs = self.decoder.pre(gt_mel_shifted)# -> [B, mel_T//n_frames_per_step, prenet_dim]
        
        tf_frames = (torch.rand(prenet_outputs.shape[1]) < teacher_force_prob).tolist()# list[mel_T]
        out['tf_frames'] = tf_frames
        
        tf_states = None; tf_attention_output = None
        declstm_outputs  = []; attention_outputs = []; alignments = []
        for i, (gt_mel_frame, next_gt_mel_frame, prenet_frame, b_tf) in enumerate(zip(gt_mel_shifted.unbind(1), gt_mel.unbind(2), prenet_outputs.unbind(1), tf_frames)):
            tf_declstm_output, tf_attention_output, alignment, tf_states = self.decoder(prenet_frame, tf_attention_output, tf_states)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            if (inference_mode or (not b_tf)) and i > 0:# inference
                if self.decoder.prenet.batchnorms is not None and self.training: [x.eval() for x in self.decoder.prenet.batchnorms]
                pr_melgate = self.decoder.post(declstm_output.detach(), attention_output.detach(), reset_kv=False)
                pr_mel_frame = pr_melgate.view(pr_melgate.shape[0], self.n_frames_per_step, -1)[:, :, 1:].reshape(pr_melgate.shape[0], -1)# [B, n_frames_per_step*(n_mel+1)] -> [B, n_frames_per_step, n_mel+1] -> [B, n_frames_per_step*n_mel]
                fr_prenet_frame = self.decoder.pre(pr_mel_frame.detach().unsqueeze(1)).squeeze(1)
                if self.decoder.prenet.batchnorms is not None and self.training: [x.train() for x in self.decoder.prenet.batchnorms]
                
                declstm_output, attention_output, alignment, states = self.decoder(fr_prenet_frame, attention_output, states)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            else:# teacher_force
                declstm_output, attention_output, states = tf_declstm_output, tf_attention_output, tf_states
            declstm_outputs.append(declstm_output); attention_outputs.append(attention_output); alignments.append(alignment)# list.append([B, H])
        
        pr_melgate = self.decoder.post(declstm_outputs, attention_outputs, reset_kv=True)# -> [B, mel_T//n_frames_per_step, (n_mel+1)*n_frames_per_step]
        pr_mel, pr_gate, alignments = self.reshape_outputs(pr_melgate, alignments)
        out['pr_mel_a']   = pr_mel  # [B, n_mel, mel_T]
        out['pr_gate']  = pr_gate # [B, mel_T]
        
        dec_lengths = (mel_lengths//self.n_frames_per_step).clamp(max=alignments.shape[1])
        att_mask_e = get_mask_from_lengths(text_lengths)# [B, txt_T]
        att_mask_d = get_mask_from_lengths(dec_lengths) # [B, mel_T]
        att_mask = (att_mask_e.unsqueeze(1) * att_mask_d.unsqueeze(2)).bool()# [B, txt_T] * [B, mel_T] -> [B, mel_T, txt_T]
        alignments = alignments.masked_fill_(~att_mask, 0.0)
        out['soft_alignments'] = alignments# [B, mel_T, txt_T]
        
        if True:
            hard_alignments = self.onehot(alignments).masked_fill_(~att_mask, 0.0)
        else:
            hard_alignments = self.viterbi(alignments.detach().log().cpu().transpose(1, 2).float(), text_lengths.cpu(), dec_lengths.cpu())
            hard_alignments = hard_alignments.transpose(1, 2).to(alignments)
        out['hard_alignments'] = hard_alignments# [B, mel_T, txt_T]
        out['gt_dur'] = hard_alignments.sum(1, keepdim=True).detach()# [B, 1, txt_T]
        
        out['attention_contexts'] = hard_alignments @ encoder_outputs # [B, mel_T, txt_T] @ [B, txt_T, memory_dim] -> [B, mel_T, memory_dim]
        out['attention_contexts'] = out['attention_contexts'].transpose(1, 2)# [B, mel_T, memory_dim] -> [B, memory_dim, mel_T]
        return out
    
    def get_blank_frame(self, batch_size):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.n_mel*self.n_frames_per_step, device=device, dtype=dtype)
    
    def onehot(self, alignments):
        B, mel_T, txt_T = alignments.shape
        alignments_oh = alignments.new_zeros(alignments.shape)# [B, mel_T, txt_T]
        alignments_oh.view(-1, txt_T)[torch.arange(B*mel_T), alignments.view(-1, txt_T).argmax(dim=1)] = 1.0
        return alignments_oh
    
    @torch.jit.script
    def viterbi(log_prob_matrix, text_lengths, mel_lengths, pad_mag:float=1e12):
        B, L, T = log_prob_matrix.size()# [B, txt_T, mel_T]
        log_beta = torch.ones(B, L, T, device=log_prob_matrix.device, dtype=log_prob_matrix.dtype)*(-pad_mag)
        log_beta[:, 0, 0] = log_prob_matrix[:, 0, 0]
        
        for t in range(1, T):
            prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-pad_mag)], dim=-1).max(dim=-1)[0]
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
        
        indices = torch.arange(path.max()+1).view(1,1,-1).to(path) # 1, 1, L
        align = (indices==path.unsqueeze(-1)).to(path) # B, T, L
        
        for i in range(align.size(0)):
            pad= T-int(mel_lengths[i].item())
            align[i] = F.pad(align[i], (0,0,-pad,pad))
        
        return align.transpose(1,2)# [B, txt_T, mel_T]

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
        padded_mask = ~get_mask_from_lengths(lengths).unsqueeze(-1)
        x = x.masked_fill(padded_mask, 0.0)
        
        if len(self.pre2ds):
            pre_x = x
            x = x.unsqueeze(1)# [B, L, D] -> [B, 1, L, D]
            for conv2d in self.pre2ds:
                x = conv2d(x)# [B, init*(2**i), L, D//(2**i)]    
                x = F.leaky_relu(x, 0.1, inplace=False)
            x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)# -> [B, L, init*(2**i), D//(2**i)] -> [B, L, init*D]
            x = torch.cat((pre_x, x), dim=2).masked_fill_(padded_mask, 0.0)
        
        if hasattr(self, 'pre'):
            x = self.pre(x).masked_fill_(padded_mask, 0.0)
        
        pre_x = x
        for pre in self.pres:
            x = F.leaky_relu(pre(x.transpose(1, 2)).transpose(1, 2), 0.1, inplace=False).masked_fill_(padded_mask, 0.0)# [B, L, D] * [B, L, 1]
        if len(self.pres):
            x = (x + pre_x).div_(1.41421356237).masked_fill_(padded_mask, 0.0)
        
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
            if return_alignments: alignments.append(align.unsqueeze(1))
        
        x = x.transpose(0, 1)# -> [B, L, D]
        
        if hasattr(self, 'post'):
            x = self.post(x).masked_fill_(padded_mask, 0.0)
        
        if return_alignments:
            return x, torch.cat(alignments, 1)# [B, L, D], [L, B, n_layers, L]
        else:
            return x# [B, L, D]


class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.cf0_sigmoid = getattr(h, 'cf0_out_sigmoid', False)
        self.encoder = FeedForwardBlock(h, output_dim=h.mem_dim)
        
        self.durgenerator = FFT(h.dur_hidden_dim, h.dur_n_heads, h.dur_ff_dim, h.dur_n_layers, ff_kernel_size=h.dur_ff_kernel_size, learned_noise=True, add_position_encoding=True, input_dim=h.mem_dim,   output_dim=h.mem_dim)
        self.durproj = LinearNorm(h.mem_dim, 1)
        self.cf0generator = FFT(h.cf0_hidden_dim, h.cf0_n_heads, h.cf0_ff_dim, h.cf0_n_layers, ff_kernel_size=h.cf0_ff_kernel_size, learned_noise=True, add_position_encoding=True, input_dim=h.mem_dim+1, output_dim=h.mem_dim)
        self.cf0proj = LinearNorm(h.mem_dim, 2)
        self.melgenerator = FFT(h.mel_hidden_dim, h.mel_n_heads, h.mel_ff_dim, h.mel_n_layers, ff_kernel_size=h.mel_ff_kernel_size, learned_noise=True, add_position_encoding=True, input_dim=h.mem_dim+1, output_dim=h.n_mel_channels)
    
    def predict_cf0(self, encoder_outputs, logdur, text_lengths):
        cf0gen_input = torch.cat((encoder_outputs, logdur.unsqueeze(2)), dim=2)# -> [B, txt_T, mem_dim+1]
        cf0gen_outputs = self.cf0generator(cf0gen_input, text_lengths)# -> [B, txt_T, mem_dim]
        pr_logcf0, pr_cvoiced = self.cf0proj(cf0gen_outputs).unbind(2)
        if self.cf0_sigmoid:
            pr_logcf0 = pr_logcf0.sigmoid()*9.0
        pr_cvoiced = pr_cvoiced.sigmoid()
        pr_logcf0 = (pr_logcf0*pr_cvoiced + (0.0)*(1.0-pr_cvoiced)).masked_fill_(~get_mask_from_lengths(text_lengths), 0.0)# [B, txt_T]
        return cf0gen_outputs, pr_logcf0
    
    def predict_dur(self, encoder_outputs, text_lengths):
        durgen_outputs = self.durgenerator(encoder_outputs, text_lengths)# -> [B, txt_T, mem_dim]
        pr_logdur = self.durproj(durgen_outputs).squeeze(2)
        return durgen_outputs, pr_logdur
    
    def forward(self, gt_char_logf0,# FloatTensor[B, mel_T]
                     gt_char_voiced,# FloatTensor[B, mel_T]
               gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                 text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                         speaker_id,#  LongTensor[B]
                 speaker_f0_meanstd,# FloatTensor[B, 2]
              speaker_slyps_meanstd,# FloatTensor[B, 2]
                           gt_sylps,# FloatTensor[B]
                      torchmoji_hdn,# FloatTensor[B, embed]
                   freq_grad_scalar,# FloatTensor[B]
                    hard_alignments,# FloatTensor[B, mel_T, txt_T]
                   ):
        out = {}
        
        # Text Encoder + Speaker Decoder + Emotion Decoder
        encoder_outputs, spkrenc_outputs, emotion_embed = self.encoder(text, text_lengths, torchmoji_hdn, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
        out['encoder_outputs'] = encoder_outputs
        out['spkrenc_outputs'] = spkrenc_outputs
        out['emotion_embed']   = emotion_embed
        
        # Dur Generator
        durgen_outputs, pr_logdur = self.predict_dur(encoder_outputs, text_lengths)# -> [B, txt_T, mem_dim]
        encoder_outputs = (encoder_outputs+durgen_outputs)/(2**0.5)# [B, txt_T, mem_dim]
        out['pr_logdur'] = pr_logdur# [B, txt_T]
        
        # get GT Dur
        gt_logdur = hard_alignments.sum(dim=1).clamp(min=0.4).log()
        
        # cF0 Generator
        cf0gen_outputs, pr_logcf0 = self.predict_cf0(encoder_outputs, gt_logdur, text_lengths)# -> [B, txt_T, mem_dim+1]
        encoder_outputs = (encoder_outputs+cf0gen_outputs)/(2**0.5)
        out['pr_logcf0'] = pr_logcf0
        
        # Attention
        attention_contexts     = encoder_outputs.transpose(1, 2) @ hard_alignments.transpose(1, 2)# [B, txt_T, mem_dim] @ [B, txt_T, mel_T] -> [B, mem_dim, mel_T]
        gt_expanded_char_logf0 =   gt_char_logf0.unsqueeze(1)    @ hard_alignments.transpose(1, 2)# [B, txt_T,       1] @ [B, txt_T, mel_T] -> [B,       1, mel_T]
        
        # Mel Generator
        melgen_input = torch.cat((attention_contexts, gt_expanded_char_logf0), dim=1)# -> [B, mem_dim+1, mel_T]
        pr_mel = self.melgenerator(melgen_input.transpose(1, 2), mel_lengths).transpose(1, 2)# -> [B, n_mel, mel_T]
        out['pr_mel'] = pr_mel# [B, n_mel, mel_T]
        
        return out
    
    def infer(self, text, text_lengths, torchmoji_hdn, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd, hard_alignments=None, sigma=1.0, randn_max=99.9, max_decode_length=1024):
        out = {}
        # Text Encoder + Speaker Decoder + Emotion Decoder
        encoder_outputs, spkrenc_outputs, emotion_embed = self.encoder(text, text_lengths, torchmoji_hdn, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
        
        # Dur Generator
        durgen_outputs, pr_logdur = self.predict_dur(encoder_outputs, text_lengths)# -> [B, txt_T, mem_dim]
        encoder_outputs = (encoder_outputs+durgen_outputs)/(2**0.5)# [B, txt_T, mem_dim]
        out['pr_logdur'] = pr_logdur# [B, txt_T]
        
        pred_dur = F.relu(pr_logdur.exp()).round().long()*get_mask_from_lengths(text_lengths)# -> [B, txt_T]
        mel_lengths = pred_dur.sum(1).long()# -> LongTensor[B]
        out['mel_lengths'] = mel_lengths
        
        encoder_outputs = (encoder_outputs+durgen_outputs)/(2**0.5)# [B, txt_T, mem_dim]
        
        # cF0 Generator
        cf0gen_outputs, pr_logcf0 = self.predict_cf0(encoder_outputs, pr_logdur, text_lengths)# -> [B, txt_T, mem_dim+1]
        encoder_outputs = (encoder_outputs+cf0gen_outputs)/(2**0.5)
        pr_logcf0[pr_logcf0<0.5] = 0.0
        out['pr_logcf0'] = pr_logcf0
        
        # Attention
        if hard_alignments is not None:
            attention_contexts = encoder_outputs.transpose(1, 2) @ hard_alignments.transpose(1, 2)# [B, mem_dim, txt_T] @ [B, txt_T, mel_T] -> [B, mem_dim, mel_T]
        else:
            attention_contexts = self.get_attention_from_lengths(encoder_outputs, pred_dur, text_lengths).transpose(1, 2)# -> [B, mem_dim, mel_T]
        
        # F0s Generator
        if hard_alignments is not None:
            pr_expanded_char_logf0 = pr_logcf0.unsqueeze(1) @ hard_alignments.transpose(1, 2)# [B, 1, txt_T] @ [B, txt_T, mel_T] -> [B, 1, mel_T]
        else:
            pr_expanded_char_logf0 = self.get_attention_from_lengths(pr_logcf0.unsqueeze(2), pred_dur, text_lengths).transpose(1, 2)# -> [B, 1, mel_T]
        
        # Mel Generator
        melgen_input = torch.cat((attention_contexts, pr_expanded_char_logf0), dim=1)# -> [B, mem_dim+1, mel_T]
        pr_mel = self.melgenerator(melgen_input.transpose(1, 2), mel_lengths).transpose(1, 2)# -> [B, n_mel, mel_T]
        out['pred_mel'] = pr_mel# [B, n_mel, mel_T]
        
        return out
    
    def get_attention_from_lengths(self,
            seq        : Tensor,# FloatTensor[B, seq_T, enc_dim]
            seq_dur    : Tensor,# FloatTensor[B, seq_T]
            seq_masklen: Tensor,#  LongTensor[B]
        ):
        B, seq_T, seq_dim = seq.shape
        
        mask = get_mask_from_lengths(seq_masklen)
        seq_dur = seq_dur.masked_fill_(~mask, 0.0)
        
        if seq_dur.dtype is torch.float:
            seq_dur = seq_dur.round()#  [B, seq_T]
        dec_T = int(seq_dur.sum(dim=1).max().item())# [B, seq_T] -> int
        
        attention_contexts = torch.zeros(B, dec_T, seq_dim, device=seq.device, dtype=seq.dtype)# [B, dec_T, enc_dim]
        for i in range(B):
            mem_temp = []
            for j in range(int(seq_masklen[i].item())):
                duration = int(seq_dur[i, j].item())
                if duration == 0: continue
                # [B, seq_T, enc_dim] -> [1, enc_dim] -> [duration, enc_dim]
                mem_temp.append( seq[i, j:j+1].repeat(duration, 1) )
            mem_temp = torch.cat(mem_temp, dim=0)# [[duration, enc_dim], ...] -> [dec_T, enc_dim]
            min_len = min(attention_contexts.shape[1], mem_temp.shape[0])
            attention_contexts[i, :min_len] = mem_temp[:min_len]
        
        return attention_contexts# [B, dec_T, enc_dim]

class Discriminator(nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.mem_dim = h.mem_dim
        self.out_tanh = getattr(h, 'd_out_tanh', False)
        self.cf0_noise = 1.0
        self.encoder = FeedForwardBlock(h, output_dim=h.mem_dim)
        self.encoder_weight = nn.Parameter(torch.zeros(1))
        
        self.durdiscriminator = FFT(h.d_dur_hidden_dim, h.d_dur_n_heads, h.d_dur_ff_dim, h.d_dur_n_layers, ff_kernel_size=h.d_dur_ff_kernel_size, n_pre_layers=getattr(h, 'd_dur_n_pre_layers', 0), add_position_encoding=True, input_dim=h.mem_dim+1,      output_dim=h.mem_dim+1)
        self.cf0discriminator = FFT(h.d_cf0_hidden_dim, h.d_cf0_n_heads, h.d_cf0_ff_dim, h.d_cf0_n_layers, ff_kernel_size=h.d_cf0_ff_kernel_size, n_pre_layers=getattr(h, 'd_cf0_n_pre_layers', 0), add_position_encoding=True, input_dim=h.mem_dim+1,      output_dim=h.mem_dim+1)
        self.meldecoder       = FFT(h.d_mel_hidden_dim, h.d_mel_n_heads, h.d_mel_ff_dim,                0, ff_kernel_size=h.d_mel_ff_kernel_size, n_pre_layers= 0                                 , add_position_encoding=True, input_dim=h.n_mel_channels, output_dim=h.d_mel_hidden_dim,
                                    n_pre2d_layers=h.d_mel_n_pre2d_layers, pre2d_kernel_size=h.d_mel_pre2d_kernel_size, pre2d_separable=h.d_mel_pre2d_separable, pre2d_init_channels=h.d_mel_pre2d_init_channels)
        self.meldiscriminator = FFT(h.d_mel_hidden_dim, h.d_mel_n_heads, h.d_mel_ff_dim, h.d_mel_n_layers, ff_kernel_size=h.d_mel_ff_kernel_size, n_pre_layers=getattr(h, 'd_mel_n_pre_layers', 0), add_position_encoding=True, input_dim=h.mem_dim+1+h.d_mel_hidden_dim, output_dim=1)
    
    def discriminator_loss(self, gt, pr_encoder_outputs, pr_logdur, pr_logcf0, pr_mel, hard_alignments):
        ((pr_logdur_realness, gt_logdur_realness),
         (pr_logcf0_realness, gt_logcf0_realness),
         (pr_mel_realness,    gt_mel_realness   )) = self(gt, pr_encoder_outputs, pr_logdur, pr_logcf0, pr_mel, hard_alignments)
        
        dur_loss = discriminator_loss(pr_logdur_realness, gt_logdur_realness, gt['text_lengths'])
        cf0_loss = discriminator_loss(pr_logcf0_realness, gt_logcf0_realness, gt['text_lengths'])
        mel_loss = discriminator_loss(pr_mel_realness,    gt_mel_realness   , gt[ 'mel_lengths'])
        
        return dur_loss, cf0_loss, mel_loss, (
            (pr_logdur_realness, gt_logdur_realness),
            (pr_logcf0_realness, gt_logcf0_realness),
            (pr_mel_realness,    gt_mel_realness   )
        )
    
    def forward(self, gt, pr_encoder_outputs, pr_logdur, pr_logcf0, pr_mel, hard_alignments):
        gt_logdur = hard_alignments.sum(dim=1).clamp(min=0.4).log()# [B, mel_T, txt_T] -> [B, txt_T]
        gt_logdur += gt_logdur.new_empty(gt_logdur.shape).uniform_(-0.5, 0.5)
        gt_logcf0 = gt['gt_char_logf0'] # [B, txt_T]
        gt_mel    = gt['gt_mel']
        mel_noise = gt_mel.new_empty(gt_mel.shape).normal_(std=0.2)
        
        encoder_outputs, spkrenc_outputs, emotion_embed = self.encoder(gt['text'], gt['text_lengths'], gt['torchmoji_hdn'], gt['speaker_id'], gt['speaker_f0_meanstd'], gt['speaker_slyps_meanstd'])
        
        if False:
            encoder_outputs = encoder_outputs + (pr_encoder_outputs*self.encoder_weight)
        
        pr_logdur_realness, _          = self.durdiscriminator(torch.cat((encoder_outputs, pr_logdur.unsqueeze(-1)), dim=2), gt['text_lengths']).split((1, self.mem_dim), dim=2)
        gt_logdur_realness, durdis_out = self.durdiscriminator(torch.cat((encoder_outputs, gt_logdur.unsqueeze(-1)), dim=2), gt['text_lengths']).split((1, self.mem_dim), dim=2)
        
        pr_logcf0_noisy = pr_logcf0 + pr_logcf0.new_empty(pr_logcf0.shape).normal_(std=self.cf0_noise)
        gt_logcf0_noisy = gt_logcf0 + gt_logcf0.new_empty(gt_logcf0.shape).normal_(std=self.cf0_noise)
        cf0dis_input = (encoder_outputs+durdis_out)/(2**0.5)
        pr_logcf0_realness, _          = self.cf0discriminator(torch.cat((cf0dis_input, pr_logcf0_noisy.unsqueeze(-1)), dim=2), gt['text_lengths']).split((1, self.mem_dim), dim=2)
        gt_logcf0_realness, cf0dis_out = self.cf0discriminator(torch.cat((cf0dis_input, gt_logcf0_noisy.unsqueeze(-1)), dim=2), gt['text_lengths']).split((1, self.mem_dim), dim=2)
        
        meldis_char_input = (cf0dis_input+cf0dis_out)/(2**0.5)
        
        meldis_input = meldis_char_input.transpose(1, 2) @ hard_alignments.transpose(1, 2)# [B, mem_dim, txt_T] @ [B, txt_T, mel_T] -> [B, mem_dim, mel_T]
        gt_logf0     =         gt_logcf0.unsqueeze(1)    @ hard_alignments.transpose(1, 2)# [B,       1, txt_T] @ [B, txt_T, mel_T] -> [B,       1, mel_T]
        
        pr_mel_decoded = self.meldecoder((pr_mel+mel_noise).transpose(1, 2), gt['mel_lengths']).transpose(1, 2)
        gt_mel_decoded = self.meldecoder((gt_mel+mel_noise).transpose(1, 2), gt['mel_lengths']).transpose(1, 2)
        
        pr_mel_realness = self.meldiscriminator(torch.cat((pr_mel_decoded, gt_logf0, meldis_input), dim=1).transpose(1, 2), gt['mel_lengths'])
        gt_mel_realness = self.meldiscriminator(torch.cat((gt_mel_decoded, gt_logf0, meldis_input), dim=1).transpose(1, 2), gt['mel_lengths'])
        
        if self.out_tanh:
            pr_logdur_realness, gt_logdur_realness = pr_logdur_realness.tanh()*0.5, gt_logdur_realness.tanh()*0.5
            pr_logcf0_realness, gt_logcf0_realness = pr_logcf0_realness.tanh()*0.5, gt_logcf0_realness.tanh()*0.5
            pr_mel_realness,    gt_mel_realness    = pr_mel_realness   .tanh()*0.5, gt_mel_realness   .tanh()*0.5
        
        return ((pr_logdur_realness, gt_logdur_realness),
                (pr_logcf0_realness, gt_logcf0_realness),
                (pr_mel_realness,    gt_mel_realness   ))


class Model(nn.Module):
    def __init__(self, h):
        super(Model, self).__init__()
        self.fp16_run = h.fp16_run
        self.aligner_enable = h.aligner_enable
        self.GAN_enable = h.GAN_enable
        
        if h.aligner_enable:
            self.aligner = Aligner(h)
        self.generator = Generator(h)
        if h.GAN_enable:
            self.discriminator = Discriminator(h)
    
    def parse_batch(self, batch, device='cuda'):
        if self.fp16_run:# convert data to half-precision before giving to GPU (to reduce wasted bandwidth)
            batch = {k: v.half() if type(v) is torch.Tensor and v.dtype is torch.float else v for k,v in batch.items()}
        batch = {k: v.to(device) if type(v) is torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def generator_loss(self, gt, pr_encoder_outputs, pr_logdur, pr_logcf0, pr_mel, hard_alignments):
        ((pr_logdur_realness, gt_logdur_realness),
         (pr_logcf0_realness, gt_logcf0_realness),
         (pr_mel_realness,    gt_mel_realness   )) = self.discriminator(gt, pr_encoder_outputs, pr_logdur, pr_logcf0, pr_mel, hard_alignments)
        
        dur_loss = generator_loss(pr_logdur_realness, gt['text_lengths'])
        cf0_loss = generator_loss(pr_logcf0_realness, gt['text_lengths'])
        mel_loss = generator_loss(pr_mel_realness   ,  gt['mel_lengths'])
        
        return dur_loss, cf0_loss, mel_loss, (
            (pr_logdur_realness, gt_logdur_realness),
            (pr_logcf0_realness, gt_logcf0_realness),
            (pr_mel_realness,    gt_mel_realness   )
        )
    
    def forward(self, gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                     gt_frame_voiceds,# FloatTensor[B,     4, mel_T]
                     gt_char_logf0   ,# FloatTensor[B,     4, txt_T]
                     gt_char_voiced  ,# FloatTensor[B,     4, txt_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd,# FloatTensor[B, 2]
                             gt_sylps,# FloatTensor[B]
                        torchmoji_hdn,# FloatTensor[B, embed]
                     freq_grad_scalar,# FloatTensor[B]
                            alignment,# FloatTensor[B, mel_T, txt_T]
                 inference_mode=False,
             teacher_force_prob=1.0):
        out = {}
        
        if alignment is None and self.aligner_enable:
            out_tmp = self.aligner(gt_mel, mel_lengths, text, text_lengths, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd, gt_sylps, torchmoji_hdn, freq_grad_scalar, inference_mode, teacher_force_prob)
            alignment = out['hard_alignments']
            out = {**out, **out_tmp}
        
        out_tmp = self.generator(gt_char_logf0, gt_char_voiced, gt_mel, mel_lengths, text, text_lengths, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd, gt_sylps, torchmoji_hdn, freq_grad_scalar, alignment)
        out = {**out, **out_tmp}
        
        return out

#def feature_loss(fmap_r, fmap_g):
#    loss = 0
#    for dr, dg in zip(fmap_r, fmap_g):
#        loss += F.l1_loss(dr, dg)
#    return loss*2


def masked_l1_loss(targ, pred, lengths):
    numel = lengths.sum()
    loss = F.l1_loss(*torch.broadcast_tensors(targ, pred), reduction='none').masked_fill_(~get_mask_from_lengths(lengths)[:, :pred.shape[1]].unsqueeze(2), 0.0).sum(dtype=torch.float)/numel
    return loss


def masked_mse_loss(targ, pred, lengths):
    numel = lengths.sum()
    loss = F.mse_loss(*torch.broadcast_tensors(targ, pred), reduction='none').masked_fill_(~get_mask_from_lengths(lengths)[:, :pred.shape[1]].unsqueeze(2), 0.0).sum(dtype=torch.float)/numel
    return loss


def discriminator_loss(dg, dr, mel_lengths):# [B, mel_T, 1], [B, mel_T, 1], [B]
    real_target = torch.tensor( 0.5, device=dr.device, dtype=dr.dtype)
    fake_target = torch.tensor(-0.5, device=dg.device, dtype=dg.dtype)
    
    r_loss = masked_mse_loss(real_target, dr, mel_lengths)# torch.mean((1-dr)**2)
    g_loss = masked_mse_loss(fake_target, dg, mel_lengths)# torch.mean(dg**2)
    return (r_loss + g_loss)


def generator_loss(dg, mel_lengths):# [B, mel_T, 1], [B]
    real_target = torch.tensor( 0.5, device=dg.device, dtype=dg.dtype)
    fake_target = torch.tensor(-0.5, device=dg.device, dtype=dg.dtype)
    
    g_loss = masked_mse_loss(real_target, dg, mel_lengths)# torch.mean((1-dg)**2)
    return 2.*g_loss

