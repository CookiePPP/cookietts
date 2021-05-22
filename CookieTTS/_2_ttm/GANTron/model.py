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
    def __init__(self, h, input_dim=None, output_dim=None, proj_logvar=False):
        super(Projection, self).__init__()
        self.n_frames_per_step = h.n_frames_per_step
        self.proj_logvar = proj_logvar
        self.linear = LinearNorm(input_dim, (output_dim or h.n_mel_channels+1)*self.n_frames_per_step*(1+proj_logvar), bias=True)
    
    def reparameterize(self, mulogvar):
        mu, logvar = mulogvar.chunk(2, dim=-1)
        std = logvar.exp().sqrt()
        return mu + mu.new_empty(mu.shape).normal_(std=1.0)*std
    
    def forward(self, x):
        x = self.linear(x) 
        if self.proj_logvar:
            x = self.reparameterize(x)
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, h, randn_dim=0, proj_logvar=False, output_dim=None):
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
        
        self.projnet = Projection(h, h.att_value_dim+h.declstm_dim, output_dim, proj_logvar)
    
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
    def __init__(self, h):
        super(FeedForwardBlock, self).__init__()
        self.spkrenc = SpeakerEncoder(h)
        self.textenc = TextEncoder(h)
        self.tmenc = TorchMojiEncoder(h)
        
        input_dim = h.speaker_embed_dim + 2*h.textenc_lstm_dim + h.torchmoji_expanded_dim
        self.GLU = GLU(input_dim, h.att_value_dim)
    
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


class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.encoder = FeedForwardBlock(h)
        self.decoder = RecurrentBlock(h, randn_dim=h.fproj_randn_dim if h.GAN_enable else 0)
        self.n_frames_per_step = h.n_frames_per_step
        self.n_mel = h.n_mel_channels
    
    def reshape_outputs(self, pred_melgate, alignments):
        pred_melgate = pred_melgate.view(pred_melgate.shape[0], -1, pred_melgate.shape[2]//self.n_frames_per_step)# [B, mel_T, (n_mel+1)*n_frames_per_step] -> [B, mel_T, n_mel+1]
        pred_mel  = pred_melgate[:, :, 1:].transpose(1, 2)# -> [B, n_mel, mel_T]
        pred_gate = pred_melgate[:, :, 0]                 # -> [B, mel_T]
        alignments = torch.stack(alignments, dim=1)       # -> [B, mel_T, txt_T]
        return pred_mel, pred_gate, alignments
    
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
        
        gt_mel_shifted = F.pad(gt_mel[..., :(gt_mel.shape[-1]//self.n_frames_per_step)*self.n_frames_per_step], (self.n_frames_per_step, -self.n_frames_per_step))
        gt_mel_shifted = gt_mel_shifted.transpose(1, 2).reshape(gt_mel.shape[0], -1, self.n_mel*self.n_frames_per_step)# [B, mel_T, n_mel] -> [B, mel_T//n_frames_per_step, n_frames_per_step*n_mel]
        prenet_outputs = self.decoder.pre(gt_mel_shifted)# -> [B, mel_T//n_frames_per_step, prenet_dim]
        if (~torch.isfinite(prenet_outputs)).any():
            print("458 : prenet_outputs has non-finite elements")
        
        tf_frames = (torch.rand(prenet_outputs.shape[1]) < teacher_force_prob).tolist()# list[mel_T]
        out['tf_frames'] = tf_frames
        
        tf_states = None; tf_attention_output = None
        declstm_outputs  = []; attention_outputs = []; alignments = []
        for i, (gt_mel_frame, next_gt_mel_frame, prenet_frame, b_tf) in enumerate(zip(gt_mel_shifted.unbind(1), gt_mel.unbind(2), prenet_outputs.unbind(1), tf_frames)):
            tf_declstm_output, tf_attention_output, alignment, tf_states = self.decoder(prenet_frame, tf_attention_output, tf_states)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            if (inference_mode or (not b_tf)) and i > 0:# inference
                if self.decoder.prenet.batchnorms is not None and self.training: [x.eval() for x in self.decoder.prenet.batchnorms]
                pred_melgate = self.decoder.post(declstm_output.detach(), attention_output.detach(), reset_kv=False)
                pred_mel_frame = pred_melgate.view(pred_melgate.shape[0], self.n_frames_per_step, -1)[:, :, 1:].reshape(pred_melgate.shape[0], -1)# [B, n_frames_per_step*(n_mel+1)] -> [B, n_frames_per_step, n_mel+1] -> [B, n_frames_per_step*n_mel]
                fr_prenet_frame = self.decoder.pre(pred_mel_frame.detach().unsqueeze(1)).squeeze(1)
                if self.decoder.prenet.batchnorms is not None and self.training: [x.train() for x in self.decoder.prenet.batchnorms]
                
                declstm_output, attention_output, alignment, states = self.decoder(fr_prenet_frame, attention_output, states)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            else:# teacher_force
                declstm_output, attention_output, states = tf_declstm_output, tf_attention_output, tf_states
            declstm_outputs.append(declstm_output); attention_outputs.append(attention_output); alignments.append(alignment)# list.append([B, H])
            
            if (~torch.isfinite(declstm_output)).any():
                print(f"480 : {i} : declstm_output has non-finite elements")
            if (~torch.isfinite(attention_output)).any():
                print(f"482 : {i} : attention_output has non-finite elements")
        
        pred_melgate = self.decoder.post(declstm_outputs, attention_outputs, reset_kv=True)# -> [B, mel_T//n_frames_per_step, (n_mel+1)*n_frames_per_step]
        pred_mel, pred_gate, alignments = self.reshape_outputs(pred_melgate, alignments)
        out['pred_mel']   = pred_mel  # [B, n_mel, mel_T]
        out['pred_gate']  = pred_gate # [B, mel_T]
        out['alignments'] = alignments# [B, mel_T, txt_T]
        return out
    
    def get_blank_frame(self, batch_size):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.n_mel*self.n_frames_per_step, device=device, dtype=dtype)
    
    def infer(self,   text,#  LongTensor[B, txt_T],
              text_lengths,#  LongTensor[B],
                speaker_id,#  LongTensor[B]
        speaker_f0_meanstd,# FloatTensor[B, 2]
     speaker_slyps_meanstd,# FloatTensor[B, 2]
             torchmoji_hdn,# FloatTensor[B, embed]
                 sigma=0.66,
             randn_max=100.0,
       max_decode_length=512,
          gate_threshold=0.5,):
        out = {}
        
        encoder_outputs, spkrenc_outputs, emotion_embed = self.encoder(text, text_lengths, torchmoji_hdn, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
        self.decoder.update_kv(encoder_outputs, text_lengths)
        
        self.decoder.sigma     = sigma
        self.decoder.randn_max = randn_max
        pred_mel, pred_gate, alignments = self.infer_decoder(batch_size=encoder_outputs.shape[0], max_decode_length=max_decode_length, gate_threshold=gate_threshold)
        out['pred_mel']  = pred_mel
        out['pred_gate'] = pred_gate.sigmoid()
        out['alignments'] = alignments
        return out
    
    def infer_decoder(self, gt_mel_frame=None, batch_size=None, max_decode_length=512, gate_threshold=0.5):# uses less VRAM, runs slower
        if gt_mel_frame is None:
            assert batch_size is not None
            gt_mel_frame = self.get_blank_frame(batch_size)
        
        states = None; attention_output = None; max_gate = torch.zeros(gt_mel_frame.shape[0])
        pred_melgates = []; alignments = []
        for i in range(max_decode_length):
            pred_melgate, attention_output, alignment, states = self.decoder.ar_forward(gt_mel_frame, attention_output, states)# [B, H], TensorList(), [B, txt_T, C] -> [B, n_mel+1], [B, mel_T, txt_T], TensorList()
            pred_melgates.append(pred_melgate); alignments.append(alignment)# list.append([B, H])
            
            pred_melgate = pred_melgate.view(pred_melgate.shape[0], self.n_frames_per_step, -1)# [B, n_frames_per_step*(n_mel+1)] -> [B, n_frames_per_step, n_mel+1]
            
            gt_mel_frame = pred_melgate[:, :, 1:].reshape(pred_melgate.shape[0], -1)# -> [B, n_frames_per_step*n_mel]
            max_gate = torch.max(torch.max(pred_melgate[:, :, 0].data.cpu(), dim=1)[0].sigmoid(), max_gate)# -> [B]
            if max_gate.min() > gate_threshold:
                break
        else:# if loop finishes without breaking
            print(f"WARNING: Decoded {i+1} steps without finishing speaking. Breaking loop.")
        self.decoder.reset_kv()
        
        pred_melgate = torch.stack(pred_melgates, dim=1)# -> [B, mel_T//n_frames_per_step, n_frames_per_step*(n_mel+1)]
        pred_mel, pred_gate, alignments = self.reshape_outputs(pred_melgate, alignments)
        return pred_mel, pred_gate, alignments# [B, n_mel, mel_T], [B, mel_T], [B, mel_T, txt_T]


class Discriminator(nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.encoder = FeedForwardBlock(h)
        self.decoder = RecurrentBlock(h, randn_dim=0, output_dim=1)
        self.n_frames_per_step = h.n_frames_per_step
        self.n_mel = h.n_mel_channels
    
    def reshape_outputs(self, fakeness, feature_map):
        fakeness = fakeness.view(fakeness.shape[0], -1, fakeness.shape[2]//self.n_frames_per_step)# [B, mel_T//n_frames_per_step, n_frames_per_step] -> [B, mel_T, 1]
        feature_map = list(zip(*feature_map))# [mel_T, n_layers, ...] -> [n_layers, mel_T, ...] transpose list
        for layer_idx in range(len(feature_map)):
            feature_map[layer_idx] = None#torch.stack(feature_map[layer_idx], dim=1)# -> [B, mel_T, ...]
        return fakeness, feature_map
    
    def discriminator_loss(self,# gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
               pred_mel,              # FloatTensor[B, n_mel, mel_T]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd,# FloatTensor[B, 2]
                             gt_sylps,# FloatTensor[B]
                        torchmoji_hdn,# FloatTensor[B, embed]
                     freq_grad_scalar,# FloatTensor[B]
                            tf_frames):#       List[mel_T]
        
        real_fakeness, fake_fakeness, real_fm, fake_fm, alignments = self(gt_mel, mel_lengths, pred_mel, text, text_lengths, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd, gt_sylps, torchmoji_hdn, freq_grad_scalar, tf_frames, verbose=False)
        loss_disc = discriminator_loss(real_fakeness, fake_fakeness, mel_lengths.clamp(max=real_fakeness.shape[1]))
        
        return loss_disc, real_fakeness, fake_fakeness, alignments
    
    def forward(self,# gt_frame_logf0s, FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
               pred_mel,              # FloatTensor[B, n_mel, mel_T]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd,# FloatTensor[B, 2]
                             gt_sylps,# FloatTensor[B]
                        torchmoji_hdn,# FloatTensor[B, embed]
                     freq_grad_scalar,# FloatTensor[B]
                            tf_frames,#        List[mel_T]
                       verbose=False):
        
        encoder_outputs, spkrenc_outputs, emotion_embed = self.encoder(text, text_lengths, torchmoji_hdn, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
        self.decoder.update_kv(encoder_outputs, text_lengths)
        
        gt_mel_t = gt_mel[:, :, :(gt_mel.shape[-1]//self.n_frames_per_step)*self.n_frames_per_step]
        gt_mel_t = gt_mel_t.transpose(1, 2).reshape(gt_mel.shape[0], -1, self.n_mel*self.n_frames_per_step)# [B, n_mel, mel_T] -> [B, mel_T//n_frames_per_step, n_frames_per_step*n_mel]
        real_prenet_outputs = self.decoder.pre(gt_mel_t)# -> [B, mel_T//n_frames_per_step, prenet_dim]
        
        pred_mel = pred_mel[:, :, :(gt_mel.shape[-1]//self.n_frames_per_step)*self.n_frames_per_step]
        pred_mel = pred_mel.transpose(1, 2).reshape(gt_mel.shape[0], -1, self.n_mel*self.n_frames_per_step)# [B, n_mel, mel_T] -> [B, mel_T//n_frames_per_step, n_frames_per_step*n_mel]
        fake_prenet_outputs = self.decoder.pre(pred_mel)# -> [B, mel_T//n_frames_per_step, prenet_dim]
        #fake_prenet_outputs = real_prenet_outputs.roll(shifts=1, dims=0)
        
        r_declstm_output, r_attention_output, r_states = None, None, None
        real_declstm_outputs = []; real_attention_outputs = []; real_fm = []
        for i, (real_prenet_frame, fake_prenet_frame, b_tf) in enumerate(zip(real_prenet_outputs.unbind(1), fake_prenet_outputs.unbind(1), tf_frames)):
            r_declstm_output, r_attention_output, alignment, r_states = self.decoder(real_prenet_frame, r_attention_output, r_states)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            real_declstm_outputs.append(r_declstm_output); real_attention_outputs.append(r_attention_output); real_fm.append(r_states)# list.append([B, H]), list.append(TensorList([B, C, ...], [B, C, ...], ...))
        real_fakeness = self.decoder.post(real_declstm_outputs, real_attention_outputs, reset_kv=False)# -> [B, mel_T//n_frames_per_step, n_frames_per_step]
        real_fakeness += 0.5
        
        alignments = []
        f_declstm_output, f_attention_output, f_states = None, None, None
        fake_declstm_outputs = []; fake_attention_outputs = []; fake_fm = []
        for i, (real_prenet_frame, fake_prenet_frame, r_attention_output, r_states, b_tf) in enumerate(zip(real_prenet_outputs.unbind(1), fake_prenet_outputs.unbind(1), [None, *real_attention_outputs[:-1]], [None, *real_fm[:-1]], tf_frames)):
            f_declstm_output, f_attention_output, alignment, f_states = self.decoder(fake_prenet_frame, f_attention_output, f_states)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            fake_declstm_outputs.append(f_declstm_output); fake_attention_outputs.append(f_attention_output); fake_fm.append(f_states)# list.append([B, H]), list.append(TensorList([B, C, ...], [B, C, ...], ...))
            alignments.append(alignment)
        fake_fakeness = self.decoder.post(fake_declstm_outputs, fake_attention_outputs, reset_kv=True)# -> [B, mel_T//n_frames_per_step, n_frames_per_step]
        fake_fakeness += 0.5
        
        real_fakeness, real_fm = self.reshape_outputs(real_fakeness, real_fm)# -> ([B, mel_T, 1], [[B, mel_T, ...],]*n_layers)
        fake_fakeness, fake_fm = self.reshape_outputs(fake_fakeness, fake_fm)# -> ([B, mel_T, 1], [[B, mel_T, ...],]*n_layers)
        
        return real_fakeness, fake_fakeness, real_fm, fake_fm, torch.stack(alignments, dim=1)# [B, mel_T, 1], [B, mel_T, 1], [B, mel_T], [B, mel_T, txt_T]


class Model(nn.Module):
    def __init__(self, h):
        super(Model, self).__init__()
        self.generator = Generator(h)
        if h.GAN_enable:
            self.discriminator = Discriminator(h)
        self.fp16_run = h.fp16_run
    
    def parse_batch(self, batch, device='cuda'):
        if self.fp16_run:# convert data to half-precision before giving to GPU (to reduce wasted bandwidth)
            batch = {k: v.half() if type(v) is torch.Tensor and v.dtype is torch.float else v for k,v in batch.items()}
        batch = {k: v.to(device) if type(v) is torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def generator_loss(self,# gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                  gt_mel, mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                pred_mel,             # FloatTensor[B, n_mel, mel_T]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd,# FloatTensor[B, 2]
                             gt_sylps,# FloatTensor[B]
                        torchmoji_hdn,# FloatTensor[B, embed]
                     freq_grad_scalar,# FloatTensor[B]
                           tf_frames):#       List[mel_T]
        
        real_fakeness, fake_fakeness, real_fm, fake_fm, alignments = self.discriminator(gt_mel, mel_lengths, pred_mel, text, text_lengths, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd, gt_sylps, torchmoji_hdn, freq_grad_scalar, tf_frames, verbose=True)
        loss_fm  = None#feature_loss(real_fm, fake_fm)
        loss_gen = generator_loss(fake_fakeness, mel_lengths)
        
        return loss_gen, loss_fm
    
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
        
        out = self.generator(gt_mel, mel_lengths, text, text_lengths, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd, gt_sylps, torchmoji_hdn, freq_grad_scalar, inference_mode, teacher_force_prob)
        return out


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        loss += F.l1_loss(dr, dg)
    return loss*2


def masked_l1_loss(targ, pred, lengths):
    numel = lengths.sum()
    loss = F.l1_loss(*torch.broadcast_tensors(targ, pred), reduction='none').masked_fill_(~get_mask_from_lengths(lengths)[:, :pred.shape[1]].unsqueeze(2), 0.0).sum()/numel
    return loss


def masked_mse_loss(targ, pred, lengths):
    numel = lengths.sum()
    loss = F.mse_loss(*torch.broadcast_tensors(targ, pred), reduction='none').masked_fill_(~get_mask_from_lengths(lengths)[:, :pred.shape[1]].unsqueeze(2), 0.0).sum()/numel
    return loss


def discriminator_loss(dr, dg, mel_lengths):# [B, mel_T, 1], [B, mel_T, 1], [B]
    real_target = torch.tensor(1., device=dr.device, dtype=dr.dtype)
    fake_target = torch.tensor(0., device=dg.device, dtype=dg.dtype)
    
    r_loss = masked_mse_loss(real_target, dr, mel_lengths)# torch.mean((1-dr)**2)
    g_loss = masked_mse_loss(fake_target, dg, mel_lengths)# torch.mean(dg**2)
    return (r_loss + g_loss)


def generator_loss(dg, mel_lengths):
    real_target = torch.tensor(1., device=dg.device, dtype=dg.dtype)
    fake_target = torch.tensor(0., device=dg.device, dtype=dg.dtype)
    
    g_loss = masked_mse_loss(real_target, dg, mel_lengths)# torch.mean((1-dg)**2)
    return 2.*g_loss

