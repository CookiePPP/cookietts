import inspect
from math import sqrt
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

from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d, LnBatchNorm1d
from CookieTTS._2_ttm.MelFlow.model import FFT
from CookieTTS._2_ttm.arflow_hifigan.arflow import ARFlow
from CookieTTS._2_ttm.arflow_hifigan.HiFiGAN_wrapper import HiFiGAN
from CookieTTS.utils.model.transformer import PositionalEncoding

drop_rate = 0.5

def load_model(hparams, device='cuda'):
    model = ARFlow_HiFiGAN(hparams)
    if torch.cuda.is_available() or 'cuda' not in device:
        model = model.to(device)
    return model


class SSSUnit(nn.Module):# Dubbing this the Shift Scale Shift Unit, mixes information from input 2 into input 1.
    def __init__(self, x_dim, y_dim, enabled=True, grad_scale=1.0):
        super(SSSUnit, self).__init__()
        self.enabled = enabled
        self.grad_scale = grad_scale
        if enabled:
            self.linear = LinearNorm(y_dim, x_dim*3)
            nn.init.zeros_(self.linear.linear_layer.weight)
            nn.init.zeros_(self.linear.linear_layer.bias)
    
    def forward(self, x, y):
        if self.enabled:
            shift_scale = self.linear(y)
            shift_scale = grad_scale(shift_scale, self.grad_scale)
            preshift, logscale, postshift = shift_scale.chunk(3, dim=-1)
            return x.add(preshift).mul_(logscale.exp()).add(postshift)
        return x


class MDN(nn.Module):# Mixture Density Network. Will align a text sequence to an audio/spectrogram sequence. (Also contains an optional duration predictor for inference)
    def __init__(self, hparams, memory_dim):
        super(MDN, self).__init__()
        self.memory_efficient = hparams.memory_efficient
        
        self.MDN_mel_downscale = hparams.MDN_mel_downscale
        self.n_mel_channels    = hparams.n_mel_channels//self.MDN_mel_downscale
        self.smoothing_order   = 0
        self.smooth_left       = False
        
        self.register_buffer('pe', PositionalEncoding(memory_dim).pe)
        
        from CookieTTS._2_ttm.MelFlow.model import FFT
        self.lower  = FFT(memory_dim, hparams.mdn_n_heads, hparams.mdn_ff_dim, hparams.mdn_n_layers)
        self.higher = nn.Sequential(nn.Linear(memory_dim, memory_dim),
                           nn.LayerNorm(memory_dim),
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(memory_dim, 2*self.n_mel_channels))
        
        from CookieTTS._2_ttm.MelFlow.model import MelEncoder
        self.mel_enc = MelEncoder(hparams, hp_prepend='mdn_', output_dim=memory_dim)
        
        # duration predictors
        self.dp_lower  = FFT(memory_dim, hparams.durpred_n_heads, hparams.durpred_ff_dim, hparams.durpred_n_layers)
        self.dp_higher = nn.Linear(memory_dim, 1)
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    @torch.no_grad()
    def downsize_mel(self, gt_mel):
        if self.MDN_mel_downscale == 1:
            return gt_mel
        
        gt_mel = gt_mel.detach().exp()# [B, n_mel, mel_T]
        new_mel = gt_mel[:, 0::self.MDN_mel_downscale].clone()
        for i in range(1, self.MDN_mel_downscale):
            new_mel += gt_mel[:, i::self.MDN_mel_downscale]
        return new_mel.log()# [B, n_mel//downscale, mel_T]
    
    def infer(self, logdur, text_lengths):
        memory = memory + self.pe[:memory.shape[1]].unsqueeze(0)# [1, txt_T, hdn]
        x = self.dp_lower(memory, text_lengths)[0]# [B, txt_T, mem] -> # [B, txt_T, mem]
        logdur = self.dp_higher(x)# [B, txt_T, mem] -> # [B, txt_T, 1]
        dur = logdur.exp().squeeze(-1)# [B, txt_T]
        dur.masked_fill_(~get_mask_from_lengths(text_lengths), 0.0)
        attention_contexts, attention = self.align_duration(memory, dur, self.smoothing_order)
        mel_lengths = dur.sum(dim=1)
        return attention_contexts, attention, mel_lengths# [B, mel_T, C], [B, mel_T, txt_T], [B]
    
    def forward(self, memory, gt_mel, text_lengths, mel_lengths, mdn_align_grads=True):# [B, txt_T, mem], ...
        memory = memory + self.pe[:memory.shape[1]].unsqueeze(0)# [1, txt_T, hdn]
        
        x      = self.maybe_cp(self.dp_lower , *(memory, text_lengths))[0]# [B, txt_T, mem] -> [B, txt_T, mem]
        logdur = self.maybe_cp(self.dp_higher, *(x,)                  )   # [B, txt_T, mem] -> [B, txt_T,   1]
        
        memory = memory + self.mel_enc(gt_mel)[0].unsqueeze(1)
        memory        = self.maybe_cp(self.lower , *(memory, text_lengths))[0]# [B, txt_T, mem]
        mdn_mu_logvar = self.maybe_cp(self.higher, *(memory,)             )# [B, txt_T, 2*n_mel]
        
        new_mel = self.downsize_mel(gt_mel)
        
        if mdn_align_grads:
            mdn_loss, log_prob_matrix = self.MDNLoss(mdn_mu_logvar, new_mel, text_lengths, mel_lengths, self.n_mel_channels)# [B, txt_T, mel_T]
        else:
            with torch.no_grad():
                mdn_loss, log_prob_matrix = self.MDNLoss(mdn_mu_logvar, new_mel, text_lengths, mel_lengths, self.n_mel_channels)# [B, txt_T, mel_T]
        
        with torch.no_grad():
            alignment = self.viterbi(log_prob_matrix.detach().float().cpu(), text_lengths.cpu(), mel_lengths.cpu())
            alignment = alignment.transpose(1, 2)[:, :mel_lengths.max().item()].to(new_mel)# [B, mel_T, txt_T]
            
            for i in range(self.smoothing_order):
                pad = (-1,1) if i%2==0 or self.smooth_left else (1,-1)
                alignment += F.pad(alignment.transpose(1, 2), pad, mode='replicate').transpose(1, 2)# [B, mel_T, txt_T]
                alignment /= 2
        
        return mdn_loss, alignment, logdur# [B], [B, mel_T, txt_T], [B, txt_T, 1]
    
    #@torch.jit.script
    def MDNLoss(self, mu_logvar, z, text_lengths, mel_lengths, n_mel_channels:int=160, pad_mag:float=1e12):
        # mu, sigma: [B, txt_T, 2*n_mel]
        #         z: [B, n_mel,   mel_T]
        
        B, txt_T, _ = mu_logvar.size()
        mel_T = z.size(2)
        
        mu     = mu_logvar[:, :, :n_mel_channels ]# [B, txt_T, n_mel]
        logvar = mu_logvar[:, :,  n_mel_channels:]# [B, txt_T, n_mel]
        
        x = z.transpose(1, 2).unsqueeze(1)# [B, n_mel, mel_T] -> [B, 1, mel_T, n_mel]
        mu     = mu    .unsqueeze(2)# [B, txt_T, 1, n_mel]
        logvar = logvar.unsqueeze(2)# [B, txt_T, 1, n_mel]
        # SquaredError/logvar -> SquaredError/var -> NLL Loss
        # [B, 1, mel_T, n_mel]-[B, txt_T, 1, n_mel] -> [B, txt_T, mel_T, n_mel] -> [B, txt_T, mel_T]
        bx, bmu = torch.broadcast_tensors(x, mu)
        exponential = -0.5 * ( (F.mse_loss(bx, bmu, reduction='none')/logvar.exp())+logvar ).mean(dim=3, dtype=torch.float)
        
        log_prob_matrix = exponential# - (self.n_mel_channels/2)*torch.log(torch.tensor(2*math.pi))# [B, txt_T, mel_T] - [B, 1, mel_T]
        log_alpha = torch.ones(B, txt_T+1, mel_T, device=exponential.device, dtype=exponential.dtype)*(-pad_mag)
        log_alpha[:, 1, 0] = log_prob_matrix[:, 0, 0]
        
        for t in range(1, mel_T):
            prev_step = torch.cat([log_alpha[:, 1:, t-1:t], log_alpha[:, :-1, t-1:t]], dim=-1)
            log_alpha[:, 1:, t] = torch.logsumexp(prev_step.add_(1e-7), dim=-1).add(log_prob_matrix[:, :, t])
        
        log_alpha = log_alpha[:, 1:, :]
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1].clone()
        alpha_last = alpha_last/mel_lengths# avg by length of the log_alpha
        mdn_loss = -alpha_last
        
        return mdn_loss, log_prob_matrix
    
    @torch.jit.script
    def viterbi(log_prob_matrix, text_lengths, mel_lengths, pad_mag:float=1e12):
        B, L, T = log_prob_matrix.size()
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
    
    def align_duration(self,
            seq,                   # [B, txt_T, C]
            seq_dur,               # [B, txt_T]
            smoothing_order=4,     # int
            mel_lengths=None,      # [B, mel_T]
            seq_lengths=None,      # [B, txt_T]
            attention_override=None# [B, txt_T]
        ):
        
        if attention_override is None:
            B, txt_T, C = seq.shape# [B, Text Length, Encoder Dimension]
            
            if mel_lengths is None:# get Length of output Tensors
                mel_T = int(seq_dur.sum(dim=1).max().item())
            else:
                mel_T = mel_lengths.max().item()
            
            start_pos     = torch.zeros (B,               device=seq.device, dtype=seq.dtype)# [B]
            attention_pos = torch.arange(mel_T,           device=seq.device, dtype=seq.dtype).expand(B, mel_T)# [B, mel_T]
            attention     = torch.zeros (B, mel_T, txt_T, device=seq.device, dtype=seq.dtype)# [B, mel_T, txt_T]
            for enc_inx in range(seq_dur.shape[1]):
                dur = seq_dur[:, enc_inx]# [B]
                end_pos = start_pos + dur# [B]
                if seq_lengths is not None:# if last char in seq, extend this duration till end of non-padded area.
                    mask = (seq_lengths == (enc_inx+1))# [B]
                    if mask.any():
                        end_pos.masked_fill_(mask, mel_T)
                
                att = (attention_pos>=start_pos.unsqueeze(-1).repeat(1, mel_T)) & (attention_pos<end_pos.unsqueeze(-1).repeat(1, mel_T))
                attention[:, :, enc_inx][att] = 1.# set predicted duration values to positive
                
                start_pos = start_pos + dur # [B]
            
            for i in range(smoothing_order):
                pad = (-1,1) if i%2==0 or self.smooth_left else (1,-1)
                attention += F.pad(attention.transpose(1, 2), pad, mode='replicate').transpose(1, 2)# [B, mel_T, txt_T]
                attention /= 2
            
            if seq_lengths is not None:
                attention = attention * get_mask_3d(mel_lengths, seq_lengths)
        else:
            attention = attention_override
        return attention@seq, attention# [B, mel_T, txt_T] @ [B, txt_T, C] -> [B, mel_T, C], [B, mel_T, txt_T]


class Encoder(nn.Module):# Text Encoder, Learns stuff about text.
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__() 
        self.encoder_speaker_embed_dim = hparams.encoder_speaker_embed_dim
        if self.encoder_speaker_embed_dim:
            self.encoder_speaker_embedding = nn.Embedding(
            hparams.n_speakers, self.encoder_speaker_embed_dim)
        
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.encoder_conv_hidden_dim = hparams.encoder_conv_hidden_dim
        
        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            if _ == 0:
                if self.encoder_concat_speaker_embed == 'before_conv':
                    input_dim = hparams.symbols_embedding_dim+self.encoder_speaker_embed_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    input_dim = hparams.symbols_embedding_dim
                else:
                    raise NotImplementedError(f'encoder_concat_speaker_embed is has invalid value {hparams.encoder_concat_speaker_embed}, valid values are "before","inside".')
            else:
                input_dim = self.encoder_conv_hidden_dim
            
            if _ == (hparams.encoder_n_convolutions)-1: # last conv
                if self.encoder_concat_speaker_embed == 'before_conv':
                    output_dim = hparams.encoder_LSTM_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    output_dim = hparams.encoder_LSTM_dim-self.encoder_speaker_embed_dim
            else:
                output_dim = self.encoder_conv_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(input_dim,
                         output_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(output_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm_skip = LinearNorm(output_dim, hparams.encoder_LSTM_dim)
        self.lstm_n_layers = getattr(hparams, 'encoder_LSTM_n_layers', 1)
        self.lstm = nn.LSTM(hparams.encoder_LSTM_dim,
                            hparams.encoder_LSTM_dim//2, self.lstm_n_layers,
                            batch_first=True, bidirectional=True)
        self.LReLU = nn.LeakyReLU(negative_slope=0.01) # LeakyReLU
        
        self.sylps_layer = LinearNorm(hparams.encoder_LSTM_dim*self.lstm_n_layers, 1)
    
    def forward(self, text, text_lengths=None, speaker_ids=None):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.expand(-1, -1, text.size(2)) # extend across all encoder steps
            if self.encoder_concat_speaker_embed == 'before_conv':
                text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            text = F.dropout(self.LReLU(conv(text)), drop_rate, self.training)
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        text = text.transpose(1, 2)# -> [B, txt_T, C]
        text_skip = text
        
        if text_lengths is not None:
            # pytorch tensor are not reversible, hence the conversion
            text_lengths = text_lengths.cpu().numpy()
            text = nn.utils.rnn.pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, (hidden_state, _) = self.lstm(text)
        
        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        outputs = outputs + self.lstm_skip(text_skip)
        
        hidden_state = hidden_state.transpose(0, 1)# [2*lstm_n_layers, B, h_dim] -> [B, 2*lstm_n_layers, h_dim]
        B, _, h_dim = hidden_state.shape
        hidden_state = hidden_state.contiguous().view(B, -1)# [B, 2*lstm_n_layers, h_dim] -> [B, 2*lstm_n_layers*h_dim]
        pred_sylps = self.sylps_layer(hidden_state)# [B, 2*h_dim] -> [B, 1]
        
        return outputs, hidden_state, pred_sylps


class MemoryBottleneck(nn.Module):# Mix everything from the text half of the model into a single tensor. (This is normally the last step before the text is used for decoding)
    """
    Crushes the memory/encoder outputs dimension to save excess computation during Decoding.
    (If it works for the Attention then I don't see why it shouldn't also work for the Decoder)
    """
    def __init__(self, hparams, mem_dim):
        super(MemoryBottleneck, self).__init__()
        self.mem_output_dim = hparams.memory_bottleneck_dim
        self.bottleneck = LinearNorm(mem_dim, self.mem_output_dim, bias=hparams.memory_bottleneck_bias, w_init_gain='tanh')
        self.speaker_proj = LinearNorm(hparams.speaker_embedding_dim, self.mem_output_dim, bias=True)
    
    def forward(self, memory, speaker_embed):
        memory = self.bottleneck(memory)# [B, txt_T, input_dim] -> [B, txt_T, output_dim]
        if speaker_embed is not None:
            memory = memory * self.speaker_proj(speaker_embed).exp().unsqueeze(1)# [B, embed] -> [B, 1, output_dim]
        return memory


class Decoder(nn.Module):# Takes Text + Attention and outputs audio/spectrogram.
    def __init__(self, hparams, mem_dim):
        super(Decoder, self).__init__()
        self.memory_efficient = hparams.memory_efficient
        
        #self.norm1 = nn.LayerNorm(mem_dim, elementwise_affine=True)
        #self.norm2 = nn.LayerNorm(mem_dim, elementwise_affine=True)
        self.register_buffer('pe', PositionalEncoding(mem_dim).pe*10.)
        
        self.fft = []
        for i in range(hparams.dec_n_blocks):
            self.fft.append(FFT(mem_dim, hparams.dec_n_heads, hparams.dec_ff_dim, hparams.dec_n_layers, ff_kernel_size=3, rezero_pos_enc=False, add_position_encoding=bool(0), position_encoding_random_start=True))
        self.fft = nn.ModuleList(self.fft)
        
        self.mel_proj = LinearNorm(mem_dim, hparams.n_mel_channels)
        
        self.dec_glow = getattr(hparams, 'dec_glow', False)
        if self.dec_glow:
            self.arflow = ARFlow(mem_dim                              ,
                                 hparams.dec_glow_n_flows             ,
                                 hparams.n_mel_channels               ,
                                 hparams.memory_efficient             ,
                                 hparams.dec_glow_n_cond_layers       ,
                                 hparams.dec_glow_cond_hidden_channels,
                                 hparams.dec_glow_cond_output_channels,
                                 hparams.dec_glow_cond_kernel_size    ,
                                 hparams.dec_glow_cond_residual       ,
                                 hparams.dec_glow_cond_padding_mode   ,
                                 hparams.dec_glow_WN_config           ,)
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def fft_blocks(self, attention_contexts, mel_lengths):
        decoder_outputs = attention_contexts
        #decoder_outputs = self.norm1(decoder_outputs)
        decoder_outputs = decoder_outputs + self.pe[:decoder_outputs.shape[1]].unsqueeze(0)# [1, txt_T, hdn]
        #decoder_outputs = self.norm2(decoder_outputs)
        
        for i in range(len(self.fft)):
            decoder_outputs = self.maybe_cp(self.fft[i], *(decoder_outputs, mel_lengths))[0]# [B, mel_T, mem_dim] -> [B, mel_T, mem_dim]
        
        pred_mel = self.mel_proj(decoder_outputs)# [B, mel_T, mem_dim] -> [B, mel_T, n_mel]
        return decoder_outputs, pred_mel
    
    def forward(self, attention_contexts, mel_lengths, gt_mel):# [B, mel_T, mem_dim]
        decoder_outputs, pred_mel = self.fft_blocks(attention_contexts, mel_lengths)
        
        additional_outputs = {}
        if self.dec_glow:
            z, logdet_w, log_s = self.arflow(gt_mel, decoder_outputs.transpose(1, 2), z_lengths=mel_lengths)
            additional_outputs["melflow_pack"] = [z, logdet_w, log_s]
        
        return decoder_outputs, pred_mel, additional_outputs# [B, mel_T, mem_dim], [B, mel_T, n_mel]
    
    def infer(self, attention_contexts, mel_lengths, z_sigma=1.0):
        decoder_outputs, pred_mel = self.fft_blocks(attention_contexts, mel_lengths)
        
        if self.dec_glow:
            z = pred_mel.normal_()*z_sigma
            pred_mel = self.arflow.inverse(z.transpose(1, 2),
                             decoder_outputs.transpose(1, 2), z_lengths=mel_lengths)
        else:
            pred_mel = pred_mel.transpose(1, 2)# [B, mel_T, n_mel] -> [B, n_mel, mel_T]
        return decoder_outputs, pred_mel# [B, mel_T, mem_dim], [B, n_mel, mel_T]


class ARFlow_HiFiGAN(nn.Module):# Main module, contains all the submodules for the network.
    def __init__(self, hparams):
        super(ARFlow_HiFiGAN, self).__init__()
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels   = hparams.n_mel_channels
        self.memory_efficient = hparams.memory_efficient
        self.mdn_align_grads = True# override using 'run_every_epoch.py' or whatever.
        self.HiFiGAN_enable  = getattr(hparams, 'HiFiGAN_enable', False)
        
        if True:# Text Encoder
            self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
            std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.embedding.weight.data.uniform_(-val, val)
            
            self.encoder = Encoder(hparams)
            mem_dim = hparams.encoder_LSTM_dim
        
        if True:# Global + Local Conditioning
            self.speaker_embedding_dim = hparams.speaker_embedding_dim
            if self.speaker_embedding_dim:
                self.speaker_embedding = nn.Embedding(hparams.n_speakers, self.speaker_embedding_dim)
            
            self.tm_linear = nn.Linear(hparams.torchMoji_attDim, hparams.torchMoji_crushedDim)
            if hparams.torchMoji_BatchNorm:
                self.tm_bn = MaskedBatchNorm1d(hparams.torchMoji_attDim, eval_only_momentum=False, momentum=0.05)
            mem_dim += hparams.torchMoji_crushedDim
            
            if not hasattr(hparams, 'res_enc_n_tokens'):
                hparams.res_enc_n_tokens = 0
            if hasattr(hparams, 'use_res_enc') and hparams.use_res_enc:
                self.res_enc = ReferenceEncoder(hparams)
                mem_dim += getattr(hparams, 'res_enc_embed_dim', 128)
            
            if hparams.use_memory_bottleneck:
                self.memory_bottleneck = MemoryBottleneck(hparams, mem_dim)
                mem_dim = self.memory_bottleneck.mem_output_dim
        
        if True:# Memory Encoder
            self.mem_fft = FFT(mem_dim, hparams.mem_fft_n_heads, hparams.mem_fft_ff_dim, hparams.mem_fft_n_layers, ff_kernel_size=3, rezero_pos_enc=False, add_position_encoding=True, position_encoding_random_start=False)
        
        if True:# Mel Encoder + Mixture Density Network
            self.MDN = MDN(hparams, mem_dim)
        
        if True:# FESV+Logdur Autoregressive Flow
            self.fesv_bn = MaskedBatchNorm1d(7, eval_only_momentum=False, affine=False, momentum=0.1)
            #self.lnf0_bn = MaskedBatchNorm1d(1, eval_only_momentum=False, affine=False, momentum=0.1)
            self.ldur_bn = MaskedBatchNorm1d(1, eval_only_momentum=False, affine=False, momentum=0.1)
            self.fesdv_dim = 8
            self.arflow = ARFlow(mem_dim                            ,
                                 hparams.arflow_n_flows             ,
                                 self.fesdv_dim                     ,
                                 hparams.memory_efficient           ,
                                 hparams.arflow_n_cond_layers       ,
                                 hparams.arflow_cond_hidden_channels,
                                 hparams.arflow_cond_output_channels,
                                 hparams.arflow_cond_kernel_size    ,
                                 hparams.arflow_cond_residual       ,
                                 hparams.arflow_cond_padding_mode   ,
                                 hparams.arflow_WN_config           ,)
        
        if True:# Conditional Scale Shift the memory with FESV+logdur
            self.cond_ss = SSSUnit(mem_dim, self.fesdv_dim)
        
        if True:# Decoder
            self.decoder = Decoder(hparams, mem_dim)
        
        if False:# Vocoder
            self.hifigan = HiFiGAN(hparams)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def forward(self, gt_audio, audio_lengths,# FloatTensor[B, wav_T],  LongTensor[B]
                        gt_mel,   mel_lengths,# FloatTensor[B, mel_T],  LongTensor[B]
                          text,  text_lengths,#  LongTensor[B, txt_T],  LongTensor[B]
                               gt_char_logdur,# FloatTensor[B, txt_T]
                               gt_char_logf0 ,# FloatTensor[B, txt_T]
                               gt_char_fesv  ,# FloatTensor[B, 7, txt_T]
                                   speaker_id,#  LongTensor[B]
                                     gt_sylps,# FloatTensor[B]
                                torchmoji_hdn,# FloatTensor[B, embed]
#                                    alignment,# FloatTensor[B, mel_T, txt_T] # get alignment from file instead of recalculating.
            ):
        # package into dict for output
        outputs = {}
        
        gt_mel.requires_grad_()# <-(for gradient checkpoint func)
        
        memory = []
        
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, txt_T]
        encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]    
        outputs["encoder_outputs"] = encoder_outputs# [B, txt_T, enc_dim]
        outputs["pred_sylps"]      = pred_sylps     # [B]
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_id)# [B, embed]
            outputs["speaker_embed"] = speaker_embed# [B, embed]
        
        # (TorchMoji)
        if hasattr(self, 'tm_bn'):
            torchmoji_hdn = self.tm_bn(torchmoji_hdn).to(encoder_outputs)# [B, hdn_dim]
        torchmoji_hdn = self.tm_linear(torchmoji_hdn)          # [B, hdn_dim] -> [B, crushed_dim]
        memory.append(torchmoji_hdn[:, None].expand(-1, encoder_outputs.size(1), -1))# [B, C] -> [B, txt_T, C]
        
        # (Residual Encoder) gt_mel -> res_embed
        if hasattr(self, 'res_enc'):
            res_embed, zr, r_mu, r_logvar, r_mu_logvar = self.res_enc(res_gt_mel, mel_lengths, speaker_embed)# -> [B, embed]
            res_embed = res_embed[:, None].expand(-1, encoder_outputs.size(1), -1)# -> [B, txt_T, embed]
            memory.append(res_embed)
            outputs["res_enc_pkg"] = [r_mu, r_logvar, r_mu_logvar,]# [B, n_tokens], [B, n_tokens], [B, 2*n_tokens]
        
        # (Encoder/Attention) merge memory and calculate alignment override if used.
        memory = torch.cat(memory, dim=2)# concat along Embed dim # [B, txt_T, mem_dim]
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory, speaker_embed)
        
        # Memory FFT
        memory = self.maybe_cp(self.mem_fft, *(memory, text_lengths))[0]# [B, txt_T, mem_dim] -> [B, txt_T, mem_dim]
        
        # Mixture Density Network (aligns the chars)
        mdn_loss, alignment, logdur = self.MDN(memory, gt_mel, text_lengths, mel_lengths, mdn_align_grads=self.mdn_align_grads)
        outputs['mdn_loss']      = mdn_loss # [B]
        outputs['mdn_alignment'] = alignment# [B, mel_T, txt_T]
        outputs['pred_logdur']   = logdur   # [B, mel_T, txt_T]
        
        # ARFlow | FESV + Duration Predictor
        text_mask = get_mask_from_lengths(text_lengths)# [B, txt_T]
        bn_char_fesv   = self.fesv_bn(gt_char_fesv               , text_mask)# [B, 7, txt_T] -> [B, 7, txt_T]
        bn_char_logdur = self.ldur_bn(gt_char_logdur.unsqueeze(1), text_mask)# [B, txt_T]    -> [B, 1, txt_T]
        bn_fesvd = torch.cat((bn_char_logdur, bn_char_fesv,), dim=1)# [B, 1, txt_T], [B, 7, txt_T] -> [B, 8, txt_T]
        outputs['bn_fesvd'] = bn_fesvd
        
        z, logdet_w, log_s = self.arflow(bn_fesvd, memory.transpose(1, 2), z_lengths=text_lengths)
        outputs["fesvdglow_pack"] = [z, logdet_w, log_s]
        
        # Conditional Scale Shift
        memory = self.cond_ss(memory, bn_fesvd.transpose(1, 2))# [B, txt_T, mem_dim]
        
        # Extend memory -> attention_contexts
        attention_contexts = alignment @ memory# [B, mel_T, txt_T] @ [B, txt_T, mem_dim] -> [B, mel_T, mem_dim]
        
        # (Decoder) attention_contexts -> spectrogram
        decoder_outputs, pred_mel, aux_out = self.decoder(attention_contexts, mel_lengths, gt_mel)# [B, mel_T, mem_dim] -> [B, mel_T, mem_dim], [B, mel_T, n_mel_channels]
        outputs['decoder_outputs'] = decoder_outputs         # [B, mem_dim, n_mel]
        outputs['pred_mel']        = pred_mel.transpose(1, 2)# [B,   mel_T, n_mel] -> [B, n_mel, mel_T]
        if 'melflow_pack' in aux_out:
            outputs['melflow_pack']= aux_out['melflow_pack']
        
        if self.HiFiGAN_enable:
            pred_audio = self.hifigan(decoder_outputs)
        
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
    
    def inference(self, text_seq, text_lengths, speaker_id, torchmoji_hdn,
                    res_sigma=1.0, mel_z_sigma=1.0, char_z_sigma=1.0,
                    bn_fesvd=None, char_dur=None, gt_mel=None, alignment=None,
                    mel_lengths=None, trim_frames=8, trim_chars=4, disable_glow_decoder=False):# [B, enc_T], [B], [B], [B], [B, tm_dim]
        outputs = {}
        
        if disable_glow_decoder:
            self.dec_glow = False
        
        memory = []
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text_seq).transpose(1, 2) # [B, embed, txt_T]
        encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]    
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_id)# [B, embed]
            outputs["speaker_embed"] = speaker_embed# [B, embed]
        
        # (TorchMoji)
        if hasattr(self, 'tm_bn'):
            torchmoji_hdn = self.tm_bn(torchmoji_hdn).to(encoder_outputs)# [B, hdn_dim]
        torchmoji_hdn = self.tm_linear(torchmoji_hdn)          # [B, hdn_dim] -> [B, crushed_dim]
        memory.append(torchmoji_hdn[:, None].expand(-1, encoder_outputs.size(1), -1))# [B, C] -> [B, txt_T, C]
        
        # (Residual Encoder) gt_mel -> res_embed
        if hasattr(self, 'res_enc'):
            if gt_mel is not None:
                res_embed, zr, r_mu, r_logvar = self.res_enc(gt_mel, rand_sampling=False)# -> [B, embed]
            else:
                res_embed = self.res_enc.prior(encoder_outputs, std=res_sigma)# -> [B, embed]
            res_embed = res_embed[:, None].expand(-1, encoder_outputs.size(1), -1)# -> [B, txt_T, embed]
            memory.append(res_embed)
        
        # (Encoder/Attention) merge memory and calculate alignment override if used.
        memory = torch.cat(memory, dim=2)# concat along Embed dim # [B, txt_T, mem_dim]
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory, speaker_embed)
        
        # Memory FFT
        memory = self.maybe_cp(self.mem_fft, *(memory, text_lengths))[0]# [B, txt_T, mem_dim] -> [B, txt_T, mem_dim]
        
        # ARFlow | FESV + Duration Predictor
        if bn_fesvd is None:
            z = torch.empty(memory.shape[0], self.fesdv_dim, memory.shape[1], device=memory.device, dtype=memory.dtype).normal_()*char_z_sigma
            z_padded = F.pad(z, (trim_chars, trim_chars))# [B, txt_T, fesdv_dim] -> [B, txt_T+2*pad, fesdv_dim]
            memory_padded = F.pad(memory.transpose(1, 2), (trim_chars, trim_chars))# [B, txt_T, mem_dim] -> [B, mem_dim, txt_T+2*pad]
            pred_bn_fesvd = self.arflow.inverse(z_padded, memory_padded, z_lengths=text_lengths+(trim_chars*2))
            outputs['pred_bn_fesvd'] = bn_fesvd = pred_bn_fesvd[:, :, trim_chars:-trim_chars]# [B, fesdv_dim, txt_T]
            
        if char_dur is None:
            text_mask = get_mask_from_lengths(text_lengths)# [B, txt_T]
            pred_char_logdur = self.ldur_bn.inverse(bn_fesvd[:, :1, :], text_mask)# [B, 1, txt_T] -> [B, 1, txt_T]
            pred_char_dur    = pred_char_logdur.squeeze(1).exp().round()# [B, txt_T]
            pred_char_dur.masked_fill_(~text_mask, 0.)
            char_dur = pred_char_dur# [B, txt_T]
            char_dur[:, 0] = char_dur[:, 0].clamp(min=8.0, max=16.0) + trim_frames
            last_chars = text_mask.logical_xor(get_mask_from_lengths(text_lengths-1, max_len=text_mask.shape[1]))
            char_dur[last_chars] = char_dur[last_chars].clamp(min=8.0, max=16.0) + trim_frames
        
        # Conditional Scale Shift
        memory = self.cond_ss(memory, bn_fesvd.transpose(1, 2))# [B, txt_T, mem_dim]
        
        if mel_lengths is None:
            mel_lengths = char_dur.sum(dim=1).round().long()# [B]
        
        if alignment is None:
            # Get attention_contexts from durations
            attention_contexts = self.get_attention_from_lengths(memory, # [B, txt_T, mem_dim]
                                                          pred_char_dur, # [B, txt_T]
                                                           text_lengths,)# [B]
        else:
            # Get attention_contexts from alignment
            attention_contexts = alignment @ memory# [B, mel_T, txt_T] @ [B, txt_T, mem_dim] -> [B, mel_T, mem_dim]
            outputs["alignments"] = alignment# [B, mel_T, txt_T]
        
        # (Decoder/Attention) memory -> pred_mel
        decoder_outputs, pred_mel = self.decoder.infer(attention_contexts, mel_lengths, z_sigma=mel_z_sigma)
        decoder_outputs = decoder_outputs[:, trim_frames:-trim_frames   ]# [B, mel_T, mem_dim]
        pred_mel        = pred_mel       [:, :, trim_frames:-trim_frames]# [B, mel_T, n_mel]
        outputs["pred_mel"] = pred_mel# [B, n_mel, mel_T]
        
        #outputs["tokps"] = tokps     # [B] <- tokens per second
        
        mel_lengths -= trim_frames*2
        outputs['mel_lengths'] = mel_lengths
        return outputs
    
    def get_attention_from_lengths(self,
            seq        : Tensor,# FloatTensor[B, seq_T, enc_dim]
            seq_dur    : Tensor,# FloatTensor[B, seq_T]
            seq_masklen: Tensor,#  LongTensor[B]
        ):
        B, seq_T, seq_dim = seq.shape
        
        mask = get_mask_from_lengths(seq_masklen)
        seq_dur.masked_fill_(~mask, 0.0)
        
        seq_dur = seq_dur.round()#  [B, seq_T]
        dec_T = int(seq_dur.sum(dim=1).max().item())# [B, seq_T] -> int
        
        attention_contexts = torch.zeros(B, dec_T, seq_dim, device=seq.device, dtype=seq.dtype)# [B, dec_T, enc_dim]
        for i in range(B):
            mem_temp = []
            for j in range(int(seq_masklen[i].item())):
                duration = int(seq_dur[i, j].item())
                
                # [B, seq_T, enc_dim] -> [1, enc_dim] -> [duration, enc_dim]
                mem_temp.append( seq[i, j:j+1].repeat(duration, 1) )
            mem_temp = torch.cat(mem_temp, dim=0)# [[duration, enc_dim], ...] -> [dec_T, enc_dim]
            min_len = min(attention_contexts.shape[1], mem_temp.shape[0])
            attention_contexts[i, :min_len] = mem_temp[:min_len]
        
        return attention_contexts# [B, dec_T, enc_dim]