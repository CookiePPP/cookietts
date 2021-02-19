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
from CookieTTS._2_ttm.tacotron2_tm.model import ResBlock1d
from CookieTTS._2_ttm.VDVAETTS.HiFiGAN_wrapper import HiFiGAN
from CookieTTS.utils.model.transformer import PositionalEncoding

drop_rate = 0.5

def load_model(hparams, device='cuda'):
    model = VDVAETTS(hparams)
    if torch.cuda.is_available() or 'cuda' not in device:
        model = model.to(device)
    return model


class SSSUnit(nn.Module):# Dubbing this the Shift Scale Shift Unit, mixes information from input 2 into input 1.
    def __init__(self, x_dim, y_dim, enabled=True, grad_scale=1.0, preshift=True):
        super(SSSUnit, self).__init__()
        self.enabled = enabled
        self.preshift = preshift
        self.grad_scale = grad_scale
        if enabled:
            self.linear = LinearNorm(y_dim, x_dim*(2+preshift))
            nn.init.zeros_(self.linear.linear_layer.weight)
            nn.init.zeros_(self.linear.linear_layer.bias)
    
    def forward(self, x, y):# [..., x_dim], [..., y_dim]
        if self.enabled:
            shift_scale = self.linear(y)
            shift_scale = grad_scale(shift_scale, self.grad_scale)
            if self.preshift:
                preshift, logscale, postshift = shift_scale.chunk(3, dim=-1)
                return x.add(preshift).mul_(logscale.exp()).add(postshift)
            else:
                logscale, postshift = shift_scale.chunk(2, dim=-1)
                return x.mul(logscale.exp()).add(postshift)
        return x# [..., x_dim]


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


class VariancePredictor(nn.Module):
    """ Predicts [Duration, ...] using FFT Encoder outputs and VAE Latent. """
    def __init__(self, hparams, input_dim, cond_dim, latent_dim):
        super(VariancePredictor, self).__init__()
        self.memory_efficient = hparams.memory_efficient
        self.std = 0.0
        
        self.input_dim  = input_dim
        self.cond_dim   = cond_dim
        self.latent_dim = latent_dim
        self.lstm_dim      = hparams.varpred_lstm_dim
        self.lstm_n_layers = hparams.varpred_lstm_n_layers
        
        self.enc_lstm = nn.LSTM(cond_dim+input_dim, self.lstm_dim, num_layers=self.lstm_n_layers, bidirectional=True)
        self.enc_lstm_seq_post = LinearNorm(2*self.lstm_dim,                    2*self.latent_dim)
        self.enc_lstm_vec_post = LinearNorm(2*self.lstm_dim*self.lstm_n_layers, 2*self.latent_dim)
        
        self.dropout = hparams.varpred_lstm_dropout
        self.zoneout = hparams.varpred_lstm_zoneout
        
        inp_dim = cond_dim+input_dim+latent_dim+latent_dim
        self.lstm_cell = []
        for i in range(self.lstm_n_layers):
            self.lstm_cell.append(LSTMCellWithZoneout(inp_dim, self.lstm_dim, dropout=self.dropout, zoneout=self.zoneout))
            inp_dim = self.lstm_dim
        self.lstm_cell = nn.ModuleList(self.lstm_cell)
        self.lstm_proj = LinearNorm(inp_dim, input_dim)
        
        self.mem_cell = LSTMCellWithZoneout(cond_dim, cond_dim, dropout=0.0, zoneout=0.5)
        
        self.state_pred = []
        for i in range(self.lstm_n_layers):
            lin = ResBlock1d(cond_dim+(2*self.lstm_dim if i else 0), 4*self.lstm_dim, n_layers=3, n_dim=cond_dim+256, n_blocks=1, kernel_w=1, dropout=0.15)
            self.state_pred.append(lin)
        self.state_pred = nn.ModuleList(self.state_pred)
    
    def reparameterize(self, mu, logvar):# use for VAE sampling
        if self.std or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            if (not self.training) and self.std != 1.0:
                std *= float(self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def get_cond_summary(self, cond, text_lengths):# list[FloatTensor[B, cond_dim]], FloatTensor[B]
        txt_T = len(cond)
        B, cond_dim = cond[0].shape
        
        # Predict Final LSTM States with cond
        mem_states       = (cond[0].new_zeros(B, self.lstm_dim), cond[0].new_zeros(B, self.lstm_dim))
        mem_final_states = [cond[0].new_zeros(B, self.lstm_dim).requires_grad_(), cond[0].new_zeros(B, self.lstm_dim).requires_grad_()]
        
        for i, x in enumerate(cond):
            mem_states = self.mem_cell(x, mem_states)
            final_idx = (text_lengths==i+1)
            final_idx_sum = final_idx.sum()
            if final_idx_sum:
                mem_final_states[0] = torch.where(final_idx.unsqueeze(1), mem_states[0], mem_final_states[0])
                mem_final_states[1] = torch.where(final_idx.unsqueeze(1), mem_states[1], mem_final_states[1])
        cond_summary = mem_final_states[0]
        return cond_summary
    
    def encode(self, input, text_lengths):# [B, T, input_dim], [B]
        B, txt_T, input_dim = input.shape
        
        assert text_lengths.max() == txt_T, 'text_lengths.max() != text.shape[1] or text_lengths.max() != txt_T'
        assert text_lengths.min()  > 0, 'text_lengths.min() <= 0'
        input = nn.utils.rnn.pack_padded_sequence(input, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        input, (h, c) = self.enc_lstm(input)
        
        input = nn.utils.rnn.pad_packed_sequence(input, batch_first=True)[0]
        seq_z_mu, seq_z_logvar = self.enc_lstm_seq_post(input).chunk(2, dim=2)# -> [B, T, latent_dim], [B, T, latent_dim]
        seq_z = self.reparameterize(seq_z_mu, seq_z_logvar)
        
        h = h.view(self.enc_lstm.num_layers*2, B, self.enc_lstm.hidden_size).transpose(0, 1).reshape(B, -1)# -> [B, n_layers*2*lstm_dim]
        vec_z_mu, vec_z_logvar = self.enc_lstm_vec_post(h).chunk(2, dim=1)# -> [B, latent_dim], [B, latent_dim]
        vec_z = self.reparameterize(vec_z_mu, vec_z_logvar)
        return seq_z, seq_z_mu, seq_z_logvar, vec_z, vec_z_mu, vec_z_logvar
    
    def decode_tf(self, cond, input, text_lengths, seq_z, vec_z):
        B,  cond_dim, txt_T = cond.shape
        B, input_dim, txt_T = input.shape
        
        # Decode with LSTM Cells and Teacher-Forced Input
        states       = [(cond[0].new_zeros(B, self.lstm_dim), cond[0].new_zeros(B, self.lstm_dim)) for i in range(self.lstm_n_layers)]
        final_states = [[cond[0].new_zeros(B, self.lstm_dim),
                         cond[0].new_zeros(B, self.lstm_dim)] for i in range(self.lstm_n_layers)]
        
        tf_input = torch.cat((cond.transpose(1, 2),
             F.pad(input, (1, -1)).transpose(1, 2),
    *torch.broadcast_tensors(seq_z, vec_z.unsqueeze(1))), dim=2).unbind(1)# -> [[B, input_dim+cond_dim+latent_dim+latent_dim],]*txt_T
        
        pred_input = []
        for i, x in enumerate(tf_input):
            final_idx = (text_lengths==i+1)
            final_idx_sum = final_idx.sum()
            
            for j, lstm_cell in enumerate(self.lstm_cell):
                states[j] = lstm_cell(x, states[j])
                x = states[j][0]
                if final_idx_sum:
                    final_states[j][0] = torch.where(final_idx.unsqueeze(1), states[j][0], final_states[j][0])
                    final_states[j][1] = torch.where(final_idx.unsqueeze(1), states[j][1], final_states[j][1])
            pred_input.append(self.lstm_proj(x))# .append([B, input_dim])
        pred_input = torch.stack(pred_input, dim=2)# [B, input_dim, txt_T]
        pred_input = pred_input*get_mask_from_lengths(text_lengths).unsqueeze(1)
        
        final_states = torch.stack([torch.cat((h, c), dim=1) for h, c in final_states], dim=1)# [B, n_lstm, 2*lstm_dim]
        return pred_input, final_states
    
    def forward(self, cond, input, text_lengths):# [B, cond_dim, txt_T], [B, input_dim, txt_T], [B]
        
        # Encode -> get global and local Z from input
        enc_inp = torch.cat((cond, input), dim=1).transpose(1, 2)# -> [B, txt_T, cond_dim+input_dim]
        enc_latents = self.encode(enc_inp, text_lengths)# 3x[B, T, latent_dim], 3x[B, latent_dim]
        seq_z, seq_z_mu, seq_z_logvar, vec_z, vec_z_mu, vec_z_logvar = enc_latents
        
        # Decode with Teacher Forcing
        pred_input, final_states = self.maybe_cp(self.decode_tf, *(cond, input, text_lengths, seq_z, vec_z))
        
        return (pred_input, final_states, *enc_latents)# [B, input_dim, txt_T], [B, n_lstm, 4*lstm_dim]
    
    def infer(self, cond, text_lengths, std=None, global_std_mul=1.0, local_std_mul=1.0):# [B, txt_T, cond_dim]
        B, txt_T, cond_dim = cond.shape
        if std is not None:
            self.std = std
        
        # Predict input with cond + Autoregressive LSTM stack
        z_local  = torch.empty(B, txt_T, self.latent_dim, device=cond.device, dtype=cond.dtype).normal_()*local_std_mul
        z_global = torch.empty(B,     1, self.latent_dim, device=cond.device, dtype=cond.dtype).normal_()*global_std_mul
        z = torch.cat(torch.broadcast_tensors(z_local, z_global), dim=2)*self.std# -> [B, txt_T, 2*latent_dim]
        z_list = z.unbind(1)
        cond = cond.unbind(1)# [B, txt_T, cond_dim] -> [[B, cond_dim],]*txt_T
        
        states = [(cond[0].new_zeros(B, self.lstm_dim), cond[0].new_zeros(B, self.lstm_dim)) for i in range(self.lstm_n_layers)]
        x = cond[0].new_zeros(B, self.input_dim)# [B, input_dim]
        
        pred_input = []
        for i, (c, z) in enumerate(zip(cond, z_list)):
            x = torch.cat((c, x, z), dim=1)# [B, cond_dim], [B, input_dim], [B, 2*latent_dim] -> [B, cond_dim+input_dim+latent_dim+latent_dim]
            for j, lstm_cell in enumerate(self.lstm_cell):
                states[j] = lstm_cell(x, states[j])
                x = states[j][0]
            x = self.lstm_proj(x)
            pred_input.append(x)# .append([B, input_dim])
        pred_input = torch.stack(pred_input, dim=2)# [B, input_dim, txt_T]
        pred_input = pred_input*get_mask_from_lengths(text_lengths).unsqueeze(1)
        
        return pred_input

#@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):# uses mean and standard deviation
    return -0.5 +logsigma2 -logsigma1 + 0.5 * (logsigma1.exp().pow(2) + F.mse_loss(mu1, mu2, reduction='none')) / (logsigma2.exp().pow(2))

#@torch.jit.script
def gaussian_analytical_kl_var(mu1, mu2, logvar1, logvar2):# uses mean and variance
    """KLD between 2 sets of guassians. Returns Tensor with same shape as input."""
    return 0.5 * (-1.0 +logvar2 -logvar1 +logvar1.exp().add(F.mse_loss(mu1, mu2, reduction='none')).div(logvar2.exp()) )

class ResBlock(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=True, rezero=True, kernel_sizes=[1, 3, 3, 1]):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.rezero = rezero
        self.conv = []
        for i, kernel_size in enumerate(kernel_sizes):
            inp_dim = in_width  if   i==0                 else middle_width
            out_dim = out_width if 1+i==len(kernel_sizes) else middle_width
            self.conv.append(nn.Conv1d(inp_dim, out_dim, kernel_size, padding=(kernel_size-1)//2))
        self.conv = nn.ModuleList(self.conv)
        if self.residual and self.rezero:
            self.res_weight = nn.Parameter(torch.ones(1)*0.01)
    
    def forward(self, x):# [B, C_in, T]
        res = x
        for conv in self.conv:
            x = conv(F.gelu(x))
        if hasattr(self, 'res_weight'):
            if self.res_weight.abs() < 1e-6:
                self.res_weight.data.fill_(self.res_weight.sign().item()*0.03 or 0.03)
            x = x * self.res_weight
        if self.residual:# make res_channels match x_channels then add the Tensors.
            x_channels = x.shape[1]
            B, res_channels, T = res.shape
            if res_channels > x_channels:
                res = res[:, :x_channels, :]
            elif res_channels < x_channels:
                res = torch.cat((res, res.new_zeros(B, x_channels-res_channels, T)), dim=1)
            x = res + x
        if self.down_rate is not None:
            x = F.avg_pool2d(x, kernel_size=self.down_rate)
        return x# [B, C_out, T//down_rate]

class TopDownBlock(nn.Module):
    def __init__(self, hparams, hdn_dim, btl_dim, latent_dim, mem_dim=0):
        super().__init__()
        self.std = 0.5
        
        self.mem_dim = mem_dim
        self.btl_dim = btl_dim
        self.latent_dim = latent_dim
        self.enc    = ResBlock(2*btl_dim+mem_dim, hdn_dim,         2*latent_dim, residual=False)# get Z from target + input
        self.prior  = ResBlock(1*btl_dim+mem_dim, hdn_dim, btl_dim+2*latent_dim, residual=False)# guess Z from just input
        self.prior_weight = nn.Parameter(torch.ones(1)*0.01)
        self.z_proj = nn.Conv1d(latent_dim, btl_dim, 1)# expand Z to the input dim
        if getattr(hparams, 'topdown_resnet_enable', True):
            self.resnet = ResBlock(btl_dim, hdn_dim, btl_dim, residual=True)
    
    def reparameterize(self, mu, logsigma):# use for VAE sampling
        if self.std or self.training:
            std = torch.exp(logsigma)
            eps = torch.randn_like(std)
            if (not self.training) and self.std != 1.0:
                std *= float(self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def get_z(self, x, z_acts, attention_contexts):
        enc_input = [x, z_acts]
        if self.mem_dim:
            enc_input.append(attention_contexts)
        enc_input = torch.cat(enc_input, dim=1)
        z_mu, z_logsigma = self.enc(enc_input).chunk(2, dim=1)
        return z_mu, z_logsigma
    
    def pred_z(self, x, attention_contexts):
        res = x
        if self.mem_dim:
            x = torch.cat((x, attention_contexts), dim=1)
        x = self.prior(x)
        x, zp_mu, zp_logsigma = x.split([self.btl_dim, self.latent_dim, self.latent_dim], dim=1)
        x = res + x*self.prior_weight
        return x, zp_mu, zp_logsigma
    
    def get_z_embed(self, z_mu, z_logsigma):
        z = self.reparameterize(z_mu, z_logsigma)
        z_embed = self.z_proj(z)
        return z_embed
    
    def forward(self,  x,#      x:'Comes from previous DecBlock'                     FloatTensor[B, btl_dim, T]
                  z_acts,# z_acts:'Comes from spect ResBlocks during training'       FloatTensor[B, btl_dim, T]
 attention_contexts=None,#attent:'Comes from the Text Encoder after being expanded' FloatTensor[B, mem_dim, T]
           mel_mask=None,
         use_pred_z=False):
        z_mu, z_logsigma = self.get_z(x, z_acts, attention_contexts)
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)
        
        z_embed = self.get_z_embed(z_mu, z_logsigma)
        if use_pred_z:
            z_embed = self.get_z_embed(z_mu*0.01+zp_mu*0.99, z_logsigma*0.01+zp_logsigma*0.99)# -> [B, latent_dim]
        else:
            z_embed = self.get_z_embed(z_mu, z_logsigma)# -> [B, latent_dim]
        x = x + z_embed
        if mel_mask is not None:
            x.masked_fill_(~mel_mask, 0.0)
        
        if hasattr(self, 'resnet'):
            x = self.resnet(x)
            if mel_mask is not None:
                x.masked_fill_(~mel_mask, 0.0)
        
        kl = gaussian_analytical_kl(z_mu, zp_mu, z_logsigma, zp_logsigma)# [B, latent_dim, T]
        return x, kl# [B, btl_dim, T], [B, latent_dim, T]
    
    def infer(self, x, attention_contexts, mel_mask, std=None):
        if std is not None:
            self.std = std
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)
        
        z_embed = self.get_z_embed(zp_mu, zp_logsigma)
        x = x + z_embed
        x.masked_fill_(~mel_mask, 0.0)
        
        if hasattr(self, 'resnet'):
            x = self.resnet(x)
            x.masked_fill_(~mel_mask, 0.0)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, in_width, middle_width, out_seq_width, out_vec_width, down_rate=None, residual=False, rezero=True, output_sequence=True, output_vector=True, lstm_n_layers=1):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.rezero = rezero
        assert output_sequence or output_vector, 'this LSTM block has no outputs!'
        if self.residual:
            assert in_width==out_seq_width, 'residual is True but in_width != out_vec_width'
        self.output_sequence = output_sequence
        self.output_vector = output_vector
        
        self.conv_pre = nn.Conv1d(in_width,  middle_width, 1)
        self.lstm = nn.LSTM(middle_width, middle_width, lstm_n_layers, batch_first=True, dropout=0.0, bidirectional=True)
        if self.output_vector:
            self.lstm_proj = nn.Linear(2*middle_width*lstm_n_layers, out_vec_width)
        if self.output_sequence:
            self.conv_post = nn.Conv1d(2*middle_width, out_seq_width, 1)
        if self.residual and self.rezero:
            self.res_weight = nn.Parameter(torch.ones(1)*0.01)
    
    def forward(self, x):# [B, in_width, T]
        B, in_width, T = x.shape
        output = []
        
        res = x
        x = self.conv_pre(F.gelu(x))# -> [B, middle_width, T]
        x, (h, c) = self.lstm(x.transpose(1, 2))
        if self.output_sequence:
            x = x.transpose(1, 2)# -> [B, middle_width, T]
            x = self.conv_post(F.gelu(x))# -> [B, out_width, T]
            if hasattr(self, 'res_weight'):
                if self.res_weight.abs() < 1e-6:
                    self.res_weight.data.fill_(self.res_weight.sign().item()*0.03 or 0.03)
                x = x*self.res_weight
            if self.residual:
                x = res + x
            if self.down_rate is not None:
                x = F.avg_pool2d(x, kernel_size=self.down_rate)
            output.append(x)
        
        if self.output_vector:
            h = h.view(self.lstm.num_layers*2, B, self.lstm.hidden_size).transpose(0, 1).reshape(B, -1)# -> [B, n_layers*2*lstm_dim]
            h = self.lstm_proj(h)# -> [B, out_width]
            output.append(h)
        return tuple(output) if len(output) > 1 else output[0]

class TopDownLSTMBlock(nn.Module):
    def __init__(self, hparams, inp_dim, hdn_dim, btl_dim, latent_dim, mem_dim=0, n_layers=2):
        super().__init__()
        self.std = 0.5
        
        assert mem_dim, 'TopDownLSTMBlock requires mem_dim and attention_contexts'
        self.inp_dim = inp_dim
        self.mem_dim = mem_dim
        self.btl_dim = btl_dim
        self.latent_dim = latent_dim
        
        if hparams.exp_cond_proj:
            self.cond_proj = nn.Conv1d(mem_dim, mem_dim, 1)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        
        self.enc    = LSTMBlock(inp_dim+mem_dim+btl_dim, hdn_dim, out_seq_width=   None, out_vec_width=2*latent_dim, residual=False, output_sequence=False, output_vector=True, lstm_n_layers=n_layers)# get global Z from global z_acts + local input
        self.prior  = LSTMBlock(inp_dim+mem_dim        , hdn_dim, out_seq_width=btl_dim, out_vec_width=2*latent_dim, residual=False, output_sequence= True, output_vector=True, lstm_n_layers=n_layers)# guess global Z from just local input
        if self.inp_dim:
            self.prior_weight = nn.Parameter(torch.ones(1)*0.01)
        self.z_proj = nn.Linear(latent_dim, btl_dim, 1)# expand Z to the input dim
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)
        self.resnet = ResBlock(btl_dim, hdn_dim, btl_dim, residual=True)
    
    def reparameterize(self, mu, logsigma):# use for VAE sampling
        if self.std or self.training:
            std = torch.exp(logsigma)# NOTE THIS IS SIGMA
            eps = torch.randn_like(std)
            if (not self.training) and self.std != 1.0:
                std *= float(self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def get_z(self, x, z_acts, attention_contexts):
        assert attention_contexts is not None and z_acts is not None
        enc_input = [z_acts,]
        if self.inp_dim:
            assert x is not None
            enc_input.append(x)
        if self.mem_dim:
            enc_input.append(attention_contexts)
        enc_input = torch.cat(enc_input, dim=1) if len(enc_input) > 1 else enc_input[0]
        z_mu, z_logsigma = self.enc(enc_input).chunk(2, dim=1)
        return z_mu, z_logsigma
    
    def pred_z(self, x=None, attention_contexts=None):
        assert attention_contexts is not None
        if self.inp_dim:
            assert x is not None
            res = x
            attention_contexts = torch.cat((attention_contexts, x), dim=1)
        x, zp_mulogsigma = self.prior(attention_contexts)
        zp_mu, zp_logsigma = zp_mulogsigma.chunk(2, dim=1)
        if self.inp_dim:
            x = res + x*self.prior_weight
        return x, zp_mu, zp_logsigma
    
    def get_z_embed(self, z_mu, z_logsigma):
        z = self.reparameterize(z_mu, z_logsigma)# -> [B, latent_dim]
        z_embed = self.z_proj(z)# -> [B, embed]
        return z_embed
    
    def forward(self, x=None,#      x:'Comes from previous DecBlock'                     FloatTensor[B, btl_dim, T]
                 z_acts=None,# z_acts:'Comes from spect ResBlocks during training'       FloatTensor[B, btl_dim, T]
     attention_contexts=None,#attent:'Comes from the Text Encoder after being expanded' FloatTensor[B, mem_dim, T]
               mel_mask=None,
             use_pred_z=False,):
        assert z_acts is not None, 'z_acts is None'
        assert attention_contexts is not None, 'attention_contexts is None'
        if hasattr(self, 'cond_proj'):
            attention_contexts = attention_contexts*torch.exp(self.cond_proj(attention_contexts))
        
        z_mu, z_logsigma = self.get_z(x, z_acts, attention_contexts)# -> [B, latent_dim], [B, latent_dim]
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)# -> [B, btl_dim, T], [B, latent_dim], [B, latent_dim]
        if use_pred_z:
            z_embed = self.get_z_embed(z_mu*0.01+zp_mu*0.99, z_logsigma*0.01+zp_logsigma*0.99)# -> [B, latent_dim]
        else:
            z_embed = self.get_z_embed(z_mu, z_logsigma)# -> [B, latent_dim]
        x = x + z_embed.unsqueeze(-1)# [B, btl_dim, T] -> [B, btl_dim, T]
        if mel_mask is not None:
            x.masked_fill_(~mel_mask, 0.0)
        
        x = self.resnet(x)
        if mel_mask is not None:
            x.masked_fill_(~mel_mask, 0.0)
        
        kl = gaussian_analytical_kl(z_mu, zp_mu, z_logsigma, zp_logsigma).unsqueeze(-1)# -> [B, latent_dim, 1]
        return x, kl# [B, btl_dim, T], [B, latent_dim, 1]
    
    def infer(self, x, attention_contexts, mel_mask, std=None):
        if std is not None:
            self.std = std
        
        assert attention_contexts is not None, 'attention_contexts is None'
        if hasattr(self, 'cond_proj'):
            attention_contexts = attention_contexts*torch.exp(self.cond_proj(attention_contexts))
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)# -> [B, btl_dim, T], [B, latent_dim], [B, latent_dim]
        
        z_embed = self.get_z_embed(zp_mu, zp_logsigma)# -> [B, latent_dim]
        x = x + z_embed.unsqueeze(-1)# [B, btl_dim, T] -> [B, btl_dim, T]
        x.masked_fill_(~mel_mask, 0.0)
        
        x = self.resnet(x)
        x.masked_fill_(~mel_mask, 0.0)
        return x

class DecBlock(nn.Module):# N*ResBlock + Downsample + N*TopDownBlock + Upsample + Conv1d Grouped Cond
    def __init__(self, hparams, mem_dim, hdn_dim, btl_dim, latent_dim, n_blocks, scale=1):# hparams, hidden_dim, bottleneck_dim, latent_dim
        super().__init__()
        self.mem_dim = mem_dim
        
        if mem_dim and hparams.exp_cond_proj:
            self.res_cond_proj = nn.Conv1d(mem_dim, btl_dim, 1)
            nn.init.zeros_(self.res_cond_proj.weight)
            nn.init.zeros_(self.res_cond_proj.bias)
            
            self.td_cond_proj = nn.Conv1d(mem_dim, btl_dim, 1)
            nn.init.zeros_(self.td_cond_proj.weight)
            nn.init.zeros_(self.td_cond_proj.bias)
        
        self.n_blocks = n_blocks
        self.tdblock = []
        self.melresnet = []
        for i in range(self.n_blocks):
            is_first_block = bool(i==0)
            self.melresnet.append(ResBlock(btl_dim+(mem_dim*is_first_block), hdn_dim, btl_dim, residual=True))
            self.tdblock.append(TopDownBlock(hparams, hdn_dim, btl_dim, latent_dim, mem_dim if is_first_block else 0))
        self.melresnet = nn.ModuleList(self.melresnet)
        self.tdblock   = nn.ModuleList(self.tdblock)
        
        self.scale = scale
    
    def downsample(self, x):
        return F.avg_pool1d(x, kernel_size=self.scale)
    
    def upsample(self, x):
        return F.interpolate(x, scale_factor=self.scale)
    
    def forward_up(self, z_acts, attention_contexts):
        if self.mem_dim and hasattr(self, 'res_cond_proj'):
            z_acts = z_acts*torch.exp(self.res_cond_proj(attention_contexts))
        
        z_acts = torch.cat((z_acts, attention_contexts), dim=1)
        for i in range(self.n_blocks):
            z_acts = self.melresnet[i](z_acts)
        return z_acts
    
    def forward(self, x, z_acts, attention_contexts, mel_mask, use_pred_z=False):#     x:'Comes from previous DecBlock'
                                                     #z_acts:'Comes from spect ResBlocks during training'
                                                     #attent:'Comes from the Text Encoder after being expanded'
        if self.mem_dim and hasattr(self, 'td_cond_proj'):
            x = x*torch.exp(self.td_cond_proj(attention_contexts))
        
        kls = []
        for i in range(self.n_blocks):
            x, kl = self.tdblock[i](x, z_acts, attention_contexts, mel_mask, use_pred_z=use_pred_z)
            kls.append(kl)
        return x, kls# [B, btl_dim, T], list([B, latent_dim, T])
    
    def infer(self, x, attention_contexts, mel_mask, std=None):
        if std is not None:
            self.std = std
        
        if self.mem_dim and hasattr(self, 'td_cond_proj'):
            x = x*torch.exp(self.td_cond_proj(attention_contexts))
        
        for i in range(self.n_blocks):
            x = self.tdblock[i].infer(x, attention_contexts, mel_mask, std)
        
        return x

class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_dim=0, n_layers=1, dropout=0.0, zoneout=0.1):
        super(AutoregressiveLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.zoneout = zoneout
        
        inp_dim = input_dim+cond_dim
        self.lstm_cell = []
        for i in range(self.lstm_n_layers):
            self.lstm_cell.append(LSTMCellWithZoneout(inp_dim, self.hidden_dim, dropout=self.dropout, zoneout=self.zoneout))
            inp_dim = self.hidden_dim
        self.lstm_cell = nn.ModuleList(self.lstm_cell)
        self.lstm_proj = LinearNorm(inp_dim, input_dim)
    
    def forward(self, x, x_lengths, cond=None):# [B, T, input_dim], [B], [B, T, cond_dim]
        if self.cond_dim:
            assert cond is not None, 'self.cond_dim > 0 but cond is None'
        assert x.shape[2] == self.input_dim
        B, input_dim, T = x.shape
        
        cond = cond.unbind(2)          # [B,  cond_dim, T] -> [[B,  cond_dim],]*T
        x = F.pad(x, (1, -1)).unbind(2)# [B, input_dim, T] -> [[B, input_dim],]*T
        
        states       = [(x[0].new_zeros(B, self.hidden_dim), x[0].new_zeros(B, self.hidden_dim)) for i in range(self.n_layers)]
        final_states = [[x[0].new_zeros(B, self.hidden_dim),
                         x[0].new_zeros(B, self.hidden_dim)] for i in range(self.n_layers)]
        
        pred_x = []
        for i, (xi, ci) in enumerate(zip(x, cond)):
            final_idx = (x_lengths==i+1)
            final_idx_sum = final_idx.sum()
            
            xi = torch.cat((xi, ci), dim=1)# [B, input_dim], [B, cond_dim] -> [B, input_dim+cond_dim]
            for j, lstm_cell in enumerate(self.lstm_cell):
                states[j] = lstm_cell(xi, states[j])
                xi = states[j][0]
                if final_idx_sum:
                    final_states[j][0] = torch.where(final_idx.unsqueeze(1), states[j][0], final_states[j][0])
                    final_states[j][1] = torch.where(final_idx.unsqueeze(1), states[j][1], final_states[j][1])
            pred_x.append(self.lstm_proj(xi))# .append([B, input_dim])
        pred_x = torch.stack(pred_x, dim=2)# [B, input_dim, T]
        pred_x = pred_x*get_mask_from_lengths(x_lengths).unsqueeze(1)
        
        return pred_x
    
    def infer(self, x_lengths, cond=None):# [B], [B, T, cond_dim]
        states = [(x[0].new_zeros(B, self.hidden_dim), x[0].new_zeros(B, self.hidden_dim)) for i in range(self.n_layers)]
        xi = x[0].new_zeros(B, self.input_dim)
        
        pred_x = []
        for i, ci in enumerate(cond):
            xi = torch.cat((xi, ci), dim=1)# [B, input_dim], [B, cond_dim] -> [B, input_dim+cond_dim]
            for j, lstm_cell in enumerate(self.lstm_cell):
                states[j] = lstm_cell(xi, states[j])
                xi = states[j][0]
            xi = self.lstm_proj(xi)
            pred_x.append(xi)# .append([B, input_dim])
        pred_x = torch.stack(pred_x, dim=2)# [B, input_dim, T]
        pred_x = pred_x*get_mask_from_lengths(x_lengths).unsqueeze(1)
        
        return pred_x

class Decoder(nn.Module):
    def __init__(self, hparams, mem_dim, hdn_dim, btl_dim, latent_dim):# hparams, memory_dim, hidden_dim, bottleneck_dim, latent_dim
        super(Decoder, self).__init__()
        self.memory_efficient = False#hparams.memory_efficient
        self.use_pred_z = False
        self.n_blocks = hparams.decoder_n_blocks
        self.scale = 2
        self.downscales = [self.scale**i for i in range(self.n_blocks)]
        
        self.start = nn.Conv1d(hparams.n_mel_channels, btl_dim, 1)
        
        # project x -> spect
        self.spkr_proj = []
        for i in range(self.n_blocks):
            self.spkr_proj.append(LinearNorm(hparams.speaker_embedding_dim, mem_dim, bias=True))
        self.spkr_proj = nn.ModuleList(self.spkr_proj)
        
        # Decoder Blocks (one per timescale)
        # each contains ResBlock, TopDownBlock, upsample func, downsample func
        self.block = []
        for i in range(self.n_blocks):
            n_blocks = hparams.n_blocks_per_timescale if type(hparams.n_blocks_per_timescale) is int else hparams.n_blocks_per_timescale[i]
            self.block.append(DecBlock(hparams, mem_dim, hdn_dim, btl_dim, latent_dim, n_blocks, scale=self.scale))
        self.block = nn.ModuleList(self.block)
        
        self.lstm_up   = LSTMBlock(mem_dim+btl_dim, hdn_dim, btl_dim, out_vec_width=None, residual=False, output_vector=False)
        self.lstm_down = TopDownLSTMBlock(hparams, 0, hdn_dim, btl_dim, latent_dim, mem_dim)
        
        # project x -> spect
        self.proj = []
        for i in range(self.n_blocks):
            self.proj.append(nn.Conv1d(btl_dim, hparams.n_mel_channels, 1, bias=True))
        self.proj = nn.ModuleList(self.proj)
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def downsample_to_list(self, x):
        x_list = []
        for i, block in enumerate(self.block):
            x_list.append(x)
            
            is_last_block = (i+1==len(self.block))
            if not is_last_block:
                x = block.downsample(x)
        return x_list
    
    def get_mask_list(self, mel_lengths, type='floor'):
        mel_mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)# [B, 1, mel_T]
        mel_mask = F.pad(mel_mask, (0, self.downscales[-1]-mel_mask.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        
        mel_masks = self.downsample_to_list(mel_mask.float())
        if   type=='floor':
            mel_masks = [mask.floor().bool() for mask in mel_masks]
        elif type=='round':
            mel_masks = [mask.round().bool() for mask in mel_masks]
        elif type== 'ceil':
            mel_masks = [mask. ceil().bool() for mask in mel_masks]
        else:
            raise NotImplementedError
        return mel_mask, mel_masks
    
    def forward_up(self, gt_mel, *attention_contexts_list):
        gt_mel = F.pad(gt_mel, (0, self.downscales[-1]-gt_mel.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        z_acts = self.start(gt_mel)# [B, btl_dim, mel_T]
        z_acts_list = []
        for i, block in enumerate(self.block):# go up the ResBlock's
            z_acts = block.forward_up(z_acts, attention_contexts_list[i])
            if torch.isinf(z_acts).any() or torch.isnan(z_acts).any():
                print(f"Up ResBlock {i} has a nan or inf output.")
            z_acts_list.append(z_acts)
            
            is_last_block = (i+1==len(self.block))
            if not is_last_block:
                z_acts = block.downsample(z_acts)
        
        z_acts = torch.cat((attention_contexts_list[-1], z_acts), dim=1)
        z_acts_top = self.lstm_up(z_acts)# -> [B, btl_dim, mel_T]
        if torch.isinf(z_acts).any() or torch.isnan(z_acts).any():
            print(f"Up LSTMBlock has a nan or inf output.")
        return (z_acts_top, *z_acts_list)
    
    def forward_down(self, z_acts_top, z_acts_list, attention_contexts_list, mel_masks, use_pred_z=False):
        use_pred_z = use_pred_z or self.use_pred_z
        x, kl = self.lstm_down(None, z_acts_top, attention_contexts_list[-1], mel_masks[-1])
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(f"TopDownLSTMBlock has a nan or inf output.")
        
        x_list  = [ x,]
        kl_list = [kl,]
        for i, block in reversed(list(enumerate(self.block))):# go down the TopDownBlock stack
            x, kls = block(x, z_acts_list[i], attention_contexts_list[i], mel_masks[i], use_pred_z=use_pred_z)
            if torch.isinf(x).any() or torch.isnan(x).any():
                print(f"TopDownBlock {i} has a nan or inf output.")
            x_list.append(x)
            kl_list.extend(kls)
            
            is_first_block = (  i==0  )
            is_last_block  = (i+1==len(self.block))
            if not is_first_block:
                x = block.upsample(x)
        x_list  =  x_list[::-1]# reverse list -> [bottom, ..., top, toplstm]
        kl_list = kl_list[::-1]# reverse list -> [bottom, ..., top, toplstm]
        
        mel_list = []
        for i, proj in enumerate(self.proj):
            mel_list.append(proj(x_list[i]))
        # mel_list -> [bottom, ..., top,]
        
        return (*x_list, *mel_list, *kl_list)
    
    def forward(self, gt_mel, attention_contexts, mel_lengths, speaker_embed, use_pred_z=False):
        B,     n_mel, mel_T = gt_mel.shape
        B,   mem_dim, mel_T = attention_contexts.shape
        B, embed_dim        = speaker_embed.shape
        
        mel_mask, mel_masks = self.get_mask_list(mel_lengths)# [B, 1, mel_T], list([B, 1, mel_T//2**i] for i in range(self.n_blocks))
        
        # downsample attention_contexts
        attention_contexts = F.pad(attention_contexts, (0, self.downscales[-1]-attention_contexts.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        attention_contexts_list = self.downsample_to_list(attention_contexts)
        
        # add seperate speaker info for each attention_contexts timescale
        attention_contexts_list = [x+self.spkr_proj[i](speaker_embed).unsqueeze(2) for i, x in enumerate(attention_contexts_list)]
        
        z_acts_top, *z_acts_list = self.maybe_cp(self.forward_up, *(gt_mel, *attention_contexts_list))
        
        out = self.maybe_cp(self.forward_down, *(z_acts_top, z_acts_list, attention_contexts_list, mel_masks, use_pred_z))
        x_list, mel_list, kl_list = self.group_returned_tuple(out)
        
        return (*x_list, *mel_list, *kl_list)
    
    def group_returned_tuple(self, x):
        x_list   = x[                0:1*self.n_blocks+1]
        mel_list = x[1*self.n_blocks+1:2*self.n_blocks+1]
        kl_list  = x[2*self.n_blocks+1:                 ]
        return x_list, mel_list, kl_list
    
    def infer(self, attention_contexts, mel_lengths, speaker_embed, std):
        B,   mem_dim, mel_T = attention_contexts.shape
        B, embed_dim        = speaker_embed.shape
        
        mel_mask, mel_masks = self.get_mask_list(mel_lengths)# [B, 1, mel_T], list([B, 1, mel_T//2**i] for i in range(self.n_blocks))
        
        # downsample attention_contexts
        attention_contexts = F.pad(attention_contexts, (0, self.downscales[-1]-attention_contexts.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        attention_contexts_list = self.downsample_to_list(attention_contexts)
        
        # add seperate speaker info for each attention_contexts timescale
        attention_contexts_list = [x+self.spkr_proj[i](speaker_embed).unsqueeze(2) for i, x in enumerate(attention_contexts_list)]
        
        x = self.lstm_down.infer(None, attention_contexts_list[-1], mel_masks[-1], std)
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(f"TopDownLSTMBlock has a nan or inf output.")
        
        x_list  = [x,]
        for i, block in reversed(list(enumerate(self.block))):# go down the TopDownBlock stack
            x = block.infer(x, attention_contexts_list[i], mel_masks[i], std)
            if torch.isinf(x).any() or torch.isnan(x).any():
                print(f"TopDownBlock {i} has a nan or inf output.")
            x_list.append(x)
            
            is_first_block = (  i==0  )
            is_last_block  = (i+1==len(self.block))
            if not is_first_block:
                x = block.upsample(x)
        x_list = x_list[::-1]# reverse list -> [bottom, ..., top, toplstm]
        
        mel_list = []
        for i, proj in enumerate(self.proj):
            mel_list.append(proj(x_list[i]))
        # mel_list -> [bottom, ..., top,]
        
        pred_mel = mel_list[0]
        return pred_mel, x_list[0]# [B, n_mel, mel_T]

class Postnet(nn.Module):
    def __init__(self, hparams, hdn_dim, btl_dim, latent_dim):# hparams, memory_dim, hidden_dim, bottleneck_dim, latent_dim
        super(Postnet, self).__init__()
        self.std = 0.0
        self.f0_thresh = 0.5
        self.latent_dim = latent_dim
        self.f0s_dim = 4
        
        in_width  = btl_dim + hparams.n_mel_channels + self.f0s_dim
        out_width = 2*latent_dim
        self.encoder = nn.Sequential(*[ResBlock(in_width if i==0 else btl_dim, hdn_dim, out_width if i+1==hparams.postnet_n_blocks else btl_dim) for i in range(hparams.postnet_n_blocks)])
        
        in_width  = btl_dim
        out_width = 2*latent_dim + 2*self.f0s_dim
        self.prior = nn.Sequential(*[ResBlock(in_width if i==0 else btl_dim, hdn_dim, out_width if i+1==hparams.postnet_n_blocks else btl_dim) for i in range(hparams.postnet_n_blocks)])
        
        in_width  = latent_dim + self.f0s_dim
        out_width = hparams.n_mel_channels
        self.decoder = nn.Sequential(*[ResBlock(in_width if i==0 else btl_dim, hdn_dim, out_width if i+1==hparams.postnet_n_blocks else btl_dim) for i in range(hparams.postnet_n_blocks)])
    
    def reparameterize(self, mu, logsigma):# use for VAE sampling
        if self.std or self.training:
            std = torch.exp(logsigma)
            eps = torch.randn_like(std)
            if (not self.training) and self.std != 1.0:
                std *= float(self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x, gt_mel, gt_frame_logf0s, mel_lengths):# [B, btl_dim, mel_T], [B, n_mel, mel_T], [B, f0s_dim, mel_T]
        mel_mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)
        
        encoder_inputs = torch.cat((x, gt_mel, gt_frame_logf0s), dim=1)# -> [B, btl_dim+n_mel+f0s_dim, mel_T]
        z_mu_logsigma = self.encoder(encoder_inputs)# -> [B, 2*latent_dim, mel_T]
        z_mu_logsigma.masked_fill_(~mel_mask, 0.0)
        z_mu, z_logsigma = z_mu_logsigma.chunk(2, dim=1)# -> [B, latent_dim, mel_T], [B, latent_dim, mel_T]
        
        prior_out = self.prior(x)# -> [B, 2*latent_dim+2*f0s_dim, mel_T]
        prior_out.masked_fill_(~mel_mask, 0.0)
        pred_z_mu, pred_z_logsigma, pred_logf0s, pred_voiceds = prior_out.split([self.latent_dim, self.latent_dim, self.f0s_dim, self.f0s_dim], dim=1)
        pred_voiceds = pred_voiceds.sigmoid()
        # -> [B, latent_dim, mel_T], [B, latent_dim, mel_T], [B, f0s_dim, mel_T], [B, f0s_dim, mel_T]
        
        kld = gaussian_analytical_kl(z_mu,       pred_z_mu,
                                     z_logsigma, pred_z_logsigma)# -> [B, latent_dim, mel_T]
        
        z = self.reparameterize(z_mu, z_logsigma)# -> [B, latent_dim, mel_T]
        
        decoder_input = torch.cat((z, gt_frame_logf0s), dim=1)
        pred_mel = self.decoder(decoder_input)# [B, latent_dim, mel_T], [B, f0s_dim, mel_T] -> [B, n_mel_channels, mel_T]
        return pred_mel, kld, pred_logf0s, pred_voiceds# -> [B, n_mel_channels, mel_T], [B, latent_dim, mel_T]
    
    def infer(self, x):
        prior_out = self.prior(x)# -> [B, 2*latent_dim+2*f0s_dim, mel_T]
        pred_z_mu, pred_z_logsigma, pred_logf0s, pred_voiceds = prior_out.split([self.latent_dim, self.latent_dim, self.f0s_dim, self.f0s_dim], dim=1)
        pred_voiceds = pred_voiceds.sigmoid()
        # -> [B, latent_dim, mel_T], [B, latent_dim, mel_T], [B, f0s_dim, mel_T], [B, f0s_dim, mel_T]
        
        z = self.reparameterize(pred_z_mu, pred_z_logsigma)# -> [B, latent_dim, mel_T]
        
        # set any frames that don't have a pitch to 0.0
        pred_logf0s[pred_voiceds<self.f0_thresh] = 0.0
        
        decoder_input = torch.cat((z, pred_logf0s), dim=1)
        pred_mel = self.decoder(decoder_input)# [B, latent_dim, mel_T], [B, f0s_dim, mel_T] -> [B, n_mel_channels, mel_T]
        return pred_mel# -> [B, n_mel_channels, mel_T]

class FFTBlock(nn.Module):
    def __init__(self, hparams, hdn_dim, n_heads, ff_dim, n_layers, n_blocks):
        super(FFTBlock, self).__init__()
        self.memory_efficient = hparams.memory_efficient
        self.n_blocks = n_blocks
        self.register_buffer('pe', PositionalEncoding(hdn_dim).pe*3.)
        
        self.fft = []
        for i in range(self.n_blocks):
            self.fft.append(FFT(hdn_dim, n_heads, ff_dim, n_layers, ff_kernel_size=3, rezero_pos_enc=False, add_position_encoding=False, position_encoding_random_start=True))
        self.fft = nn.ModuleList(self.fft)
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def forward(self, x, lengths):# [B, T, hdn_dim], [B]
        x = x + self.pe[:x.shape[1]].unsqueeze(0)# [1, txt_T, hdn_dim]
        
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(f"FFT -1 has a nan or inf output.")
        for i in range(len(self.fft)):
            x = self.maybe_cp(self.fft[i], *(x, lengths))[0]# [B, T, hdn_dim] -> [B, T, hdn_dim]
            if torch.isinf(x).any() or torch.isnan(x).any():
                print(f"FFT {i} has a nan or inf output.")
        return x# [B, T, hdn_dim]


class VDVAETTS(nn.Module):# Main module, contains all the submodules for the network.
    def __init__(self, hparams):
        super(VDVAETTS, self).__init__()
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
        
        if True:# Variance Predictor
            self.var_dim = 1
            self.varpred = VariancePredictor(hparams, 1, mem_dim, hparams.varpred_n_tokens)
        
        if True:# Mel Encoder + Mixture Density Network
            self.MDN = MDN(hparams, mem_dim)
        
        if True:# FESV+Logdur Autoregressive Flow
            self.fesv_bn = MaskedBatchNorm1d(7, eval_only_momentum=False, affine=False, momentum=0.1)
            #self.lnf0_bn = MaskedBatchNorm1d(1, eval_only_momentum=False, affine=False, momentum=0.1)
            self.ldur_bn = MaskedBatchNorm1d(1, eval_only_momentum=False, affine=False, momentum=0.1)
        
        if True:# Conditional Scale Shift the memory with FESV+logdur
            self.cond_ss = SSSUnit(mem_dim, self.var_dim)
        
        if True:# Attention Decoder
            self.att_dec = FFTBlock(hparams, mem_dim, hparams.att_dec_n_heads, hparams.att_dec_ff_dim, hparams.att_dec_n_layers, hparams.att_dec_n_blocks)
        
        if True:# Decoder
            hdn_dim = hparams.dec_hidden_dim
            btl_dim = hparams.dec_bottleneck_dim
            latent_dim = hparams.dec_latent_dim
            self.decoder = Decoder(hparams, mem_dim, hdn_dim, btl_dim, latent_dim)
        
        if getattr(hparams, 'pitch_postnet_enable', False):
            latent_dim = hparams.postnet_latent_dim
            self.postnet = Postnet(hparams, hdn_dim, btl_dim, latent_dim)
        
        if self.HiFiGAN_enable:# Vocoder
            #self.hifigan_res = LinearNorm(hparams.n_mel_channels, mem_dim)
            pass
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def get_speaker_embed(self, speaker_id, y=None):
        return self.speaker_embedding(speaker_id)
    
    def forward(self, gt_audio, audio_lengths,# FloatTensor[B, wav_T],  LongTensor[B]
                        gt_mel,   mel_lengths,# FloatTensor[B, n_mel, mel_T],  LongTensor[B]
                          text,  text_lengths,#  LongTensor[B, txt_T],  LongTensor[B]
                               gt_char_dur   ,# FloatTensor[B, txt_T]
                               gt_char_logdur,# FloatTensor[B, txt_T]
                               gt_char_logf0 ,# FloatTensor[B, txt_T]
                               gt_char_fesv  ,# FloatTensor[B, 7, txt_T]
                             gt_frame_logf0s ,# FloatTensor[B, f0s_dim, mel_T]
                             gt_frame_voiceds,# FloatTensor[B, f0s_dim, mel_T]
                                   speaker_id,#  LongTensor[B]
                                     gt_sylps,# FloatTensor[B]
                                torchmoji_hdn,# FloatTensor[B, embed]
                                    alignment,# FloatTensor[B, mel_T, txt_T] # get alignment from file instead of recalculating.
                             use_pred_z=False,# Bool # Sample Z from Prior instead of Encoder. Should NEVER be used for training, only for Validation or Testing.
            ):
        try:    alignment = alignment
        except: alignment = None
        
        # package into dict for output
        outputs = {}
        
        gt_mel.requires_grad_()# <-(for gradient checkpoint func)
        
        memory = []
        
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, txt_T]
        if torch.isinf(embedded_text).any() or torch.isnan(embedded_text).any():
            print(f"embedded_text has a nan or inf output.")
        encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]
        if torch.isinf(encoder_outputs).any() or torch.isnan(encoder_outputs).any():
            print(f"encoder_outputs has a nan or inf output.")
        outputs["encoder_outputs"] = encoder_outputs# [B, txt_T, enc_dim]
        outputs["pred_sylps"]      = pred_sylps     # [B]
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.maybe_cp(self.get_speaker_embed, *(speaker_id, encoder_outputs))# [B, embed]
            if torch.isinf(speaker_embed).any() or torch.isnan(speaker_embed).any():
                print(f"speaker_embed has a nan or inf output.")
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
        if torch.isinf(memory).any() or torch.isnan(memory).any():
            print(f"memory has a nan or inf output.")
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory, speaker_embed)
            if torch.isinf(memory).any() or torch.isnan(memory).any():
                print(f"memory_bottleneck output has a nan or inf output.")
        
        # Memory FFT
        memory = self.maybe_cp(self.mem_fft, *(memory, text_lengths))[0]# [B, txt_T, mem_dim] -> [B, txt_T, mem_dim]
        if torch.isinf(memory).any() or torch.isnan(memory).any():
            print(f"mem_fft output has a nan or inf output.")
        
        # Mixture Density Network (aligns the chars)
        if alignment is None:
            mdn_loss, alignment, logdur = self.MDN(memory, gt_mel, text_lengths, mel_lengths, mdn_align_grads=self.mdn_align_grads)
            outputs['mdn_loss']      = mdn_loss # [B]
            outputs['mdn_alignment'] = alignment# [B, mel_T, txt_T]
            outputs['pred_logdur']   = logdur   # [B, mel_T, txt_T]
        
        # get BatchNorm FESVD
        text_mask = get_mask_from_lengths(text_lengths)# [B, txt_T]
        bn_char_fesv   = self.fesv_bn(gt_char_fesv               , text_mask)# [B, 7, txt_T] -> [B, 7, txt_T]
        bn_char_logdur = self.ldur_bn(gt_char_logdur.unsqueeze(1), text_mask)# [B, txt_T]    -> [B, 1, txt_T]
        outputs['bn_logdur'] = bn_char_logdur
        bn_fesvd = torch.cat((bn_char_logdur, bn_char_fesv,), dim=1)# [B, 1, txt_T], [B, 7, txt_T] -> [B, 8, txt_T]
        outputs['bn_fesvd'] = bn_fesvd
        
        if hasattr(self, 'varpred'):
            pred_logdur, final_states, *vp_enc_latents = self.varpred(memory.transpose(1, 2), bn_char_logdur, text_lengths)# [B, 8, txt_T], [B, n_lstm, 4*lstm_dim]
            outputs['varpred_pred'] = pred_logdur# [B, 1, txt_T]
            outputs['varpred_latents'] = vp_enc_latents
            outputs['varpred_hidden'] = final_states# [B, n_lstm, 4*lstm_dim]
        
        # Char Conditional Scale Shift
        memory = self.cond_ss(memory, bn_char_logdur.transpose(1, 2))# [B, txt_T, mem_dim], [B, 1, txt_T] -> [B, txt_T, mem_dim]
        if torch.isinf(memory).any() or torch.isnan(memory).any():
            print(f"cond_ss output has a nan or inf output.")
        
        # Extend memory -> attention_contexts
        attention_contexts = alignment @ memory# [B, mel_T, txt_T] @ [B, txt_T, mem_dim] -> [B, mel_T, mem_dim]
        
        # (Attention Decoder) FFT
        attention_contexts = self.att_dec(attention_contexts, mel_lengths)# [B, mel_T, mem_dim] -> [B, mel_T, mem_dim]
        
        self.decoder.use_pred_z = use_pred_z
        dec_out = self.maybe_cp(self.decoder, *(gt_mel, attention_contexts.transpose(1, 2), mel_lengths, speaker_embed))
        x_list, mel_list, kl_list = self.decoder.group_returned_tuple(dec_out)
        outputs['dec_x_list'   ] = x_list
        outputs['dec_kl_list'  ] = kl_list
        outputs['pred_mel_list'] = mel_list
        outputs['pred_mel'] = mel_list[0][:, :, :mel_lengths.max()]
        
        x = x_list[0][:, :, :mel_lengths.max()]
        if hasattr(self, 'postnet'):
            postnet_out = self.maybe_cp(self.postnet, *(x, gt_mel, gt_frame_logf0s, mel_lengths))
            pred_mel, kld, pred_frame_f0s, pred_frame_voiceds = postnet_out
            outputs['postnet_pred_mel'] = pred_mel# [B, n_mel, mel_T]
            outputs['postnet_kld']      = kld     # [B, n_mel, mel_T]
            outputs['postnet_pred_logf0s']  = pred_frame_f0s    # [B, f0s_dim, mel_T]
            outputs['postnet_pred_voiceds'] = pred_frame_voiceds# [B, f0s_dim, mel_T]
        
        if self.HiFiGAN_enable:
            outputs['hifigan_inputs'] = x# [B, btl_dim, mel_T]
        
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
                    char_sigma=1.0, frame_sigma=1.0,
                    bn_logdur=None, char_dur=None, gt_mel=None, alignment=None,
                    mel_lengths=None,):# [B, enc_T], [B], [B], [B], [B, tm_dim]
        outputs = {}
        
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
        memory = self.mem_fft(memory, text_lengths)[0]# [B, txt_T, mem_dim] -> [B, txt_T, mem_dim]
        
        # Char Variance Predictor
        if hasattr(self, 'varpred') and bn_logdur is None:
            bn_logdur = self.varpred.infer(memory, text_lengths, std=char_sigma)# [B, 8, txt_T], [B, n_lstm, 4*lstm_dim]
        
        if char_dur is None:
            text_mask = get_mask_from_lengths(text_lengths)# [B, txt_T]
            char_logdur = self.ldur_bn.inverse(bn_logdur, text_mask)# -> [B, 1, txt_T]
            char_dur = char_logdur.squeeze(1).exp()
            char_dur.masked_fill_(~text_mask, 0.0)
            char_dur = char_dur.round().long()
        
        # Char Conditional Scale Shift
        memory = self.cond_ss(memory, bn_logdur.transpose(1, 2))# [B, txt_T, mem_dim]
        
        if mel_lengths is None:
            if char_dur.dtype is torch.float:
                char_dur = char_dur.round().long()
            mel_lengths = char_dur.sum(dim=1)# [B]
        outputs['mel_lengths'] = mel_lengths
        
        if alignment is None:
            # Get attention_contexts from durations
            attention_contexts = self.get_attention_from_lengths(memory, # [B, txt_T, mem_dim]
                                                               char_dur, # [B, txt_T]
                                                           text_lengths,)# [B]
        else:
            # Get attention_contexts from alignment
            attention_contexts = alignment @ memory# [B, mel_T, txt_T] @ [B, txt_T, mem_dim] -> [B, mel_T, mem_dim]
            outputs["alignments"] = alignment# [B, mel_T, txt_T]
        
        # (Attention Decoder) FFT
        attention_contexts = self.att_dec(attention_contexts, mel_lengths)# [B, mel_T, mem_dim] -> [B, mel_T, mem_dim]
        
        # (Decoder/Attention) memory -> pred_mel
        pred_mel, hifigan_inputs = self.decoder.infer(attention_contexts.transpose(1, 2), mel_lengths, speaker_embed, frame_sigma)
        outputs["pred_mel"] = pred_mel[:, :, :mel_lengths.max()]# [B, n_mel, mel_T]
        if self.HiFiGAN_enable:
            outputs['hifigan_inputs'] = hifigan_inputs[:, :, :mel_lengths.max()]# [B, btl_dim, mel_T]
        
        return outputs
    
    def get_attention_from_lengths(self,
            seq        : Tensor,# FloatTensor[B, seq_T, enc_dim]
            seq_dur    : Tensor,# FloatTensor[B, seq_T]
            seq_masklen: Tensor,#  LongTensor[B]
        ):
        B, seq_T, seq_dim = seq.shape
        
        mask = get_mask_from_lengths(seq_masklen)
        seq_dur.masked_fill_(~mask, 0.0)
        
        if seq_dur.dtype is torch.float:
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
