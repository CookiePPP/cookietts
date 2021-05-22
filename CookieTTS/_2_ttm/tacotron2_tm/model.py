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

from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm, LSTMCellWithZoneout, GMMAttention, DynamicConvolutionAttention
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame, freeze_grads, grad_scale, elapsed_timer

from CookieTTS._2_ttm.tacotron2_ssvae.nets.SylpsNet import SylpsNet
from CookieTTS._2_ttm.tacotron2_tm.modules_vae import ReferenceEncoder
from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d, LnBatchNorm1d

drop_rate = 0.5

def load_model(hparams, device='cuda'):
    model = Tacotron2(hparams)
    if torch.cuda.is_available() or 'cuda' not in device:
        model = model.to(device)
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min
    return model


class SSSUnit(nn.Module):# Dubbing this the Shift Scale Shift Unit
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

class MDN(nn.Module):
    def __init__(self, hparams, memory_dim):
        super(MDN, self).__init__()
        self.MDN_mel_downscale = hparams.MDN_mel_downscale
        self.n_mel_channels    = hparams.n_mel_channels//self.MDN_mel_downscale
        self.smoothing_order   = 4
        self.smooth_left       = False
        
        from CookieTTS.utils.model.transformer import PositionalEncoding
        self.register_buffer('pe', PositionalEncoding(memory_dim).pe)
        
        from CookieTTS._2_ttm.MelFlow.model import FFT
        self.lower  = FFT(memory_dim, hparams.mdn_n_heads, hparams.mdn_ff_dim, hparams.mdn_n_layers, legacy=True)
        self.higher = nn.Sequential(nn.Linear(memory_dim, memory_dim),
                           nn.LayerNorm(memory_dim),
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(memory_dim, 2*self.n_mel_channels))
        
        from CookieTTS._2_ttm.MelFlow.model import MelEncoder
        self.mel_enc = MelEncoder(hparams, hp_prepend='mdn_', output_dim=memory_dim)
        
        # duration predictors
        self.dp_lower  = FFT(memory_dim, hparams.durpred_n_heads, hparams.durpred_ff_dim, hparams.durpred_n_layers, legacy=True)
        self.dp_higher = nn.Linear(memory_dim, 1)
    
    @torch.no_grad()
    def downsize_mel(self, gt_mel):
        if self.MDN_mel_downscale == 1:
            return gt_mel
        
        gt_mel = gt_mel.detach().exp()# [B, n_mel, mel_T]
        new_mel = gt_mel[:, 0::self.MDN_mel_downscale].clone()
        for i in range(1, self.MDN_mel_downscale):
            new_mel += gt_mel[:, i::self.MDN_mel_downscale]
        return new_mel.log()# [B, n_mel//downscale, mel_T]
    
    def infer(self, memory, text_lengths):
        memory = memory + self.pe.roll(random.randint(0, 4999), 0)[:memory.shape[1]].unsqueeze(0)# [1, txt_T, hdn]
        x = self.dp_lower(memory, text_lengths)[0]# [B, txt_T, mem] -> # [B, txt_T, mem]
        logdur = self.dp_higher(x)# [B, txt_T, mem] -> # [B, txt_T, 1]
        dur = logdur.exp().squeeze(-1)# [B, txt_T]
        dur.masked_fill_(~get_mask_from_lengths(text_lengths), 0.0)
        attention_contexts, attention = self.align_duration(memory, dur, self.smoothing_order)
        mel_lengths = dur.sum(dim=1)
        return attention_contexts, attention, mel_lengths# [B, mel_T, C], [B, mel_T, txt_T], [B]
    
    def forward(self, memory, gt_mel, text_lengths, mel_lengths, mdn_align_grads=True):# [B, txt_T, mem], ...
        memory = memory + self.pe.roll(random.randint(0, 4999), 0)[:memory.shape[1]].unsqueeze(0)# [1, txt_T, hdn]
        
        x = self.dp_lower(memory, text_lengths)[0]# [B, txt_T, mem] -> # [B, txt_T, mem]
        logdur = self.dp_higher(x)# [B, txt_T, mem] -> # [B, txt_T, 1]
        
        memory = memory + self.mel_enc(gt_mel)[0].unsqueeze(1)
        memory = self.lower(memory, text_lengths)[0]# [B, txt_T, mem]
        mdn_mu_logvar = self.higher(memory)# [B, txt_T, 2*n_mel]
        
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
        exponential = -0.5 * ( ((x-mu).pow_(2)/logvar.exp())+logvar ).mean(dim=3, dtype=torch.float)
        
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


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim, out_bias=False):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=out_bias, w_init_gain='tanh')
    
    def forward(self, attention_weights_cat): # [B, 2, enc]
        processed_attention = self.location_conv(attention_weights_cat) # [B, 2, enc] -> [B, n_filters, enc]
        processed_attention = processed_attention.transpose(1, 2) # [B, n_filters, enc] -> [B, enc, n_filters]
        processed_attention = self.location_dense(processed_attention) # [B, enc, n_filters] -> [B, enc, attention_dim]
        return processed_attention # [B, enc, attention_dim]


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 windowed_attention_range: int=0,
                 windowed_att_pos_learned: bool=True,
                 windowed_att_pos_offset: float=0.,
                 attention_learned_temperature: bool=False):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, # Crushes the Encoder outputs to Attention Dimension used by this module
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.windowed_attention_range = windowed_attention_range
        if windowed_att_pos_learned is True:
            self.windowed_att_pos_offset = nn.Parameter( torch.zeros(1) )
        else:
            self.windowed_att_pos_offset = windowed_att_pos_offset
        
        self.softmax_temp = nn.Parameter(torch.tensor(0.0)) if attention_learned_temperature else None
        self.score_mask_value = -float("inf")
    
    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat,
                mask             : Optional[Tensor] = None,
                memory_lengths   : Optional[Tensor] = None,
                attention_weights: Optional[Tensor] = None,
                current_pos      : Optional[Tensor] = None,
                score_mask_value : float = -float('inf'),
                return_override  : bool = False) -> Tuple[Tensor, Tensor]:
        """
        PARAMS
        ------
        attention_hidden_state:
            [B, AttRNN_dim] FloatTensor
                attention rnn last output
        memory:
            [B, txt_T, enc_dim] FloatTensor
                encoder outputs
        processed_memory:
            [B, txt_T, proc_enc_dim] FloatTensor
                processed encoder outputs
        attention_weights_cat:
            [B, 2 (or 3), txt_T] FloatTensor
                previous, cummulative (and sometimes exp_avg) attention weights
        mask:
            [B, txt_T] BoolTensor
                mask for padded data
        attention_weights: (Optional)
            [B, txt_T] FloatTensor
                optional override attention_weights
                useful for duration predictor attention or perfectly copying a clip with an alternative speaker.
        """
        B, txt_T, enc_dim = memory.shape
        
        if return_override:
            pred_attention_weights = attention_weights
        else:
            processed = self.location_layer(attention_weights_cat) # [B, 2, txt_T] # conv1d, matmul
            processed.add_( self.query_layer(attention_hidden_state.unsqueeze(1)).expand_as(processed_memory) ) # unsqueeze, matmul, expand_as, add_
            processed.add_( processed_memory ) # add_
            alignment = self.v( torch.tanh( processed ) ).squeeze(-1) # tanh, matmul, squeeze
            
            if mask is not None:
                if self.windowed_attention_range > 0 and current_pos is not None:
                    if self.windowed_att_pos_offset:
                        current_pos = current_pos + self.windowed_att_pos_offset
                    max_end = memory_lengths - 1 - self.windowed_attention_range
                    min_start = self.windowed_attention_range
                    current_pos = torch.min(current_pos.clamp(min=min_start), max_end.to(current_pos))
                    
                    mask_start = (current_pos-self.windowed_attention_range).clamp(min=0).round() # [B]
                    mask_end = mask_start+(self.windowed_attention_range*2)                       # [B]
                    pos_mask = torch.arange(txt_T, device=current_pos.device).unsqueeze(0).expand(B, -1)  # [B, txt_T]
                    pos_mask = (pos_mask >= mask_start.unsqueeze(1).expand(-1, txt_T)) & (pos_mask <= mask_end.unsqueeze(1).expand(-1, txt_T))# [B, txt_T]
                    
                    # attention_weights_cat[pos_mask].view(B, self.windowed_attention_range*2+1) # for inference masked_select later
                    
                    mask = mask | ~pos_mask# [B, txt_T] & [B, txt_T] -> [B, txt_T]
                alignment.data.masked_fill_(mask, score_mask_value)#    [B, txt_T]
            
            softmax_temp = self.softmax_temp
            if softmax_temp is not None:
                alignment = alignment*softmax_temp.exp()
            
            pred_attention_weights = F.softmax(alignment, dim=1)# [B, txt_T] # softmax along encoder tokens dim
        if attention_weights is None:
            attention_weights = pred_attention_weights
        
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)# unsqueeze, bmm
                                      # [B, 1, txt_T] @ [B, txt_T, enc_dim] -> [B, 1, enc_dim]
        
        attention_context = attention_context.squeeze(1)# [B, 1, enc_dim] -> [B, enc_dim] # squeeze
        
        new_pos = (pred_attention_weights*torch.arange(txt_T, device=pred_attention_weights.device).expand(B, -1)).sum(1)
                       # ([B, txt_T] * [B, txt_T]).sum(1) -> [B]
        
        return attention_context, pred_attention_weights, new_pos# [B, enc_dim], [B, txt_T]


class Prenet(nn.Module):
    def __init__(self, hparams, in_dim, sizes):
        super(Prenet, self).__init__()
        self.in_dim = in_dim
        self.sizes  = sizes
        
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.use_speaker_embed = getattr(hparams, 'prenet_speaker_embed', False)
        if self.use_speaker_embed:
            in_dim+=self.speaker_embedding_dim
        
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.p_prenet_dropout = hparams.p_prenet_dropout
        self.prenet_batchnorm = getattr(hparams, 'prenet_batchnorm', False)
        self.p_prenet_input_dropout = 0.
        
        self.speaker_linears = None
        if getattr(hparams, 'prenet_speaker_embed_latents', False):
            self.speaker_linears = nn.ModuleList([LinearNorm(self.speaker_embedding_dim, 3*size) for size in sizes])
        
        self.bn_momentum = hparams.prenet_bn_momentum
        self.batchnorms = None
        if self.prenet_batchnorm:
            self.batchnorms = nn.ModuleList([ MaskedBatchNorm1d(size, eval_only_momentum=False, momentum=self.bn_momentum) for size in sizes ])
            self.batchnorms.append(MaskedBatchNorm1d(in_dim, eval_only_momentum=False, momentum=self.bn_momentum))
    
    def forward(self, x, speaker_embed=None, disable_dropout=False):
        if self.use_speaker_embed:
            if len(x.shape) == 3 and len(speaker_embed.shape) == 2:
                speaker_embed = speaker_embed.unsqueeze(0).expand(x.shape[0], -1, -1)# [B, embed] -> [T, B, embed]
            x = torch.cat((x, speaker_embed), dim=-1)# [..., C] +[..., embed] -> [..., C+embed]
        
        #if self.batchnorms is not None:
        #    x = self.batchnorms[-1](x)
        
        if self.p_prenet_input_dropout and (not disable_dropout):# dropout from the input, definitely a dangerous idea, but I think it would be very interesting to try values like 0.05 and see the effect
            x = F.dropout(x, self.p_prenet_input_dropout, self.training)
        
        for i, linear in enumerate(self.layers):
            x = F.relu(linear(x))
            if self.p_prenet_dropout > 0 and (not disable_dropout):
                x = F.dropout(x, p=self.p_prenet_dropout, training=True)
            if self.batchnorms is not None:
                x = self.batchnorms[i](x)
            if self.speaker_linears is not None:
                shift_scale = self.speaker_linears[i](speaker_embed)
                preshift, scale, postshift = shift_scale.chunk(3, dim=-1)
                x = x.add(preshift).mul(scale).add_(postshift)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.b_res = hparams.postnet_residual_connections if hasattr(hparams, 'postnet_residual_connections') else False
        self.convolutions = nn.ModuleList()
        
        prev_output_layer = True
        for i in range(hparams.postnet_n_convolutions):
            is_output_layer = (bool(self.b_res) and bool( i % self.b_res == 0 )) or (i+1 == hparams.postnet_n_convolutions)
            layers = [ ConvNorm(hparams.n_mel_channels if prev_output_layer else hparams.postnet_embedding_dim,
                             hparams.n_mel_channels if is_output_layer else hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='linear' if is_output_layer else 'tanh'), ]
            if not is_output_layer:
                layers.append(nn.BatchNorm1d(hparams.postnet_embedding_dim))
            prev_output_layer = is_output_layer
            self.convolutions.append(nn.Sequential(*layers))
    
    def forward(self, x):
        x_orig = x.clone()
        len_convs = len(self.convolutions)
        for i, conv in enumerate(self.convolutions):
            if (bool(self.b_res) and bool( i % self.b_res == 0 )) or (i+1 == len_convs):
                x_orig = x_orig + conv(x)
                x = x_orig
            else:
                x = F.dropout(torch.tanh(conv(x)), drop_rate, self.training)
        
        return x_orig


class Encoder(nn.Module):
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
        assert not (torch.isnan(text).any() or torch.isinf(text).any())
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        assert not (torch.isnan(text).any() or torch.isinf(text).any())
        
        text = text.transpose(1, 2)# [B, txt_T, C]
        
        if text_lengths is not None:
            # pytorch tensor are not reversible, hence the conversion
            text_lengths = text_lengths.cpu().numpy()
            text = nn.utils.rnn.pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, (hidden_state, _) = self.lstm(text)
        assert not (torch.isnan(hidden_state).any() or torch.isinf(hidden_state).any())
        
        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        assert not (torch.isnan(outputs).any() or torch.isinf(outputs).any())
        
        hidden_state = hidden_state.transpose(0, 1)# [2*lstm_n_layers, B, h_dim] -> [B, 2*lstm_n_layers, h_dim]
        B, _, h_dim = hidden_state.shape
        hidden_state = hidden_state.contiguous().view(B, -1)# [B, 2*lstm_n_layers, h_dim] -> [B, 2*lstm_n_layers*h_dim]
        pred_sylps = self.sylps_layer(hidden_state)# [B, 2*h_dim] -> [B, 1]
        assert not (torch.isnan(pred_sylps).any() or torch.isinf(pred_sylps).any())
        
        return outputs, hidden_state, pred_sylps


class MemoryBottleneck(nn.Module):
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


@torch.jit.ignore
def HeightGaussianBlur(inp, blur_strength=1.0):
    """
    inp: [B, H, W] FloatTensor
    blur_strength: Float - min 0.0, max 5.0"""
    inp = inp.unsqueeze(1) # [B, height, width] -> [B, 1, height, width]
    var_ = blur_strength
    norm_dist = torch.distributions.normal.Normal(0, var_)
    conv_kernel = torch.stack([norm_dist.cdf(torch.tensor(i+0.5)) - norm_dist.cdf(torch.tensor(i-0.5)) for i in range(int(-var_*3-1),int(var_*3+2))], dim=0)[None, None, :, None]
    input_padding = (conv_kernel.shape[2]-1)//2
    out = F.conv2d(F.pad(inp, (0,0,input_padding,input_padding), mode='reflect'), conv_kernel).squeeze(1) # [B, 1, height, width] -> [B, height, width]
    return out

class VariationalEncoder(nn.Module):
    def __init__(self, hparams):
        super(VariationalEncoder, self).__init__()
        self.std = 0.95
        self.n_tokens = hparams.ve_n_tokens
        self.lstm = nn.LSTM(hparams.n_mel_channels, hparams.ve_lstm_dim, hparams.ve_n_lstm, batch_first=False)
        self.token_linear = nn.Linear(hparams.ve_lstm_dim, hparams.ve_n_tokens*2)
        self.embed_linear = nn.Linear(hparams.ve_n_tokens, hparams.ve_embed_dim)
    
    def sample_randomly(self, x:bool):
        self.std=float(x)
    
    def reparameterize(self, mu, logvar):# use for VAE sampling
        if self.std or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            if (not self.training) and self.std != 1.0:
                std *= float(self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, spect: Optional[torch.Tensor], B:int=0):
        squeeze_t = False
        if spect is not None and len(spect.shape) == 2:# [B, n_mel]
            spect = spect.unsqueeze(0)
            squeeze_t = True
        
        if spect is not None:# if spect is a Tensor
            spect = self.lstm(spect)[0]# [mel_T, B, n_mel] -> [mel_T, B, lstm_dim]
            spect = self.token_linear(spect)# [mel_T, B, lstm_dim] -> [mel_T, B, n_tokens*2]
            _ = spect.chunk(2, dim=-1)# -> [[mel_T, B, n_tokens], [mel_T, B, n_tokens]]
            mu, logvar = _# -> [mel_T, B, n_tokens], [mel_T, B, n_tokens]
            tokens = self.reparameterize(mu, logvar)
        else:# else, if spect is None
            assert B != 0# batch size required is spect is missing
            device=next(self.parameters()).device
            dtype =next(self.parameters()).dtype
            tokens = torch.randn(B, self.n_tokens, device=device, dtype=dtype)# [B, n_tokens]
            mu     = torch.zeros(B, self.n_tokens, device=device, dtype=dtype)# [B, n_tokens]
            logvar = torch. ones(B, self.n_tokens, device=device, dtype=dtype)# [B, n_tokens]
        
        embed = self.embed_linear(tokens)# [..., n_tokens] -> [..., embed]
        if squeeze_t:
            embed = embed.squeeze(0)# [1, B, embed]
        return embed, mu, logvar

class Decoder(nn.Module):
    attention_hidden:        Optional[torch.Tensor]# init self vars
    attention_cell:          Optional[torch.Tensor]# init self vars
    second_attention_hidden: Optional[torch.Tensor]# init self vars
    second_attention_cell:   Optional[torch.Tensor]# init self vars
    
    decoder_hidden:        Optional[torch.Tensor]# init self vars
    decoder_cell:          Optional[torch.Tensor]# init self vars
    second_decoder_hidden: Optional[torch.Tensor]# init self vars
    second_decoder_cell:   Optional[torch.Tensor]# init self vars
    third_decoder_hidden:  Optional[torch.Tensor]# init self vars
    third_decoder_cell:    Optional[torch.Tensor]# init self vars
    
    attention_weights:     Optional[torch.Tensor]# init self vars
    attention_weights_cum: Optional[torch.Tensor]# init self vars
    saved_attention_weights:     Optional[torch.Tensor]# init self vars
    saved_attention_weights_cum: Optional[torch.Tensor]# init self vars
    
    attention_context: Optional[torch.Tensor]# init self vars
    previous_location: Optional[torch.Tensor]# init self vars
    
    memory:           Optional[torch.Tensor]# init self vars
    processed_memory: Optional[torch.Tensor]# init self vars
    
    mask: Optional[torch.Tensor]# init self vars
    gate_delay: int
    
    def __init__(self, hparams, mem_dim):
        super(Decoder, self).__init__()
        self.n_mel_channels     = hparams.n_mel_channels
        self.stft_clamp_val     = hparams.stft_clamp_val
        self.n_frames_per_step  = hparams.n_frames_per_step
        self.context_frames     = hparams.context_frames
        self.n_frames_per_frame = getattr(hparams, 'n_frames_per_frame', 1)
        self.n_fpf_per_elem     = getattr(hparams, 'n_fpf_per_elem', False)
        self.max_decoder_steps  = hparams.max_decoder_steps
        self.gate_threshold     = hparams.gate_threshold
        self.p_teacher_forcing  = hparams.p_teacher_forcing
        self.teacher_force_till = hparams.teacher_force_till
        self.use_speaker_everywhere = getattr(hparams, 'use_speaker_everywhere', False)
        sss_grad_scale = 0.25
        
        self.memory_dim = mem_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.use_DecoderRNG  = getattr(hparams, 'use_DecoderRNG', False)
        self.DecoderRNG_dim  = getattr(hparams, 'DecoderRNG_dim', 0)
        
        self.DecRNN_hidden_dropout_type = hparams.DecRNN_hidden_dropout_type
        self.p_DecRNN_hidden_dropout    = hparams.p_DecRNN_hidden_dropout
        
        self.windowed_attention_range = hparams.windowed_attention_range if hasattr(hparams, 'windowed_attention_range') else 0
        self.windowed_att_pos_offset  = hparams.windowed_att_pos_offset  if hasattr(hparams, 'windowed_att_pos_offset' ) else 0
        if self.windowed_attention_range:
            self.exp_smoothing_factor = nn.Parameter( torch.ones(1) * 0.0 )
        
        
        self.inference_mode = False# use during actual free-running inference, will disable predicted attention
        
        self.half_inference_mode = False# updated by "run_every_epoch.py" when InferenceGAN is enabled.
        self.HiFiGAN_enable      = getattr(hparams, 'HiFiGAN_enable', False)
        self.gradient_checkpoint = getattr(hparams, 'gradient_checkpoint', False)
        self.decode_chunksize    = getattr(hparams, 'checkpoint_decode_chunksize', 32)
        self.hide_startstop_tokens = hparams.hide_startstop_tokens
        
        self.dump_attention_weights = False# dump as in, not store in VRAM
        self.return_attention_rnn_outputs = False
        self.return_decoder_rnn_outputs   = False
        self.return_attention_contexts    = False
        
        ######################
        ##  Default States  ##
        ######################
        self.attention_hidden = None
        self.attention_cell   = None
        self.second_attention_hidden = None
        self.second_attention_cell   = None
        
        self.decoder_hidden = None
        self.decoder_cell   = None
        self.second_decoder_hidden = None
        self.second_decoder_cell   = None
        self.third_decoder_hidden = None
        self.third_decoder_cell   = None
        
        self.attention_weights     = None
        self.attention_weights_cum = None
        self.saved_attention_weights     = None
        self.saved_attention_weights_cum = None
        self.attention_position = None
        
        self.attention_context = None
        self.previous_location = None
        self.attention_weights_scaler = None
        
        self.memory = None
        self.processed_memory = None
        self.mask = None
        self.gate_delay = 0
        
        ###############
        ##  Modules  ##
        ###############
        
        # Prenet
        self.prenet_dim = hparams.prenet_dim
        self.prenet_speaker_embed_dim = getattr(hparams, 'prenet_speaker_embed_dim', 0)
        self.prenet = Prenet(hparams,
            (hparams.n_mel_channels+self.prenet_speaker_embed_dim)*hparams.n_frames_per_step*self.context_frames,
            [hparams.prenet_dim]*hparams.prenet_layers,
        )
        
        self.prenet_noise    = hparams.prenet_noise
        self.prenet_blur_min = hparams.prenet_blur_min
        self.prenet_blur_max = hparams.prenet_blur_max
        
        # Variational Encoder
        if getattr(hparams, 'use_ve', False):
            self.variational_encoder = VariationalEncoder(hparams)
        
        # Attention/Input LSTM
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.AttRNN_hidden_dropout_type = hparams.AttRNN_hidden_dropout_type
        self.p_AttRNN_hidden_dropout    = hparams.p_AttRNN_hidden_dropout
        self.AttRNN_extra_decoder_input = hparams.AttRNN_extra_decoder_input
        self.AttRNN_use_global_cond     = getattr(hparams, 'AttRNN_use_global_cond', False)
        
        AttRNN_Dimensions = hparams.prenet_dim + self.memory_dim
        if self.AttRNN_extra_decoder_input:
            AttRNN_Dimensions += hparams.decoder_rnn_dim
        if self.use_DecoderRNG:
            AttRNN_Dimensions += self.DecoderRNG_dim
        if  getattr(hparams, 'use_ve', False):
            AttRNN_Dimensions += hparams.ve_embed_dim
        global_cond_dim = hparams.speaker_embedding_dim+hparams.torchMoji_crushedDim+1
        if self.AttRNN_use_global_cond:
            AttRNN_Dimensions += global_cond_dim
        
        self.attention_rnn = LSTMCellWithZoneout(
            AttRNN_Dimensions, hparams.attention_rnn_dim, bias=True,
            zoneout=self.p_AttRNN_hidden_dropout if self.AttRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_AttRNN_hidden_dropout if self.AttRNN_hidden_dropout_type == 'dropout' else 0.0)
        AttRNN_output_dim = hparams.attention_rnn_dim
        self.attention_rnn_sss = SSSUnit(AttRNN_output_dim, global_cond_dim, self.use_speaker_everywhere, sss_grad_scale)
        
        self.second_attention_rnn = None
        self.second_attention_rnn_dim                 = getattr(hparams, 'second_attention_rnn_dim', 0)
        self.second_attention_rnn_residual_connection = getattr(hparams, 'second_attention_rnn_residual_connection', False)
        if self.second_attention_rnn_dim > 0:
            if self.second_attention_rnn_residual_connection:
                assert self.second_attention_rnn_dim == hparams.attention_rnn_dim, "if using 'second_attention_rnn_residual_connection', both Attention RNN dimensions must match."
            self.second_attention_rnn = LSTMCellWithZoneout(
            hparams.attention_rnn_dim, self.second_attention_rnn_dim, bias=True,
            zoneout=self.p_AttRNN_hidden_dropout if self.AttRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_AttRNN_hidden_dropout if self.AttRNN_hidden_dropout_type == 'dropout' else 0.0)
            AttRNN_output_dim = self.second_attention_rnn_dim
            self.second_attention_rnn_sss = SSSUnit(AttRNN_output_dim, global_cond_dim, self.use_speaker_everywhere, sss_grad_scale)
        
        # Location/Attention Layer
        self.attention_layer = Attention(
            AttRNN_output_dim, self.memory_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size,
            self.windowed_attention_range, hparams.windowed_att_pos_learned,
            self.windowed_att_pos_offset,  hparams.attention_learned_temperature)
        
        if hasattr(hparams, 'use_cum_attention_scaler') and hparams.use_cum_attention_scaler:
            self.attention_weights_scaler = nn.Parameter(torch.ones(1)*0.25)
        
        self.decoder_residual_connection = hparams.decoder_residual_connection
        if self.decoder_residual_connection:
            assert (AttRNN_output_dim + self.memory_dim) == hparams.decoder_rnn_dim, f"if using 'decoder_residual_connection', decoder_rnn_dim must equal attention_rnn_dim + memory_dim ({hparams.attention_rnn_dim + self.memory_dim})."
        self.decoder_rnn = LSTMCellWithZoneout(
            AttRNN_output_dim + self.memory_dim, hparams.decoder_rnn_dim, bias=True,
            zoneout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'dropout' else 0.0)
        decoder_rnn_output_dim = hparams.decoder_rnn_dim
        self.decoder_rnn_sss = SSSUnit(decoder_rnn_output_dim, global_cond_dim, self.use_speaker_everywhere, sss_grad_scale)
        
        self.second_decoder_rnn   = None
        if hparams.second_decoder_rnn_dim > 0:
            self.second_decoder_rnn_dim = hparams.second_decoder_rnn_dim
            self.second_decoder_residual_connection = hparams.second_decoder_residual_connection
            if self.second_decoder_residual_connection:
                assert self.second_decoder_rnn_dim == hparams.decoder_rnn_dim, "if using 'second_decoder_residual_connection', both DecoderRNN dimensions must match."
            self.second_decoder_rnn = LSTMCellWithZoneout(
            hparams.decoder_rnn_dim, hparams.second_decoder_rnn_dim, bias=True,
            zoneout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'dropout' else 0.0)
            decoder_rnn_output_dim = hparams.second_decoder_rnn_dim
            self.second_decoder_rnn_sss = SSSUnit(decoder_rnn_output_dim, global_cond_dim, self.use_speaker_everywhere, sss_grad_scale)
        
        self.third_decoder_rnn   = None
        if getattr(hparams, 'third_decoder_rnn_dim', 0) > 0:
            assert hparams.second_decoder_rnn_dim > 0, 'second_decoder required to use third!'
            self.third_decoder_rnn_dim = hparams.third_decoder_rnn_dim
            self.third_decoder_residual_connection = hparams.third_decoder_residual_connection
            if self.third_decoder_residual_connection:
                assert self.third_decoder_rnn_dim == hparams.decoder_rnn_dim, "if using 'third_decoder_residual_connection', both DecoderRNN dimensions must match."
            self.third_decoder_rnn = LSTMCellWithZoneout(
            hparams.decoder_rnn_dim, hparams.third_decoder_rnn_dim, bias=True,
            zoneout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'dropout' else 0.0)
            decoder_rnn_output_dim = hparams.third_decoder_rnn_dim
            self.third_decoder_rnn_sss = SSSUnit(decoder_rnn_output_dim, global_cond_dim, self.use_speaker_everywhere, sss_grad_scale)
        
        lin_proj_input_dim = decoder_rnn_output_dim + self.memory_dim
        self.decoder_input_residual = getattr(hparams, 'decoder_input_residual', False)
        if self.decoder_input_residual:
            lin_proj_input_dim+=self.prenet_dim
        self.gate_layer        = LinearNorm(lin_proj_input_dim, 1, w_init_gain='sigmoid')
        self.linear_projection = LinearNorm(lin_proj_input_dim, hparams.n_mel_channels*hparams.n_frames_per_step*self.n_frames_per_frame)
        if self.n_frames_per_frame > 1:
            class_dim = hparams.n_mel_channels*hparams.n_frames_per_step*self.n_frames_per_frame if self.n_fpf_per_elem else hparams.n_frames_per_step*self.n_frames_per_frame
            self.linear_classifier = LinearNorm(lin_proj_input_dim, class_dim)
        self.noisel_projection = LinearNorm(lin_proj_input_dim, hparams.n_mel_channels*hparams.n_frames_per_step*self.n_frames_per_frame) if getattr(hparams, 'noise_projector', False) else None
        if self.noisel_projection is not None:
            self.noisel_projection.linear_layer.weight.data.mul_(0.01)
        self.hifigan_projection= LinearNorm(lin_proj_input_dim, hparams.n_mel_channels*hparams.n_frames_per_step) if self.HiFiGAN_enable else None
    
    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = memory.new_zeros(B, self.n_mel_channels * self.n_frames_per_step)
        return decoder_input
    
    def get_decoder_states(self, memory, preserve: Optional[Tensor] = None):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory  : Encoder outputs
        preserve: Batch shape bool tensor of decoder states to preserve
        """
        B, MAX_ENCODE, *_ = memory.shape
        
        if preserve is not None:
            preserve = preserve.float()
            if len(preserve.shape) < 2:
                preserve = preserve[:, None]
            assert preserve.shape[0] == B
        
        attention_hidden = self.attention_hidden # https://github.com/pytorch/pytorch/issues/22155#issue-460050914
        attention_cell   = self.attention_cell
        if attention_hidden is not None and attention_cell is not None and preserve is not None:
            attention_hidden *= preserve
            attention_cell   *= preserve
            attention_hidden.detach_().requires_grad_()
            attention_cell  .detach_().requires_grad_()
        else:
            attention_hidden = memory.new_zeros(B, self.attention_rnn_dim)# attention hidden state
            attention_cell   = memory.new_zeros(B, self.attention_rnn_dim)# attention cell state
        
        second_attention_rnn    = self.second_attention_rnn
        second_attention_hidden = self.second_attention_hidden
        second_attention_cell   = self.second_attention_cell
        if second_attention_rnn is not None:
            if second_attention_hidden is not None and second_attention_cell is not None and preserve is not None:
                second_attention_hidden *= preserve
                second_attention_cell   *= preserve
                second_attention_hidden.detach_().requires_grad_()
                second_attention_cell  .detach_().requires_grad_()
            else:
                second_attention_hidden = memory.new_zeros(B, self.second_attention_rnn_dim)# LSTM attention hidden state
                second_attention_cell   = memory.new_zeros(B, self.second_attention_rnn_dim)# LSTM attention cell state
        elif self.gradient_checkpoint:
            second_attention_hidden = memory.new_zeros(1, requires_grad=True)# Dummy LSTM attention cell state
            second_attention_cell   = memory.new_zeros(1, requires_grad=True)# Dummy LSTM attention cell state
        
        decoder_hidden = self.decoder_hidden
        decoder_cell   = self.decoder_cell
        if decoder_hidden is not None and decoder_cell is not None and preserve is not None:
            decoder_hidden *= preserve
            decoder_cell   *= preserve
            decoder_hidden.detach_().requires_grad_()
            decoder_cell  .detach_().requires_grad_()
        else:
            decoder_hidden = memory.new_zeros(B, self.decoder_rnn_dim, requires_grad=True)# LSTM decoder hidden state
            decoder_cell   = memory.new_zeros(B, self.decoder_rnn_dim, requires_grad=True)# LSTM decoder cell state
        
        second_decoder_rnn    = self.second_decoder_rnn
        second_decoder_hidden = self.second_decoder_hidden
        second_decoder_cell   = self.second_decoder_cell
        if second_decoder_rnn is not None:
            if second_decoder_hidden is not None and second_decoder_cell is not None and preserve is not None:
                second_decoder_hidden *= preserve
                second_decoder_cell   *= preserve
                second_decoder_hidden.detach_().requires_grad_()
                second_decoder_cell  .detach_().requires_grad_()
            else:
                second_decoder_hidden = memory.new_zeros(B, self.second_decoder_rnn_dim, requires_grad=True)# LSTM decoder hidden state
                second_decoder_cell   = memory.new_zeros(B, self.second_decoder_rnn_dim, requires_grad=True)# LSTM decoder cell state
        elif self.gradient_checkpoint:
            second_decoder_hidden = memory.new_zeros(1, requires_grad=True)# Dummy LSTM attention cell state
            second_decoder_cell   = memory.new_zeros(1, requires_grad=True)# Dummy LSTM attention cell state
        
        third_decoder_rnn    = self.third_decoder_rnn
        third_decoder_hidden = self.third_decoder_hidden
        third_decoder_cell   = self.third_decoder_cell
        if third_decoder_rnn is not None:
            if third_decoder_hidden is not None and third_decoder_cell is not None and preserve is not None:
                third_decoder_hidden *= preserve
                third_decoder_cell   *= preserve
                third_decoder_hidden.detach_().requires_grad_()
                third_decoder_cell  .detach_().requires_grad_()
            else:
                third_decoder_hidden = memory.new_zeros(B, self.third_decoder_rnn_dim, requires_grad=True)# LSTM decoder hidden state
                third_decoder_cell   = memory.new_zeros(B, self.third_decoder_rnn_dim, requires_grad=True)# LSTM decoder cell state
        elif self.gradient_checkpoint:
            third_decoder_hidden = memory.new_zeros(1, requires_grad=True)# Dummy LSTM attention cell state
            third_decoder_cell   = memory.new_zeros(1, requires_grad=True)# Dummy LSTM attention cell state
        
        saved_attention_weights     = None
        saved_attention_weights_cum = None
        attention_weights     = self.attention_weights
        attention_weights_cum = self.attention_weights_cum
        if attention_weights is not None and attention_weights_cum is not None and preserve is not None: # save all the encoder possible
            saved_attention_weights     = attention_weights
            saved_attention_weights_cum = attention_weights_cum
        
        attention_weights     = memory.new_zeros(B, MAX_ENCODE, requires_grad=True)# attention weights of that frame
        attention_weights_cum = memory.new_zeros(B, MAX_ENCODE, requires_grad=True)# cumulative weights of all frames during that inferrence
        
        if (saved_attention_weights     is not None and
            saved_attention_weights_cum is not None and
            attention_weights     is not None and
            attention_weights_cum is not None and
            preserve is not None):
            COMMON_ENCODE = min(MAX_ENCODE, saved_attention_weights.shape[1]) # smallest MAX_ENCODE of the saved and current encodes
            attention_weights    [:, :COMMON_ENCODE] = saved_attention_weights    [:, :COMMON_ENCODE]# preserve any encoding weights possible (some will be part of the previous iterations padding and are gone)
            attention_weights_cum[:, :COMMON_ENCODE] = saved_attention_weights_cum[:, :COMMON_ENCODE]
            attention_weights     *= preserve
            attention_weights_cum *= preserve
            attention_weights    .detach_().requires_grad_()
            attention_weights_cum.detach_().requires_grad_()
        
        attention_context = self.attention_context
        if attention_context is not None and preserve is not None:
            attention_context *= preserve
            attention_context = attention_context.detach().requires_grad_()
        else:
            attention_context = memory.new_zeros(B, self.memory_dim, requires_grad=True)# attention output
        
        processed_memory = self.attention_layer.memory_layer(memory) # Linear Layer, [B, txt_T, enc_dim] -> [B, txt_T, attention_dim]
        
        attention_position = self.attention_position
        if attention_position is not None and preserve is not None:
            attention_position *= preserve.squeeze(1)
            attention_position.detach_().requires_grad_()
        else:
            attention_position = memory.new_zeros(B, requires_grad=True)# [B]
        
        return [attention_hidden,  attention_cell, second_attention_hidden, second_attention_cell,
                decoder_hidden,    decoder_cell,   second_decoder_hidden,   second_decoder_cell,
                                                   third_decoder_hidden,    third_decoder_cell,
                attention_context, attention_position, attention_weights,   attention_weights_cum,
                memory, processed_memory]
    
    def lstm_cp(self, func, input_, states):
        hidden_state, cell_state = states
        def new_func(input_, hidden_state, cell_state):
            return func(input_, (hidden_state, cell_state))
        return checkpoint(new_func, input_, hidden_state, cell_state)
    
    def att_cp(self, func, attention_rnn_output, memory, processed_memory, attention_weights_cat, mask, memory_lengths, _, attention_position):
        def new_func(attention_rnn_output, memory, processed_memory, attention_weights_cat, mask, memory_lengths, attention_position):
            return func(attention_rnn_output, memory, processed_memory, attention_weights_cat, mask, memory_lengths, None, attention_position)
        return checkpoint(new_func, attention_rnn_output, memory, processed_memory, attention_weights_cat, mask, memory_lengths, attention_position)
    
    def decode(self, decoder_input, gt_attention_contexts, gt_attention_weights, var_embed, memory_lengths, mask: Optional[Tensor],
        attention_hidden,  attention_cell, second_attention_hidden, second_attention_cell,
        decoder_hidden,    decoder_cell,   second_decoder_hidden,   second_decoder_cell,
                                            third_decoder_hidden,    third_decoder_cell,
        attention_context, attention_position, attention_weights,   attention_weights_cum,
        memory, processed_memory, global_cond):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
#        attention_hidden, attention_cell, second_attention_hidden, second_attention_cell,
#        decoder_hidden, decoder_cell, second_decoder_hidden, second_decoder_cell,
#        attention_context, attention_weights, attention_weights_cum, memory, processed_memory, mask
        
        assert attention_hidden is not None
        assert attention_cell is not None
        
        assert attention_context is not None
        assert attention_position is not None
        assert attention_weights is not None
        assert attention_weights_cum is not None
        
        assert memory           is not None
        assert processed_memory is not None
        
        assert decoder_hidden is not None
        assert decoder_cell is not None
        
        # Attention LSTM(s)
        cell_input = [decoder_input, attention_context]# [B, prenet_dim], [B, memory_dim]
        if self.AttRNN_extra_decoder_input:
            cell_input.append(decoder_hidden)# [B, DecRNN_output_dim]
        if self.use_DecoderRNG:
            cell_input.append(torch.randn(decoder_input.shape[0], self.DecoderRNG_dim, device=decoder_input.device, dtype=decoder_input.dtype))
        if hasattr(self, 'variational_encoder'):
            if var_embed is not None:
                cell_input.append(var_embed)
            else:
                cell_input.append(self.variational_encoder(var_embed, decoder_input.shape[0])[0])
        if self.AttRNN_use_global_cond:
            cell_input.append(global_cond)# [B, embed]
        cell_input = torch.cat(cell_input, -1)# [Processed Previous Spect Frame, Last input Taken from Text/Att]
        
        if False and self.training and self.gradient_checkpoint:
            _ = self.lstm_cp(self.attention_rnn.__call__, cell_input, (attention_hidden, attention_cell))
        else:
            _ = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = _[0]
        attention_cell   = _[1]
        attention_rnn_output = self.attention_rnn_sss(attention_hidden, global_cond)
        
        if self.second_attention_rnn is not None:
            assert second_attention_hidden is not None
            assert second_attention_cell is not None
            if False and self.training and self.gradient_checkpoint:
                second_attention_state = self.lstm_cp(self.second_attention_rnn, attention_rnn_output, (second_attention_hidden, second_attention_cell))
            else:
                second_attention_state = self.second_attention_rnn(attention_rnn_output, (second_attention_hidden, second_attention_cell))
            second_attention_hidden = second_attention_state[0]
            second_attention_cell   = second_attention_state[1]
            if self.second_attention_rnn_residual_connection:
                attention_rnn_output = attention_rnn_output + second_attention_hidden
            else:
                attention_rnn_output = second_attention_hidden
            attention_rnn_output = self.second_attention_rnn_sss(attention_rnn_output, global_cond)
        elif self.gradient_checkpoint:
            _ = torch.nn.parameter.Parameter(second_attention_hidden.new_ones(1))
            second_attention_hidden=second_attention_hidden*_;second_attention_cell=second_attention_cell*_
        
        scaled_attention_weights_cum = attention_weights_cum.unsqueeze(1)
        if self.attention_weights_scaler is not None:
            scaled_attention_weights_cum *= self.attention_weights_scaler
        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), scaled_attention_weights_cum), dim=1)
        # [B, 1, txt_T] cat [B, 1, txt_T] -> [B, 2, txt_T]
        
        # Attention
        if True and self.training and self.gradient_checkpoint:
            _ = self.att_cp(self.attention_layer, *(attention_rnn_output, memory, processed_memory, attention_weights_cat, mask, memory_lengths, gt_attention_weights, attention_position, -float('inf'), self.inference_mode))
        else:
            _ = self.attention_layer(attention_rnn_output, memory, processed_memory, attention_weights_cat, mask, memory_lengths, gt_attention_weights, attention_position, -float('inf'), self.inference_mode)
        attention_context = _[0] if gt_attention_contexts is None else gt_attention_contexts
        attention_weights = _[1]
        new_pos = _[2]
        
        exp_smoothing_factor = self.exp_smoothing_factor
        assert exp_smoothing_factor is not None
        
        smooth_factor = torch.sigmoid(exp_smoothing_factor)
        attention_position = (attention_position*smooth_factor) + (new_pos*(1-smooth_factor))
        
        attention_weights_cum = attention_weights_cum + attention_weights
        
        # Decoder LSTM(s)
        decoder_rnn_input = torch.cat( ( attention_rnn_output, attention_context), -1) # cat 6.475ms
        if False and self.training and self.gradient_checkpoint:
            decoderrnn_state = self.lstm_cp(self.decoder_rnn, decoder_rnn_input, (decoder_hidden, decoder_cell))
        else:
            decoderrnn_state = self.decoder_rnn(decoder_rnn_input, (decoder_hidden, decoder_cell))# lstmcell 12.789ms
        decoder_hidden = decoderrnn_state[0]
        decoder_cell   = decoderrnn_state[1]
        if self.decoder_residual_connection:
            decoder_rnn_output = decoder_hidden + decoder_rnn_input
        else:
            decoder_rnn_output = decoder_hidden
        decoder_rnn_output = self.decoder_rnn_sss(decoder_rnn_output, global_cond)
        
        if self.second_decoder_rnn is not None:
            assert second_decoder_hidden is not None
            assert second_decoder_cell is not None
            if False and self.training and self.gradient_checkpoint:
                second_decoder_state = self.lstm_cp(self.second_decoder_rnn, decoder_rnn_output, (second_decoder_hidden, second_decoder_cell))
            else:
                second_decoder_state = self.second_decoder_rnn(decoder_rnn_output, (second_decoder_hidden, second_decoder_cell))
            second_decoder_hidden = second_decoder_state[0]
            second_decoder_cell   = second_decoder_state[1]
            if self.second_decoder_residual_connection:
                decoder_rnn_output = decoder_rnn_output + second_decoder_hidden
            else:
                decoder_rnn_output = second_decoder_hidden
            decoder_rnn_output = self.second_decoder_rnn_sss(decoder_rnn_output, global_cond)
        elif self.gradient_checkpoint:
            second_decoder_hidden.requires_grad_()
            second_decoder_cell  .requires_grad_()
        
        if self.third_decoder_rnn is not None:
            assert third_decoder_hidden is not None
            assert third_decoder_cell is not None
            if False and self.training and self.gradient_checkpoint:
                third_decoder_state = self.lstm_cp(self.third_decoder_rnn, decoder_rnn_output, (third_decoder_hidden, third_decoder_cell))
            else:
                third_decoder_state = self.third_decoder_rnn(decoder_rnn_output, (third_decoder_hidden, third_decoder_cell))
            third_decoder_hidden = third_decoder_state[0]
            third_decoder_cell   = third_decoder_state[1]
            if self.third_decoder_residual_connection:
                decoder_rnn_output = decoder_rnn_output + third_decoder_hidden
            else:
                decoder_rnn_output = third_decoder_hidden
            decoder_rnn_output = self.third_decoder_rnn_sss(decoder_rnn_output, global_cond)
        elif self.gradient_checkpoint:
            third_decoder_hidden.requires_grad_()
            third_decoder_cell  .requires_grad_()
        
        if self.decoder_input_residual:
            decoder_hidden_attention_context = torch.cat((decoder_rnn_output, attention_context, decoder_input), dim=1) # -> [B, dim] cat 6.555ms
        else:
            decoder_hidden_attention_context = torch.cat((decoder_rnn_output, attention_context), dim=1) # -> [B, dim] cat 6.555ms
        
        gate_prediction = self.gate_layer(       decoder_hidden_attention_context)# -> [B, 1] addmm 5.762ms
        decoder_output  = self.linear_projection(decoder_hidden_attention_context)# -> [B, context_frames*n_mel*n_frames] addmm 5.621ms
        if hasattr(self, 'noisel_projection') and self.noisel_projection is not None and self.training:
            decoder_output += self.noisel_projection(decoder_hidden_attention_context)*torch.randn(decoder_output.shape[0], decoder_output.shape[1], device=decoder_output.device, dtype=decoder_output.dtype)
        
        hifigan_output = None
        if hasattr(self, 'hifigan_projection') and self.hifigan_projection is not None:
            hifigan_output = self.hifigan_projection(decoder_hidden_attention_context)
        
        decoder_output_logprob = None
        if hasattr(self, 'linear_classifier') and self.linear_classifier is not None:
            decoder_output_logprob = self.linear_classifier(decoder_hidden_attention_context)
        
        decode_args = [attention_hidden,  attention_cell,     second_attention_hidden, second_attention_cell,
                       decoder_hidden,    decoder_cell,       second_decoder_hidden,   second_decoder_cell,
                                                               third_decoder_hidden,    third_decoder_cell,
                       attention_context, attention_position, attention_weights,       attention_weights_cum,
                       memory, processed_memory, global_cond,
                       decoder_output_logprob, hifigan_output, attention_rnn_output, decoder_rnn_output,]# these last 2 are only used for GANs.
        return decoder_output, gate_prediction, attention_weights, decode_args
    
    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-spects
        
        RETURNS
        -------
        inputs: processed decoder inputs
        
        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs
    
    def parse_decoder_outputs(self,
            mel_outputs:  List[Tensor],
            gate_outputs: List[Tensor],
            alignments:   List[Tensor],):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        """
        if type(alignments) in (list, tuple):
            alignments = torch.stack(alignments)
        alignments = alignments.transpose(0, 1)
        
        # (T_out, B) -> (B, T_out)
        if type(gate_outputs) in (list, tuple):
            gate_outputs = torch.stack(gate_outputs)
        gate_outputs = gate_outputs.transpose(0, 1) if len(gate_outputs.size()) > 1 else gate_outputs[None]
        gate_outputs = gate_outputs.contiguous()
        
        if type(mel_outputs) in (list, tuple):
            mel_outputs = torch.stack([frame for mel_output in mel_outputs for frame in mel_output.chunk(self.n_frames_per_step, dim=1)], dim=2)# [B, n_mel]*mel_T -> [B, n_mel, mel_T]
        
        return mel_outputs, gate_outputs, alignments
    
    def reshape_output(self, mel_outputs):
        if type(mel_outputs) in (list, tuple):
            # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
            mel_outputs = torch.stack(mel_outputs)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view( mel_outputs.size(0), -1, self.n_mel_channels )
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        return mel_outputs.transpose(1, 2)
    
    def predict_best_frames(self,
            pred_mels,  # [B, n_frames*n_mel]
            logprobs,   # [B, n_frames*(1 OR n_mel)]
            mode='max'):# 'rand' or 'max' or 'mix'
        if self.n_frames_per_frame > 1:
            probs = F.softmax(logprobs, dim=1)# [B, n_frames*(1 OR n_mel)]
            if mode == 'mix':
                if not self.n_fpf_per_elem:
                    probs = probs.repeat_interleave(self.n_mel_channels, dim=1)
                pred_mels = sum((probs*pred_mels).chunk(self.n_frames_per_frame, dim=1))# [[B, n_mel],]*n_frames
            elif mode == 'rand':
                rand_num = torch.rand(probs.shape[0], 1, device=probs.device, dtype=probs.dtype)# [B, 1]
                cumprobs = probs.cumsum(dim=1)# [B, n_frames*(1 OR n_mel)]
                indexes = ((rand_num > cumprobs-probs) & (rand_num < cumprobs)).half().argmax(dim=1)# [B]
                
                pred_mels_ = []
                for i, index in enumerate(indexes):
                    pred_mels_.append(pred_mels[i].chunk(self.n_frames_per_frame, dim=0)[index])# .append( [n_mel] ) -> [n_mel,]*B
                pred_mels = torch.stack(pred_mels_, dim=0)# [n_mel,]*B -> [B, n_mel]
            elif mode == 'max':
                indexes = probs.argmax(dim=1)# [B, n_frames] -> [B]
                
                # [B, n_mel*n_frames] -> [B, n_mel]
                pred_mels = torch.stack([pred_mels[i, (index)*self.n_mel_channels:(index+1)*self.n_mel_channels] for i, index in enumerate(indexes)], dim=0)
            else:
                raise NotImplementedError
        
        return pred_mels# [B, n_mel]
    
    def best_frames(self, pred_mels, gt_mel, frame_mean=True):# [B, n_mel*n_frames], [B, n_mel]
        """Output best frames from a set of predicted mels and the target mel."""
        B, n_mel_n_frames = pred_mels.shape
        pred_mels = pred_mels.chunk(self.n_frames_per_frame, dim=1)# [[B, n_mel],]*n_frames
        aerr_mels = []
        for i in range(self.n_frames_per_frame):
            aerr_mel = F.l1_loss(pred_mels[i].detach(), gt_mel.detach(), reduction='none')
            if frame_mean:
                aerr_mel = aerr_mel.mean(dim=1, keepdim=True)
            aerr_mels.append(aerr_mel)# -> [B, n_mel]
        
        pred_mel = pred_mels[0]
        aerr_mel = aerr_mels[0]
        for i in range(1, self.n_frames_per_frame):
            pred_mel = pred_mel.where(aerr_mel < aerr_mels[i], pred_mels[i])
            aerr_mel = aerr_mel.where(aerr_mel < aerr_mels[i], aerr_mels[i])
        
        return pred_mel# [B, n_mel, mel_T]
    
    def forward(self, gt_mel, decoder_inputs, gt_attention_contexts, gt_attention_weights, var_embeds, memory_lengths, 
                mel_outputs, n_already_generated,
                mask, speaker_embed, *decode_args,):
        """
        Decoder truncated forward pass.
        This function takes part of the decoder sequence and processes it and can be used for gradient checkpointing.
        """
        teacher_force_till  = self.teacher_force_till
        p_teacher_forcing   = self.p_teacher_forcing
        
        gate_outputs, alignments = [], []
        attention_rnn_outputs = [] if self.return_attention_rnn_outputs else None
        decoder_rnn_outputs   = [] if self.return_decoder_rnn_outputs   else None
        attention_contexts    = [] if self.return_attention_contexts    else None
        hifigan_outputs       = [] if self.HiFiGAN_enable               else None
        mel_outputs_logprob   = [] if self.n_frames_per_frame > 1       else None
        for i in range(len(decoder_inputs)):
            if self.half_inference_mode:
                decoder_input = decoder_inputs[ i].chunk(2, dim=0)[0]# [B/2, C] use teacher forced input
                prenet_input  =    mel_outputs[-1].chunk(2, dim=0)[1]# [B/2, n_mel]
                prenet_input  = self.prenet(prenet_input.detach(), speaker_embed.chunk(2, dim=0)[1])# [B/2, C] use last output for next input (like inference)
                decoder_input = torch.cat((decoder_input, prenet_input), dim=0)# [B/2, ...] + [B/2, ...] -> [B]
                del prenet_input
            else:
                if teacher_force_till >= len(mel_outputs)+n_already_generated-1 or p_teacher_forcing >= torch.rand(1):
                    decoder_input = decoder_inputs[i] # use teacher forced input
                else:
                    prenet_input  = self.best_frames(mel_outputs[-1].detach(), gt_mel[:, :, i], frame_mean=not self.n_fpf_per_elem)# [B, n_mel*n_frames], [B, n_mel]
                    decoder_input = self.prenet(prenet_input, speaker_embed)# [B, n_mel] use last output for next input (like inference)
                    del prenet_input
            
            var_embed = var_embeds[i] if var_embeds is not None else None
            mel_output, gate_output, attention_weights, decode_args = self.decode(decoder_input,
                                                       gt_attention_contexts[:, i] if gt_attention_contexts   is not None else None,
                                                        gt_attention_weights[:, i] if gt_attention_weights is not None else None,
                                                                                    var_embed, memory_lengths, mask, *decode_args)
            if self.dump_attention_weights:
                attention_weights = attention_weights.cpu()
            if i == 0:
                mel_outputs = []
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            if not self.dump_attention_weights or len(mel_outputs) < 2:
                alignments += [attention_weights]
            if self.return_decoder_rnn_outputs:
                decoder_rnn_outputs  += [decode_args[-1],]# decoder_rnn_output
            if self.return_attention_rnn_outputs:
                attention_rnn_outputs+= [decode_args[-2],]# attention_rnn_output
            if self.HiFiGAN_enable:
                hifigan_outputs      += [decode_args[-3],]# hifigan_output
            if self.n_frames_per_frame > 1:
                mel_outputs_logprob  += [decode_args[-4],]# decoder classification outputs
            if self.return_attention_contexts:
                attention_contexts   += [decode_args[ 8],]# attention_contexts
            
            del decode_args[-4:]# delete attention_rnn_outputs, decoder_rnn_outputs, hifi_outputs and mel_outputs_logprob from decoder_args
        
        if self.output_tensor:
            mel_outputs  = torch.stack(mel_outputs)
            gate_outputs = torch.stack(gate_outputs)
            alignments   = torch.stack(alignments)
        attention_rnn_outputs = torch.stack(attention_rnn_outputs, dim=2) if self.return_attention_rnn_outputs else None# [[B, C], [B, C], ...] -> [B, C, T]
        decoder_rnn_outputs   = torch.stack(decoder_rnn_outputs,   dim=2) if self.return_decoder_rnn_outputs   else None# [[B, C], [B, C], ...] -> [B, C, T]
        attention_contexts    = torch.stack(attention_contexts,    dim=2) if self.return_attention_contexts    else None# [[B, C], [B, C], ...] -> [B, C, T]
        hifigan_outputs       = torch.stack(hifigan_outputs,       dim=0) if self.HiFiGAN_enable               else None# [[B, C], [B, C], ...] -> [B, C, T]
        mel_outputs_logprob   = torch.stack(mel_outputs_logprob,   dim=2) if self.n_frames_per_frame > 1       else None# [[B, C], [B, C], ...] -> [B, C, T]
        
#        attention_hidden, attention_cell, second_attention_hidden, second_attention_cell,\
#        decoder_hidden,   decoder_cell,   second_decoder_hidden,   second_decoder_cell,\
#                                           third_decoder_hidden,    third_decoder_cell,\
#        attention_context, attention_position, attention_weights, attention_weights_cum,\
#        memory, processed_memory, global_cond = decode_args
        outputs = [mel_outputs, gate_outputs, alignments, *decode_args]
        if self.return_attention_rnn_outputs:
            outputs.append(attention_rnn_outputs)
        if self.return_decoder_rnn_outputs:
            outputs.append(decoder_rnn_outputs)
        if self.return_attention_contexts:
            outputs.append(attention_contexts)
        if self.HiFiGAN_enable:
            outputs.append(self.reshape_output(hifigan_outputs))
        if self.n_frames_per_frame > 1:
            outputs.append(mel_outputs_logprob)
        return outputs
    
    def RNN_train(self):
        try: self.attention_rnn.train(); self.second_attention_rnn.train()
        except: pass
        try: self.decoder_rnn.train(); self.second_decoder_rnn.train(); self.third_decoder_rnn.train()
        except: pass
    
    def RNN_eval(self):
        try: self.attention_rnn.eval(); self.second_attention_rnn.eval()
        except: pass
        try: self.decoder_rnn.eval(); self.second_decoder_rnn.eval(); self.third_decoder_rnn.eval()
        except: pass
    
    def forward_init(self, memory, global_cond, speaker_embed, decoder_inputs, memory_lengths,
                preserve_decoder:   Optional[Tensor] = None,
                decoder_input:      Optional[Tensor] = None,
                attention_weights:  Optional[Tensor] = None,
             gt_attention_contexts: Optional[Tensor] = None,
                teacher_force_till:            int = 0,
                p_teacher_forcing:           float = 1.0,
                return_hidden_state:          bool = False):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory             : Encoder outputs
        decoder_inputs     : Decoder inputs for teacher forcing. i.e. mel-spects
        memory_lengths     : Encoder output lengths for attention masking.
        preserve_decoder   : [B] Tensor - Preserve model state for True items in batch/Tensor
        decoder_input      : [B, n_mel, context] FloatTensor
        teacher_force_till : INT - Beginning X frames where Teacher Forcing is forced ON.
        p_teacher_forcing  : Float - 0.0 to 1.0 - Change to use Teacher Forcing during training/validation.
        
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        if self.hide_startstop_tokens: # remove first/last tokens from Memory # I no longer believe this is useful.
            memory = memory[:,1:-1,:]
            memory_lengths = memory_lengths-2
        
        #if self.prenet_blur_max > 0.0:
        #    rand_blur_strength = torch.rand(1).uniform_(self.prenet_blur_min, self.prenet_blur_max)
        #    decoder_inputs = HeightGaussianBlur(decoder_inputs, blur_strength=rand_blur_strength)# [B, n_mel, mel_T]
        
        gt_mel = decoder_inputs# [B, n_mel, mel_T]
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)# [B, n_mel, mel_T] -> [mel_T, B, n_mel]
        
        if hasattr(self, 'variational_encoder'):
            var_embeds, var_mu, var_logvar = self.variational_encoder(decoder_inputs)# [mel_T, B, n_mel]
        
        if decoder_input is None:
            decoder_input = self.get_go_frame(memory).unsqueeze(0) # create blank starting frame
            if self.context_frames > 1:
                decoder_input = decoder_input.expand(self.context_frames, -1, -1)
        else:
            decoder_input = decoder_input.permute(2, 0, 1) # [B, n_mel, context_frames] -> [context_frames, B, n_mel]
        # memory -> (1, B, n_mel) <- which is all 0's
        
        decoder_inputs = torch.cat((decoder_input, decoder_inputs[:-1]), dim=0) # [mel_T, B, n_mel] concat T_out
        
        if self.prenet_noise and self.training:
            prenet_noise = self.prenet_noise* torch.randn(decoder_inputs.shape, device=decoder_inputs.device, dtype=decoder_inputs.dtype)
            decoder_inputs += prenet_noise
            decoder_inputs = decoder_inputs.clamp(min=log(self.stft_clamp_val)+0.1)
        
        decoder_inputs = self.prenet(decoder_inputs, speaker_embed)# [T, B, C]
        
        decode_args = self.get_decoder_states(memory, preserve=preserve_decoder) + [global_cond,]
#        attention_hidden, attention_cell, second_attention_hidden, second_attention_cell,\
#        decoder_hidden,   decoder_cell,   second_decoder_hidden,   second_decoder_cell,\
#                                           third_decoder_hidden,    third_decoder_cell,\
#        attention_context, attention_position, attention_weights, attention_weights_cum,\
#        memory, processed_memory, global_cond = decode_args
        
        mask = ~get_mask_from_lengths(memory_lengths)
        
        decode_chunksize         = self.decode_chunksize
        self.output_tensor       = self.training and self.gradient_checkpoint
        self.teacher_force_till  = teacher_force_till
        self.p_teacher_forcing   = p_teacher_forcing
        
        n_frames = len(decoder_inputs)
        n_frames_generated = 0
        mel_outputs, gate_outputs, alignments = [], [], []
        
        hiddens = {}
        if hasattr(self, 'variational_encoder'):
            hiddens["var_mu"    ] = var_mu
            hiddens["var_logvar"] = var_logvar
        attention_contexts    = [] if self.return_attention_contexts    else None
        decoder_rnn_outputs   = [] if self.return_decoder_rnn_outputs   else None
        attention_rnn_outputs = [] if self.return_attention_rnn_outputs else None
        hifigan_outputs       = [] if self.HiFiGAN_enable               else None
        mel_outputs_logprob   = [] if self.n_frames_per_frame > 1       else None
        
        while n_frames_generated < n_frames:
            var_embeds_sliced = var_embeds[n_frames_generated:n_frames_generated+decode_chunksize] if hasattr(self, 'variational_encoder') else None
            func_args = (gt_mel[:, :, n_frames_generated:n_frames_generated+decode_chunksize],
                 decoder_inputs[n_frames_generated:n_frames_generated+decode_chunksize],
          gt_attention_contexts[:, n_frames_generated:n_frames_generated+decode_chunksize] if gt_attention_contexts is not None else None,
              attention_weights[:, n_frames_generated:n_frames_generated+decode_chunksize] if attention_weights is not None else None,
                   var_embeds_sliced, memory_lengths, mel_outputs[-1].unsqueeze(0) if n_frames_generated else decoder_input,
                   torch.tensor(n_frames_generated), mask, speaker_embed, *decode_args)
            if self.training and self.gradient_checkpoint:
                out = checkpoint(self, *func_args)
            else:
                out = self(*func_args)
            del func_args
            mel_outputs .extend(out[0])
            gate_outputs.extend(out[1])
            alignments  .extend(out[2])
            if self.n_frames_per_frame > 1:
                mel_outputs_logprob.append(out.pop())
            if self.HiFiGAN_enable:
                hifigan_outputs.append(out.pop())
            if self.return_attention_contexts:
                attention_contexts.append(out.pop())
            if self.return_decoder_rnn_outputs:
                decoder_rnn_outputs.append(out.pop())
            if self.return_attention_rnn_outputs:
                attention_rnn_outputs.append(out.pop())
            decode_args = out[3:]
            n_frames_generated = len(mel_outputs)
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        if self.n_frames_per_frame > 1:
            mel_outputs_logprob = torch.cat(mel_outputs_logprob, dim=2)# [B, C, mel_T]
            hiddens['mel_outputs_logprob'] = mel_outputs_logprob
        if self.HiFiGAN_enable:
            hifigan_outputs = torch.cat(hifigan_outputs, dim=2)# [B, C, mel_T]
            hiddens['hifigan_inputs'] = hifigan_outputs
        if self.return_attention_contexts:
            attention_contexts    = torch.cat(attention_contexts,    dim=2)# [B, C, mel_T]
            hiddens['attention_contexts'] = attention_contexts
        if self.return_decoder_rnn_outputs:
            decoder_rnn_outputs   = torch.cat(decoder_rnn_outputs,   dim=2)# [B, C, mel_T]
            hiddens['decoder_rnn_outputs'] = decoder_rnn_outputs
        if self.return_attention_rnn_outputs:
            attention_rnn_outputs = torch.cat(attention_rnn_outputs, dim=2)# [B, C, mel_T]
            hiddens['attention_rnn_outputs'] = attention_rnn_outputs
        
        self.save_decoder_args(decode_args)# used for TBPTT, doesn't use param so shouldn't be saved in state_dict
        
        return mel_outputs, gate_outputs, alignments, memory, hiddens
    
    def save_decoder_args(self, decoder_args):
        self.attention_hidden        = decoder_args[ 0].detach()
        self.attention_cell          = decoder_args[ 1].detach()
        self.second_attention_hidden = decoder_args[ 2].detach() if decoder_args[ 2] is not None else None
        self.second_attention_cell   = decoder_args[ 3].detach() if decoder_args[ 3] is not None else None
        
        self.decoder_hidden          = decoder_args[ 4].detach()
        self.decoder_cell            = decoder_args[ 5].detach()
        self.second_decoder_hidden   = decoder_args[ 6].detach() if decoder_args[ 6] is not None else None
        self.second_decoder_cell     = decoder_args[ 7].detach() if decoder_args[ 7] is not None else None
        self.third_decoder_hidden    = decoder_args[ 8].detach() if decoder_args[ 8] is not None else None
        self.third_decoder_cell      = decoder_args[ 9].detach() if decoder_args[ 9] is not None else None
        
        self.attention_context       = decoder_args[10].detach()
        self.attention_position      = decoder_args[11].detach()
        self.attention_weights       = decoder_args[12].detach()
        self.attention_weights_cum   = decoder_args[13].detach()
    
    @torch.jit.export
    def inference(self, memory, attention_contexts, gt_attention_weights, global_cond, speaker_embed, memory_lengths: Tensor,):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        
        RETURNS
        -------
        mel_outputs  : mel outputs from the decoder
        gate_outputs : gate outputs from the decoder
        alignments   : sequence of attention weights from the decoder
        """
        if gt_attention_weights is not None:
            self.inference_mode = True
        if self.hide_startstop_tokens: # remove start/stop token from Decoder
            memory = memory[:,1:-1,:]
            memory_lengths = memory_lengths-2
        
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory)
        
        decoder_input = self.get_go_frame(memory)
        
        decode_args = self.get_decoder_states(memory) + [global_cond,]
        mask = ~get_mask_from_lengths(memory_lengths)
        
        max_length = float('inf') if gt_attention_weights is None else gt_attention_weights.shape[1]
        sig_max_gates = torch.zeros(decoder_input.size(0))
        mel_outputs, gate_outputs, alignments, break_point = [], [], [], self.max_decoder_steps
        for i in range(self.max_decoder_steps):
            decoder_input = self.prenet(decoder_input, speaker_embed)
            
            mel_output, gate_output_gpu, alignment, decode_args = self.decode(decoder_input,
                                                          attention_contexts[:, i] if attention_contexts   is not None else None,
                                                        gt_attention_weights[:, i] if gt_attention_weights is not None else None,
                                                                                None, memory_lengths, mask, *decode_args)
            
            mel_logprob = decode_args[-4]
            mel_output = self.predict_best_frames(mel_output, mel_logprob)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_output_cpu = gate_output_gpu.cpu().float() # small operations e.g min(), max() and sigmoid() are faster on CPU # also .float() because Tensor.min() doesn't work on half precision CPU
            gate_outputs += [gate_output_gpu.squeeze(1)]
            alignments += [alignment]
            
            # once ALL batch predictions have gone over gate_threshold at least once, set break_point
            if i > 4: # model has very *interesting* starting predictions
                sig_max_gates = torch.max(torch.sigmoid(gate_output_cpu), sig_max_gates)# sigmoid -> max
            if sig_max_gates.min() > self.gate_threshold: # min()  ( implicit item() as well )
                break_point = min(break_point, i+self.gate_delay)
            
            if i >= break_point or i+1 >= max_length:
                break
            
            decoder_input = mel_output
            del decode_args[-4:]# delete attention_rnn_outputs, decoder_rnn_outputs and hifi_outputs from decoder_args
        else:
            print("Warning! Reached max decoder steps")
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        # apply sigmoid to the GPU as well.
        gate_outputs = torch.sigmoid(gate_outputs)
        
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.n_frames_per_frame = getattr(hparams, 'n_frames_per_frame', 1)
        self.n_fpf_per_elem     = getattr(hparams, 'n_fpf_per_elem', False)
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.teacher_force_till = hparams.teacher_force_till
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.gradient_checkpoint = getattr(hparams, 'gradient_checkpoint', False)
        self.mdn_align_grads = True# override using 'run_every_epoch.py' or whatever.
        
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        self.drop_frame_rate = hparams.drop_frame_rate
        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        
        self.encoder = Encoder(hparams)
        mem_dim = hparams.encoder_LSTM_dim
        
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(hparams.n_speakers, self.speaker_embedding_dim)
            mem_dim += hparams.speaker_embedding_dim
        
        self.sylps_net = SylpsNet(hparams)
        mem_dim += 1
        
        self.tm_linear = nn.Linear(hparams.torchMoji_attDim, hparams.torchMoji_crushedDim)
        if hparams.torchMoji_BatchNorm:
            self.tm_bn = MaskedBatchNorm1d(hparams.torchMoji_attDim, eval_only_momentum=True, affine=False, momentum=0.05)
        mem_dim += hparams.torchMoji_crushedDim
        
        if not hasattr(hparams, 'res_enc_n_tokens'):
            hparams.res_enc_n_tokens = 0
        if hasattr(hparams, 'use_res_enc') and hparams.use_res_enc:
            self.res_enc = ReferenceEncoder(hparams)
            mem_dim += getattr(hparams, 'res_enc_embed_dim', 128)
        
        if hparams.use_memory_bottleneck:
            self.memory_bottleneck = MemoryBottleneck(hparams, mem_dim)
            mem_dim = self.memory_bottleneck.mem_output_dim
        
        if getattr(hparams, 'enable_MDN', False):
            self.MDN = MDN(hparams, self.memory_bottleneck.mem_output_dim)
            
            if getattr(hparams, 'enable_align_fft', False):
                from CookieTTS._2_ttm.MelFlow.model import FFT
                self.attention_fft = FFT(mem_dim, hparams.align_fft_n_heads,
                        hparams.align_fft_ff_dim, hparams.align_fft_n_layers,
                        add_position_encoding=True, position_encoding_random_start=True, legacy=True)
        
        self.decoder = Decoder(hparams, mem_dim)
        
        if not hasattr(hparams, 'use_postnet') or hparams.use_postnet:
            self.postnet = Postnet(hparams)
        
        if getattr(hparams, 'HiFiGAN_enable', False):
            self.postnet_hifigan = Postnet(hparams)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self, gt_mel, mel_lengths, text, text_lengths, speaker_id, gt_sylps,
                torchmoji_hdn, pres_prev_state, cont_next_iter, init_mel,
                teacher_force_till=None, p_teacher_forcing=None, drop_frame_rate=None, return_hidden_state=False):
        assert not (torch.isnan(gt_mel).any() or torch.isinf(gt_mel).any())
        assert not (torch.isnan(text).any() or torch.isinf(text).any())
        assert not (torch.isnan(gt_sylps).any() or torch.isinf(gt_sylps).any())
        assert not (torch.isnan(torchmoji_hdn).any() or torch.isinf(torchmoji_hdn).any())
        assert not (torch.isnan(init_mel).any() or torch.isinf(init_mel).any())
        # package into dict for output
        outputs = {}
        
        with torch.no_grad():
            p_teacher_forcing  = self.p_teacher_forcing  if teacher_force_till is None else p_teacher_forcing
            teacher_force_till = self.teacher_force_till if p_teacher_forcing  is None else teacher_force_till
            drop_frame_rate    = self.drop_frame_rate    if drop_frame_rate    is None else drop_frame_rate
            
            if drop_frame_rate > 0. and self.training:
                gt_mel = dropout_frame(gt_mel, self.global_mean, mel_lengths, drop_frame_rate)# [B, n_mel, mel_T]
            
            if True:
                res_gt_mel = dropout_frame(gt_mel, self.global_mean, mel_lengths, 0.5)# [B, n_mel, mel_T]
        
        gt_mel.requires_grad_()# <-(for gradient checkpoint func)
        
        memory = []
        
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, txt_T]
        if False and self.training and self.gradient_checkpoint:
            encoder_outputs, hidden_state, pred_sylps = checkpoint(self.encoder, *(embedded_text, text_lengths, speaker_id)) # [B, txt_T, enc_dim]
        else:
            encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]    
        outputs["encoder_outputs"] = encoder_outputs# [B, txt_T, enc_dim]
        outputs["pred_sylps"]      = pred_sylps     # [B]
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_id)# [B, embed]
            memory.append( speaker_embed[:, None].expand(-1, encoder_outputs.size(1), -1) )
            outputs["speaker_embed"] = speaker_embed# [B, embed]
        
        # (SylpsNet) Sylps -> sylzu, mu, logvar
        sylzu, syl_mu, syl_logvar = self.sylps_net(gt_sylps)
        assert not (torch.isnan(sylzu).any() or torch.isinf(sylzu).any())
        assert not (torch.isnan(syl_mu).any() or torch.isinf(syl_mu).any())
        assert not (torch.isnan(syl_logvar).any() or torch.isinf(syl_logvar).any())
        memory.append( sylzu[:, None].expand(-1, encoder_outputs.size(1), -1) )
        outputs["pred_sylps_mu"]     = syl_mu    # [B]
        outputs["pred_sylps_logvar"] = syl_logvar# [B]
        
        # (TorchMoji)
        if hasattr(self, 'tm_bn'):
            torchmoji_hdn = self.tm_bn(torchmoji_hdn).to(sylzu)# [B, hdn_dim]
            assert not (torch.isnan(torchmoji_hdn).any() or torch.isinf(torchmoji_hdn).any())
        torchmoji_hdn = self.tm_linear(torchmoji_hdn)          # [B, hdn_dim] -> [B, crushed_dim]
        assert not (torch.isnan(torchmoji_hdn).any() or torch.isinf(torchmoji_hdn).any())
        memory.append(torchmoji_hdn[:, None].expand(-1, encoder_outputs.size(1), -1))# [B, C] -> [B, txt_T, C]
        
        # (Residual Encoder) gt_mel -> res_embed
        if hasattr(self, 'res_enc'):
            res_embed, zr, r_mu, r_logvar, r_mu_logvar = self.res_enc(res_gt_mel, mel_lengths, speaker_embed)# -> [B, embed]
            assert not (torch.isnan(res_embed).any() or torch.isinf(res_embed).any())
            res_embed = res_embed[:, None].expand(-1, encoder_outputs.size(1), -1)# -> [B, txt_T, embed]
            memory.append(res_embed)
            outputs["res_enc_pkg"] = [r_mu, r_logvar, r_mu_logvar,]# [B, n_tokens], [B, n_tokens], [B, 2*n_tokens]
        
        # (Encoder/Attention) merge memory and calculate alignment override if used.
        memory = torch.cat(memory, dim=2)# concat along Embed dim # [B, txt_T, dim]
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory, speaker_embed)
        with torch.no_grad():
            assert not (torch.isnan(memory).any() or torch.isinf(memory).any())
        
        attention_contexts = alignment = None
        if hasattr(self, 'MDN'):
            mdn_loss, alignment, pred_logdur = self.MDN(memory, gt_mel, text_lengths, mel_lengths, mdn_align_grads=self.mdn_align_grads)
            outputs['mdn_loss'] = mdn_loss      # [B]
            outputs['mdn_alignment'] = alignment# [B, mel_T, txt_T]
            outputs['pred_logdur'] = pred_logdur# [B, txt_T]
            
            if alignment is not None and hasattr(self, 'attention_fft'):
                attention_contexts = alignment @ memory # [B, mel_T, txt_T] @ [B, txt_T, dim] -> [B, mel_T, dim]
                attention_contexts = self.attention_fft(attention_contexts, mel_lengths)[0]# [B, mel_T, dim]
        
        # (Decoder/Attention) memory -> pred_mel
        global_cond = torch.cat((speaker_embed, torchmoji_hdn, sylzu), dim=1)
        _ = self.decoder.forward_init(memory, global_cond, speaker_embed, gt_mel, memory_lengths=text_lengths, preserve_decoder=pres_prev_state, decoder_input=init_mel,
                      attention_weights=alignment, gt_attention_contexts=attention_contexts, teacher_force_till=teacher_force_till, p_teacher_forcing=p_teacher_forcing, return_hidden_state=return_hidden_state)
        pred_mels, pred_gate_logits, alignments, memory, hiddens = _
        
        outputs["pred_mel"        ] = pred_mels       # [B, n_mel*n_frames, mel_T]
        outputs["pred_gate_logits"] = pred_gate_logits# [B, mel_T]
        outputs["alignments"      ] = alignments      # [B, mel_T, txt_T]
        
        if hasattr(self, 'postnet_hifigan'):
            hiddens['hifigan_inputs'] = self.postnet_hifigan(hiddens['hifigan_inputs'])
        if hasattr(self.decoder, 'variational_encoder'):
            outputs["var_mu"    ] = hiddens["var_mu"    ]
            outputs["var_logvar"] = hiddens["var_logvar"]
        if self.decoder.return_attention_rnn_outputs or 'attention_rnn_output' in hiddens:
            outputs['attention_rnn_output'] = hiddens['attention_rnn_outputs']# [B, 2ndAttRNNDim or 1stAttRNNDim, mel_T]
        if self.decoder.return_decoder_rnn_outputs   or 'decoder_rnn_output'   in hiddens:
            outputs['decoder_rnn_output'  ] = hiddens['decoder_rnn_outputs'  ]# [B, 2ndDecRNNDim or 1stDecRNNDim, mel_T]
        if self.decoder.return_attention_contexts    or 'attention_contexts'   in hiddens:
            outputs['attention_contexts'  ] = hiddens['attention_contexts'   ]# [B, memory_dim, mel_T]
        if 'hifigan_inputs' in hiddens:
            outputs['hifigan_inputs'      ] = hiddens['hifigan_inputs'       ]# [B, memory_dim, mel_T]
        if 'mel_outputs_logprob' in hiddens:
            outputs['mel_outputs_logprob' ] = hiddens['mel_outputs_logprob'  ]# [B, n_frames_per_frame, mel_T]
        
        # (Postnet) pred_mel -> pred_mel_postnet (learn a modifier for the output)
        if hasattr(self, 'postnet'):
            pred_mel = self.best_frames(pred_mels, gt_mel, frame_mean=not self.n_fpf_per_elem)
            if self.training and self.gradient_checkpoint:
                pred_mel_postnet = checkpoint(self.postnet, pred_mel)
            else:
                pred_mel_postnet = self.postnet(pred_mel)
            outputs["pred_mel_postnet"] = pred_mel_postnet# [B, n_mel, mel_T]
        
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
    
    def best_frames(self, pred_mels, gt_mel, frame_mean=True):# [B, n_mel*n_frames, mel_T], [B, n_mel, mel_T]
        """Output best frames from a set of predicted mels and the target mel."""
        B, n_mel_n_frames, mel_T = pred_mels.shape
        pred_mels = pred_mels.clone().chunk(self.n_frames_per_frame, dim=1)# [[B, n_mel, mel_T],]*n_frames
        aerr_mels = []
        for i in range(self.n_frames_per_frame):
            aerr_mel = F.l1_loss(pred_mels[i].detach(), gt_mel.detach(), reduction='none')
            if frame_mean:
                aerr_mel = aerr_mel.mean(dim=1, keepdim=True)
            aerr_mels.append(aerr_mel)# -> [B, n_mel, mel_T]
        
        pred_mel = pred_mels[0]
        aerr_mel = aerr_mels[0]
        for i in range(1, self.n_frames_per_frame):
            pred_mel = pred_mel.where(aerr_mel < aerr_mels[i], pred_mels[i])
            aerr_mel = aerr_mel.where(aerr_mel < aerr_mels[i], aerr_mels[i])
        
        return pred_mel# [B, n_mel, mel_T]
    
    def inference(self, text_seq, text_lengths, speaker_id, torchmoji_hdn, gt_sylps=None, gt_mel=None, res_std=0.0, return_hidden_state=False):# [B, enc_T], [B], [B], [B], [B, tm_dim]
        memory = []
        
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text_seq).transpose(1, 2) # [B, embed, txt_T]
        encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_id)
            memory.append( speaker_embed[:, None].expand(-1, encoder_outputs.size(1), -1) )
        
        # (SylpsNet) Sylps -> sylzu, mu, logvar
        sylzu = self.sylps_net.infer_auto(gt_sylps or pred_sylps, rand_sampling=False)
        memory.append( sylzu[:, None].expand(-1, encoder_outputs.size(1), -1) )
        
        # (TorchMoji)
        if hasattr(self, 'tm_bn'):
            torchmoji_hdn = self.tm_bn(torchmoji_hdn).to(sylzu)# [B, hdn_dim]
        torchmoji_hdn = self.tm_linear(torchmoji_hdn)#       [B, hdn_dim] -> [B, crushed_dim]
        memory.append(torchmoji_hdn[:, None].expand(-1, encoder_outputs.size(1), -1))
        
        # (Residual Encoder) gt_mel -> res_embed
        if hasattr(self, 'res_enc'):
            if gt_mel is not None:
                res_embed, zr, r_mu, r_logvar = self.res_enc(gt_mel, rand_sampling=False)# -> [B, embed]
            else:
                res_embed = self.res_enc.prior(encoder_outputs, std=res_std)# -> [B, embed]
            res_embed = res_embed[:, None].expand(-1, encoder_outputs.size(1), -1)# -> [B, txt_T, embed]
            memory.append(res_embed)
        
        # (Encoder/Attention) merge memory and calculate alignment override if used.
        memory = torch.cat(memory, dim=2)# concat along Embed dim # [B, txt_T, dim]
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory, speaker_embed)
        global_cond = torch.cat((speaker_embed, torchmoji_hdn, sylzu), dim=1)
        
        mel_lengths = alignment = None
        if hasattr(self, 'MDN'):
            _, alignment, mel_lengths = self.MDN.infer(memory, text_lengths)
            
            if alignment is not None and hasattr(self, 'attention_fft'):
                attention_contexts = alignment @ memory # [B, mel_T, txt_T] @ [B, txt_T, dim] -> [B, mel_T, dim]
                attention_contexts = self.attention_fft(attention_contexts, mel_lengths)[0]# [B, mel_T, dim]
        
        # (Decoder/Attention) memory -> pred_mel
        pred_mel, pred_gate, alignments = self.decoder.inference(memory, attention_contexts, alignment, global_cond, speaker_embed, memory_lengths=text_lengths)
        
        # (Postnet) pred_mel -> pred_mel_postnet (learn a modifier for the output)
        pred_mel_postnet = self.postnet(pred_mel) if hasattr(self, 'postnet') else mel_outputs
        
        outputs = {
           "pred_mel_postnet": pred_mel_postnet,# [B, n_mel, mel_T]
                  "pred_gate": pred_gate,       # [B, mel_T]
                 "alignments": alignments,      # [B, mel_T, txt_T]
                 "pred_sylps": pred_sylps,      # [B]
        }
        return outputs

class ResBlock1d(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_dim, n_blocks, kernel_w, bias=True, act_func=nn.LeakyReLU(negative_slope=0.2, inplace=False), dropout=0.0, stride=1, res=True):
        super(ResBlock1d, self).__init__()
        self.blocks = nn.ModuleList()
        for j in range(n_blocks):
            layers = nn.ModuleList()
            for i in range(n_layers):
                in_dim  = input_dim  if (i==0) and (j==0) else n_dim
                out_dim = output_dim if (i+1==n_layers) and (j+1==n_blocks) else n_dim
                pad = (kernel_w - 1)//2
                conv = nn.Conv1d(in_dim, out_dim, kernel_w, padding=pad, stride=stride, bias=bias)
                layers.append(conv)
            self.blocks.append(layers)
        self.act_func = act_func
        self.dropout = dropout
        self.res = res
    
    def forward(self, x): # [B, in_dim, T]
        if len(x.shape) == 4 and x.shape[1] == 1:# if [B, 1, H, W]
            x = x.squeeze(1)# [B, 1, H, W] -> [B, H, W]
        
        squeeze_T = False
        if len(x.shape) == 2:
            squeeze_T = True
            x = x.unsqueeze(-1)# [B, C] -> [B, C, 1]
        
        skip = x
        for j, block_layers in enumerate(self.blocks):
            for i, layer in enumerate(block_layers):
                is_last_layer = bool( i+1 == len(block_layers) )
                x = layer(x)
                if not is_last_layer:
                    x = self.act_func(x)
                if self.dropout > 0.0 and self.training:
                    x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
            if self.res:
                if skip.shape[1]==x.shape[1]:
                    x = x + skip
                skip = x
        
        if squeeze_T:
            x = x.squeeze(2)
        return x# [B, out_dim, T]


class ResBlock2d(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_dim, kernel_h, kernel_w, stride=1, bias=True, act_func=nn.LeakyReLU(negative_slope=0.2, inplace=True), dropout=0.0, res=False):
        super(ResBlock2d, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim  =  input_dim if i == 0 else n_dim
            out_dim = output_dim if i+1 == n_layers else n_dim
            pad_h = (kernel_h - 1)//2
            pad_w = (kernel_w - 1)//2
            conv = nn.Conv2d(in_dim, out_dim, [kernel_h, kernel_w], padding=[pad_h, pad_w], stride=stride, bias=bias)
            self.layers.append(conv)
        self.act_func = act_func
        self.dropout = dropout
        self.res = res
        if self.res:
            assert input_dim == output_dim, 'residual connection requires input_dim and output_dim to match.'
    
    def forward(self, x): # [B, in_dim, n_mel_channels, T]
        if len(x.shape) == 3:# [B, in_dim, n_mel_channels] -> [B, in_dim, n_mel_channels, T]
            x = x.unsqueeze(-1)
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
        return x # [B, out_dim, n_mel_channels, T]


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n_gpus
    return rt

class DebluraGANSubModule(nn.Module):# contains the actual Layers/Modules, does not contain any optimizers or saving/loading stuff
    def __init__(self, hparams):
        super(DebluraGANSubModule, self).__init__()
        self.prenet = []
        assert len(hparams.dbGAN_prenet_stride) == hparams.dbGAN_prenet_n_blocks, 'n_blocks != len(stride)'
        viewed_prenet_dim = hparams.n_mel_channels
        for i in range(hparams.dbGAN_prenet_n_blocks):
            is_first_block = bool(i   == 0)
            is_last_block  = bool(i+1 == hparams.dbGAN_prenet_n_blocks)
            self.prenet.append(ResBlock2d(1 if is_first_block else hparams.dbGAN_prenet_dim, hparams.dbGAN_prenet_dim,
                                          hparams.dbGAN_prenet_n_layers, hparams.dbGAN_prenet_dim,
                                          kernel_h=hparams.dbGAN_prenet_kernel_h, kernel_w=hparams.dbGAN_prenet_kernel_w, stride=hparams.dbGAN_prenet_stride[i]))
            for j in range(hparams.dbGAN_prenet_n_layers):
                viewed_prenet_dim = ((viewed_prenet_dim +2*((hparams.dbGAN_prenet_kernel_h-1)//2) -1*(hparams.dbGAN_prenet_kernel_h-1) -1)//hparams.dbGAN_prenet_stride[i][0])+1
        self.prenet = nn.Sequential(*self.prenet)
        
        viewed_prenet_dim *= hparams.dbGAN_prenet_dim
        
        self.postnet = []
        for i in range(hparams.dbGAN_n_blocks):
            is_first_block = bool(i   == 0)
            is_last_block  = bool(i+1 == hparams.dbGAN_n_blocks)
            self.postnet.append(ResBlock1d(viewed_prenet_dim+hparams.speaker_embedding_dim if is_first_block else hparams.dbGAN_dim, 1 if is_last_block else hparams.dbGAN_dim,
                                           hparams.dbGAN_n_layers, hparams.dbGAN_dim, kernel_w=hparams.dbGAN_kernel_w, stride=hparams.dbGAN_stride))
        self.postnet = nn.Sequential(*self.postnet)
    
    def forward(self, mels: Tensor, speaker_embed: Tensor):
        assert not (torch.isnan(mels) | torch.isinf(mels)).any(), 'NaN or Inf value found in computation'
        speaker_embed = speaker_embed.clone().detach()
        mels = mels + 5.0
        mels = mels + mels.clone().normal_(std=3.5)# add random noise to the input to force the discriminator to focus on all aspects of the input instead of just focusing on the really small bits it can reliably use.
        
        if len(speaker_embed.shape) == 2:
            speaker_embed = speaker_embed.unsqueeze(-1)# -> [B, embed, 1]
        B, embed_dim, *_ = speaker_embed.shape
        
        if len(mels.shape) == 3:
            mels.unsqueeze(1)# [B, C, T] -> [B, 1, C, T]
        
        mels = self.prenet(mels)# [B, 1, C, T] -> [B, prenet_dim, C//4, T//4]
        assert not (torch.isnan(mels) | torch.isinf(mels)).any(), 'NaN or Inf value found in computation'
        B, *_, mel_T = mels.shape
        mels = mels.view(B, -1, mel_T)# [B, prenet_dim, C//4, T//4] -> [B, prenet_dim*C//4, T//4]
        mels = torch.cat((mels, speaker_embed.to(mels).expand(B, embed_dim, mels.shape[2])), dim=1)# [B, C//4, T//4] + [B, speaker_embed] -> [B, C//4+spkr_embed, T//4]
        pred_fakeness = self.postnet(mels)# [B, C//4+spkr_embed, T//4] -> [B, 1, T//4]
        assert not (torch.isnan(mels) | torch.isinf(mels)).any(), 'NaN or Inf value found in computation'
        
        pred_fakeness = pred_fakeness.squeeze(1)# [B, 1, T//4] -> [B, T//4]
        return pred_fakeness# [B, (mel_T//prenet_stride^(prenet_n_blocks*prenet_n_layers))//(n_blocks*stride)]

class DebluraGAN(nn.Module):# GAN loss on spectrogram to reduce blur from predicted outputs
    def __init__(self, hparams):
        super(DebluraGAN, self).__init__()
        self.optimizer     = None
        self.discriminator = DebluraGANSubModule(hparams)
        self.fp16_run    = hparams.fp16_run
        self.n_gpus      = hparams.n_gpus
        self.gradient_checkpoint = hparams.gradient_checkpoint
    
    def state_dict(self):
        dict_ = {
        "discriminator_state_dict": self.discriminator.state_dict(),
            "optimzier_state_dict": self.optimizer.state_dict(),
        }
        return dict_
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_file(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def load_state_dict(self, dict_):
        local_dict = self.discriminator.state_dict()
        new_dict   = dict_['discriminator_state_dict']
        local_dict.update({k: v for k,v in new_dict.items() if k in local_dict and local_dict[k].shape == new_dict[k].shape})
        n_missing_keys = len([k for k,v in new_dict.items() if not (k in local_dict and local_dict[k].shape == new_dict[k].shape)])
        self.discriminator.load_state_dict(local_dict)
        del local_dict, new_dict
        
        if 1 and n_missing_keys == 0:
            self.optimizer.load_state_dict(dict_['optimzier_state_dict'])
    
    def forward(self, pred, gt, reduced_loss_dict, loss_dict, loss_scalars, tfB=None):
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        
        if tfB is None:
            tfB = pred['speaker_embed'].shape[0]# Does this actually pick up the tfB? or just B?
        pred_mel         = pred['pred_mel_postnet'][:tfB].float().detach().unsqueeze(1)# [B, 1, n_mel, mel_T]
        pred_mel_postnet = pred['pred_mel_b'      ][:tfB].float().detach().unsqueeze(1)# [B, 1, n_mel, mel_T]
        gt_mel           =   gt['gt_mel'          ][:tfB].float().detach().unsqueeze(1)# [B, 1, n_mel, mel_T]
        
        B, _, n_mel, mel_T = gt_mel.shape
        mels = torch.cat((gt_mel, pred_mel, pred_mel_postnet), dim=0)# -> [3*B, 1, n_mel, mel_T]
        assert not (torch.isnan(mels) | torch.isinf(mels)).any(), 'NaN or Inf value found in computation'
        
        #if self.training and self.gradient_checkpoint:
        #    pred_fakeness = checkpoint(self.discriminator, mels, gt['speaker_id'].repeat(3)).squeeze(1)# -> [B, mel_T//?]
        #else:
        pred_fakeness = self.discriminator(mels, pred['speaker_embed'][:tfB].repeat(3, 1)).squeeze(1)# -> [3*B, mel_T//?]
        
        gt_fakeness, pred_fakeness, postnet_fakeness = pred_fakeness.chunk(3, dim=0)# -> [B, mel_T//?], [B, mel_T//?], [B, mel_T//?]
        B, mel_T = gt_fakeness.shape
        fake_label = torch.ones(B, mel_T, device=gt_mel.device, dtype=gt_mel.dtype)     # [B, mel_T//?]
        real_label = torch.ones(B, mel_T, device=gt_mel.device, dtype=gt_mel.dtype)*-1.0# [B, mel_T//?]
        
        loss_dict['dbGAN_dLoss'] = F.mse_loss(real_label, gt_fakeness)*0.5 + F.mse_loss(fake_label, pred_fakeness)*0.25 + F.mse_loss(fake_label, postnet_fakeness)*0.25
        loss = loss_dict['dbGAN_dLoss'] * loss_scalars['dbGAN_dLoss_weight']
        
        reduced_loss_dict['dbGAN_dLoss'] = reduce_tensor(loss_dict['dbGAN_dLoss'].data, self.n_gpus).item() if self.n_gpus > 1 else loss_dict['dbGAN_dLoss'].item()
        
        with torch.no_grad():
            B_mel_T = gt_fakeness.numel()# B*mel_T
            reduced_loss_dict['dbGAN_accuracy'   ] = ((   pred_fakeness>0.0).sum()+(gt_fakeness<0.0).sum()).item()/(2*B_mel_T)
            reduced_loss_dict['dbGAN_accuracy_pn'] = ((postnet_fakeness>0.0).sum()+(gt_fakeness<0.0).sum()).item()/(2*B_mel_T)
        
        if self.fp16_run:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        self.optimizer.step()


class ResGAN(nn.Module):
    def __init__(self, hparams):
        super(ResGAN, self).__init__()
        self.optimizer     = None
        self.discriminator = ResBlock1d(hparams.res_enc_n_tokens*2, hparams.n_speakers+hparams.n_symbols,
                                        hparams.res_enc_n_layers,   hparams.res_enc_dis_dim, kernel_w=1)
        self.gt_speakers = None
        self.gt_sym_durs = None
        self.fp16_run    = hparams.fp16_run
        self.n_symbols, self.n_speakers = hparams.n_symbols, hparams.n_speakers
    
    def state_dict(self):
        dict_ = {
        "discriminator_state_dict": self.discriminator.state_dict(),
            "optimzier_state_dict": self.optimizer.state_dict(),
                     "gt_speakers": self.gt_speakers,
                     "gt_sym_durs": self.gt_sym_durs,
                        "fp16_run": self.fp16_run,
                       "n_symbols": self.n_symbols,
                      "n_speakers": self.n_speakers,
        }
        return dict_
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_file(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def load_state_dict(self, dict_):
        local_dict = self.discriminator.state_dict()
        new_dict   = dict_['discriminator_state_dict']
        local_dict.update({k: v for k,v in new_dict.items() if k in local_dict and local_dict[k].shape == new_dict[k].shape})
        n_missing_keys = len([k for k,v in new_dict.items() if not (k in local_dict and local_dict[k].shape == new_dict[k].shape)])
        self.discriminator.load_state_dict(local_dict)
        del local_dict, new_dict
        
        if n_missing_keys == 0:
            self.optimizer.load_state_dict(dict_['optimzier_state_dict'])
        
        if False:
            self.gt_speakers = dict_["gt_speakers"]
            self.gt_sym_durs = dict_["gt_sym_durs"]
            self.fp16_run    = dict_["fp16_run"]
            self.n_symbols   = dict_["n_symbols"]
            self.n_speakers  = dict_["n_speakers"]
    
    def forward(self, pred, reduced_loss_dict, loss_dict, loss_scalars):
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        
        _, _, mulogvar = pred['res_enc_pkg']
        out = self.discriminator(mulogvar.detach())
        B = out.shape[0]
        pred_sym_durs, pred_speakers = out.squeeze(-1).split([self.n_symbols, self.n_speakers], dim=1)
        pred_speakers = torch.nn.functional.softmax(pred_speakers, dim=1)
        loss_dict['res_enc_dMSE'] = (nn.MSELoss(reduction='sum')(pred_sym_durs, self.gt_sym_durs)*0.0001 + nn.MSELoss(reduction='sum')(pred_speakers, self.gt_speakers))/B
        loss = loss_dict['res_enc_dMSE'] * loss_scalars['res_enc_dMSE_weight']
        reduced_loss_dict['res_enc_dMSE'] = loss_dict['res_enc_dMSE'].item()
        
        if self.fp16_run:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        self.optimizer.step()
        

