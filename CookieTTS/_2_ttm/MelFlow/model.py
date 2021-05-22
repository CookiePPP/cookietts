import inspect
from math import sqrt, ceil, log, pi
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
from CookieTTS._2_ttm.MelFlow.arglow import ARFlow

from CookieTTS.utils.model.transformer import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding

drop_rate = 0.5

def load_model(hparams):
    model = MelFlow(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

class Prenet(nn.Module):
    def __init__(self, hp):
        super(Prenet, self).__init__()
        # text embed
        self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        # speaker embed + linear
        self.speaker_embedding = nn.Embedding(hp.n_speakers, hp.speaker_embedding_dim)
        self.speaker_linear = nn.Linear(hp.speaker_embedding_dim, hp.symbols_embedding_dim)
        self.speaker_linear.weight.data *= 0.1
        
        self.register_buffer('pe', PositionalEncoding(hp.hidden_dim).pe)
        self.pos_enc_weight = nn.Parameter(torch.ones(1)*0.5)
        
        self.dropout = nn.Dropout(0.1)
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
    def __init__(self, hidden_dim, n_heads, ff_dim, n_layers, ff_kernel_size=1, rezero_pos_enc=True, add_position_encoding=False, position_encoding_random_start=False, rezero_transformer=True, legacy=False):
        super(FFT, self).__init__()
        self.FFT_layers = nn.ModuleList( [TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=ff_dim, ff_kernel_size=ff_kernel_size, rezero=rezero_transformer, legacy=legacy) for _ in range(n_layers)] )
        
        self.add_position_encoding = add_position_encoding
        if self.add_position_encoding:
            self.register_buffer('pe', PositionalEncoding(hidden_dim).pe)
            self.position_encoding_random_start = position_encoding_random_start
            self.rezero_pos_enc = rezero_pos_enc
            if self.rezero_pos_enc:
                self.pos_enc_weight = nn.Parameter(torch.ones(1)*1.0)
            if legacy:
                self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, lengths, return_alignments=False):# [B, L, D], [B]
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
        
        x = x.masked_fill_(~get_mask_from_lengths(lengths).unsqueeze(-1), 0.0)# [B, L, D] * [B, L, 1]
        
        alignments = []
        x = x.transpose(0,1)# [B, L, D] -> [L, B, D]
        mask = ~get_mask_from_lengths(lengths)# -> [B, L]
        for layer in self.FFT_layers:
            x, align = layer(x, src_key_padding_mask=mask)# -> [L, B, D], ???
            if save_alignments: alignments.append(align.unsqueeze(1))
        
        if return_alignments:
            return x.transpose(0,1), torch.cat(alignments, 1)# [B, L, D], [L, B, n_layers, L]
        else:
            return x.transpose(0,1)# [B, L, D]


class MelEncoder(nn.Module):
    def __init__(self, hparams, hp_prepend='', input_dim=None, output_dim=None):
        super(MelEncoder, self).__init__()
        self.std = 0.95# sigma for sampling
        self.variational_tokens = getattr(hparams, hp_prepend+'mel_enc_variational_tokens', True)
        
        conv_dim = getattr(hparams, hp_prepend+'mel_enc_conv_dim')
        self.conv = ConvNorm(input_dim or hparams.n_mel_channels, conv_dim,
                             kernel_size=3, stride=3, padding=0,)
        
        lstm_dim = getattr(hparams, hp_prepend+'mel_enc_lstm_dim')
        self.lstm = nn.LSTM(conv_dim, lstm_dim, num_layers=1, bidirectional=True)
        
        self.n_tokens = getattr(hparams, hp_prepend+'mel_enc_n_tokens')
        self.bottleneck = LinearNorm(lstm_dim*2, self.n_tokens*2 if self.variational_tokens else self.n_tokens)
        
        self.embed_fc = LinearNorm(self.n_tokens, output_dim or hparams.symbols_embedding_dim)
    
    def prior(self, x, std=0.0):
        if self.variational_tokens:
            zr = torch.randn(x.shape[0], self.n_tokens, device=x.device, dtype=next(self.parameters()).dtype) * std
        else:
            zr = torch.zeros(x.shape[0], self.n_tokens, device=x.device, dtype=next(self.parameters()).dtype)
        embed = self.embed_fc(zr)# [B, n_tokens] -> [B, embed]
        return embed
    
    def reparameterize(self, mu, logvar):# use for VAE sampling
        if self.std or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            if (not self.training) and self.std != 1.0:
                std *= float(self.std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, gt_mel):
        gt_mel = self.conv(gt_mel)# [B, n_mel, mel_T] -> [B, conv_dim, mel_T//3]
        
        _, states = self.lstm(gt_mel.permute(2, 0, 1))
        state = states[0]# -> [2, B, lstm_dim]
        
        state = state.permute(1, 0, 2).reshape(state.shape[1], -1)# [2, B, lstm_dim] -> [B, 2*lstm_dim]
        if self.variational_tokens:
            mulogvar = self.bottleneck(state)    #                [B, 2*lstm_dim] -> [B, 2*n_tokens]
            mu, logvar = mulogvar.chunk(2, dim=1)#                [B, 2*n_tokens] -> [B,   n_tokens], [B, n_tokens]
            out = self.reparameterize(mu, logvar)# [B, n_tokens], [B,   n_tokens] -> [B,   n_tokens]
        else:
            mulogvar = None
            out = self.bottleneck(state)# [B, 2*lstm_dim] -> [B, n_tokens]
        
        out = self.embed_fc(out)# [B, n_tokens] -> [B, hdn_dim]
        return out, mulogvar


class MelFlow(nn.Module):
    def __init__(self, hparams):
        super(MelFlow, self).__init__()
        self.n_mel_channels   = hparams.n_mel_channels
        self.memory_efficient = hparams.memory_efficient
        self.use_flow_conds   = getattr(hparams, 'use_flow_conds', False)
        
        # Variational Global Cond (to reduce impact of volume/speaker on MDN outputs accuracy)
        self.mel_encoder = MelEncoder(hparams)
        
        # Text / Embeddings
        self.Prenet = Prenet(hparams)
        
        # Text global cond(s)
        if hparams.torchMoji_BatchNorm:
            self.tm_bn = MaskedBatchNorm1d(hparams.torchMoji_attDim, eval_only_momentum=False, momentum=0.2)
        self.tm_linear = nn.Sequential(
            nn.Dropout(hparams.torchMoji_dropout, inplace=True),
            nn.Linear(hparams.torchMoji_attDim, hparams.torchMoji_crushedDim),
            nn.Linear(hparams.torchMoji_crushedDim, hparams.hidden_dim),)
        
        # Encoder + MDN + Duration Predictor
        self.FFT_lower   = FFT(hparams.hidden_dim, hparams.n_heads, hparams.ff_dim, hparams.n_layers)
        if self.use_flow_conds:
            self.cond_FFTs = nn.ModuleList([FFT(hparams.hidden_dim, hparams.n_heads, hparams.ff_dim, hparams.n_layers) for i in range(hparams.cond_n_FFTs)])
        
        self.MDN_lower = FFT(hparams.hidden_dim, hparams.mdn_n_heads, hparams.mdn_ff_dim, hparams.mdn_n_layers)
        self.MDN = nn.Sequential(nn.Linear(hparams.hidden_dim, hparams.hidden_dim),
                                 nn.LayerNorm(hparams.hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(hparams.hidden_dim, 2*hparams.n_mel_channels))
        
        if not self.use_flow_conds:
            self.z_MDN_lower = FFT(hparams.hidden_dim, hparams.mdn_n_heads, hparams.mdn_ff_dim, hparams.mdn_n_layers)
            self.z_MDN = nn.Sequential(nn.Linear(hparams.hidden_dim, hparams.hidden_dim),
                                   nn.LayerNorm(hparams.hidden_dim),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(hparams.hidden_dim, 2*hparams.n_mel_channels*self.z_channel_multiplier))
        
        self.duration_predictor = FFT(hparams.hidden_dim, hparams.durpred_n_heads, hparams.durpred_ff_dim, hparams.durpred_n_layers)
        self.durpred_post = nn.Linear(hparams.hidden_dim, 1)
        
        # Causal Autoregressive Glow - Spectrogram Decoder
        self.z_channel_multiplier = getattr(hparams, 'z_channel_multiplier', 1)
        self.decoder = ARFlow(hparams.hidden_dim          ,
                              hparams.n_flows             ,
                              hparams.n_group             ,
                              hparams.n_mel_channels      ,# update this with VFlow scalar later
                              hparams.n_early_every       ,
                              hparams.n_early_size        ,
                              hparams.memory_efficient    ,
                              hparams.n_cond_layers       ,
                              hparams.cond_hidden_channels,
                              hparams.cond_output_channels,
                              hparams.cond_kernel_size    ,
                              hparams.cond_residual       ,
                              hparams.cond_padding_mode   ,
                              hparams.WN_config           ,
               channel_mixing=hparams.channel_mixing      ,
                    mix_first=hparams.mix_first           ,
                  shift_spect=hparams.shift_spect         ,
                  scale_spect=hparams.scale_spect         ,
         z_channel_multiplier=self.z_channel_multiplier   ,)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    #@torch.jit.script
    def MDNLoss(self, mu_logvar, z, text_lengths, mel_lengths, n_mel_channels:int=160, latent_logp:bool=False, pad_mag:float=1e12):
        # mu, sigma: [B, txt_T, 2*n_mel]
        #         z: [B, n_mel,   mel_T]
        
        B, txt_T, _ = mu_logvar.size()
        mel_T = z.size(2)
        
        mu     = mu_logvar[:, :, :n_mel_channels ]# [B, txt_T, n_mel]
        logvar = mu_logvar[:, :,  n_mel_channels:]# [B, txt_T, n_mel]
        
        if latent_logp:
            x_s_sq_r = (-2*logvar).exp_()# [B, txt_T, n_mel]
            zpm =  z.pow(2).mul_(-0.5)
            mpm = mu.pow(2).mul_(-0.5)
            logp1 = ((-0.5*log(2*pi)) -logvar).sum(2, True)#      [] - [B, txt_T, n_mel] -> [B, txt_T,     1]
            logp2 = (    x_s_sq_r) @ zpm         # [B, txt_T, n_mel] @ [B, n_mel, mel_T] -> [B, txt_T, mel_T]
            logp3 = ( mu*x_s_sq_r) @ z           # [B, txt_T, n_mel] @ [B, n_mel, mel_T] -> [B, txt_T, mel_T]
            logp4 = (mpm*x_s_sq_r).sum(2, True)  # [B, txt_T, n_mel] * [B, txt_T, n_mel] -> [B, txt_T,     1]
            logp = logp1.add(logp2).add_(logp3).add_(logp4)# -> [B, txt_T, mel_T]
            exponential = logp
        else:
            x = z.transpose(1, 2).unsqueeze(1)# [B, n_mel, mel_T] -> [B, 1, mel_T, n_mel]
            mu     = mu    .unsqueeze(2)# [B, txt_T, 1, n_mel]
            logvar = logvar.unsqueeze(2)# [B, txt_T, 1, n_mel]
            # SquaredError/logstd -> SquaredError/std -> SquaredError/var -> NLL Loss
            # [B, 1, mel_T, n_mel]-[B, txt_T, 1, n_mel] -> [B, txt_T, mel_T, n_mel] -> [B, txt_T, mel_T]
            exponential = -0.5 * ( ((x-mu).pow_(2)/logvar.exp())+logvar ).mean(dim=3)
        
        log_prob_matrix = exponential# - (self.n_mel_channels/2)*torch.log(torch.tensor(2*math.pi))# [B, txt_T, mel_T] - [B, 1, mel_T]
        log_alpha = torch.ones(B, txt_T+1, mel_T, device=mu_logvar.device, dtype=mu_logvar.dtype)*(-pad_mag)
        log_alpha[:, 1, 0] = log_prob_matrix[:, 0, 0]
        
        for t in range(1, mel_T):
            prev_step = torch.cat([log_alpha[:, 1:, t-1:t], log_alpha[:, :-1, t-1:t]], dim=-1)
            log_alpha[:, 1:, t] = torch.logsumexp(prev_step.add_(1e-7), dim=-1).add(log_prob_matrix[:, :, t])
        
        log_alpha = log_alpha[:, 1:, :]
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1].clone()
        alpha_last = alpha_last/mel_lengths# avg by length of the log_alpha
        mdn_loss = -alpha_last.mean()
        
        return mdn_loss, log_prob_matrix
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def forward(self, text, text_lengths, gt_mel, mel_lengths, speaker_id, torchmoji_hdn, align_with_z, mdn_align_grads=True, log_viterbi=False, cpu_viterbi=False):
        outputs = {}
        text_mask = get_mask_from_lengths(text_lengths).unsqueeze(2)# [B, txt_T, 1]
        
        mel_cond, melenc_mu_logvar = self.maybe_cp(self.mel_encoder, *(gt_mel.clone().requires_grad_(),))# [B, hdn_dim], ...
        if not mdn_align_grads:
            melenc_mu_logvar.detach_()
        outputs["melenc_mu_logvar"] = melenc_mu_logvar# [B, 2*melenc_n_tokens]
        
        encoder_input = self.Prenet(text, speaker_id)
        hidden_states = self.maybe_cp(self.FFT_lower, *(encoder_input, text_lengths))[0]# [B, txt_T, hdn]
        
        dp_input = hidden_states.detach().clone().requires_grad_() if (self.use_flow_conds and (not mdn_align_grads)) else hidden_states# kill gradients to Lower Encoder if the alignment gradients are disabled to preserve the alignment accuracy.
        dp_out = self.maybe_cp(self.duration_predictor, *(dp_input, text_lengths))[0]# [B, txt_T, hdn]
        dp_out = self.durpred_post(dp_out)# [B, txt_T, hdn] -> # [B, txt_T, 1]
        outputs["pred_durations"] = dp_out
        
        mdn_inputs, _ = self.maybe_cp(self.MDN_lower, *(hidden_states+mel_cond.unsqueeze(1), text_lengths))
        mdn_mu_logvar = self.maybe_cp(self.MDN,       *(mdn_inputs,))
        outputs["mdn_mu_logvar"] = mdn_mu_logvar# [B, txt_T, 2*n_mel]
        
        if not self.use_flow_conds:
            z_mdn_inputs, _ = self.maybe_cp(self.z_MDN_lower, *(hidden_states+mel_cond.unsqueeze(1), text_lengths))
            z_mdn_mu_logvar = self.maybe_cp(self.z_MDN,       *(z_mdn_inputs,))
            outputs["z_mdn_mu_logvar"] = z_mdn_mu_logvar# [B, txt_T, 2*n_mel]
        
        if True:# run decoder and get alignments
            B, n_mel, mel_T = gt_mel.shape
            if self.use_flow_conds:# add encoder conditioning
                if mdn_align_grads:
                    mdn_loss, log_prob_matrix = self.MDNLoss(mdn_mu_logvar, gt_mel, text_lengths, mel_lengths, self.n_mel_channels)# [B, txt_T, mel_T]
                else:
                    with torch.no_grad():
                        mdn_loss, log_prob_matrix = self.MDNLoss(mdn_mu_logvar, gt_mel, text_lengths, mel_lengths, self.n_mel_channels)# [B, txt_T, mel_T]
                outputs['mdn_loss'] = mdn_loss
                with torch.no_grad():
                    alignment = self.viterbi(log_prob_matrix.detach().float().cpu(), text_lengths.cpu(), mel_lengths.cpu())
                    alignment = alignment.transpose(1, 2)[:, :mel_lengths.max().item()].to(gt_mel)# [B, mel_T, txt_T]
                    outputs["alignment"] = alignment# [B, mel_T, txt_T]
                
                torchmoji_hdn = self.tm_bn(torchmoji_hdn) if hasattr(self, 'tm_bn') else torchmoji_hdn
                
                hidden_states = hidden_states.detach() + encoder_input.detach().clone().requires_grad_()
                hidden_states = hidden_states + 0.1*self.tm_linear(torchmoji_hdn).unsqueeze(1)
                hidden_states *= text_mask
                for cond_FFT in self.cond_FFTs:
                    hidden_states = self.maybe_cp(cond_FFT, *(hidden_states, text_lengths))[0]# [B, txt_T, hdn]
                expanded_hidden_states  = self.maybe_cp(self.expand_seq, *(hidden_states, alignment)).transpose(1, 2)# -> [B, hdn, mel_T]
                expanded_hidden_states += self.Prenet.pe.roll(random.randint(0, 4999), 0)[:mel_T].transpose(0, 1).unsqueeze(0)# += [mel_T, 1, hdn]
            else:# use blank conditioning
                B, txt_T, hdn_dim = hidden_states.shape
                expanded_hidden_states = gt_mel.new_zeros(B, hdn_dim, mel_T).requires_grad_()# [B, hdn, mel_T]
            
            # decode gt_mel into Z
            z, logdet_w, log_s = self.decoder(expanded_hidden_states, gt_mel, mel_lengths)
            log_s = [x.transpose(2, 3).reshape(x.shape[0], x.shape[1], -1) for x in log_s]# [B, n_mel, n_group, mel_T//n_group] -> [B, n_mel, mel_T]
            outputs["melglow_pack"] = [z, logdet_w, log_s]
            
            if not self.use_flow_conds:
                if align_with_z:
                    with torch.no_grad():
                        log_prob_matrix = self.MDNLoss(z_mdn_mu_logvar, z, text_lengths, mel_lengths, self.n_mel_channels, latent_logp=True)[1].detach()# [B, txt_T, mel_T]
                else:
                    if mdn_align_grads:
                        mdn_loss, log_prob_matrix = self.MDNLoss(mdn_mu_logvar, gt_mel, text_lengths, mel_lengths, self.n_mel_channels)# [B, txt_T, mel_T]
                    else:
                        with torch.no_grad():
                            mdn_loss, log_prob_matrix = self.MDNLoss(mdn_mu_logvar, gt_mel, text_lengths, mel_lengths, self.n_mel_channels)# [B, txt_T, mel_T]
                    outputs['mdn_loss'] = mdn_loss
                with torch.no_grad():
                    alignment = self.viterbi(log_prob_matrix.detach().float().cpu(), text_lengths.cpu(), mel_lengths.cpu())
                    alignment = alignment.transpose(1, 2)[:, :mel_lengths.max().item()].to(gt_mel)# [B, mel_T, txt_T]
                    outputs["alignment"] = alignment# [B, mel_T, txt_T]
                
                # conditional Z target distribution
                outputs['z_mu_logvar'] = alignment @ z_mdn_mu_logvar# [B, mel_T, txt_T] @ [B, txt_T, 2*n_mel] -> [B, mel_T, 2*n_mel]
        
        return outputs
    
    def infer(self, text, text_lengths, speaker_id, torchmoji_hdn, alignment=None, mel_lengths=None, sigma_scale=0.95):
        outputs = {}
        text_mask = get_mask_from_lengths(text_lengths).unsqueeze(2)# [B, txt_T, 1]
        
        mel_cond = self.mel_encoder.prior(text)# -> [B, 2*melenc_n_tokens]
        
        encoder_input = self.Prenet(text, speaker_id)
        hidden_states = self.maybe_cp(self.FFT_lower, *(encoder_input, text_lengths))[0]# [B, txt_T, hdn]
        
        dp_out = self.maybe_cp(self.duration_predictor, *(hidden_states, text_lengths))[0]# [B, txt_T, hdn]
        log_dur = self.durpred_post(dp_out)# [B, txt_T, hdn] -> # [B, txt_T, 1]
        
        dur = log_dur.squeeze(-1).exp()# [B, txt_T]
        mask = get_mask_from_lengths(text_lengths)
        dur.masked_fill_(~mask, 0.0)
        outputs["pred_durations"] = dur
        
        if alignment is None:
            mel_lengths = dur.round().sum(dim=1).long()# [B, txt_T] -> # [B]
            outputs["mel_lengths"] = mel_lengths
            
            #expanded_seq = self.get_attention_from_lengths(seq, dur, text_lengths)# [B, seq_T, enc_dim], [B, seq_T], [B] -> [B, trg_T, enc_dim]
        
        mdn_inputs, _ = self.maybe_cp(self.MDN_lower, *(hidden_states+mel_cond.unsqueeze(1), text_lengths))
        mdn_mu_logvar = self.maybe_cp(self.MDN,       *(mdn_inputs,))
        outputs["mdn_mu_logvar"] = mdn_mu_logvar# [B, txt_T, 2*n_mel]
        
        B, txt_T, hdn_dim = hidden_states.shape
        mel_T = int(mel_lengths.max().item())
        if not self.use_flow_conds:
            z_mdn_inputs, _ = self.maybe_cp(self.z_MDN_lower, *(hidden_states+mel_cond.unsqueeze(1), text_lengths))
            z_mdn_mu_logvar = self.maybe_cp(self.z_MDN,       *(z_mdn_inputs,))
            outputs["z_mdn_mu_logvar"] = z_mdn_mu_logvar# [B, txt_T, 2*n_mel]
            
            if alignment is not None:
                z_mu_logvar = (alignment @ z_mdn_mu_logvar).transpose(1, 2)# [B, mel_T, txt_T] @ [B, txt_T, 2*n_mel] -> [B, 2*n_mel, mel_T]
            else:
                z_mu_logvar = self.get_attention_from_lengths(z_mdn_mu_logvar, dur, text_lengths)# [B, txt_T, 2*n_mel], [B, txt_T], [B] -> [B, mel_T, enc_dim]
            z_mu     = z_mu_logvar.chunk(2, dim=1)[0]# [B, n_mel, mel_T]
            z_logvar = z_mu_logvar.chunk(2, dim=1)[1]# [B, n_mel, mel_T]
            
            z = (z_mu + torch.randn_like(z_mu)*z_logvar.exp()*sigma_scale)# [B, n_mel, mel_T]
            
            cond = z.new_zeros(B, hdn_dim, mel_T)# [B, hdn, mel_T]
        else:
            z = torch.randn(B, self.n_mel_channels*self.z_channel_multiplier, mel_T, device=hidden_states.device, dtype=hidden_states.dtype)*sigma_scale# [B, n_mel, mel_T]
            
            torchmoji_hdn = self.tm_bn(torchmoji_hdn) if hasattr(self, 'tm_bn') else torchmoji_hdn
            
            hidden_states = hidden_states.detach() + encoder_input.detach().clone()
            hidden_states  = hidden_states + 0.1*self.tm_linear(torchmoji_hdn).unsqueeze(1)
            hidden_states *= text_mask
            for cond_FFT in self.cond_FFTs:
                hidden_states = self.maybe_cp(cond_FFT, *(hidden_states, text_lengths))[0]# [B, txt_T, hdn]
            if alignment is not None:
                expanded_hidden_states = self.maybe_cp(self.expand_seq, *(hidden_states, alignment)).transpose(1, 2)# -> [B, hdn, mel_T]
            else:
                expanded_hidden_states = self.get_attention_from_lengths(hidden_states, dur, text_lengths).transpose(1, 2)# [B, txt_T, hdn], [B, txt_T], [B] -> [B, mel_T, enc_dim] -> [B, enc_dim, mel_T]
            expanded_hidden_states += self.Prenet.pe.roll(random.randint(0, 4999), 0)[:mel_T].transpose(0, 1).unsqueeze(0)# += [mel_T, 1, hdn]
            cond = expanded_hidden_states
        
        if mel_lengths is not None:
            z_mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)# [B, 1, mel_T]
            z *= z_mask
        
        outputs['pred_mel'] = self.decoder.inverse(z, cond, mel_lengths)
        if mel_lengths is not None:
            outputs['pred_mel'].masked_fill_(~get_mask_from_lengths(mel_lengths).unsqueeze(1), 0.0).clamp_(min=-11.52, max=4.0)
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
    
    def expand_seq(self, h, alignment, duration_mode=False):# [B, txt_T, hdn], [B, mel_T, txt_T]
        assert h.shape[1] == alignment.shape[2], 'hidden_states and alignment have different text lengths'
        assert h.shape[0] == alignment.shape[0], 'hidden_states and alignment have different num of items in dim 0'
        B, txt_T, hdn   = h.shape
        B, mel_T, txt_T = alignment.shape
        
        if duration_mode:# duration_mode should use less VRAM and run faster, but the input alignment must be monotonic.
            hy = torch.zeros(B, hdn_dim, mel_T)# [B, hdn, mel_T]
            for enc_idx in range(txt_T):
                pass
        else:
            hy = alignment@h# [B, mel_T, txt_T] @ [B, txt_T, hdn] -> [B, mel_T, hdn]
        return hy
    
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
        
        log_beta = _log_prob_matrix.new_ones(B, L, T)*(-1e3)
        log_beta[:, 0, 0] = _log_prob_matrix[:, 0, 0]
        
        for t in range(1, T):
            prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-1e3)], dim=-1).max(dim=-1)[0]
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
