import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import numpy as np
import math
from CookieTTS.utils.model.utils import get_mask_from_lengths, alignment_metric, get_first_over_thresh, freeze_grads
from typing import Optional


class MDNLoss(nn.Module):
    def __init__(self, n_mel_channels):
        super(MDNLoss, self).__init__()
        self.n_mel_channels = n_mel_channels
    
    def forward(self, mu_logvar, melspec, text_lengths, mel_lengths):
        # mu, sigma: [B, txt_T, 2*n_mel]
        #   melspec: [B, n_mel,   mel_T]
        
        B, txt_T, _ = mu_logvar.size()
        mel_T = melspec.size(2)
        
        x = melspec.transpose(1,2).unsqueeze(1) # [B, 1, mel_T, n_mel]
        mu     = mu_logvar[:, :, :self.n_mel_channels ].unsqueeze(2)# [B, txt_T, 1, n_mel]
        logvar = mu_logvar[:, :,  self.n_mel_channels:].unsqueeze(2)# [B, txt_T, 1, n_mel]
        
        # SquaredError/logstd -> SquaredError/std -> SquaredError/var -> NLL Loss
        
        # [B, 1, mel_T, n_mel]-[B, txt_T, 1, n_mel] -> [B, txt_T, mel_T, n_mel] -> [B, txt_T, mel_T]
        exponential = -0.5 * ( ((x-mu).pow(2)/logvar.exp())+logvar ).mean(dim=3)
        
        log_prob_matrix = exponential# - (self.n_mel_channels/2)*torch.log(torch.tensor(2*math.pi))# [B, txt_T, mel_T] - [B, 1, mel_T]
        log_alpha = mu_logvar.new_ones(B, txt_T, mel_T)*(-1e30)
        log_alpha[:,0, 0] = log_prob_matrix[:,0, 0]
        
        for t in range(1, mel_T):
            prev_step = torch.cat([log_alpha[:, :, t-1:t], F.pad(log_alpha[:, :, t-1:t], (0,0,1,-1), value=-1e30)], dim=-1)
            log_alpha[:, :, t] = torch.logsumexp(prev_step+1e-30, dim=-1)+log_prob_matrix[:, :, t]
        
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1]
        alpha_last = alpha_last/mel_lengths# avg by length of the log_alpha
        mdn_loss = -alpha_last.mean()
        
        return mdn_loss, log_prob_matrix


class GlowLoss(nn.Module):
    def __init__(self, sigma=1., n_group=None):
        super().__init__()
        self.var     = sigma**2
        self.n_group = n_group
    
    def forward(self, z, log_s_, logdet_w, mel_lengths, z_mu:Optional[torch.Tensor]=None, z_logvar:Optional[torch.Tensor]=None):
        B, z_dim, mel_T = z.shape
        
        log_s_lengths = (mel_lengths//self.n_group)*(self.n_group-1)
        
        numel = mel_lengths.sum()*z_dim
        numfr = mel_lengths.sum()
        mask = get_mask_from_lengths(mel_lengths)# [B, mel_T]
        
        # z_loss
        z = z[:, :, :mel_lengths.max().item()]
        z = z.masked_select(mask.unsqueeze(1))# [B, z_dim, mel_T] -> [B*z_dim*mel_T]
        
        if z_mu is not None:
            z_mu = z_mu.masked_select(mask.unsqueeze(1))# [B, z_dim, mel_T] -> [B*z_dim*mel_T]
            if z_logvar is not None:
                z_logvar = z_logvar.masked_select(mask.unsqueeze(1))# [B, z_dim, mel_T] -> [B*z_dim*mel_T]
                
                z_loss  = torch.sum(z_logvar) + 0.5*torch.sum((-2.*z_logvar).exp()*(z-z_mu).pow(2))# neg normal likelihood w/o the constant term
                z_loss /= numel
            else:
                z_loss = ((z-z_mu).pow(2)/(2*self.var)).mean(dtype=torch.float)# []
        else:
            z_loss = z.pow(2).mean(dtype=torch.float)/(2*self.var)# []
        
        # s_loss
        log_s = log_s_[0].mean(dim=1, dtype=torch.float).view(B, -1)# [B, mel_T+8-mel_T%8]
        for log_s_k in log_s_[1:]:
            log_s += log_s_k.mean(dim=1, dtype=torch.float).view(B, -1)# [B, mel_T+8-mel_T%8]
        log_s = log_s[:, :log_s_lengths.max().item()]# [B, mel_T+8-mel_T%8] -> [B, mel_T]
        
        mask = get_mask_from_lengths(log_s_lengths)# [B, mel_T]
        log_s = log_s.masked_select(mask)# [B*mel_T]
        
        s_loss = -(log_s.sum(dtype=torch.float)/(numfr))# [B, mel_T] -> []
        
        # w_loss
        logdet_w_sum = logdet_w[0].float()
        for log_det_w_k in logdet_w[1:]:
            logdet_w_sum += log_det_w_k
        w_loss = -(logdet_w_sum/self.n_group)# []
        
        loss = z_loss+s_loss+w_loss
        
        return loss, z_loss, s_loss, w_loss


class MelFlowLoss(nn.Module):
    def __init__(self, hparams):
        super(MelFlowLoss, self).__init__()
        self.rank    = hparams.rank
        self.MDNLoss = MDNLoss(hparams.n_mel_channels)
        
        self.GlowLoss = GlowLoss(hparams.sigma, hparams.n_group)
        self.melglow_z_weight = 0.0
        self.melglow_s_weight = 0.0
        self.melglow_w_weight = 0.0
        
        self.uncond_melglow_total_loss_weight = 1.0
        self.uncond_melglow_z_weight = 0.0
        self.uncond_melglow_s_weight = 0.0
        self.uncond_melglow_w_weight = 0.0
    
    def colate_losses(self, loss_dict, loss_scalars, loss=None):
        for k, v in loss_dict.items():
            loss_scale = loss_scalars.get(f'{k}_weight', None)
            if loss_scale is None:
                loss_scale = getattr(self, f'{k}_weight', None)
            if loss_scale is None:
                loss_scale = 1.0
                print(f'{k} is missing loss weight')
            if loss_scale > 0.0:
                assert not torch.isnan(v), f'{k} is NaN'
                assert not torch.isinf(v), f'{k} is Inf'
                if loss is not None:
                    loss = loss + v*loss_scale
                else:
                    loss = v*loss_scale
            if False and self.rank == 0:
                print(f'{k:20} {loss_scale:05.2f} {loss:05.2f} {loss_scale*v:+010.6f}', v)
        loss_dict['loss'] = loss
        return loss_dict
    
    def forward(self, model, pred, gt, loss_scalars, calc_alignments=False, save_alignments=False):
        loss_dict = {}
        file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        for i in range(B):
            current_time = time.time()
            if gt['audiopath'][i] not in file_losses:
                file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        #####################
        ##  MelFlow Loss   ##
        #####################
        if ("pred_durations" in pred) and ("alignment" in pred):
            pred_durs = pred['pred_durations'].squeeze(-1)# [B, txt_T]
            mask = get_mask_from_lengths(gt['text_lengths'])# [B, txt_T]
            pred_durs.masked_fill_(~mask, 0.0)
            loss_dict['dur_loss'] = F.mse_loss(pred['alignment'].sum(dim=1).detach().clamp(min=1.).log().masked_fill(~mask, 0.0), pred_durs)# [B, txt_T], [B, txt_T] -> []
        
        if "melenc_mu_logvar" in pred:
            logvar, mu = pred['melenc_mu_logvar'].chunk(2, dim=1)# [B, 2*melenc_n_tokens] -> [B, melenc_n_tokens], [B, melenc_n_tokens]
            B, n_tokens = mu.shape
            loss_dict['melenc_kld'] = -0.5 * ( 1 +logvar -logvar.exp() -mu.pow(2) ).sum()/B
        
        if "mdn_loss" in pred:
            loss_dict['mdn_loss'] = pred['mdn_loss']# []
        
        if "melglow_pack" in pred:
            z, logdet_w, log_s = pred['melglow_pack']
            mel_lengths = gt['mel_lengths']
            z_mu = z_logvar = None
            if "z_mu_logvar" in pred:
                z_mu, z_logvar = pred["z_mu_logvar"].transpose(1, 2).chunk(2, dim=1)# [B, mel_T, 2*n_mel] -> [B, n_mel, mel_T], [B, n_mel, mel_T]
            loss, z_loss, s_loss, w_loss = self.GlowLoss(z, log_s, logdet_w, mel_lengths, z_mu, z_logvar)
            loss_dict['melglow_total_loss'] = loss
            loss_dict['melglow_z']        = z_loss
            loss_dict['melglow_s']        = s_loss
            loss_dict['melglow_w']        = w_loss
        
        with torch.no_grad():
            B = len(gt['audiopath'])
            align = pred['mdn_alignment'].detach().cpu()# [B, mel_T, txt_T]
            if save_alignments:
                for i in range(B):
                    if gt['arpa'][i]:
                        alignpath = os.path.splitext(audiopath[i])[0]+'_palign.pt'
                    else:
                        alignpath = os.path.splitext(audiopath[i])[0]+'_galign.pt'
                    torch.save(alignpath, align[i].clone())# [mel_T, txt_T]
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        return loss_dict, file_losses, align

