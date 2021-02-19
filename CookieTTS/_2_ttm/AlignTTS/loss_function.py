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
    
    def forward(self, mu_logvar, melspec, text_lengths, mel_lengths, pad_mag=1e12):
        # mu, sigma: [B, txt_T, 2*n_mel]
        #   melspec: [B, n_mel,   mel_T]
        
        B, txt_T, _ = mu_logvar.size()
        mel_T = melspec.size(2)
        
        x = melspec.transpose(1,2).unsqueeze(1) # [B, 1, mel_T, n_mel]
        mu     = mu_logvar[:, :, :self.n_mel_channels ].unsqueeze(2)# [B, txt_T, 1, n_mel]
        logvar = mu_logvar[:, :,  self.n_mel_channels:].unsqueeze(2)# [B, txt_T, 1, n_mel]
        
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
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1]
        alpha_last = alpha_last/mel_lengths# avg by length of the log_alpha
        mdn_loss = -alpha_last.mean()
        
        return mdn_loss, log_prob_matrix



    
@torch.no_grad()
def downsize_mel(gt_mel, MDN_mel_downscale=1):
    if MDN_mel_downscale == 1:
        return gt_mel
    
    gt_mel = gt_mel.detach().exp()# [B, n_mel, mel_T]
    new_mel = gt_mel[:, 0::MDN_mel_downscale].clone()
    for i in range(1, MDN_mel_downscale):
        new_mel += gt_mel[:, i::MDN_mel_downscale]
    return new_mel.log()# [B, n_mel//downscale, mel_T]

class AlignTTSLoss(nn.Module):
    def __init__(self, hparams):
        super(AlignTTSLoss, self).__init__()
        self.rank    = hparams.rank
        self.text_repeat_interleave = hparams.text_repeat_interleave
        self.mel_downsize = hparams.mel_downsize
        
        self.MDNLoss = MDNLoss(hparams.n_mel_channels//self.mel_downsize)
        self.mdn_loss_weight = 1.0
    
    def colate_losses(self, loss_dict, loss_scalars, loss=None):
        for k, v in loss_dict.items():
            loss_scale = loss_scalars.get(f'{k}_weight', None)
            if loss_scale is None:
                loss_scale = getattr(self, f'{k}_weight', None)
            if loss_scale is None:
                loss_scale = 1.0
                print(f'{k} is missing loss weight')
            if loss_scale > 0.0:
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
        ##  Tacotron Loss  ##
        #####################
        if True:
            mu_logvar    = pred['mu_logvar']
            gt_mel       =   gt['gt_mel']
            text_lengths =   gt['text_lengths']
            mel_lengths  =   gt['mel_lengths']
            gt_mel = downsize_mel(gt_mel, self.mel_downsize)
            B, n_mel, mel_T = gt_mel.shape
            assert n_mel == mu_logvar.shape[2]//2, f'got {mu_logvar.shape[2]} channels from mu_logvar, expected {n_mel*2}.'
            mdn_loss, log_prob_matrix = self.MDNLoss(mu_logvar, gt_mel, text_lengths, mel_lengths)
            loss_dict['mdn_loss'] = mdn_loss
        
        #align = None
        #if calc_alignments or save_alignments:
        #    with torch.no_grad():
        #        B = len(gt['audiopath'])
        #        mdn_alignment = model.viterbi(log_prob_matrix.cpu(), text_lengths.cpu(), mel_lengths.cpu()).to(torch.long)
        #        align = align.detach().cpu()# [B, txt_T, mel_T]
        #        if save_alignments:
        #            for i in range(B):
        #                if gt['arpa'][i]:
        #                    alignpath = os.path.splitext(audiopath[i])[0]+'_palign.pt'
        #                else:
        #                    alignpath = os.path.splitext(audiopath[i])[0]+'_galign.pt'
        #                torch.save(alignpath, align[i].detach().clone())# [txt_T, mel_T]
        
        with torch.no_grad():
            mdn_alignment = model.viterbi(log_prob_matrix.cpu(), text_lengths.cpu(), mel_lengths.cpu()).to(torch.long)
            mdn_alignment = mdn_alignment.transpose(1, 2)# [B, txt_T, mel_T] -> [B, mel_T, txt_T]
            assert mdn_alignment.shape[1] == gt['mel_lengths'].max(), f"got shape of {mdn_alignment.shape}, expected size(1) == {gt['mel_lengths'].max()}"
            assert mdn_alignment.shape[2] == gt['text_lengths'].max(), f"got shape of {mdn_alignment.shape}, expected size(2) == {gt['text_lengths'].max()}"
            if calc_alignments or save_alignments:
                not_truncated = ~(gt['pres_prev_state'] | gt['cont_next_iter'])
                for i, not_trunc in enumerate(not_truncated):
                    if not_trunc:
                        mel_length  = gt['mel_lengths'][i]
                        text_length = gt['text_lengths'][i]
                        alignment = mdn_alignment[i, :mel_length, :text_length]# -> [mel_T, txt_T]
                        if alignment.sum() != mel_length.item():
                            print(f'alignment.sum() = {alignment.sum()}, expected {mel_length}'); continue
                        
                        audiopath = gt['audiopath'][i]
                        is_arpa   = gt['arpa'][i]
                        alignpath = os.path.splitext(audiopath)[0]+('_palign.pt' if is_arpa else '_galign.pt')
                        torch.save(alignment.detach().clone().half(), alignpath)
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        return loss_dict, file_losses, mdn_alignment

