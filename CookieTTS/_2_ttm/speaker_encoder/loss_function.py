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


class AutoVCLoss(nn.Module):
    def __init__(self, hparams):
        super(AutoVCLoss, self).__init__()
        self.rank       = hparams.rank
        self.n_speakers = hparams.n_speakers
    
    def vae_kl_anneal_function(self, anneal_function, lag, step, k, x0, upper):
        if anneal_function == 'logistic': # https://www.desmos.com/calculator/neksnpgtmz
            return float(upper/(upper+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            if step > lag:
                return min(upper, step/x0)
            else:
                return 0
        elif anneal_function == 'cycle':
            return min(1,(max(0,(step%x0)-lag))/k) * upper
        elif anneal_function == 'constant':
            return upper or 0.001
    
    def colate_losses(self, loss_dict, loss_scalars, loss=None):
        for k, v in loss_dict.items():
            loss_scale = loss_scalars.get(f'{k}_weight', None)
            if loss_scale is None:
                loss_scale = getattr(self, f'{k}_weight', 1.0)
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
    
    def forward(self, model, pred, gt, loss_scalars,):
        loss_dict = {}
        file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        for i in range(B):
            current_time = time.time()
            if gt['audiopath'][i] not in file_losses:
                file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        if True:
            pred_mel_postnet = pred['pred_mel_postnet']
            pred_mel         = pred['pred_mel']
            gt_mel           =   gt['gt_mel']
            mel_lengths      =   gt['mel_lengths']
            
            mask = get_mask_from_lengths(mel_lengths, max_len=gt_mel.size(2))
            mask = mask.expand(gt_mel.size(1), *mask.shape).permute(1, 0, 2)
            pred_mel_postnet.masked_fill_(~mask, 0.0)
            pred_mel        .masked_fill_(~mask, 0.0)
            
            with torch.no_grad():
                assert not torch.isnan(pred_mel).any(), 'mel has NaNs'
                assert not torch.isinf(pred_mel).any(), 'mel has Infs'
                assert not torch.isnan(pred_mel_postnet).any(), 'mel has NaNs'
                assert not torch.isinf(pred_mel_postnet).any(), 'mel has Infs'
            
            B, n_mel, mel_T = gt_mel.shape
            
            # spectrogram / decoder loss
            pred_mel_selected = torch.masked_select(pred_mel, mask)
            gt_mel_selected   = torch.masked_select(gt_mel,   mask)
            spec_SE = nn.MSELoss(reduction='none')(pred_mel_selected, gt_mel_selected)
            loss_dict['spec_MSE'] = spec_SE.mean()
            
            losses = spec_SE.split([x*n_mel for x in mel_lengths.cpu()])
            for i in range(B):
                audiopath = gt['audiopath'][i]
                file_losses[audiopath]['spec_MSE'] = losses[i].mean().item()
            
            # postnet
            pred_mel_postnet_selected = torch.masked_select(pred_mel_postnet, mask)
            loss_dict['postnet_MSE'] = nn.MSELoss()(pred_mel_postnet_selected, gt_mel_selected)
            
            # squared by frame, mean postnet
            mask = mask.transpose(1, 2)[:, :, :1]# [B, mel_T, n_mel] -> [B, mel_T, 1]
            
            spec_AE = nn.L1Loss(reduction='none')(pred_mel, gt_mel).transpose(1, 2)# -> [B, mel_T, n_mel]
            spec_AE = spec_AE.masked_select(mask).view(mel_lengths.sum(), n_mel)   # -> [B* mel_T, n_mel]
            loss_dict['spec_MFSE'] = (spec_AE * spec_AE.mean(dim=1, keepdim=True)).mean()# multiply by frame means (similar to square op from MSE) and get the mean of the losses
            
            post_AE = nn.L1Loss(reduction='none')(pred_mel_postnet, gt_mel).transpose(1, 2)# -> [B, mel_T, n_mel]
            post_AE = post_AE.masked_select(mask).view(mel_lengths.sum(), n_mel)# -> [B*mel_T, n_mel]
            loss_dict['postnet_MFSE'] = (post_AE * post_AE.mean(dim=1, keepdim=True)).mean()# multiply by frame means (similar to square op from MSE) and get the mean of the losses
            del gt_mel, spec_AE, post_AE,#pred_mel_postnet, pred_mel
        
        if True:
            # Code semantic loss.
            code_reconst = model(pred_mel_postnet, gt['speaker_embeds'], None)
            loss_dict['code_L1'] = F.l1_loss(pred['bottleneck_codes'], code_reconst)
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        return loss_dict, file_losses

