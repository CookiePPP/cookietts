import time
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import inspect
import torch.distributed as dist
from typing import Optional
import numpy as np
import math
from CookieTTS.utils.model.utils import get_mask_from_lengths, alignment_metric, get_first_over_thresh, freeze_grads
from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d
from CookieTTS.utils.audio.stft import TacotronSTFT

class Loss(nn.Module):
    def __init__(self, hparams):
        super(Loss, self).__init__()
        self.rank       = hparams.rank
        self.n_gpus     = hparams.n_gpus
        
        self.STFT = TacotronSTFT(
                 filter_length = hparams.HiFiGAN_filter_length,
                 hop_length    = hparams.HiFiGAN_hop_length,
                 win_length    = hparams.HiFiGAN_win_length,
                 n_mel_channels= hparams.HiFiGAN_n_mel_channels,
                 sampling_rate = hparams.sampling_rate,
                 clamp_val     = hparams.HiFiGAN_clamp_val,
                 mel_fmin=0.0,
                 mel_fmax=min(hparams.sampling_rate/2., 20000.),
                 stft_dtype=torch.float32,)
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def colate_losses(self, loss_dict, loss_scalars, loss=None, loss_key_append=''):
        for k, v in loss_dict.items():
            loss_scale = loss_scalars.get(f'{k}_weight', None)
            
            if loss_scale is None:
                loss_scale = getattr(self, f'{k}_weight', None)
            
            if loss_scale is None:
                loss_scale = 1.0
                print(f'{k} is missing loss weight')
            
            if loss_scale > 0.0:
                new_loss = v*loss_scale
                if new_loss > 40.0 or math.isnan(new_loss) or math.isinf(new_loss):
                    print(f'{k} reached {v}.')
                if loss is not None:
                    loss = loss + new_loss
                else:
                    loss = new_loss
            if False and self.rank == 0:
                print(f'{k:20} {loss_scale:05.2f} {loss:05.2f} {loss_scale*v:+010.6f}', v)
        loss_dict['loss'+loss_key_append] = loss
        return loss_dict
    
    def d_forward(self, iteration, model, pred, gt, loss_scalars):
        loss_dict = {}
        file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        
        #with torch.no_grad():
        #    current_time = time.time()
        #    for i in range(B):
        #        if gt['audiopath'][i] not in file_losses:
        #            file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        if True:
            dtype = next(model.discriminator.parameters()).dtype
            gt_audio   =   gt['gt_audio'  ].unsqueeze(1).to(dtype)
            pred_audio = pred['pred_audio'].detach()    .to(dtype)
            _, loss_disc_s, loss_disc_f = model.discriminator.discriminator_loss(gt_audio, pred_audio)
            loss_dict['d_msd_class'] = loss_disc_s
            loss_dict['d_mpd_class'] = loss_disc_f
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars, loss_key_append='_d')
        return loss_dict, file_losses
    
    def g_forward(self, iteration, model, pred, gt, loss_scalars, file_losses={}):
        loss_dict = {}
        
        if True:
            dtype = next(model.parameters()).dtype
            gt_audio = gt['gt_audio'].unsqueeze(1)
            pred_audio = pred['pred_audio']
            
            gt_mel   = self.STFT.mel_spectrogram(gt_audio.squeeze(1), filter_pad=False)
            pred_mel = self.STFT.mel_spectrogram_with_grad(pred_audio.squeeze(1), filter_pad=False)
            pred['pred_mel'] = pred_mel
            #print(*[[x.dtype, x.device] for x in (gt_audio, pred_audio, gt_mel, pred_mel)])
            _ = model.generator_loss(gt_audio.to(dtype), pred_audio.to(dtype), gt_mel, pred_mel)
            loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_mel = _
            loss_dict['g_msd_class'] = loss_gen_s
            loss_dict['g_mpd_class'] = loss_gen_f
            loss_dict['g_msd_fm']    = loss_fm_s
            loss_dict['g_mpd_fm']    = loss_fm_f
            loss_dict['g_mel_mae']   = loss_mel
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        return loss_dict, file_losses

