import os
import random
import json
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict

from CookieTTS.utils.model.utils import get_mask_from_lengths

from CookieTTS._4_mtw.hifigan.env import AttrDict, build_env
from CookieTTS._4_mtw.hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss,\
    generator_loss, discriminator_loss

from CookieTTS._4_mtw.hifigan.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, del_old_checkpoints, save_checkpoint

from CookieTTS.utils.audio.stft import TacotronSTFT

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n_gpus
    return rt

# Should predict "inferenceness" by predicting whether the spectrogram came from teacher forcing or inference.
# The generator will then attempt to reduce inferenceness and increase teacher-forcedness outputs on the discriminator.
class HiFiGAN_wrapper(nn.Module):
    def __init__(self, hparams):
        super(HiFiGAN_wrapper, self).__init__()
        assert hparams.batch_size>=hparams.HiFiGAN_batch_size, 'HiFiGAN batch size must be even or greater than Tacotron2 batch size!'
        
        self.g_optimizer = None
        self.d_optimizer = None
        
        self.generator        = None
        self.mp_discriminator = None
        self.ms_discriminator = None
        
        self.STFT = TacotronSTFT(
                 filter_length = hparams.HiFiGAN_filter_length,
                 hop_length    = hparams.HiFiGAN_hop_length,
                 win_length    = hparams.HiFiGAN_win_length,
                 n_mel_channels= hparams.HiFiGAN_n_mel_channels,
                 sampling_rate = hparams.sampling_rate,
                 clamp_val     = hparams.HiFiGAN_clamp_val,
                 mel_fmin=0.0,
                 mel_fmax=min(hparams.sampling_rate/2., 19000.),
                 stft_dtype=torch.float32)
        
        self.n_models_to_keep = 9# sync with tacotron n_models_to_keep if it's added
        self.fp16_run    = hparams.fp16_run
        self.n_gpus      = hparams.n_gpus
        self.gradient_checkpoint = hparams.gradient_checkpoint
    
    def load_config(self, path):
        assert os.path.exists(path), f'config path does not exist!\path = "{path}"'
        with open(path) as f:
            data = f.read()
        
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
    
    def save_state_dict(self, path=None):
        if path is None:
            assert hasattr(self, 'cp_path'), 'tried to save HiFiGAN but no checkpoint path found!'
            path = self.cp_path
        
        # checkpointing
        steps = self.steps
        
        gpath = path
        if os.path.isdir(path):
            gpath = "{}/g_{:08d}".format(gpath, steps)
        save_checkpoint(gpath, {'generator': self.generator.state_dict(), 'steps': self.steps, 'ft_steps': self.ft_steps})
        
        dopath = path
        if os.path.isdir(path):
            dopath = "{}/do_{:08d}".format(dopath, steps)
        save_checkpoint(dopath,
                        {'mpd'     : self.mp_discriminator.state_dict(),
                         'msd'     : self.ms_discriminator.state_dict(),
                         'optim_g' : self.g_optimizer.state_dict(),
                         'optim_d' : self.d_optimizer.state_dict(),
                         'steps'   : self.steps,
                         'ft_steps': self.ft_steps})
        if os.path.isdir(path):
            del_old_checkpoints(path, 'g_' , self.n_models_to_keep)
            del_old_checkpoints(path, 'do_', self.n_models_to_keep)
    
    def load_state_dict_from_file(self, path:str, device='cuda'):
        assert os.path.exists(path), f'path does not exist!\path = "{path}"'
        if os.path.isdir(path):
            self.cp_path = path
            self.load_config(os.path.join(path, 'config.json'))
            h = self.h
            
            self.generator        = Generator(h).to(device)
            self.mp_discriminator = MultiPeriodDiscriminator(h["discriminator_periods"] if "discriminator_periods" in h.keys() else None).to(device)
            self.ms_discriminator = MultiScaleDiscriminator().to(device)
            
            g_path = scan_checkpoint(path, 'g_' )
            if g_path is not None:
                state_dict_g = load_checkpoint(g_path, 'cpu')
                gsd = self.generator.state_dict()
                gsd.update({k: v for k,v in state_dict_g['generator'].items() if k in gsd and state_dict_g['generator'][k].shape == gsd[k].shape})
                missing_keys = {k: v for k,v in state_dict_g['generator'].items() if not (k in gsd and state_dict_g['generator'][k].shape == gsd[k].shape)}.keys()
                self.generator.load_state_dict(gsd)
                del gsd, state_dict_g
            
            self.steps = 0
            self.ft_steps = -1
            do_path = scan_checkpoint(path, 'do_')
            if do_path is None or len(missing_keys):
                state_dict_do = None
            else:
                state_dict_do = load_checkpoint(do_path, 'cpu')
                self.mp_discriminator.load_state_dict(state_dict_do['mpd'])
                self.ms_discriminator.load_state_dict(state_dict_do['msd'])
                del state_dict_do['msd'], state_dict_do['mpd']
                self.steps = state_dict_do['steps']
                if 'ft_steps' in state_dict_do:
                    self.ft_steps = state_dict_do['ft_steps']
            
            self.g_optimizer = torch.optim.AdamW(
                               self.generator.parameters(),
                               h.learning_rate, betas=[h.adam_b1, h.adam_b2])
            self.d_optimizer = torch.optim.AdamW(
                               itertools.chain(self.ms_discriminator.parameters(), self.mp_discriminator.parameters()),
                               h.learning_rate, betas=[h.adam_b1, h.adam_b2])
            
            if state_dict_do is not None:
                #self.g_optimizer.load_state_dict(state_dict_do['optim_g'])
                #self.d_optimizer.load_state_dict(state_dict_do['optim_d'])
                del state_dict_do
            
            #self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=h.lr_decay, last_epoch=self.ft_steps)
            #self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=h.lr_decay, last_epoch=self.ft_steps)
        else:
            raise NotImplementedError('HiFiGAN_cp_folder must be a folder/directory!')
    
    def discriminator_loss(self, gt_audio, pred_audio):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mp_discriminator(gt_audio, pred_audio)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.ms_discriminator(gt_audio, pred_audio)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_disc_all = loss_disc_s + loss_disc_f
        return loss_disc_all, loss_disc_s, loss_disc_f
    
    def generator_loss(self, gt_audio, pred_audio, gt_mel, pred_mel, loss_dict):
        # Generator
        self.g_optimizer.zero_grad()
        
        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(gt_mel.detach(), pred_mel)
        
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mp_discriminator(gt_audio, pred_audio)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.ms_discriminator(gt_audio, pred_audio)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        #loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45
        
        loss_dict['HiFiGAN_g_msd_class'] = loss_gen_s
        loss_dict['HiFiGAN_g_mpd_class'] = loss_gen_f
        loss_dict['HiFiGAN_g_all_class'] = loss_gen_s+loss_gen_f
        loss_dict['HiFiGAN_g_msd_featuremap'] = loss_fm_s
        loss_dict['HiFiGAN_g_mpd_featuremap'] = loss_fm_f
        loss_dict['HiFiGAN_g_all_featuremap'] = loss_fm_s+loss_fm_f
        loss_dict['HiFiGAN_g_all_mel_mae'   ] = loss_mel
    
    def g_optimizer_step_and_clear(self):
        self.g_optimizer.step()
        self.g_optimizer.zero_grad()
    
    def forward(self, tt2_model, pred:dict, gt:dict, reduced_loss_dict:dict, loss_dict:dict, loss_scalars:dict):
        #self.g_optimizer.step() # run this **before** gradient clipping?
        self.d_optimizer.zero_grad()
        
        if pred['hifigan_indexes'].sum():
            gt_audio   = gt['hifigan_gt_audio']
            gt_mel     = gt['hifigan_gt_mel']
            pred_audio = self.generator(pred['hifigan_inputs'])
            pred_mel   = self.STFT.mel_spectrogram(pred_audio.squeeze(1))
            
            # discriminator
            loss, loss_disc_s, loss_disc_f = self.discriminator_loss(gt_audio, pred_audio.detach())
            loss = loss * loss_scalars.get('HiFiGAN_d_all_class_weight', 1.0)
            
            if self.fp16_run:
                with self.amp.scale_loss(loss, self.d_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            self.g_optimizer.zero_grad()
            self.d_optimizer.step()
            
            loss_dict['HiFiGAN_d_msd_class'] = loss_disc_s.detach()
            loss_dict['HiFiGAN_d_mpd_class'] = loss_disc_f.detach()
            reduced_loss_dict['HiFiGAN_d_msd_class'] = reduce_tensor(loss_dict['HiFiGAN_d_msd_class'].data, self.n_gpus).item() if self.n_gpus > 1 else loss_dict['HiFiGAN_d_msd_class'].item()
            reduced_loss_dict['HiFiGAN_d_mpd_class'] = reduce_tensor(loss_dict['HiFiGAN_d_mpd_class'].data, self.n_gpus).item() if self.n_gpus > 1 else loss_dict['HiFiGAN_d_mpd_class'].item()
            reduced_loss_dict['HiFiGAN_d_all_class'] = reduced_loss_dict['HiFiGAN_d_msd_class'] + reduced_loss_dict['HiFiGAN_d_mpd_class']
        
        self.steps    += 1
        self.ft_steps += 1
        #self.scheduler_g.step() # f*** it, just use 'run_every_epoch'. This pytorch scheduler is inflexible as hell
        #self.scheduler_d.step() # f*** it, just use 'run_every_epoch'. This pytorch scheduler is inflexible as hell

