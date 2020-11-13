import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from distributed import reduce_tensor
from CookieTTS.utils.audio.stft import STFT
from CookieTTS.utils.audio.audio_processing import window_sumsquare, dynamic_range_compression, dynamic_range_decompression

class MultiResSpect(nn.Module):
    def __init__(self, win_lens, fil_lens, hop_lens):
        super(MultiResSpect, self).__init__()
        self.max_channels = max(fil_lens)//2
        self.stfts = nn.ModuleList()
        for win_len, fil_len, hop_len in zip(win_lens, fil_lens, hop_lens):
            stft = STFT(filter_length=fil_len,
                           hop_length=hop_len,
                           win_length=win_len,)
            self.stfts.append(stft)
    
    def get_mel(self, audio):# [B, T]
        """Take audio and convert to multi-res spectrogram"""
        min_CT = int(9e9)
        melspec = []
        for stft in self.stfts:
            spect = stft.transform(audio, return_phase=False)[0]# -> [B, n_mel, spec_T]
            spect = spect[:, :-1, :]
            B, C, T = spect.shape
            min_CT = min(min_CT, (C*T)//self.max_channels*self.max_channels)
            melspec.append(spect)
        
        melspec = [x[:, :, :min_CT//x.shape[1]] for x in melspec]# cut-off remainders
        melspec = [x.reshape(B, self.max_channels, -1) for x in melspec]# reshaping different hop-lengths for concatenation
        return torch.cat(melspec, dim=1)# [[B, n_mel, spec_T], ...] -> [B, n_stft*n_mel, spec_T]


class HiFiGANLoss(nn.Module):
    def __init__(self, DW_config, DS_config, WN_config=None, postnet_config=None, sampling_rate=48000, stage=0):
        super(HiFiGANLoss, self).__init__()
        self.needs_reduction=True
        self.stage = stage
        
        self.MRS = MultiResSpect(DS_config['window_lengths'], DS_config['filter_lengths'], DS_config['hop_lengths'])
        
        if stage >= 2:
            from modules import DS
            #self.f_max = DS_config['max_freq']
            #self.f_min = DS_config['min_freq']
            #n_spect_channels = (DS_config['filter_length']//2+1)
            #self.mask = torch.ones(n_spect_channels)
            #self.mask[round((self.f_max/sampling_rate)*n_spect_channels):] = 0.
            #self.mask[:round((self.f_min/sampling_rate)*n_spect_channels)] = 0.
            self.discriminatorS = DS(**DS_config)
            
            from modules import DW
            self.discriminatorW = DW(**DW_config)
    
    def forward(self, pred_audio, gt_audio, amp, model, optimizer, optimizer_d, num_gpus, use_grad_clip, grad_clip_thresh): # optional cond input
        """
        pred_audio: [B, T]
        gt_audio: [B, T]
        """
        metrics = {}
        pred_spect = self.MRS.get_mel(pred_audio)
        pred_spect = dynamic_range_compression(pred_spect)# linear -> log magnitudes
        gt_spec = self.MRS.get_mel(gt_audio)
        gt_spec = dynamic_range_compression(gt_spec)# linear -> log magnitudes
        if self.stage >= 2:
            real_labels = torch.zeros(gt_audio.shape[0], device=gt_audio.device, dtype=gt_audio.dtype)# [B]
            fake_labels = torch.ones( gt_audio.shape[0], device=gt_audio.device, dtype=gt_audio.dtype)# [B]
            
            if False:# (optional) mask frequencies that humans can't hear in STFT
                pred_spect *= self.mask
                gt_spec *= self.mask
            
            #############################
            ###    Generator Stuff    ###
            #############################
            mel_fake_pred_fakeness = self.discriminatorS(pred_spect)# [B] predict fakeness of generated spectrogram
            wav_fake_pred_fakeness = self.discriminatorW(pred_audio)# [B] predict fakeness of generated audio
            fake_pred_fakeness = (mel_fake_pred_fakeness+wav_fake_pred_fakeness).sigmoid()# Average and range between 0.0 and 1.0
            loss = nn.BCELoss()(fake_pred_fakeness, real_labels) # [B] -> [] calc loss to decrease fakeness of model
            
            metrics['g_train_loss'] = reduce_tensor(loss.data, num_gpus).item() if num_gpus > 1 else loss.item()
            
            if amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if use_grad_clip:
                if amp is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), grad_clip_thresh)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_thresh)
                if type(grad_norm) == torch.Tensor:
                    grad_norm = grad_norm.item()
                metrics['is_overflow'] = math.isinf(grad_norm) or math.isnan(grad_norm)
                if not metrics['is_overflow']:
                    metrics['grad_norm'] = grad_norm
            else: metrics['is_overflow'] = False; grad_norm=1e-6
            
            optimizer.step()
            
            #############################
            ###  Discriminator Stuff  ###
            #############################
            optimizer_d.zero_grad()
            
            mel_real_pred_fakeness = self.discriminatorS(gt_spec) # [B] predict fakeness of real spectrogram
            wav_real_pred_fakeness = self.discriminatorW(gt_audio)# [B] predict fakeness of real audio
            real_pred_fakeness = (mel_real_pred_fakeness+wav_real_pred_fakeness).sigmoid()# Average and range between 0.0 and 1.0
            real_d_loss = nn.BCELoss()(real_pred_fakeness, real_labels)# [B] -> [] loss to decrease distriminated fakeness of real samples
            
            mel_fake_pred_fakeness = self.discriminatorS(pred_spect.detach())# [B] predict fakeness of generated spectrogram
            wav_fake_pred_fakeness = self.discriminatorW(pred_audio.detach())# [B] predict fakeness of generated audio
            fake_pred_fakeness = (mel_fake_pred_fakeness+wav_fake_pred_fakeness).sigmoid()# Average and range between 0.0 and 1.0
            fake_d_loss = nn.BCELoss()(fake_pred_fakeness, fake_labels)# [B] -> [] loss to increase distriminated fakeness of fake samples
            
            d_loss = (real_d_loss + fake_d_loss) / 2
            metrics['d_train_loss'] = reduce_tensor(d_loss.data, num_gpus).item() if num_gpus > 1 else d_loss.item()
            
            if amp is not None:
                with amp.scale_loss(d_loss, optimizer_d) as scaled_d_loss:
                    scaled_d_loss.backward()
            else:
                d_loss.backward()
            
            if use_grad_clip:
                if amp is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer_d), grad_clip_thresh)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.parameters(), grad_clip_thresh)
                if type(grad_norm) == torch.Tensor:
                    grad_norm = grad_norm.item()
                metrics['is_overflow'] = math.isinf(grad_norm) or math.isnan(grad_norm)
                if not metrics['is_overflow']:
                    metrics['grad_norm'] = grad_norm
            else: metrics['is_overflow'] = False; grad_norm=1e-6
            
            optimizer_d.step()
        else:
            loss = F.l1_loss(pred_spect, gt_spec)
            loss += F.l1_loss(pred_audio, gt_audio)
            
            metrics['train_loss'] = reduce_tensor(loss.data, num_gpus).item() if num_gpus > 1 else loss.item()
            
            if amp is not None:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if use_grad_clip:
                if amp is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), grad_clip_thresh)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_thresh)
                if type(grad_norm) == torch.Tensor:
                    grad_norm = grad_norm.item()
                metrics['is_overflow'] = math.isinf(grad_norm) or math.isnan(grad_norm)
                if not metrics['is_overflow']:
                    metrics['grad_norm'] = grad_norm
            else: metrics['is_overflow'] = False; grad_norm=0.00001
            
            optimizer.step()
        return metrics


class HiFiGAN(nn.Module):
    def __init__(self, WN_config, postnet_config, DW_config=None, DS_config=None, sampling_rate=48000, stage=0):
        super(HiFiGAN, self).__init__()
        self.sampling_rate = sampling_rate
        self.stage = stage
        
        from modules import WN
        in_channels = 1
        out_channels = max(WN_config['n_channels'], postnet_config['n_channels'])
        self.WN = WN(in_channels, out_channels, **WN_config)
        self.WN_end = nn.Conv1d(out_channels, 1, 1)# crush last dim if stage == 0
        
        if self.stage >= 1:
            in_channels = out_channels
            from modules import PostNet
            self.postnet = PostNet(in_channels, out_channels, **postnet_config)
            self.postnet_end = nn.Conv1d(out_channels, 1, 1)# crush last dim if stage == 1
    
    def forward(self, audio): # optional cond input
        """
        audio: [B, T]
        """
        B, T = audio.shape
        audio = audio.unsqueeze(1) # [B, 1, T]
        
        audio = self.WN(audio) # [B, 1, T] -> [B, C, T]
        
        if self.stage == 0:
            return self.WN_end(audio).squeeze(1) # [B, C, T] -> [B, T]
        
        elif self.stage >= 1:
            audio = self.postnet(audio) # [B, C, T] -> [B, C, T]
            return self.postnet_end(audio).squeeze(1) # [B, C, T] -> [B, T]
    
    def remove_weightnorm(self):
        recursive_remove_weightnorm(self)
    
    def apply_weightnorm(self):
        recursive_apply_weightnorm(self)


def recursive_remove_weightnorm(model, name='weight'):
    for module in model.children():
        if hasattr(module, f'{name}_g') or hasattr(module, f'{name}_v'):
            torch.nn.utils.remove_weight_norm(module, name=name) # inplace remove weight_norm
        recursive_remove_weightnorm(module, name=name)


def recursive_apply_weightnorm(model, name='weight'):
    for module in model.children():
        if hasattr(module, f'{name}') and not (hasattr(module, f'{name}_g') or hasattr(module, f'{name}_v')):
            torch.nn.utils.weight_norm(module, name=name) # inplace remove weight_norm
        recursive_apply_weightnorm(module, name=name)