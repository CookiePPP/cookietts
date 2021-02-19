import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from CookieTTS.utils.model.utils import get_mask_from_lengths


class Glow(nn.Module):
    def __init__(self, WN_cond_channels, n_flows, n_group, n_in_channels, n_early_every, n_early_size, memory_efficient,
    n_cond_layers, cond_hidden_channels, cond_output_channels, cond_kernel_size, cond_residual, cond_padding_mode,
    WN_config, cond_res_rezero=False, cond_activation_func=nn.LeakyReLU(0.1, inplace=True),
    autoregressive=True, shift_spect=0., scale_spect=1., z_channel_multiplier=1):
        super(Glow, self).__init__()
        assert(n_group % 2 == 0)
        if autoregressive and (not n_flows % 4 == 0):
            print("Warning! n_flows must be a multiple of 4 to align latent space to inputs!")
        
        self.channel_mixing = '1x1conv' if not autoregressive else 'permuteheight'#'1x1conv' if channel_mixing.lower() in "1x1convinvertibleconv1x1invconv" else ('permuteheight' if channel_mixing.lower() in "waveflowpermuteheightpermutechannelpermute" else None)
        self.mix_first = not autoregressive
        self.z_channel_multiplier = z_channel_multiplier
        self.n_in_channels = n_in_channels*z_channel_multiplier
        
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size  = n_early_size
        
        self.shift_spect = shift_spect
        self.scale_spect = scale_spect
        
        self.nan_asserts = False # check z Tensor is finite
        self.ignore_nan  =  True # (inference) replace NaNs in z tensor with 0.0
        
        #####################
        ###  Cond Layers  ###
        #####################
        self.cond_residual = cond_residual
        if self.cond_residual is True or self.cond_residual == 1: # override conditional output size if using residuals
            cond_output_channels = WN_cond_channels
        
        self.cond_res_rezero = cond_res_rezero
        if self.cond_res_rezero:
            self.res_weight = nn.Parameter(torch.ones(1)*0.01)# rezero initial state (0.01)
        
        self.cond_activation_func = cond_activation_func
        self.cond_layers = nn.ModuleList()
        if n_cond_layers:
            if self.cond_residual == '1x1conv':
                self.res_conv = nn.Conv1d(WN_cond_channels, cond_output_channels, 1)
            cond_pad = (cond_kernel_size-1)//2
            dimensions = [WN_cond_channels,]+[cond_hidden_channels]*(n_cond_layers-1)+[cond_output_channels,]
            for i, (indim, outim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=cond_pad, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                self.cond_layers.append(weight_norm(cond_layer))
            WN_cond_channels = cond_output_channels
        
        ########################
        ###  Channel Mixing  ###
        ########################
        if self.channel_mixing == '1x1conv':
            raise NotImplementedError
            from CookieTTS._4_mtw.waveglow.efficient_modules import InvertibleConv1x1
            self.convinv = nn.ModuleList()
        elif self.channel_mixing == 'permuteheight':
            from CookieTTS.utils.model.arglow_modules import PermuteHeight
            self.convinv = list()
        
        # + Coupling Blocks
        if autoregressive:
            from CookieTTS.utils.model.arglow_modules import WaveFlowCoupling as AffineCouplingBlock
            from CookieTTS.utils.model.arglow_modules import WN_2d as WN
        else:
            raise NotImplementedError
            from CookieTTS._4_mtw.waveglow.efficient_modules import AffineCouplingBlock
            from CookieTTS._4_mtw.waveglow.glow_ax import WN
        
        ###############
        ###  Flows  ###
        ###############
        self.WN = nn.ModuleList()
        n_remaining_height = n_group
        self.z_split_sizes = []
        for i in range(n_flows):
            if i % self.n_early_every == 0 and i > 0:
                n_remaining_height -= n_early_size
                self.z_split_sizes.append(n_early_size)
                assert n_remaining_height > 0, "n_remaining_height is 0. (increase n_group or decrease n_early_every/n_early_size)"
            
            mem_eff_layer = bool(memory_efficient and (i+1)/n_flows <= memory_efficient)
            
            if self.channel_mixing == '1x1conv':
                self.convinv.append( InvertibleConv1x1(n_remaining_height, memory_efficient=mem_eff_layer) )
            elif self.channel_mixing == 'permuteheight':
                self.convinv.append( PermuteHeight(n_remaining_height, i, n_flows, sigma=1.0) ) # sigma is hardcoded here.
            
            self.WN.append( AffineCouplingBlock(WN, memory_efficient=mem_eff_layer, n_in_channels=n_in_channels*z_channel_multiplier, n_in_height=n_remaining_height,
                                cond_in_channels=WN_cond_channels, **WN_config) )
        self.z_split_sizes.append(n_remaining_height)
    
    def cond_forward(self, cond):
        cond_res = cond
        for layer in self.cond_layers:
            cond = layer(cond)
            if hasattr(self, 'cond_activation_func'):
                cond = self.cond_activation_func(cond)
        
        if self.cond_residual:
            if hasattr(self, 'res_conv'):
                cond_res = self.res_conv(cond_res)
            if hasattr(self, 'res_weight'):
                cond *= self.res_weight
            cond = cond_res+cond # modify/adjust the original input with the cond
        return cond
    
    def squeeze(self, *args, pad_val=0.0):
        out = []
        for i, arg in enumerate(args):
            B, d, mel_T = arg.shape
            if mel_T%self.n_group != 0:
                arg = F.pad(arg, (0, self.n_group-(mel_T%self.n_group)), value=pad_val)
            arg = arg.reshape(B, d, -1, self.n_group).transpose(2, 3)# -> [B, d, H, mel_T/H]
            out.append(arg)
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)
    
    def unsqueeze(self, *args, max_len=None):
        out = []
        for i, arg in enumerate(args):
            B, d, n_group, mel_T = arg.shape# [B, d, H, mel_T/H]
            arg = arg.transpose(2, 3).reshape(B, d, -1)# -> [B, d, mel_T]
            if max_len is not None:
                arg = arg[..., :max_len]
            out.append(arg)
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)
    
    def forward(self, cond, z, z_lengths=None, z_mask=None): # optional cond input
        """
        forward_input[0] = cond: [B, cond_dim, mel_T]
        forward_input[1] = z   : [B,    n_mel, mel_T]
        """
        orig_mel_T = z.shape[-1]
        cond = self.cond_forward(cond)
        
        if z_mask is None and z_lengths is not None:
            z_mask = get_mask_from_lengths(z_lengths).unsqueeze(1)# -> [B, 1, mel_T]
        
        z = (z+self.shift_spect)*self.scale_spect
        if z_mask is not None:
            z *= z_mask
        
        if z_mask is not None:
            z_mask = self.squeeze(z_mask, pad_val=0) # [B, n_mel, mel_T] -> [B, n_mel, H, mel_T/H]
        z, cond = self.squeeze(z, cond, pad_val=0.00)# [B, n_mel, mel_T] -> [B, n_mel, H, mel_T/H]
        
        if self.z_channel_multiplier > 1:
            z = z.repeat(1, self.z_channel_multiplier, 1, 1)# [B, n_mel, H, mel_T/H] -> [B, n_mel*z_mul, H, mel_T/H]
        
        z_out        = []
        logdet_W_out = []
        log_s_out    = []
        split_sections = [self.n_early_size, self.n_group]
        for k, (convinv, affine_coup) in enumerate(zip(self.convinv, self.WN)):
            if k % self.n_early_every == 0 and k > 0:
                split_sections[1] -= self.n_early_size
                early_output, z = z.split(split_sections, dim=2)
                z_out.append(early_output)
                z = z.clone()
            
            if self.mix_first:
                z, cond, log_det_W = convinv(z, cond)
                if self.nan_asserts:
                    assert not torch.isnan(z).any(), f'Flow {k} NaN Exception'
                    assert not torch.isinf(z).any(), f'Flow {k} inf Exception'
            
            z, log_s = affine_coup(z, cond, z_mask)
            
            if self.nan_asserts:
                assert not torch.isnan(z).any(), f'Flow {k} NaN Exception'
                assert not torch.isinf(z).any(), f'Flow {k} inf Exception'
            
            if not self.mix_first:
                z, cond, log_det_W = convinv(z, cond)
                if self.nan_asserts:
                    assert not torch.isnan(z).any(), f'Flow {k} NaN Exception'
                    assert not torch.isinf(z).any(), f'Flow {k} inf Exception'
            
            logdet_W_out.append(log_det_W)
            log_s_out   .append(log_s)
        
        assert split_sections[1] == self.z_split_sizes[-1]
        z_out.append(z)
        z_out = self.unsqueeze(torch.cat(z_out, 2), max_len=orig_mel_T)
        return z_out, logdet_W_out, log_s_out
    
    def inverse(self, z, cond, z_lengths=None, z_mask=None):
        orig_mel_T = z.shape[-1]
        cond = self.cond_forward(cond)
        
        if z_mask is None and z_lengths is not None:
            z_mask = get_mask_from_lengths(z_lengths).unsqueeze(1)# -> [B, 1, mel_T]
        
        if z_mask is not None:
            z_mask = self.squeeze(z_mask, pad_val=0)# [B, n_mel, mel_T] -> [B, n_mel, H, mel_T/H]
        z, cond = self.squeeze(z, cond, pad_val=0.00)# [B, n_mel, mel_T] -> [B, n_mel, H, mel_T/H]
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 2):
            remained_z.append(r.clone())
        *remained_z, z = remained_z
        
        for k, invconv, affine_coup in zip(range(self.n_flows-1, -1, -1), self.convinv[::-1], self.WN[::-1]):
            if not self.mix_first:
                z, cond, log_det_W = invconv.inverse(z, cond)
            
            z, log_s = affine_coup.inverse(z, cond, z_mask)
            
            if self.mix_first:
                z, cond, log_det_W = invconv.inverse(z, cond)
            
            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 2)
        
        if self.z_channel_multiplier > 1:
            z = sum(z.chunk(self.z_channel_multiplier, dim=1))/self.z_channel_multiplier
        
        z = self.unsqueeze(z, max_len=orig_mel_T)
        z = (z/self.scale_spect)-self.shift_spect
        return z
    
    @torch.no_grad()
    def infer(self, cond, z_mu=None, z_logvar=None, artifact_trimming=1, sigma=1.,):
        input_device, input_dtype = cond.device, cond.dtype
        self_device, self_dtype = next(self.parameters()).device, next(self.parameters()).dtype
        cond = cond.to(self_device, self_dtype)# move to GPU and correct data type
        
        if len(cond.shape) == 2:
            cond = cond[None, ...] # [cond_C, T] -> [B, cond_C, T]
        if artifact_trimming > 0:
            cond = F.pad(cond, (0, artifact_trimming), value=0.0)
        
        B, cond_C, cond_T = cond.shape # [B, cond_C, T]
        
        z = cond.new_zeros((B, cond_T)) # [B, T]
        if sigma > 0:
            z.normal_(std=sigma)
        if z_logvar is not None:
            z*=(0.5*z_logvar)
        if z_mu     is not None:
            z+=z_mu
        z, _ = self.inverse(z, cond)
        if artifact_trimming > 0:
            audio_trim = artifact_trimming*self.hop_length # amount of z to trim
            z = z[:, :-audio_trim]
        
        z = z.to(input_dtype)# output with same data type as input
        return z
    
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
