import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from CookieTTS.utils.model.utils import get_mask_from_lengths

def permute_time(x, height=None, reverse=False, bipart=False, shift=False, inverse_shift=False):
    if height is None:
        height = x.shape[-1]# [..., W] -> W
    h_permute = list(range(height))
    if bipart and reverse:
        half = len(h_permute)//2
        h_permute = h_permute[:half][::-1] + h_permute[half:][::-1] # reverse H halfs [0,1,2,3,4,5,6,7] -> [3,2,1,0] + [7,6,5,4] -> [3,2,1,0,7,6,5,4]
    elif reverse:
        h_permute = h_permute[::-1] # reverse entire H [0,1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1,0]
    if shift:
        h_permute = [h_permute[-1],] + h_permute[:-1] # shift last H into first position [0,1,2,3,4,5,6,7] -> [7,0,1,2,3,4,5,6]
    elif inverse_shift:
        h_permute = h_permute[1:] + [h_permute[0],]   # shift first H into last position [0,1,2,3,4,5,6,7] -> [1,2,3,4,5,6,7,0]
    return x[..., h_permute]


class PermuteHeight():
    """
    The layer outputs the permuted channel dim, and a placeholder log determinant.
    class.inverse() performs the permutation in reverse.
    """
    def __init__(self, n_remaining_channels, k, n_flows, sigma):
        assert n_flows%2==0, "PermuteHeight requires even n_flows"
        self.sigma = sigma
        self.const = float(-(0.5 * np.log(2 * np.pi) + np.log(self.sigma))) if k==0 else 0.0
        
        self.reverse = False
    
    def __call__(self, z, cond, z_mask):
        z      = permute_time(     z, reverse=self.reverse)
        cond   = permute_time(  cond, reverse=self.reverse)
        z_mask = permute_time(z_mask, reverse=self.reverse)
        log_det_W = torch.tensor(self.const, device=z.device)
        return z, cond, z_mask, log_det_W
    
    def inverse(self, z, cond, z_mask):
        z      = permute_time(     z, reverse=self.reverse)
        cond   = permute_time(  cond, reverse=self.reverse)
        z_mask = permute_time(z_mask, reverse=self.reverse)
        log_det_W = None
        return z, cond, z_mask, log_det_W


class ARFlow(nn.Module):
    def __init__(self, WN_cond_channels, n_flows, n_in_channels, memory_efficient,
    n_cond_layers, cond_hidden_channels, cond_output_channels, cond_kernel_size, cond_residual, cond_padding_mode,
    WN_config, cond_res_rezero=False, cond_activation_func=nn.LeakyReLU(0.1, inplace=True),
    mix_first=False, shift_spect=0., scale_spect=1.,):
        super(ARFlow, self).__init__()
        assert n_flows % 2 == 0, "Warning! n_flows must be a multiple of 2 to align latent space to inputs!"
        
        self.mix_first = mix_first
        self.n_in_channels = n_in_channels
        
        self.n_flows = n_flows
        
        self.shift_spect = shift_spect
        self.scale_spect = scale_spect
        assert self.scale_spect != 0.0
        
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
        self.convinv = list()
        
        ###############
        ###  Flows  ###
        ###############
        self.WN = nn.ModuleList()
        for i in range(n_flows):
            mem_eff_layer = bool(memory_efficient and (i+1)/n_flows <= memory_efficient)
            
            self.convinv.append( PermuteHeight(0, i, n_flows, sigma=1.0) ) # sigma is hardcoded here.
            
            self.WN.append( AffineCouplingBlock(WN, memory_efficient=mem_eff_layer, n_in_channels=n_in_channels,
                                cond_in_channels=WN_cond_channels, **WN_config) )
    
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
    
    def forward(self, z, cond, z_lengths=None, z_mask=None):# [B, n_mel, mel_T], [B, C, mel_T], [B], [B, 1, mel_T]
        """
        forward_input[0] = z   : [B,    n_mel, mel_T]
        forward_input[1] = cond: [B, cond_dim, mel_T]
        """
        orig_mel_T = z.shape[-1]
        
        if z_mask is None and z_lengths is not None:
            z_mask = get_mask_from_lengths(z_lengths).unsqueeze(1)# -> [B, 1, mel_T]
        
        cond = self.cond_forward(cond)
        if z_mask is not None:
            cond *= z_mask
        
        z = (z+self.shift_spect)*self.scale_spect
        if z_mask is not None:
            z *= z_mask
        
        z_out        = []
        logdet_W_out = []
        log_s_out    = []
        for k, (convinv, affine_coup) in enumerate(zip(self.convinv, self.WN)):
            if self.mix_first:
                z, cond, z_mask, log_det_W = convinv(z, cond, z_mask)
                if self.nan_asserts:
                    assert not torch.isnan(z).any(), f'Flow {k} NaN Exception'
                    assert not torch.isinf(z).any(), f'Flow {k} inf Exception'
            
            assert z.pow(2).sum(dtype=torch.float) > 1e-8, f'Flow {k} Before Coupling, Z is empty'
            z, log_s = affine_coup(z, cond, z_mask)
            assert z.pow(2).sum(dtype=torch.float) > 1e-8, f'Flow {k} After Coupling, Z is empty'
            if self.nan_asserts:
                assert not torch.isnan(z).any(), f'Flow {k} NaN Exception'
                assert not torch.isinf(z).any(), f'Flow {k} inf Exception'
            
            if not self.mix_first:
                z, cond, z_mask, log_det_W = convinv(z, cond, z_mask)
                if self.nan_asserts:
                    assert not torch.isnan(z).any(), f'Flow {k} NaN Exception'
                    assert not torch.isinf(z).any(), f'Flow {k} inf Exception'
            
            logdet_W_out.append(log_det_W)
            log_s_out   .append(log_s)
        
        return z, logdet_W_out, log_s_out
    
    def inverse(self, z, cond, z_lengths=None, z_mask=None):# [B, n_mel, mel_T], [B, C, mel_T], [B], [B, 1, mel_T]
        
        if z_mask is None and z_lengths is not None:
            z_mask = get_mask_from_lengths(z_lengths).unsqueeze(1)# -> [B, 1, mel_T]
        
        cond = self.cond_forward(cond)
        if z_mask is not None:
            cond *= z_mask
        
        for k, invconv, affine_coup in zip(range(self.n_flows-1, -1, -1), self.convinv[::-1], self.WN[::-1]):
            if not self.mix_first:
                z, cond, z_mask, log_det_W = invconv.inverse(z, cond, z_mask)
            
            z, log_s = affine_coup.inverse(z, cond, z_mask)
            
            if self.mix_first:
                z, cond, z_mask, log_det_W = invconv.inverse(z, cond, z_mask)
        
        z = (z/self.scale_spect)-self.shift_spect
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



import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_grads
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
import numpy as np

from functools import reduce
from operator import mul

from CookieTTS._2_ttm.MelFlow.arglow_units import get_gate_func#, GTU, GTRU, GTLRU, GLU, TTU, STU, GTSU, SPTU, GSIU, GSIRU, GTSRU, GSIRRU, GSIRLRU, GSIRRLRU

class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()
        self.memory_efficient = memory_efficient
        self.WN = transform_type(**kwargs)
    
    def forward(self, z, cond, z_mask=None):# [B, n_mel, mel_T], [B, C, mel_T], [B, 1, mel_T]
        z_previous  =      z[:, :,:-1 ]# last value not needed (as there is nothing left to output, the entire sequence will be generated without the last input)
        cond        =   cond[:, :,  1:]
        
        args = [z_previous, cond]
        if z_mask is not None: args.append(z_mask[:, :,  1:])
        if self.memory_efficient and self.training:
            acts = checkpoint_grads(self.WN.__call__, *args)
        else:
            acts = self.WN(*args)
        log_s, t = acts.chunk(2, dim=1)# [B, 2*C, mel_T] -> [B, C, mel_T], [B, C, mel_T]
        
        z_out = torch.cat((z[:,:,:1], z[:,:,1:].mul(log_s.exp()).add(t)), dim=2)# ignore changes to first sample since that conv has only padded information (so no gradients to flow anyway)
        if z_mask is not None: z_out *= z_mask
        return z_out, log_s
    
    def inverse(self, zx, cond, z_mask=None):# [B, n_mel, mel_T], [B, C, mel_T], [B, 1, mel_T]
        if hasattr(self, 'efficient_inverse'):
            print("gradient checkpointing not implemented for WaveFlowCoupling module inverse"); raise NotImplementedError
            #z, log_s = self.efficient_inverse(zx, cond, self.WN, *self.param_list)
            #zx.storage().resize_(0)
            return z, log_s
        else:
            z = [  zx[:,:,0:1],] # z is used as output tensor # initial sample # [B, n_mel, 1]
            
            input_queues = [None,]*self.WN.n_layers # create blank audio queue to flag for conv-queue
            
            mel_T = zx.shape[2]
            for i in range(mel_T-1):# [0,1,2...mel_T-1]
                curr_z  =  z[-1]# [B, n_mel, 1]
                next_z  = zx[:, :, i+1:i+2]# [B, n_mel, mel_T] -> [B, n_mel, 1]
                next_cond   =   cond[:, :, i+1:i+2]# [B, C, mel_T] -> [B, C, 1]
                next_z_mask = z_mask[:, :, i+1:i+2] if z_mask is not None else z_mask
                
                acts, input_queues = self.WN(curr_z, next_cond, next_z_mask, input_queues=input_queues) # get next sample
                
                log_s, t = acts[:,:,-1:].chunk(2, dim=1) # [B, 2*C, 1] -> [B, C, 1], [B, C, 1]
                z.append( (next_z-t).div_(log_s.exp()) ) # [B, 1, 1] save predicted next sample
            z = torch.cat(z, dim=2)# [B, n_mel, mel_T]
            if z_mask is not None: z *= z_mask
            return z, -log_s


class WN(nn.Module):
    """ This is the WaveNet like layer for the affine coupling. The primary difference from WaveNet is the convolutions are causal on the height dimension and non-causal on the width dim. There is also no dilation size reset. The dilation only doubles on each layer. """
    def __init__(self, n_in_channels, cond_in_channels, n_cond_layers, cond_hidden_channels, cond_kernel_size, cond_padding_mode, seperable_conv, merge_res_skip, n_layers, n_channels, # audio_channels, mel_channels*n_group, n_layers, n_conv_channels
                 kernel_size_w, cond_activation_func=nn.LeakyReLU(0.1), n_layers_dilations_w=None, n_layers_dilations_h=1, res_skip=True, upsample_first=None, cond_out_activation_func=True, gated_unit='GTU', dilation_rate=2, use_weight_norm=False):
        super(WN, self).__init__()
        assert(n_channels % 2 == 0)
        assert res_skip or merge_res_skip, "Cannot remove res_skip without using merge_res_skip"
        self.n_layers       = n_layers
        self.n_channels     = n_channels
        self.merge_res_skip = merge_res_skip
        self.gated_unit = get_gate_func(gated_unit)
        
        if use_weight_norm:
            from torch.nn.utils import weight_norm
        else:
            def weight_norm(args):
                return args
        
        start = nn.Conv1d(n_in_channels, n_channels, 1)
        self.start = weight_norm(start)
        
        self.cond_out_activation_func = cond_out_activation_func
        if n_cond_layers:
            cond_output_channels = 2*n_channels*n_layers
            dimensions = [cond_in_channels,]+[cond_hidden_channels]*(n_cond_layers-1)+[cond_output_channels,]
            
            self.cond_layers = nn.ModuleList()
            for indim, outim in zip(dimensions[:-1], dimensions[1:]):
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=(cond_kernel_size-1)//2, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                self.cond_layers.append(weight_norm(cond_layer))
        
        if type(n_layers_dilations_w) == int:
            n_layers_dilations_w = [n_layers_dilations_w,]*n_layers # constant dilation if using int
            print("WARNING: Using constant dilation factor for WN in_layer dilation width. This is not normally recommended.")
        
        self.in_layers       = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        self.h_dilate = n_layers_dilations_h
        self.padding_w = []
        for i in range(n_layers):
            dilation_w = dilation_rate**i if n_layers_dilations_w is None else n_layers_dilations_w[i]
            self.padding_w.append((kernel_size_w-1)*dilation_w) # causal padding https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
            
            if (not seperable_conv) or (kernel_size_w == 1 and kernel_size_h == 1):
                in_layer = nn.Conv1d(n_channels, 2*n_channels, kernel_size_w,
                                           dilation=dilation_w, padding=0, padding_mode='zeros')
            else:
                depthwise = nn.Conv1d(n_channels, n_channels, kernel_size_w,
                                    dilation=dilation_w, padding=0, padding_mode='zeros', groups=n_channels)
                pointwise = nn.Conv1d(n_channels, 2*n_channels, 1, dilation=1, padding=0)
                in_layer = torch.nn.Sequential(weight_norm(depthwise), weight_norm(pointwise))
            self.in_layers.append(weight_norm(in_layer))
            
            # output dim of res_skip layer. Last layer does not require residual output so that layer will output 1x n_channels
            res_skip_channels = n_channels*2 if (i < n_layers-1 and not self.merge_res_skip) else n_channels
            
            if res_skip:
                res_skip_layer = nn.Conv1d(n_channels, res_skip_channels, 1)
                res_skip_layer = weight_norm(res_skip_layer)
                self.res_skip_layers.append(res_skip_layer)
        
        end = nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_(); end.bias.data.zero_()# Initializing last layer to 0 makes the affine coupling layers do nothing at first. This helps with training stability.
        self.end = end
    
    def forward(self, input, cond, z_mask=None, input_queues=None, cond_queues=None):
        input = self.start(input) # [B, n_mel, mel_T] -> [B, n_channels, mel_T]
        if z_mask is not None:
            input*=z_mask
        
        output = input if self.merge_res_skip else torch.zeros_like(input)
        
        is_missing_cond = cond_queues is None or any([x is None for x in cond_queues])
        if is_missing_cond and hasattr(self, 'cond_layers') and self.cond_layers:
            if z_mask is not None:
                cond=cond*z_mask# [B, 1, mel_T]
            for i, layer in enumerate(self.cond_layers): # [B, cond_channels, mel_T] -> ... -> [B, n_channels*n_layers, mel_T]
                cond = layer(cond)
                if hasattr(self, 'cond_activation_func') and (self.cond_out_activation_func or (i+1 != len(self.cond_layers))):
                    cond = self.cond_activation_func(cond)
                if z_mask is not None:
                    cond*=z_mask# [B, 1, mel_T]
        
        for i in range(self.n_layers):
            if is_missing_cond: # if cond layer has been ran.
                layer_cond = cond[:, i*2*self.n_channels:(i+1)*2*self.n_channels] # [B, 2*n_channels*n_layers, mel_T] -> [B, 2*n_channels, mel_T]
                if cond_queues is not None and cond_queues[i] is None: # is cond_queues exists but this index is empty...
                    cond_queues[i] = layer_cond # save layer_cond into this index.
            else:                         # else...
                layer_cond = cond_queues[i] # get layer_cond from this index.
            
            if input_queues is None:# if training/validation...
                input_cpad = F.pad(input, (self.padding_w[i],0)) # apply causal seq padding -> [B, n_channels, pad+mel_T]
            else: # else, if conv-queue and inference/autoregressive sampling.
                if input_queues[i] is None: # if first sample in autoregressive sequence, pad start with zeros
                    B, n_channels, mel_T = input.shape
                    input_queues[i] = input.new_zeros( size=[B, n_channels, self.padding_w[i]] )
                
                # [B, n_channels, mel_T]
                input_queues[i] = input_cpad = torch.cat((input_queues[i], input), dim=2)[:,:,-(self.padding_w[i]+1):] # pop old samples and append new sample to end of mel_T dim
            
            acts = self.in_layers[i](input_cpad)# [B, n_channels, pad+mel_T] -> [B, 2*n_channels, mel_T]
            acts = self.gated_unit(
                acts,      # [B, 2*n_channels, mel_T]
                layer_cond,# [B, 2*n_channels, mel_T]
                self.n_channels)
            # acts.shape <- [B, n_channels, mel_T]
            
            res_skip_acts = self.res_skip_layers[i](acts) if ( hasattr(self, 'res_skip_layers') and len(self.res_skip_layers) ) else acts# -> [B, 2*n_channels, mel_T]
            
            if i == 0:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):
                    input += res_skip_acts[:,:self.n_channels ,:]
                    output = res_skip_acts[:, self.n_channels:,:]
                else:
                    output = res_skip_acts
            else:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):# if res_skip and not last layer
                    input  += res_skip_acts[:,:self.n_channels ,:]
                    output += res_skip_acts[:, self.n_channels:,:]
                else:
                    output += res_skip_acts
            if z_mask is not None:
                input*=z_mask
        
        func_out = self.end(output) # [B, n_channels, mel_T] -> [B, 2*n_mel, mel_T]
        if z_mask is not None:
            func_out*=z_mask
        
        if input_queues is not None:
            func_out = [func_out,]
            func_out.append(input_queues)
        if cond_queues is not None:
            func_out.append(cond_queues)
        return func_out


@torch.jit.script
def ignore_nan(input):
    """Replace NaN values with 0.0"""
    input.masked_fill_(torch.isnan(input), 0.)