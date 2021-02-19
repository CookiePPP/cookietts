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



def permute_channels(x, height=None, reverse=False, bipart=False, shift=False, inverse_shift=False):
    if height is None:
        height = x.shape[-2]# [..., H, W] -> H
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
    return x[..., h_permute, :]


class PermuteHeight():
    """
    The layer outputs the permuted channel dim, and a placeholder log determinant.
    class.inverse() performs the permutation in reverse.
    """
    def __init__(self, n_remaining_channels, k, n_flows, sigma):
        assert n_flows%2==0, "PermuteHeight requires even n_flows"
        self.sigma = sigma
        self.const = float(-(0.5 * np.log(2 * np.pi) + np.log(self.sigma))/n_flows)
        
        # b) we reverse Z(7), Z(6), Z(5), Z(4) over the height dimension as before, but bipartition Z(3), Z(2), Z(1), Z(0) in the middle of the height dimension then reverse each part respectively.
        if k%4 in (2,3): # Flows (2,3, 6,7, 10,11)
            self.bipart_reverse = True
            self.reverse = True
        else:            # Flows (0,1, 4,5, 8,9)
            self.bipart_reverse = False
            self.reverse = True
    
    def __call__(self, z, cond):
        height = z.shape[-2]
        z    = permute_channels(   z, reverse=self.reverse, bipart=self.bipart_reverse)
        cond = permute_channels(cond, reverse=self.reverse, bipart=self.bipart_reverse)
        log_det_W = torch.tensor(self.const, device=z.device)
        return z, cond, log_det_W
    
    def inverse(self, z, cond):
        z    = permute_channels(   z, reverse=self.reverse, bipart=self.bipart_reverse)
        cond = permute_channels(cond, reverse=self.reverse, bipart=self.bipart_reverse)
        log_det_W = None
        return z, cond, log_det_W


class WaveFlowCoupling(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()
        self.memory_efficient = memory_efficient
        self.WN = transform_type(**kwargs)
    
    def forward(self, z, cond, z_mask=None):
        z_previous  =      z[:, :,:-1 ]# last value not needed (as there is nothing left to output, the entire sequence will be generated without the last input)
        cond        =   cond[:, :,  1:]
        z_mask      = z_mask[:, :,  1:]
        
        args = [z_previous, cond]
        if z_mask is not None: args.append(z_mask)
        if self.memory_efficient and self.training:
            _ = checkpoint_grads(self.WN.__call__, *args)
        else:
            _ = self.WN(*args)
        log_s, t = _.chunk(2, dim=1)# [B, 2*C, H, mel_T//H] -> [B, C, H, mel_T//H], [B, C, H, mel_T//H]
        
        audio_out = torch.cat((z[:,:,:1], z[:,:,1:].mul(log_s.exp()).add(t)), dim=2)# ignore changes to first sample since that conv has only padded information (so no gradients to flow anyway)
        return audio_out, log_s
    
    def inverse(self, zx, cond, z_mask=None):
        if hasattr(self, 'efficient_inverse'):
            print("gradient checkpointing not implemented for WaveFlowCoupling module inverse"); raise NotImplementedError
            #z, log_s = self.efficient_inverse(zx, cond, self.WN, *self.param_list)
            #zx.storage().resize_(0)
            return z, log_s
        else:
            z = [  zx[:,:,0:1],] # z is used as output tensor # initial sample # [B, 1, T//n_group]
            
            input_queues = [None,]*self.WN.n_layers # create blank audio queue to flag for conv-queue
            
            n_group = zx.shape[2]
            for i in range(n_group-1):# [0,1,2...22]
                curr_audio  = z[-1]# just generated sample [B, C, 1, T//n_group]
                next_audio  =     zx[:, :, i+1:i+2]# [B, C, n_group, T//n_group] -> [B, C, 1, T//n_group]
                next_cond   =   cond[:, :, i+1:i+2]# [B, C, n_group, T//n_group] -> [B, C, 1, T//n_group]
                next_z_mask = z_mask[:, :, i+1:i+2] if z_mask is not None else z_mask
                
                acts, input_queues = self.WN(curr_audio, next_cond, next_z_mask, input_queues=input_queues) # get next sample
                
                log_s, t = acts[:,:,-1:].chunk(2, dim=1) # [B, 2*C, 1, T//n_group] -> [B, C, 1, 1], [B, C, 1, 1]
                z.append( (next_audio-t).div_(log_s.exp()) ) # [B, 1, 1] save predicted next sample
            z = torch.cat(z, dim=2)# [B, n_group, T//n_group]
            return z, -log_s


class WN_2d(nn.Module):
    """ This is the WaveNet like layer for the affine coupling. The primary difference from WaveNet is the convolutions are causal on the height dimension and non-causal on the width dim. There is also no dilation size reset. The dilation only doubles on each layer. """
    def __init__(self, n_in_channels, n_in_height, cond_in_channels, n_cond_layers, cond_hidden_channels, cond_kernel_size, cond_padding_mode, seperable_conv, merge_res_skip, n_layers, n_channels, # audio_channels, mel_channels*n_group, n_layers, n_conv_channels
                 kernel_size_w, kernel_size_h, cond_activation_func=nn.LeakyReLU(0.1), n_layers_dilations_w=None, n_layers_dilations_h=1, res_skip=True, upsample_first=None, cond_out_activation_func=True, gated_unit='GTU', dilation_rate=2, use_weight_norm=False):
        super(WN_2d, self).__init__()
        assert(kernel_size_w % 2 == 1)
        assert(n_channels    % 2 == 0)
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
        
        start = nn.Conv2d(n_in_channels, n_channels, (1,1))
        self.start = weight_norm(start)
        
        self.cond_out_activation_func = cond_out_activation_func
        if n_cond_layers:
            cond_output_channels = 2*n_channels*n_layers
            dimensions = [cond_in_channels,]+[cond_hidden_channels]*(n_cond_layers-1)+[cond_output_channels,]
            
            self.cond_layers = nn.ModuleList()
            for indim, outim in zip(dimensions[:-1], dimensions[1:]):
                cond_layer = nn.Conv2d(indim, outim, cond_kernel_size, padding=(cond_kernel_size-1)//2, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                self.cond_layers.append(weight_norm(cond_layer))
        
        if type(n_layers_dilations_w) == int:
            n_layers_dilations_w = [n_layers_dilations_w,]*n_layers # constant dilation if using int
            print("WARNING: Using constant dilation factor for WN in_layer dilation width. This is not normally recommended.")
        
        self.in_layers       = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        self.h_dilate = n_layers_dilations_h
        self.padding_h = []
        for i in range(n_layers):
            dilation_h = n_layers_dilations_h if (type(n_layers_dilations_h) == int) else n_layers_dilations_h[i]
            self.padding_h.append((kernel_size_h-1)*dilation_h) # causal padding https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
            
            dilation_w = dilation_rate**i if n_layers_dilations_w is None else n_layers_dilations_w[i]
            padding_w  = ((kernel_size_w-1)*dilation_w)//2
            
            if (not seperable_conv) or (kernel_size_w == 1 and kernel_size_h == 1):
                in_layer = nn.Conv2d(n_channels, 2*n_channels, (kernel_size_h,kernel_size_w),
                                           dilation=(dilation_h,dilation_w), padding=(0,padding_w), padding_mode='zeros')
            else:
                depthwise = nn.Conv2d(n_channels, n_channels, (kernel_size_h,kernel_size_w),
                                    dilation=(dilation_h,dilation_w), padding=(0,padding_w), padding_mode='zeros', groups=n_channels)
                pointwise = nn.Conv2d(n_channels, 2*n_channels, (1,1), dilation=(1,1), padding=(0,0))
                in_layer = torch.nn.Sequential(weight_norm(depthwise), weight_norm(pointwise))
            self.in_layers.append(weight_norm(in_layer))
            
            # output dim of res_skip layer. Last layer does not require residual output so that layer will output 1x n_channels
            res_skip_channels = n_channels*2 if (i < n_layers-1 and not self.merge_res_skip) else n_channels
            
            if res_skip:
                res_skip_layer = nn.Conv2d(n_channels, res_skip_channels, (1,1))
                res_skip_layer = weight_norm(res_skip_layer)
                self.res_skip_layers.append(res_skip_layer)
        
        end = nn.Conv2d(n_channels, 2*n_in_channels, (1,1))
        end.weight.data.zero_(); end.bias.data.zero_()# Initializing last layer to 0 makes the affine coupling layers do nothing at first. This helps with training stability.
        self.end = end
    
    def forward(self, input, cond, z_mask=None, input_queues=None, cond_queues=None):
        input = self.start(input) # [B, C, n_group, T//n_group] -> [B, n_channels, n_group, T//n_group]
        if z_mask is not None:
            input*=z_mask
        
        output = input if self.merge_res_skip else torch.zeros_like(input)
        
        is_missing_cond = cond_queues is None or any([x is None for x in cond_queues])
        if is_missing_cond and hasattr(self, 'cond_layers') and self.cond_layers:
            for i, layer in enumerate(self.cond_layers): # [B, cond_channels, n_group, T//n_group] -> ... -> [B, n_channels*n_layers, n_group, T//n_group]
                cond = layer(cond)
                if hasattr(self, 'cond_activation_func') and (self.cond_out_activation_func or (i+1 != len(self.cond_layers))):
                    cond = self.cond_activation_func(cond)
            #if z_mask is not None:
            #    cond*=z_mask# [B, 1, n_group, T//n_group]
        
        for i in range(self.n_layers):
            if is_missing_cond: # if cond layer has been ran.
                layer_cond = cond[:, i*2*self.n_channels:(i+1)*2*self.n_channels] # [B, 2*n_channels*n_layers, n_group, T//n_group] -> [B, 2*n_channels, n_group, T//n_group]
                if cond_queues is not None and cond_queues[i] is None: # is cond_queues exists but this index is empty...
                    cond_queues[i] = layer_cond # save layer_cond into this index.
            else:                         # else...
                layer_cond = cond_queues[i] # get layer_cond from this index.
            
            if input_queues is None:# if training/validation...
                input_cpad = F.pad(input, (0,0,self.padding_h[i],0)) # apply causal height padding (left, right, top, bottom)
            else: # else, if conv-queue and inference/autoregressive sampling.
                if input_queues[i] is None: # if first sample in autoregressive sequence, pad start with zeros
                    B, n_channels, n_group, T_group = input.shape
                    input_queues[i] = input.new_zeros( size=[B, n_channels, self.padding_h[i], T_group] )
                
                # [B, n_channels, n_group, T//n_group]
                input_queues[i] = input_cpad = torch.cat((input_queues[i], input), dim=2)[:,:,-(self.padding_h[i]+1):] # pop old samples and append new sample to end of n_group dim
            
            acts = self.in_layers[i](input_cpad) # [B, n_channels, n_group, T//n_group] -> [B, 2*n_channels, pad+n_group, T//n_group]
            acts = self.gated_unit(
                acts,      # [B, 2*n_channels, n_group, T//n_group]
                layer_cond,# [B, 2*n_channels, n_group, T//n_group]
                self.n_channels)
            # acts.shape <- [B, n_channels, n_group, T//n_group]
            
            res_skip_acts = self.res_skip_layers[i](acts) if ( hasattr(self, 'res_skip_layers') and len(self.res_skip_layers) ) else acts# -> [B, n_channels, n_group//2, T//n_group]
            
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
        
        func_out = self.end(output) # [B, n_channels, n_group, T//n_group] -> [B, 2*C, n_group, T//n_group]
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