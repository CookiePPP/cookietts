# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional

# "Gated Convolutional Neural Networks for Domain Adaptation"
#  https://arxiv.org/pdf/1905.06906.pdf
@torch.jit.script
def GTU(input_a, input_b: Optional[torch.Tensor], n_channels: int):
    """Gated Tanh Unit (GTU)"""
    if input_b is None:
        in_act = input_a
    else:
        in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GTRU(input_a, input_b, n_channels: int):# saves significant VRAM and runs faster, unstable for first 150K~ iters. (test a difference layer initialization?)
    """Gated[?] Tanh ReLU Unit (GTRU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.relu(in_act[:, n_channels:, :], inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GLU(input_a, input_b, n_channels: int):
    """Gated Linear Unit (GLU)"""
    in_act = input_a+input_b
    l_act = in_act[:, :n_channels, :]
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = l_act * s_act
    return acts

# Random units I wanted to try
@torch.jit.script
def TTU(input_a, input_b, n_channels: int):
    """Tanh Tanh Unit (TTU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    t_act2 = torch.tanh(in_act[:, n_channels:, :])
    acts = t_act * t_act2
    return acts

@torch.jit.script
def STU(input_a, input_b, n_channels: int):
    """SeLU Tanh Unit (STU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.nn.functional.selu(in_act[:, n_channels:, :], inplace=True)
    acts = t_act * s_act
    return acts

@torch.jit.script
def GTSU(input_a, input_b, n_channels: int):
    """Gated TanhShrink Unit (GTSU)"""
    in_act = input_a+input_b
    t_act = torch.nn.functional.tanhshrink(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def SPTU(input_a, input_b, n_channels: int):
    """Softplus Tanh Unit (SPTU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.nn.functional.softplus(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GSIU(input_a, input_b, n_channels: int):
    """Gated Sinusoidal Unit (GSIU)"""
    in_act = input_a+input_b
    t_act = torch.sin(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GSIRU(input_a, input_b, n_channels: int):
    """Gated SIREN Unit (GSIRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GTSRU(input_a, input_b, n_channels: int):
    """Gated[?] TanhShrink ReLU Unit (GTSRU)"""
    in_act = input_a+input_b
    t_act = torch.nn.functional.tanhshrink(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.relu(in_act[:, n_channels:, :], inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GSIRRU(input_a, input_b, n_channels: int): # best and fastest converging unit, uses a lot of VRAM.
    """Gated[?] SIREN ReLU Unit (GSIRRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.relu(in_act[:, n_channels:, :], inplace=False)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GSIRLRU(input_a, input_b, n_channels: int):
    """Gated[?] SIREN Leaky ReLU Unit (GSIRLRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.leaky_relu(in_act[:, n_channels:, :], negative_slope=0.01, inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GSIRRLRU(input_a, input_b, n_channels: int):
    """Gated[?] SIREN Randomized Leaky ReLU Unit (GSIRRLRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.rrelu(in_act[:, n_channels:, :], lower=0.01, upper=0.1, inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GTLRU(input_a, input_b, n_channels: int):
    """Gated[?] Tanh Leaky ReLU Unit (GTLRU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.leaky_relu(in_act[:, n_channels:, :], negative_slope=0.01, inplace=True)
    acts = t_act * r_act
    return acts


def get_gate_func(gated_unit_str):
    if gated_unit_str.upper() == 'GTU':
        return GTU
    elif gated_unit_str.upper() == 'GTRU':
        return GTRU
    elif gated_unit_str.upper() == 'GTLRU':
        return GTLRU
    elif gated_unit_str.upper() == 'GLU':
        return GLU
    elif gated_unit_str.upper() == 'TTU':
        return TTU
    elif gated_unit_str.upper() == 'STU':
        return STU
    elif gated_unit_str.upper() == 'GTSU':
        return GTSU
    elif gated_unit_str.upper() == 'SPTU':
        return SPTU
    elif gated_unit_str.upper() == 'GSIU':
        return GSIU
    elif gated_unit_str.upper() == 'GSIRU':
        return GSIRU
    elif gated_unit_str.upper() == 'GTSRU':
        return GTSRU
    elif gated_unit_str.upper() == 'GSIRRU':
        return GSIRRU
    elif gated_unit_str.upper() == 'GSIRLRU':
        return GSIRLRU
    elif gated_unit_str.upper() == 'GSIRRLRU':
        return GSIRRLRU
    else:
        raise Exception("gated_unit is invalid\nOptions are ('GTU','GTRU','GLU').")


class TransposedUpsampleNet(nn.Module):
    """
    Uses Transposed Convs to upsample a [B, C, T] Tensor.
    
    [B, in_channels, T] -> [B, out_channels, prod(scales)*T]
    """
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3, scales=[16, 16], use_last_layer_act_func=False, weightnorm=False, residual=False, residual_linear=False, rezero=False):
        super(TransposedUpsampleNet, self).__init__()
        self.residual = residual
        self.residual_linear = residual_linear
        self.res_weight = nn.Parameter(torch.rand(1)*0.02+0.01) if rezero else None
        self.scales = scales
        self.t_convs = nn.ModuleList()
        for i, scale in enumerate(scales):
            is_first_layer = bool(i == 0)
            is_last_layer = bool(i+1 == len(scales))
            in_dim = in_channels if is_first_layer else hidden_channels
            out_dim = out_channels if is_last_layer else hidden_channels
            k_size = kernel_size[i] if type(kernel_size) == list else kernel_size
            t_conv = nn.ConvTranspose1d(in_dim, out_dim, k_size, scale, padding=(k_size-scale)//2)
            if weightnorm:
                t_conv = torch.nn.utils.weight_norm(t_conv)
            self.t_convs.append(t_conv)
            if not is_last_layer or use_last_layer_act_func:
                self.t_convs.append( nn.LeakyReLU(negative_slope=0.4, inplace=True) )
        self.res_channels = min(in_channels, out_channels)
    
    def forward(self, x):# [B, C, T]
        if self.residual:
            scale = np.product(self.scales)
            x_interp = F.interpolate(x, scale_factor=scale, mode='linear' if self.residual_linear else 'nearest', align_corners=False)
        
        for layer in self.t_convs:
            x = layer(x)
        
        if self.residual:
            if self.res_weight:
                x *= self.res_weight
            x[:, :self.res_channels] += x_interp[:, :self.res_channels]
        return x


class WN(nn.Module):
    """
    This is the non-causal WaveNet like layer.
    """
    def __init__(self, n_in_channels, n_out_channels, n_layers, n_channels, kernel_size=3, end_kernel_size=5, dilations=None, seperable_conv=False, res_skip=True, merge_res_skip=False, gated_unit='GTU'):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.gated_unit = get_gate_func(gated_unit)
        self.merge_res_skip = merge_res_skip
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        start = nn.Conv1d(n_in_channels, n_channels, 1)
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        end = nn.Conv1d(n_channels, n_out_channels, end_kernel_size, padding=(end_kernel_size-1)//2)
        end = nn.utils.weight_norm(end, name='weight')
        self.end = end
        
        if type(dilations) == int:
            dilations = [dilations,]*n_layers # constant dilation if using int
            print("WARNING!! Using constant dilation factor for WN in_layer dilation width.")
        for i in range(n_layers):
            dilation = 2 ** i if dilations is None else dilations[i]
            padding = int((kernel_size*dilation - dilation)/2)
            if (not seperable_conv) or (kernel_size == 1):
                in_layer = nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                           dilation=dilation, padding=padding, padding_mode='zeros')
                in_layer = nn.utils.weight_norm(in_layer, name='weight')
            else:
                depthwise = nn.Conv1d(n_channels, n_channels, kernel_size,
                                    dilation=dilation, padding=padding, padding_mode='zeros', groups=n_channels)
                depthwise = nn.utils.weight_norm(depthwise, name='weight')
                pointwise = nn.Conv1d(n_channels, 2*n_channels, 1,
                                    dilation=dilation, padding=0)
                pointwise = nn.utils.weight_norm(pointwise, name='weight')
                in_layer = torch.nn.Sequential(depthwise, pointwise)
            self.in_layers.append(in_layer)
            
            # last one is not necessary
            if i < n_layers - 1 and not self.merge_res_skip:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            
            if res_skip:
                res_skip_layer = nn.Conv1d(n_channels, res_skip_channels, 1)
                res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
                self.res_skip_layers.append(res_skip_layer)
    
    def forward(self, audio, speaker_id=None):
        audio = self.start(audio)
        
        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
            acts = self.gated_unit( self.in_layers[i](audio), None, self.n_channels )
            
            res_skip_acts = self.res_skip_layers[i](acts) if ( hasattr(self, 'res_skip_layers') and len(self.res_skip_layers) ) else acts
            
            if i == 0:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):
                    audio = audio + res_skip_acts[:,:self.n_channels,:]
                    output = res_skip_acts[:,self.n_channels:,:]
                else:
                    output = res_skip_acts
            else:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):# if res_skip and not last layer
                    audio = audio + res_skip_acts[:,:self.n_channels,:]
                    output = output + res_skip_acts[:,self.n_channels:,:]
                else:
                    output = output + res_skip_acts
        
        output = self.end(output)
        return output

class PostNet(nn.Module):
    """
    This is the PostNet.
    """
    def __init__(self, n_in_channels, n_out_channels, n_layers, n_channels, kernel_size=32):
        super(PostNet, self).__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList()
        self.res_weights = nn.Parameter( torch.ones(n_layers)*0.01 )
        for i in range(n_layers):
            b_first_layer = bool(i == 0)
            b_last_layer = bool(i+1 == n_layers)
            conv = nn.Conv1d(
                n_in_channels if b_first_layer else n_channels,
                n_out_channels if b_last_layer else n_channels,
                kernel_size,
            )
            self.convs.append(conv)
    
    def forward(self, audio):
        for i, conv in enumerate(self.convs):
            left_pad, right_pad = (self.kernel_size-1)//2, -(-(self.kernel_size-1)//2) # one pad may be larger than the other if kernel_size is even
            if i%2 == 1:
                left_pad, right_pad = right_pad, left_pad
            audio_pad = F.pad(audio, (left_pad, right_pad))
            audio += self.res_weights[i] * torch.tanh(conv(audio_pad))
        return audio


class nn_GLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(nn_GLU, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*2, 1)
    
    @torch.jit.script
    def jit_forward(x, out_channels: int, conv_w, conv_b):
        x = torch.nn.functional.conv2d(x, conv_w, conv_b)
        x = x[:, :out_channels, :] * torch.sigmoid(x[:, out_channels:, :])
        return x
    
    def forward(self, audio):
        return self.jit_forward(audio, self.out_channels, self.conv.weight.to(audio.dtype), self.conv.bias.to(audio.dtype))


class StarGAN_Block(nn.Module):
    def __init__(self, n_in_channels, n_channels, kernel_h, kernel_w, stride_h, stride_w):
        super(StarGAN_Block, self).__init__()
        self.conv = nn.Conv2d(n_in_channels, n_channels, (kernel_h, kernel_w), stride=(stride_h, stride_w))
        self.bn = nn.BatchNorm2d(n_channels)
        self.GLU = nn_GLU(n_channels, n_channels)
    
    def forward(self, audio):# [B, in_dim, in_T]
        audio = self.conv(audio)
        audio = self.bn(audio)
        audio = self.GLU(audio)
        return audio# [B, out_dim, out_T]


class DS(nn.Module):
    """
    https://arxiv.org/pdf/1806.02169.pdf
    The spectrogram discriminator follows the same configuration as in StarGAN-VC [30]:
    kernel sizes of (3, 9), (3, 8), (3, 8), (3, 6);
    stride sizes of (1, 2), (1, 2), (1, 2), (1, 2);
    and channel sizes of 32 across the layers.
    """
    def __init__(self, window_lengths, filter_lengths, hop_lengths, block_confs, max_freq=None, min_freq=None):
        super(DS, self).__init__()
        remaining_h = len(window_lengths) * max(filter_lengths)//2
        in_channels = 1
        self.blocks = nn.ModuleList()
        for i, block_conf in enumerate(block_confs):
            kernel_w = block_conf['kernel_w'] if 'kernel_w' in block_conf else 3
            kernel_h = block_conf['kernel_h'] if 'kernel_h' in block_conf else 9
            stride_w = block_conf['stride_w'] if 'stride_w' in block_conf else 1
            stride_h = block_conf['stride_h'] if 'stride_h' in block_conf else 1
            n_channels = block_conf['n_channels'] if 'n_channels' in block_conf else 32
            block = StarGAN_Block(in_channels, n_channels, kernel_h, kernel_w, stride_h, stride_w)
            self.blocks.append(block)
            remaining_h = remaining_h//stride_h#(remaining_h-(kernel_h-1))//stride_h
            in_channels = n_channels
        
        self.end_conv = nn.Conv2d(in_channels, 1, (remaining_h, 3))# Crush all the dims [B, C, n_mel, T] -> [B, 1, 1, T]
    
    def forward(self, spect):
        spect = spect.unsqueeze(1) # [B, n_mel, T] -> [B, 1, n_mel, T]
        for block in self.blocks:
            spect = block(spect)# [B, C, n_mel, T] -> [B, C, n_mel, T]
        pred_fakeness = self.end_conv(spect)# [B, C, n_mel, T] -> [B, 1, 1, T//prod(stride_w)]
        pred_fakeness = pred_fakeness.mean(dim=3).squeeze(1).squeeze(1)# [B, 1, 1, T//prod(kernel_w)] -> [B, 1, 1] -> [B]
        return pred_fakeness#.sigmoid()# [B]


class DW_Module(nn.Module):
    """
    https://arxiv.org/pdf/1910.06711.pdf
    waveform discriminators, respectively operating on sampled versions of waveform, for discrimination at any configured frequency range.
    """
    def __init__(self, kernel_sizes, strides, n_channels, group_sizes, act_func=nn.LeakyReLU(0.2, inplace=True)):
        super(DW_Module, self).__init__()
        assert n_channels[-1] == 1
        assert group_sizes[-1] == 1
        
        n_in_channel = 1
        self.convs = nn.ModuleList()
        for i, (kernel_size, stride, n_channel, group_size) in enumerate(zip(kernel_sizes, strides, n_channels, group_sizes)):
            conv = nn.Conv1d(n_in_channel, n_channel, kernel_size, stride=stride, groups=group_size)
            conv = nn.utils.weight_norm(conv)
            self.convs.append(conv)
            n_in_channel = n_channel
        
        self.act_func = act_func
        self.res_weights = nn.Parameter( torch.rand(len(kernel_sizes))*0.1+0.01 )
        self.layr_weights = nn.Parameter( torch.ones(len(kernel_sizes)) )
    
    def forward(self, audio):# [B, 1, T]
        for i, conv in enumerate(self.convs):
            if audio.shape[2] < conv.kernel_size[0]:
                audio = torch.nn.functional.pad(audio, (0, conv.kernel_size[0]-audio.shape[2]))
            res = self.act_func( conv(audio) )
            
            left_diff, right_diff = (audio.shape[2]-res.shape[2])//2, -(-(audio.shape[2]-res.shape[2])//2)
            min_channel = min(res.shape[1], audio.shape[1])
            audio_skip = audio[:, :min_channel, left_diff:-right_diff]
            
            audio = self.res_weights[i] * res
            audio[:, :min_channel] += self.layr_weights[i] * audio_skip
        pred_fakeness = audio.squeeze(1).mean(dim=1)# [B, 1, T//x] -> [B]
        return pred_fakeness# [B]


class DW(nn.Module):
    """
    https://arxiv.org/pdf/1910.06711.pdf
    The design of discriminators is inspired by MelGAN [21] which uses multi-scale discrimination on waveform in speech synthesis. Similarly, we use three waveform discriminators, respectively operating at 16kHz, 8kHz and 4kHz sampled versions of waveform, for discrimination at different frequency ranges.
    """
    def __init__(self, n_discriminators,
                            # [1 , 2 ,  3 ,  4  ,  5  ,  6  , 7]
                 kernel_sizes=[15, 41,  41,   41,   41,    5, 3],
                 strides=     [ 1,  4,   4,    4,    4,    1, 1],
                 n_channels=  [16, 64, 256, 1024, 1024, 1024, 1],
                 group_sizes= [ 1,  4,  16,   64,  256,    1, 1],
                 ):
        super(DW, self).__init__()
        self.dw_modules = nn.ModuleList()
        for i in range(n_discriminators):
            module = DW_Module(kernel_sizes, strides, n_channels, group_sizes)
            self.dw_modules.append(module)
    
    def forward(self, audio):
        audio = audio.unsqueeze(1)
        pred_fakeness_es = []
        for module in self.dw_modules:
            pred_fakeness = module(audio)# [B, T] -> [B]
            pred_fakeness_es.append(pred_fakeness)
            audio = F.avg_pool1d(audio, kernel_size=4, stride=2)# [B, T] -> [B, T//2]
        pred_fakeness = torch.stack(pred_fakeness_es, dim=1).sum(dim=1)# -> [B, n_modules] -> [B]
        return pred_fakeness#.sigmoid()# [B]