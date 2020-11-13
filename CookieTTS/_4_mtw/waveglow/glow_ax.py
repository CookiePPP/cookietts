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

# "Gated Convolutional Neural Networks for Domain Adaptation"
#  https://arxiv.org/pdf/1905.06906.pdf
@torch.jit.script
def GTU(input_a, input_b, n_channels: int):
    """Gated Tanh Unit (GTU)"""
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
            if hasattr(layer, 'inplace'):
                layer.inplace = False if self.training else True
            x = layer(x)
        
        if self.residual:
            if self.res_weight:
                x *= self.res_weight
            x[:, :self.res_channels] += x_interp[:, :self.res_channels]
        return x


class WN(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, cond_in_channels, cond_layers, cond_hidden_channels, cond_kernel_size, cond_padding_mode, seperable_conv, merge_res_skip, upsample_mode, n_layers, n_channels, # audio_channels, mel_channels*n_group, n_layers, n_conv_channels
                 speaker_embed_dim, rezero, cond_activation_func='none', negative_slope=None, kernel_size=None, kernel_size_w=None, n_layers_dilations_w=None, n_layers_dilations_h=None, res_skip=True, cond_out_activation_func=True, upsample_first=None, gated_unit='GTU', upsample_factor=None,
                 transposed_conv_hidden_dim=256, transposed_conv_kernel_size=4, transposed_conv_scales=None): # bool: ReZero
        super(WN, self).__init__()
        cond_in_channels += speaker_embed_dim
        kernel_size = kernel_size_w or kernel_size
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        assert (res_skip or merge_res_skip) or n_layers == 1, "Cannot remove res_skip without using merge_res_skip"
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.speaker_embed_dim = speaker_embed_dim
        self.merge_res_skip = merge_res_skip
        self.upsample_first = upsample_first
        self.upsample_mode = upsample_mode
        self.gated_unit = get_gate_func(gated_unit)
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        assert (not rezero), "WN ReZero is depreciated"
        
        start = nn.Conv1d(n_in_channels, n_channels, 1)
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        
        if self.speaker_embed_dim:
            max_speakers = 512
            self.speaker_embed = nn.Embedding(max_speakers, self.speaker_embed_dim)
        
        self.interpolation_required = True
        t_in_dim = None
        t_out_dim = None
        if transposed_conv_scales is not None and len(transposed_conv_scales) > 0 and transposed_conv_hidden_dim and transposed_conv_kernel_size:
            t_in_dim = transposed_conv_hidden_dim if (self.upsample_first is False) else cond_in_channels
            t_out_dim = 2*n_channels*n_layers if (self.upsample_first is False) else cond_in_channels
            self.upsample_net = TransposedUpsampleNet(t_in_dim, t_out_dim, transposed_conv_hidden_dim, transposed_conv_kernel_size, transposed_conv_scales, use_last_layer_act_func=False)
            self.interpolation_required = bool( np.product(transposed_conv_scales) != upsample_factor )
        
        self.cond_out_activation_func = cond_out_activation_func
        if cond_layers:
            self.cond_layers = nn.ModuleList()
            cond_kernel_size = 2*cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            cond_pad = int((cond_kernel_size - 1)/2)
            cond_output_channels = 2*n_channels*n_layers if (t_in_dim is None) else transposed_conv_hidden_dim
            # messy initialization for arbitrary number of layers, input dims and output dims
            cond_in_channels = t_out_dim if (t_out_dim is not None and self.upsample_first == 'before_wn_cond') else cond_in_channels
            dimensions = [cond_in_channels,]+[cond_hidden_channels]*(cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            # 'zeros','replicate'
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=cond_pad, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            
            cond_activation_func = cond_activation_func.lower()
            if cond_activation_func == 'none':
                pass
            elif cond_activation_func == 'lrelu':
                self.cond_activation_func = torch.nn.functional.relu
            elif cond_activation_func == 'relu':
                assert negative_slope, "negative_slope not defined in wn_config"
                self.cond_activation_func = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
            elif cond_activation_func == 'tanh':
                self.cond_activation_func = torch.nn.functional.tanh
            elif cond_activation_func == 'sigmoid':
                self.cond_activation_func = torch.nn.functional.sigmoid
            else:
                raise NotImplementedError
        
        if type(n_layers_dilations_w) == int:
            n_layers_dilations_w = [n_layers_dilations_w,]*n_layers # constant dilation if using int
            print("WARNING: Using constant dilation factor for WN in_layer dilation width.")
        for i in range(n_layers):
            dilation = 2 ** i if n_layers_dilations_w is None else n_layers_dilations_w[i]
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
    
    def _upsample_mels(self, cond, audio_size, force_interpolate=False):
        if hasattr(self, 'upsample_net') and self.upsample_net is not None:
            cond = self.upsample_net(cond)
        if force_interpolate or (self.interpolation_required and not (cond.shape[2] == audio_size[2])):
            cond = F.interpolate(cond, size=audio_size[2], mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None)
        else:
            cond_len = cond.shape[2]
            audio_len = audio_size[2]
            pad_l = (cond_len-audio_len)//2
            pad_r = -(audio_len-cond_len)//2
            cond = cond[:, :, pad_l:-pad_r]
        return cond
    
    def forward(self, audio, spect, speaker_id=None):
        audio = self.start(audio)
        
        if self.speaker_embed_dim and speaker_id != None: # add speaker embeddings to spectrogram (channel dim)
            speaker_embeddings = self.speaker_embed(speaker_id)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, spect.shape[2]) # shape like spect
            spect = torch.cat([spect, speaker_embeddings], dim=1) # and concat them
        
        if hasattr(self, 'cond_layers') and self.cond_layers:
            for i, layer in enumerate(self.cond_layers):
                spect = layer(spect)
                if hasattr(self, 'cond_activation_func') and (self.cond_out_activation_func or (i != len(self.cond_layers)-1)):
                    spect = self.cond_activation_func(spect)
        
        if self.upsample_first is False: # if spectrogram hasn't been upsampled in an earlier stage
            spect = self._upsample_mels(spect, audio.shape)
        
        for i in range(self.n_layers): # note, later layers learn lower frequency information
                                       # receptive field = 2**(n_layers-1)*kernel_size*n_group
                                       # If segment length < receptive field expect trouble learning lower frequencies as other layers try to compensate.
                                       # Since my audio is high-passed at 40Hz, (theoretically) you can expect 48000/(40*2) = 600 samples receptive field minimum required to learn.
            spect_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
            spec = spect[:,spect_offset[0]:spect_offset[1],:]
            acts = self.gated_unit(
                self.in_layers[i](audio),
                spec,
                self.n_channels)
            
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
        
        return self.end(output).chunk(2, 1)


class WN_2d(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions are causal on the height dimension and non-causal on the width dim.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, cond_in_channels, cond_layers, cond_hidden_channels, cond_kernel_size, cond_padding_mode, seperable_conv, merge_res_skip, upsample_mode, n_layers, n_channels, # audio_channels, mel_channels*n_group, n_layers, n_conv_channels
                 kernel_size_w, kernel_size_h, speaker_embed_dim, rezero, cond_activation_func='none', negative_slope=None, n_layers_dilations_w=None, n_layers_dilations_h=1, res_skip=True, upsample_first=None, cond_out_activation_func=True, gated_unit='GTU', transposed_conv_hidden_dim=256, transposed_conv_kernel_size=4, transposed_conv_scales=None, upsample_factor=None):
        super(WN_2d, self).__init__()
        cond_in_channels += speaker_embed_dim
        assert(kernel_size_w % 2 == 1)
        assert(n_channels % 2 == 0)
        assert res_skip or merge_res_skip, "Cannot remove res_skip without using merge_res_skip"
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.kernel_size_h = kernel_size_h
        self.speaker_embed_dim = speaker_embed_dim
        self.merge_res_skip = merge_res_skip
        self.upsample_first = upsample_first
        self.upsample_mode = upsample_mode
        self.gated_unit = get_gate_func(gated_unit)
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        assert (not rezero), "WN ReZero is depreciated"
        
        start = nn.Conv2d(1, n_channels, (1,1))
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = nn.Conv2d(n_channels, 2, (1,1))
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        
        if self.speaker_embed_dim:
            max_speakers = 512
            self.speaker_embed = nn.Embedding(max_speakers, self.speaker_embed_dim)
        
        self.interpolation_required = True
        t_in_dim = None
        t_out_dim = None
        if transposed_conv_scales is not None and len(transposed_conv_scales) > 0 and transposed_conv_hidden_dim and transposed_conv_kernel_size:
            t_in_dim = transposed_conv_hidden_dim if (self.upsample_first is False) else cond_in_channels
            t_out_dim = 2*n_channels*n_layers if (self.upsample_first is False) else cond_in_channels
            self.upsample_net = TransposedUpsampleNet(t_in_dim, t_out_dim, transposed_conv_hidden_dim, transposed_conv_kernel_size, transposed_conv_scales, use_last_layer_act_func=False)
            self.interpolation_required = bool( np.product(transposed_conv_scales) != upsample_factor )
        
        self.cond_out_activation_func = cond_out_activation_func
        if cond_layers:
            self.cond_layers = nn.ModuleList()
            cond_kernel_size = 2*cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            cond_pad = int((cond_kernel_size - 1)/2)
            cond_output_channels = 2*n_channels*n_layers if (t_in_dim is None or self.upsample_first == 'before_wn_cond') else transposed_conv_hidden_dim
            # messy initialization for arbitrary number of layers, input dims and output dims
            cond_in_channels = t_out_dim if (t_out_dim is not None and self.upsample_first == 'before_wn_cond') else cond_in_channels
            dimensions = [cond_in_channels,]+[cond_hidden_channels]*(cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            # 'zeros','replicate'
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=cond_pad, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            
            cond_activation_func = cond_activation_func.lower()
            if cond_activation_func == 'none':
                pass
            elif cond_activation_func == 'lrelu':
                self.cond_activation_func = torch.nn.functional.relu
            elif cond_activation_func == 'relu':
                assert negative_slope, "negative_slope not defined in wn_config"
                self.cond_activation_func = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
            elif cond_activation_func == 'tanh':
                self.cond_activation_func = torch.nn.functional.tanh
            elif cond_activation_func == 'sigmoid':
                self.cond_activation_func = torch.nn.functional.sigmoid
            else:
                raise NotImplementedError
        
        if type(n_layers_dilations_w) == int:
            n_layers_dilations_w = [n_layers_dilations_w,]*n_layers # constant dilation if using int
            print("WARNING: Using constant dilation factor for WN in_layer dilation width.")
        if type(n_layers_dilations_h) == int:
            n_layers_dilations_h = [n_layers_dilations_h,]*n_layers # constant dilation if using int
        
        self.h_dilate = n_layers_dilations_h
        self.padding_h = []
        for i in range(n_layers):
            dilation_h = n_layers_dilations_h[i]
            dilation_w = 2 ** i if n_layers_dilations_w is None else n_layers_dilations_w[i]
            
            padding_w = ((kernel_size_w-1)*dilation_w)//2
            self.padding_h.append((kernel_size_h-1)*dilation_h) # causal padding https://theblog.github.io/post/convolution-in-autoregressive-neural-networks/
            if (not seperable_conv) or (kernel_size_w == 1 and kernel_size_h == 1):
                in_layer = nn.Conv2d(n_channels, 2*n_channels, (kernel_size_h,kernel_size_w),
                                           dilation=(dilation_h,dilation_w), padding=(0,padding_w), padding_mode='zeros')
                in_layer = nn.utils.weight_norm(in_layer, name='weight')
            else:
                depthwise = nn.Conv2d(n_channels, n_channels, (kernel_size_h,kernel_size_w),
                                    dilation=(dilation_h,dilation_w), padding=(0,padding_w), padding_mode='zeros', groups=n_channels)
                depthwise = nn.utils.weight_norm(depthwise, name='weight')
                pointwise = nn.Conv2d(n_channels, 2*n_channels, (1,1),
                                    dilation=(1,1), padding=(0,0))
                pointwise = nn.utils.weight_norm(pointwise, name='weight')
                in_layer = torch.nn.Sequential(depthwise, pointwise)
            self.in_layers.append(in_layer)
            
            # last one is not necessary
            if i < n_layers - 1 and not self.merge_res_skip:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            
            if res_skip:
                res_skip_layer = nn.Conv2d(n_channels, res_skip_channels, (1,1))
                res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
                self.res_skip_layers.append(res_skip_layer)
    
    def _upsample_mels(self, cond, audio_size, force_interpolate=False):
        if hasattr(self, 'upsample_net') and self.upsample_net is not None:
            cond = self.upsample_net(cond)
        if force_interpolate or (self.interpolation_required and not (cond.shape[2] == audio_size[3])):
            cond = F.interpolate(cond, size=audio_size[3], mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None)
        else:
            pad = (cond.shape[2] - audio_size[3])//2
            remainder = pad%2
            cond = cond[:, :, pad:-(pad+remainder)]
        return cond
    
    def forward(self, audio, spect, speaker_id=None, audio_queues=None, spect_queues=None):
        audio = audio.unsqueeze(1) #   [B, n_group//2, T//n_group] -> [B, 1, n_group//2, T//n_group]
        audio = self.start(audio) # [B, 1, n_group//2, T//n_group] -> [B, n_channels, n_group//2, T//n_group]
        if self.merge_res_skip:
            output = audio
        else:
            output = torch.zeros_like(audio)
        
        exec_cond_layer = (spect_queues is None) or ( any([x is None for x in spect_queues]) )
        if exec_cond_layer: # process spectrograms
            if self.speaker_embed_dim and speaker_id != None: # add speaker embeddings to spectrogram (channel dim)
                speaker_embeddings = self.speaker_embed(speaker_id)
                speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, spect.shape[2]) # shape like spect
                spect = torch.cat([spect, speaker_embeddings], dim=1) # and concat them
            
            if hasattr(self, 'cond_layers') and self.cond_layers:
                for i, layer in enumerate(self.cond_layers): # [B, cond_channels, T//hop_length] -> [B, n_channels*n_layers, T//hop_length]
                    spect = layer(spect)
                    if hasattr(self, 'cond_activation_func') and (self.cond_out_activation_func or (i != len(self.cond_layers)-1)):
                        spect = self.cond_activation_func(spect)
            
            if not self.upsample_first: # if spectrogram hasn't been upsampled in an earlier stage
                spect = self._upsample_mels(spect, audio.shape)# [B, n_channels*n_layers, T//hop_length] -> [B, n_channels*n_layers, T//n_group]
        
        if len(spect.shape) < 4:
            spect = spect.unsqueeze(2)# [B, n_channels*n_layers, T//n_group] -> [B, n_channels*n_layers, 1, T//n_group]
            #assert audio.size(3) == spect.size(3), f"audio size of {audio.size(3)} != spect size of {spect.size(3)}"
        
        for i in range(self.n_layers):
            if exec_cond_layer: # if cond layer has been ran.
                spect_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
                spec = spect[:,spect_offset[0]:spect_offset[1]] # [B, 2*n_channels*n_layers, 1, T//n_group] -> [B, 2*n_channels, 1, T//n_group]
                if spect_queues is not None: # is spect_queues exists...
                    if spect_queues[i] is None: # but this index is empty...
                        spect_queues[i] = spec # save spec into this index.
            else:                       # else...
                spec = spect_queues[i] # load spec from this index.
            
            if audio_queues is None:# if training/validation...
                audio_cpad = F.pad(audio, (0,0,self.padding_h[i],0)) # apply causal height padding (left, right, top, bottom)
            else: # else, if conv-queue and inference/autoregressive sampling.
                if audio_queues[i] is None: # if first sample in autoregressive sequence, pad start with zeros
                    B, n_channels, n_group, T_group = audio.shape
                    audio_queues[i] = audio.new_zeros( size=[B, n_channels, self.padding_h[i], T_group] )
                
                # [B, n_channels, n_group, T//n_group]
                audio_queues[i] = audio_cpad = torch.cat((audio_queues[i], audio), dim=2)[:,:,-(self.padding_h[i]+1):] # pop old samples and append new sample to end of n_group dim
            
            acts = self.in_layers[i](audio_cpad) # [B, n_channels, n_group//2, T//n_group] -> [B, 2*n_channels, pad+n_group//2, T//n_group]
            acts = self.gated_unit(
                acts, # [B, 2*n_channels, n_group//2, T//n_group]
                spec, # [B, 2*n_channels, 1, T//n_group]
                self.n_channels)
            # acts.shape <- [B, n_channels, n_group//2, T//n_group]
            
            res_skip_acts = self.res_skip_layers[i](acts) if ( hasattr(self, 'res_skip_layers') and len(self.res_skip_layers) ) else acts
            # if merge_res_skip: [B, n_channels, n_group//2, T//n_group] -> [B, n_channels, n_group//2, T//n_group]
            # else: [B, n_channels, n_group//2, T//n_group] -> [B, 2*n_channels, n_group//2, T//n_group]
            
            if i == 0:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):
                    audio += res_skip_acts[:,:self.n_channels,:]
                    output = res_skip_acts[:,self.n_channels:,:]
                else:
                    output = res_skip_acts
            else:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):# if res_skip and not last layer
                    audio += res_skip_acts[:,:self.n_channels,:]
                    output += res_skip_acts[:,self.n_channels:,:]
                else:
                    output += res_skip_acts
        
        func_out = self.end(output).transpose(1,0) # [B, n_channels, n_group//2, T//n_group] -> [B, 2, n_group//2, T//n_group] -> [2, B, n_group//2, T//n_group]
        
        if audio_queues is not None:
            func_out = [func_out,]
            func_out.append(audio_queues)
        if spect_queues is not None:
            func_out.append(spect_queues)
        return func_out