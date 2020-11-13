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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from CookieTTS._2_ttm.untts.waveglow.modules import AffineCouplingBlock, InvertibleConv1x1

@torch.jit.script
def GTU(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


weight_norm = True
class WN(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, cond_in_channels, hparams):
        super(WN, self).__init__()
        assert(hparams.wn_kernel_size % 2 == 1), 'kernel_size must be an odd number'
        assert(hparams.wn_n_channels % 2 == 0), 'n_channels must be a multiple of 2'
        assert(hparams.wn_cond_layers > 0), 'cond_layers must be greater than 0'
        assert hparams.wn_res_skip or hparams.wn_merge_res_skip, "Cannot remove res_skip without using merge_res_skip"
        self.n_layers = hparams.wn_n_layers
        self.n_channels = hparams.wn_n_channels
        self.speaker_embed_dim = 0
        self.merge_res_skip = hparams.wn_merge_res_skip
        self.first_wn = hparams.first_wn
        
        self.in_layers = nn.ModuleList()
        
        start = nn.Conv1d(n_in_channels, self.n_channels, 1)# first 1x1 Conv
        if weight_norm: start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        end = nn.Conv1d(self.n_channels, 2*n_in_channels, 1) # last 1x1 Conv
        end.weight.data.zero_()
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        
        self.cond_layers = nn.ModuleList()
        if hparams.wn_cond_layers:
            cond_in_channels = cond_in_channels + self.speaker_embed_dim
            cond_pad = int((hparams.wn_cond_kernel_size - 1)/2)
            cond_output_channels = 2*self.n_channels*self.n_layers
            # messy initialization for arbitrary number of layers, input dims and output dims
            dimensions = [cond_in_channels,]+[hparams.wn_cond_hidden_channels]*(hparams.wn_cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, hparams.wn_cond_kernel_size, padding=cond_pad, padding_mode=hparams.wn_cond_padding_mode)
                if weight_norm: cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            
            hparams.wn_cond_act_func = hparams.wn_cond_act_func.lower()
            if hparams.wn_cond_act_func == 'none':
                pass
            elif hparams.wn_cond_act_func == 'lrelu':
                self.wn_cond_act_func = torch.nn.functional.relu
            elif hparams.wn_cond_act_func == 'relu':
                assert negative_slope, "negative_slope not defined in wn_config"
                self.wn_cond_act_func = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)
            elif hparams.wn_cond_act_func == 'tanh':
                self.wn_cond_act_func = torch.nn.functional.tanh
            elif hparams.wn_cond_act_func == 'sigmoid':
                self.wn_cond_act_func = torch.nn.functional.sigmoid
            else:
                raise NotImplementedError('hparams.wn_cond_act_func is invalid')
        
        self.res_skip_layers = nn.ModuleList()
        if type(hparams.wn_dilations_w) == int:
            hparams.wn_dilations_w = [hparams.wn_dilations_w,]*self.n_layers # constant dilation if using int
        self.wn_padding_value = hparams.decoder_padding_value if self.first_wn else 0.0
        self.padding = []
        for i in range(self.n_layers):
            dilation = 2 ** i if hparams.wn_dilations_w is None else hparams.wn_dilations_w[i]
            self.padding.append( int((hparams.wn_kernel_size*dilation - dilation)/2) )
            if (not hparams.wn_seperable_conv) or (hparams.wn_kernel_size == 1):
                in_layer = nn.Conv1d(self.n_channels, 2*self.n_channels, hparams.wn_kernel_size, dilation=dilation, padding=0, padding_mode='zeros')
                if weight_norm: in_layer = nn.utils.weight_norm(in_layer, name='weight')
            else:
                depthwise = nn.Conv1d(self.n_channels, self.n_channels, hparams.wn_kernel_size, dilation=dilation, padding=0, padding_mode='zeros', groups=self.n_channels)
                if weight_norm: depthwise = nn.utils.weight_norm(depthwise, name='weight')
                pointwise = nn.Conv1d(self.n_channels, 2*self.n_channels, 1,
                                    dilation=dilation, padding=0)
                if weight_norm: pointwise = nn.utils.weight_norm(pointwise, name='weight')
                in_layer = torch.nn.Sequential(depthwise, pointwise)
            self.in_layers.append(in_layer)
            
            # last one is not necessary
            if i < self.n_layers - 1 and not self.merge_res_skip:
                res_skip_channels = 2*self.n_channels
            else:
                res_skip_channels = self.n_channels
            
            if hparams.wn_res_skip:
                res_skip_layer = nn.Conv1d(self.n_channels, res_skip_channels, 1)
                if weight_norm: res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
                self.res_skip_layers.append(res_skip_layer)
    
    def forward(self, spect, cond, speaker_id=None):
        spect = self.start(spect)
        if not self.merge_res_skip:
            output = torch.zeros_like(spect) # output and spect are seperate Tensors
        n_channels_tensor = torch.IntTensor([self.n_channels])
        
        if self.speaker_embed_dim and speaker_id != None: # add speaker embeddings to condrogram (channel dim)
            speaker_embeddings = self.speaker_embed(speaker_id)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        for layer in self.cond_layers:
            cond = layer(cond)
            if hasattr(self, 'wn_cond_act_func'):
                cond = self.wn_cond_act_func(cond)
        
        for i in range(self.n_layers):
            cond_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
            spect_pad = F.pad(spect, (self.padding[i],)*2, mode='constant', value=self.wn_padding_value)
            acts = GTU(self.in_layers[i](spect_pad), cond[:,cond_offset[0]:cond_offset[1],:], n_channels_tensor)
            
            if hasattr(self, 'res_skip_layers') and len(self.res_skip_layers):
                res_skip_acts = self.res_skip_layers[i](acts)
            else:
                res_skip_acts = acts
            
            if self.merge_res_skip:
                spect = spect + res_skip_acts
            else:
                if i < self.n_layers - 1:
                    spect = spect + res_skip_acts[:,:self.n_channels,:]
                    output = output + res_skip_acts[:,self.n_channels:,:]
                else:
                    output = output + res_skip_acts
        
        if self.merge_res_skip:
            output = spect
        
        return self.end(output).chunk(2, 1)


class FlowDecoder(nn.Module):
    def __init__(self, hparams):
        super(FlowDecoder, self).__init__()
        assert(hparams.n_group % 2 == 0)
        self.n_flows = hparams.n_flows
        self.n_group = hparams.n_group
        self.n_early_every = hparams.n_early_every
        self.n_early_size = hparams.n_early_size
        self.cond_in_channels = hparams.cond_input_dim
        self.n_mel_channels = hparams.n_mel_channels
        self.mix_first = hparams.mix_first
        self.speaker_embed_dim = 0
        
        self.cond_residual = hparams.cond_residual
        # override conditional output size if using residuals
        cond_output_channels = self.cond_in_channels if self.cond_residual else hparams.cond_output_channels
        
        self.cond_res_rezero = hparams.cond_res_rezero
        if self.cond_res_rezero:
            self.alpha = nn.Parameter(torch.rand(1)*0.002+0.001) # rezero initial state (0.001±0.001)
        
        self.cond_layers = nn.ModuleList()
        if hparams.cond_layers:
            # messy initialization for arbitrary number of layers, input dims and output dims
            hparams.cond_kernel_size = 2*hparams.cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            hparams.cond_pad = int((hparams.cond_kernel_size - 1)/2)
            dimensions = [self.cond_in_channels,]+[hparams.cond_hidden_channels]*(hparams.cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, hparams.cond_kernel_size, padding=hparams.cond_pad, padding_mode=hparams.cond_padding_mode)# (in_channels, out_channels, kernel_size)
                if hparams.cond_weightnorm:
                    cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            WN_cond_channels = cond_output_channels
            
            hparams.cond_act_func = hparams.cond_act_func.lower()
            if hparams.cond_act_func == 'none':
                pass
            elif hparams.cond_act_func == 'lrelu':
                self.cond_act_func = torch.nn.functional.relu
            elif hparams.cond_act_func == 'relu':
                assert negative_slope, "negative_slope not defined in wn_config"
                self.cond_act_func = torch.nn.LeakyReLU(negative_slope=0.2, inplace=False)
            elif hparams.cond_act_func == 'tanh':
                self.cond_act_func = torch.nn.functional.tanh
            elif hparams.cond_act_func == 'sigmoid':
                self.cond_act_func = torch.nn.functional.sigmoid
            else:
                raise NotImplementedError
        else:
            WN_cond_channels = self.cond_in_channels
        
        self.convinv = nn.ModuleList()
        self.WN = nn.ModuleList()
        
        n_remaining_channels = hparams.n_group
        self.z_split_sizes = []
        for k in range(hparams.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels -= self.n_early_size
                self.z_split_sizes.append(self.n_early_size)
            assert n_remaining_channels > 0, "n_remaining_channels is 0. (increase n_group or decrease n_early_every/n_early_size)"
            
            if hparams.grad_checkpoint and (k+1)/hparams.n_flows <= hparams.grad_checkpoint:
                mem_eff_layer = True
            else:
                mem_eff_layer = False
            
            if k == 0:
                hparams.first_wn = True
            else:
                hparams.first_wn = False
            self.convinv.append( InvertibleConv1x1(n_remaining_channels, memory_efficient=mem_eff_layer) )
            self.WN.append( AffineCouplingBlock(WN, memory_efficient=mem_eff_layer, n_in_channels=n_remaining_channels//2,
                                cond_in_channels=WN_cond_channels, hparams=hparams) )
            delattr(hparams, 'first_wn')
        self.z_split_sizes.append(n_remaining_channels)
    
    
    def forward(self, z, cond, speaker_ids=None): # optional cond input
        """
        z = z: batch x n_mel_channels x time
        cond = attention outputs:  batch x time x enc_embed
        """
        # Add speaker conditioning
        if self.speaker_embed_dim:
            speaker_embeddings = self.speaker_embed(speaker_ids)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        cond_res = cond
        for layer in self.cond_layers:
            cond_res = layer(cond_res)
            if hasattr(self, 'cond_act_func'):
                cond_res = self.cond_act_func(cond_res)
        
        if hasattr(self, 'alpha'):
            cond_res *= self.alpha # reZero modifier
        
        if self.cond_residual:
            cond = cond + cond_res # adjust the original input by a residual
        else:
            cond = cond_res # completely reform the input into something else
        
        batch_dim, n_mel_channels, group_steps = z.shape
        z = z.view(batch_dim, self.n_group, -1) # [B, n_mel, T] -> [B, n_mel/8, T*8]
        #cond = F.interpolate(cond, size=z.shape[-1]) # [B, enc_dim, T] -> [B, enc_dim/8, T*8]
        
        output_spect = []
        split_sections = [self.n_early_size, self.n_group]
        for k, (convinv, affine_coup) in enumerate(zip(self.convinv, self.WN)):
            if k % self.n_early_every == 0 and k > 0:
                split_sections[1] -= self.n_early_size
                early_output, z = z.split(split_sections, 1)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_spect.append(early_output)
                z = z.clone()
            
            if self.mix_first:
                z, log_det_W = convinv(z)
                assert not torch.isnan(z).any()
                assert not torch.isinf(z).any()
            
            z, log_s = affine_coup(z, cond)
            assert not torch.isnan(z).any()
            assert not torch.isinf(z).any()
            
            if not self.mix_first:
                z, log_det_W = convinv(z)
                assert not torch.isnan(z).any()
                assert not torch.isinf(z).any()
            
            if k:
                logdet_w_sum = logdet_w_sum + log_det_W
                log_s_sum = log_s_sum + log_s.float().sum((1,))
            else:
                logdet_w_sum = log_det_W
                log_s_sum = log_s.float().sum((1,))
        
        assert split_sections[1] == self.z_split_sizes[-1]
        output_spect.append(z)
        return torch.cat(output_spect, 1).contiguous().view(batch_dim, self.n_mel_channels, -1), log_s_sum, logdet_w_sum
    
    def inverse(self, z, cond, speaker_ids=None):
        # Add speaker conditioning
        if self.speaker_embed_dim:
            speaker_embeddings = self.speaker_embed(speaker_ids)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        cond_res = cond
        for layer in self.cond_layers:
            cond_res = layer(cond_res)
            if hasattr(self, 'cond_act_func'):
                cond_res = self.cond_act_func(cond_res)
        
        if hasattr(self, 'alpha'):
            cond_res *= self.alpha # reZero modifier
        
        if self.cond_residual:
            cond += cond_res # adjust the original input by a residual
        else:
            cond = cond_res # completely reform the input into something else
        
        batch_dim, n_mel_channels, group_steps = z.shape
        z = z.view(batch_dim, self.n_group, -1) # [B, n_mel, T] -> [B, n_mel/8, T*8]
        #cond = F.interpolate(cond, size=z.shape[-1]) # [B, enc_dim, T] -> [B, enc_dim/8, T*8]
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z
        
        logdet = None
        for k, invconv, affine_coup in zip(range(self.n_flows-1, -1, -1), self.convinv[::-1], self.WN[::-1]):

            if not self.mix_first:
                z, _ = invconv.inverse(z)
            
            z, _ = affine_coup.inverse(z, cond, speaker_ids=speaker_ids)
            
            if self.mix_first:
                z, _ = invconv.inverse(z)
            
            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)
        
        z = z.view(batch_dim, self.n_mel_channels, -1)
        return z, logdet
    
    @torch.no_grad()
    def infer(self, cond, speaker_ids=None, sigma=1.):
        batch_size, enc_dim, frames = cond.shape
        z = cond.new_empty((batch_size, self.n_mel_channels, frames)) # [B, n_mel, T]
        if sigma > 0.0:
            z.normal_(std=sigma)
        z, _ = self.inverse(z, cond, speaker_ids)
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