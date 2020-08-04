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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveGlowLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma
        self.sigma2 = sigma*sigma
        self.sigma2_2 = 2*sigma*sigma
    
    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]
        
        loss = torch.sum(z*z)/(self.sigma2_2) - log_s_total - log_det_W_total
        return loss/(z.size(0)*z.size(1)*z.size(2))


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W
    
    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.dtype == 'torch.float16':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            if z.dtype == 'torch.float16':
                log_det_W = batch_size * n_of_groups * torch.logdet(W.float()).half()
            else:
                log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class WN(nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, cond_in_channels, cond_layers, cond_hidden_channels, cond_kernel_size, cond_padding_mode, seperable_conv, merge_res_skip, upsample_mode, n_layers, n_channels, # audio_channels, mel_channels*n_group, n_layers, n_conv_channels
                 kernel_size, speaker_embed_dim, rezero): # bool: ReZero
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.speaker_embed_dim = speaker_embed_dim
        self.merge_res_skip = merge_res_skip
        self.upsample_mode = upsample_mode
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        if rezero:
            self.alpha_i = nn.ParameterList()
        
        start = nn.Conv2d(n_in_channels, n_channels, (1, 1))
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = nn.Conv1d(n_channels, 2*n_in_channels, (1, 1))
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        
        if self.speaker_embed_dim:
            max_speakers = 512
            self.speaker_embed = nn.Embedding(max_speakers, self.speaker_embed_dim)
        
        self.cond_layers = nn.ModuleList()
        if cond_layers:
            cond_in_channels = cond_in_channels + self.speaker_embed_dim
            cond_kernel_size = 2*cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            cond_pad = int((cond_kernel_size - 1)/2)
            cond_output_channels = 2*n_channels*n_layers
            # messy initialization for arbitrary number of layers, input dims and output dims
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
        
        for i in range(n_layers):
            dilation_h = 1
            dilation_w = 2 ** i
            
            padding_w = int((kernel_size_w*dilation - dilation)/2)
            padding_h = 0
            if (not seperable_conv) or (kernel_size == 1):
                in_layer = nn.Conv2d(n_channels, 2*n_channels, (kernel_size, kernel_height),
                                     dilation=(dilation_h, dilation_w),
                                     padding=(padding_h, padding_w))
                in_layer = nn.utils.weight_norm(in_layer, name='weight')
            else:
                # todo
                depthwise = nn.Conv2d(n_channels, n_channels, kernel_size,
                                    dilation=dilation,
                                    padding=(padding_h, padding_w),
                                    groups=n_channels)
                depthwise = nn.utils.weight_norm(depthwise, name='weight')
                pointwise = nn.Conv2d(n_channels, 2*n_channels, (1, 1),
                                    dilation=dilation,
                                    padding=(padding_h, 0))
                pointwise = nn.utils.weight_norm(pointwise, name='weight')
                in_layer = torch.nn.Sequential(depthwise, pointwise)
            self.in_layers.append(in_layer)
            
            # last one is not necessary
            if i < n_layers - 1 and not self.merge_res_skip:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = nn.Conv2d(n_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            
            if rezero:
                alpha_ = nn.Parameter(torch.rand(1)*0.02+0.09) # rezero initial state (0.1±0.01)
                self.alpha_i.append(alpha_)
            self.res_skip_layers.append(res_skip_layer)
    
    def _upsample_mels(self, cond, audio_size):
        cond = F.interpolate(cond, size=audio_size[2], mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None)
        return cond
    
    def forward(self, audio, spect, speaker_id=None):
        audio = audio.unsqueeze(1) # [B, n_group//2, T//n_group] -> [B, 1, n_group//2, T//n_group]
        audio = self.start(audio) # [B, 1, n_group//2, T//n_group] -> [B
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])
        
        if self.speaker_embed_dim and speaker_id != None: # add speaker embeddings to spectrogram (channel dim)
            speaker_embeddings = self.speaker_embed(speaker_id)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, spect.shape[2]) # shape like spect
            spect = torch.cat([spect, speaker_embeddings], dim=1) # and concat them
        
        for layer in self.cond_layers:
            spect = layer(spect)
        
        if audio.size(2) > spect.size(2): # if spectrogram hasn't been upsampled yet
            spect = self._upsample_mels(spect, audio.shape)
            assert audio.size(2) == spect.size(2)
        
        for i in range(self.n_layers): # note, later layers learn lower frequency information
                                       # receptive field = 2**(n_layers-1)*kernel_size*n_group
                                       # If segment length < receptive field expect trouble learning lower frequencies as other layers try to compensate.
                                       # Since my audio is high-passed at 40Hz, you can expect 48000/(40*2) = 600 samples receptive field minimum required to learn.
            spect_offset = i*2*self.n_channels, (i+1)*2*self.n_channels
            spec = spect[:,spect_offset[0]:spect_offset[1],:]
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spec,
                n_channels_tensor)
            
            if hasattr(self, 'alpha_i'): # if rezero
                res_skip_acts = self.res_skip_layers[i](acts) * self.alpha_i[i]
            else:
                res_skip_acts = self.res_skip_layers[i](acts)
            
            if self.merge_res_skip:
                output = audio = res_skip_acts
            else:
                if i < self.n_layers - 1:
                    audio = audio + res_skip_acts[:,:self.n_channels,:]
                    output = output + res_skip_acts[:,self.n_channels:,:]
                else:
                    output = output + res_skip_acts
        
        return self.end(output).chunk(2, 1)


class WaveGlow(nn.Module):
    def __init__(self, yoyo, yoyo_WN, n_mel_channels, n_flows, n_group, n_early_every,
                 n_early_size, memory_efficient, spect_scaling, upsample_mode, WN_config, win_length, hop_length):
        super(WaveGlow, self).__init__()
        self.spect_scaling = spect_scaling
        self.multispeaker = WN_config['speaker_embed_dim'] > 0
        
        #if self.spect_scaling:
        if False: # Untested with multiple learning rate thing. Parameters of WaveGlow are not params of children and thus probably won't learn
            self.spect_scale = nn.Parameter(torch.rand(1, n_mel_channels, 1) * 0.1 + 0.95) # init between 0.95 and 1.05
            self.spect_shift = nn.Parameter(torch.rand(1, n_mel_channels, 1) * 0.2 - 0.1) # init between -0.1 and 0.1
        
        #upsample_mode = 'normal' # options: 'normal','simple','simple_half'
        self.upsample = nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 win_length, stride=hop_length,
                                                 groups=1 if upsample_mode == 'normal' else (n_mel_channels if upsample_mode == 'simple' else (n_mel_channels/2 if upsample_mode == 'simple_half' else print("upsample_mode = {upsample_mode} invalid"))) )
        
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = nn.ModuleList()
        self.convinv = nn.ModuleList()
        
        n_half = int(n_group/2)
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            if not memory_efficient: # normal
                self.convinv.append(Invertible1x1Conv(n_remaining_channels))
                self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
            else: # mem_efficient
                pass
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, spect, audio, speaker_id=None):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        if self.spect_scaling:
            spect.mul_(self.spect_scale).add_(self.spect_shift) # adjust each spectogram channel by a param
        
        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]
        
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # "Squeeze to Vectors"
        output_audio = []
        log_s_list = []
        log_det_W_list = []
        
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:,:self.n_early_size,:])
                audio = audio[:,self.n_early_size:,:]#.clone() # memory efficient errors.
            
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)
            
            audio_0, audio_1 = audio.chunk(2,1)
            #n_half = int(audio.size(1)/2)
            #audio_0 = audio[:,:n_half,:]
            #audio_1 = audio[:,n_half:,:]
            
            #output = self.WN[k]((audio_0, spect))
            #log_s = output[:, n_half:, :]
            #b = output[:, :n_half, :]
            b, log_s = self.WN[k](audio_0, spect, speaker_id=speaker_id)
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)
            
            audio = torch.cat([audio_0, audio_1],1)
        
        output_audio.append(audio)
        return torch.cat(output_audio,1), log_s_list, log_det_W_list

    def infer(self, spect, speaker_id=None, sigma=1.0):
        if self.spect_scaling:
            spect.mul_(self.spect_scale).add_(self.spect_shift) # adjust each spectogram channel by a param

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        audio = torch.ones(spect.size(0), self.n_remaining_channels, spect.size(2), device=spect.device, dtype=spect.dtype).normal_(std=sigma)
        
        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]
            
            b, s = self.WN[k](audio_0, spect, speaker_id=speaker_id)
            
            #s = output[:, n_half:, :]
            #b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1],1)
            
            audio = self.convinv[k](audio, reverse=True)
            
            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma*z, audio),1)
        
        audio = audio.permute(0,2,1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = nn.ModuleList()
    for old_conv in conv_list:
        old_conv = nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
