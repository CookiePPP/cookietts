import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# utils
from waveglow_utils import PreEmphasis
from scipy import signal

@torch.jit.script
def ignore_nan(input):
    """Replace NaN values with 0.0"""
    input[torch.isnan(input)] = torch.zeros(1, device=input.device, dtype=input.dtype)

class WaveGlow(nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every,
                n_early_size, memory_efficient, spect_scaling, upsample_mode, upsample_first, speaker_embed, cond_layers, cond_hidden_channels, cond_output_channels, cond_kernel_size, cond_residual, cond_padding_mode, WN_config, win_length, hop_length, cond_res_rezero=False, cond_activation_func='none', negative_slope=None, channel_mixing='1x1conv', mix_first=True, preceived_vol_scaling=False, waveflow=True, yoyo='depreciated', yoyo_WN='depreciated', shift_spect=0., scale_spect=1., preempthasis=None):
        super(WaveGlow, self).__init__()
        assert(n_group % 2 == 0)
        assert(hop_length % n_group == 0), "hop_length is not int divisible by n_group"
        assert(any(channel_mixing.lower() in x for x in ("1x1convinvertibleconv1x1invconv", "waveflowpermuteheightpermutechannelpermute"))), "channel_mixing option is invalid. Options are '1x1conv' or 'permuteheight'"
        self.channel_mixing = '1x1conv' if channel_mixing.lower() in "1x1convinvertibleconv1x1invconv" else ('permuteheight' if channel_mixing.lower() in "waveflowpermuteheightpermutechannelpermute" else None)
        self.mix_first = mix_first
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        
        self.win_size = win_length
        self.hop_length = hop_length
        self.n_mel_channels = n_mel_channels
        self.vol_scaling = preceived_vol_scaling
        self.preempthasis = preempthasis
        if preempthasis:
            self.preempthasise = PreEmphasis(preempthasis)
        self.shift_spect = shift_spect
        self.scale_spect = scale_spect
        
        self.upsample_early = WN_config['upsample_first'] = upsample_first
        self.upsample_mode = WN_config['upsample_mode']
        self.nan_asserts = False # check audio Tensor is finite
        self.ignore_nan = True # replace NaNs in audio tensor with 0.0
        
        self.speaker_embed_dim = speaker_embed
        self.multispeaker = self.speaker_embed_dim > 0 or WN_config['speaker_embed_dim'] > 0
        
        if waveflow:
            from efficient_modules import WaveFlowCoupling as AffineCouplingBlock
            from glow_ax import WN_2d as WN
        else:
            from efficient_modules import AffineCouplingBlock
            from glow_ax import WN
        
        if self.speaker_embed_dim:
            max_speakers = 512
            self.speaker_embed = nn.Embedding(max_speakers, self.speaker_embed_dim)
        
        self.cond_residual = cond_residual
        if self.cond_residual: # override conditional output size if using residuals
            cond_output_channels = self.n_mel_channels+self.speaker_embed_dim
        
        self.cond_res_rezero = cond_res_rezero
        if self.cond_res_rezero:
            self.alpha = nn.Parameter(torch.rand(1)*0.002+0.001) # rezero initial state (0.001±0.001)
        
        self.cond_layers = nn.ModuleList()
        if cond_layers:
            # messy initialization for arbitrary number of layers, input dims and output dims
            cond_kernel_size = 2*cond_kernel_size - 1 # 1 -> 1, 2 -> 3, 3 -> 5
            cond_pad = int((cond_kernel_size - 1)/2)
            dimensions = [self.n_mel_channels+self.speaker_embed_dim,]+[cond_hidden_channels]*(cond_layers-1)+[cond_output_channels,]
            in_dims = dimensions[:-1]
            out_dims = dimensions[1:]
            #print(in_dims, out_dims, "\n")
            for i in range(len(in_dims)):
                indim = in_dims[i]
                outim = out_dims[i]
                cond_layer = nn.Conv1d(indim, outim, cond_kernel_size, padding=cond_pad, padding_mode=cond_padding_mode)# (in_channels, out_channels, kernel_size)
                cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
                self.cond_layers.append(cond_layer)
            WN_cond_channels = cond_output_channels
            
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
        else:
            WN_cond_channels = self.n_mel_channels+self.speaker_embed_dim
        
        # import channel mixing
        if self.channel_mixing == '1x1conv':
            from efficient_modules import InvertibleConv1x1
            self.convinv = nn.ModuleList()
        elif self.channel_mixing == 'permuteheight':
            from efficient_modules import PermuteHeight
            self.convinv = list()
        
        self.upsample_factor = hop_length // n_group
        sub_win_size = win_length // n_group
        
        self.WN = nn.ModuleList()
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        self.z_split_sizes = []
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels -= n_early_size
                self.z_split_sizes.append(n_early_size)
            
            assert n_remaining_channels > 0, "n_remaining_channels is 0. (increase n_group or decrease n_early_every/n_early_size)"
            
            if memory_efficient and (k+1)/n_flows <= memory_efficient:
                mem_eff_layer = True
                print(f"Flow {k} using Mem Efficient Backprop")
            else:
                mem_eff_layer = False
                print(f"Flow {k} using Normal Backprop")
            
            if self.channel_mixing == '1x1conv':
                self.convinv.append( InvertibleConv1x1(n_remaining_channels, memory_efficient=mem_eff_layer) )
            elif self.channel_mixing == 'permuteheight':
                self.convinv.append( PermuteHeight(n_remaining_channels, k, n_flows, sigma=1.0) ) # sigma is hardcoded here.
            
            self.WN.append( AffineCouplingBlock(WN, memory_efficient=mem_eff_layer, n_in_channels=n_remaining_channels//2,
                                cond_in_channels=WN_cond_channels, **WN_config) )
        self.z_split_sizes.append(n_remaining_channels)
    
    
    def forward(self, cond, audio, speaker_ids=None): # optional cond input
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        with torch.no_grad():
            if hasattr(self, 'preempthasise'):# apply preempthasis to audio signal (if used)
                audio = self.preempthasise(audio)
            
            if self.shift_spect != 0.:
                cond = cond + self.shift_spect
            if self.scale_spect != 1.:
                cond = cond * self.scale_spect
            
            if self.vol_scaling:
                audio[audio>0] = 2**(audio[audio>0].log10())
                audio[audio<0] = -(2**((-audio[audio<0]).log10()))
        
        # Add speaker conditioning
        if self.speaker_embed_dim:
            speaker_embeddings = self.speaker_embed(speaker_ids)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        cond_res = cond
        for layer in self.cond_layers:
            cond_res = layer(cond_res)
            if hasattr(self, 'cond_activation_func'):
                cond_res = self.cond_activation_func(cond_res)
        
        if hasattr(self, 'alpha'):
            cond_res *= self.alpha # reZero modifier
        
        if self.cond_residual:
            cond = cond + cond_res # adjust the original input by a residual
        else:
            cond = cond_res # completely reform the input into something else
        
        batch_dim, n_mel_channels, group_steps = cond.shape
        audio = audio.view(batch_dim, -1, self.n_group).transpose(1, 2)
        
        #  Upsample spectrogram to size of audio
        if self.upsample_early:
            cond = self._upsample_mels(cond, audio.size(2)) # [B, mels, T//n_group]
        
        #assert audio.size(2) <= cond.size(2)
        #cond = cond[..., :audio.size(2)]
        
        output_audio = []
        split_sections = [self.n_early_size, self.n_group]
        for k, (convinv, affine_coup) in enumerate(zip(self.convinv, self.WN)):
            if k % self.n_early_every == 0 and k > 0:
                split_sections[1] -= self.n_early_size
                early_output, audio = audio.split(split_sections, 1)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_audio.append(early_output)
                audio = audio.clone()
            
            if self.mix_first:
                audio, log_det_W = convinv(audio)
                if self.nan_asserts:
                    assert not torch.isnan(audio).any(), f'Flow {k} NaN Exception'
                    assert not torch.isinf(audio).any(), f'Flow {k} inf Exception'
            
            audio, log_s = affine_coup(audio, cond, speaker_ids=speaker_ids)
            
            if self.ignore_nan:
                ignore_nan(audio)
            
            if self.nan_asserts:
                assert not torch.isnan(audio).any(), f'Flow {k} NaN Exception'
                assert not torch.isinf(audio).any(), f'Flow {k} inf Exception'
            
            if not self.mix_first:
                audio, log_det_W = convinv(audio)
                if self.nan_asserts:
                    assert not torch.isnan(audio).any(), f'Flow {k} NaN Exception'
                    assert not torch.isinf(audio).any(), f'Flow {k} inf Exception'
            
            if k:
                logdet += log_det_W + log_s.float().sum((1, 2))
                logdet_w_sum += log_det_W
                log_s_sum += log_s.float().sum((1,))
            else:
                logdet = log_det_W + log_s.float().sum((1, 2))
                logdet_w_sum = log_det_W
                log_s_sum = log_s.float().sum((1,))
        
        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(audio)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet, logdet_w_sum, log_s_sum
    
    def _upsample_mels(self, cond, audio_size):
        cond = F.interpolate(cond, size=audio_size[2], mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None)
        #cond = F.interpolate(cond, scale_factor=600, mode=self.upsample_mode, align_corners=True if self.upsample_mode == 'linear' else None) # upsample by hop_length
        return cond
    
    def inverse(self, z, cond, speaker_ids=None, return_CPU=True):
        if self.shift_spect != 0.:
            cond = cond + self.shift_spect
        if self.scale_spect != 1.:
            cond = cond * self.scale_spect
        
        # Add speaker conditioning
        if self.speaker_embed_dim:
            if speaker_ids is None:
                raise Exception("This WaveFlow/WaveGlow model requires speaker ids or speaker embeddings.")
            speaker_embeddings = self.speaker_embed(speaker_ids)
            speaker_embeddings = speaker_embeddings.unsqueeze(-1).repeat(1, 1, cond.shape[2]) # shape like cond
            cond = torch.cat([cond, speaker_embeddings], dim=1) # and concat them
        
        cond_res = cond
        for layer in self.cond_layers:
            cond_res = layer(cond_res)
            if hasattr(self, 'cond_activation_func'):
                cond_res = self.cond_activation_func(cond_res)
        
        if hasattr(self, 'alpha'):
            cond_res *= self.alpha # reZero modifier
        
        if self.cond_residual:
            cond += cond_res # adjust the original input by a residual
        else:
            cond = cond_res # completely reform the input into something else
        
        batch_dim, n_mel_channels, group_steps = cond.shape
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        
        #  Upsample spectrogram to size of audio
        if self.upsample_early:
            cond = self._upsample_mels(cond, z.size(2)) # [B, mels, T//n_group]
        
        #assert z.size(2) <= cond.size(2)
        #cond = cond[..., :z.size(2)]
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z
        
        logdet = None
        for k, invconv, affine_coup in zip(range(self.n_flows - 1, -1, -1), self.convinv[::-1], self.WN[::-1]):
            
            if not self.mix_first:
                z, log_det_W = invconv.inverse(z)
            
            z, log_s = affine_coup.inverse(z, cond, speaker_ids=speaker_ids)
            
            if self.ignore_nan:
                ignore_nan(z)
            
            if self.mix_first:
                z, log_det_W = invconv.inverse(z)
            
            #if k == self.n_flows - 1:
            #    logdet = log_det_W + log_s.sum((1, 2))
            #else:
            #    logdet += log_det_W + log_s.sum((1, 2))
            
            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)
        
        if self.vol_scaling:
            z[z>0] = 10**(z[z>0].log2())
            z[z<0] = -(10**((-z[z<0]).log2()))
        
        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        
        if hasattr(self, 'preempthasise') or return_CPU:
            z = z.cpu()
        
        if hasattr(self, 'preempthasise'):# apply inverse-preempthasis to audio signal (if used)
            for i in range(z.shape[0]): # (scipy signal is faster than pytorch implementation for some reason /shrug )
                z[i] = torch.from_numpy(signal.lfilter([1], [1, -float(self.preempthasis)], z[i].numpy())).to(z)
            if not return_CPU:
                z = z.cuda()
        
        return z, logdet
    
    @torch.no_grad()
    def infer(self, spect, speaker_ids=None, artifact_trimming=1, sigma=1., t_scaler=1.0, return_CPU=True):
        if len(spect.shape) == 2:
            spect = spect[None, ...] # [n_mel, T//hop_length] -> [B, n_mel, T//hop_length]
        if artifact_trimming > 0:
            spect = F.pad(spect, (0, artifact_trimming), value=-11.512925)
        
        batch_dim, n_mel_channels, steps = spect.shape # [B, n_mel, T//hop_length]
        samples = (steps - 1) * self.hop_length * t_scaler # T = T//hop_length * hop_length
        samples = int(samples - (samples%self.n_group))
        
        z = spect.new_empty((batch_dim, samples)) # [B, T]
        if sigma > 0:
            z.normal_(std=sigma)
        audio, _ = self.inverse(z, spect, speaker_ids, return_CPU=return_CPU)
        if artifact_trimming > 0:
            audio_trim = artifact_trimming*self.hop_length # amount of audio to trim
            audio = audio[:, :-audio_trim]
        return audio
    
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


if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt

    spect, sr = librosa.load(librosa.util.example_audio_file())
    # spect = librosa.feature.melspectrogram(spect=spect, sr=sr, n_fft=1024, hop_length=256, n_mel_channels=80)
    # print(spect.shape, spect.max())
    # plt.imshow(spect ** 0.1, aspect='auto', origin='lower')
    # plt.show()

    spect = torch.Tensor(spect)
    net = WaveGlow(12, 8, 4, 2, sr, 1024, 256, 80, n_layers=5, residual_channels=64, dilation_channels=64,
                   skip_channels=64, bias=True)
    # print(net)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    spect = net.get_mel(spect[None, ...])[0]
    print(spect.shape, spect.max())
    plt.imshow(spect.numpy(), aspect='auto', origin='lower')
    plt.show()

    audio = torch.rand(2, 16000) * 2 - 1
    z, *_ = net(audio)
    print(z.shape)

    audio = net.infer(spect[:, :10])
    print(audio.shape)
