import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_util import add_weight_norms
import numpy as np

from efficient_modules import AffineCouplingBlock, InvertibleConv1x1


def permute_height(x, reverse=False, bipart=False, shift=False, inverse_shift=False):
    x = torch.split(x, 1, dim=2)
    if bipart and reverse:
        half = len(x)//2
        x = x[:half][::-1] + x[half:][::-1] # reverse H halfs [0,1,2,3,4,5,6,7] -> [3,2,1,0] + [7,6,5,4] -> [3,2,1,0,7,6,5,4]
    elif reverse:
        x = x[::-1] # reverse entire H [0,1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1,0]
    else:
        raise NotImplementedError
    if shift:
        x = (x[-1],) + x[:-1] # shift last H into first position [0,1,2,3,4,5,6,7] -> [7,0,1,2,3,4,5,6]
    if inverse_shift:
        x = x[1:] + (x[0],)   # shift first H into last position [0,1,2,3,4,5,6,7] -> [1,2,3,4,5,6,7,0]
    return torch.stack(x, dim=2).squeeze(-1)


class WaveGlow(nn.Module):
    def __init__(self, yoyo, yoyo_WN, n_mel_channels, n_flows, n_group, n_early_every,
                n_early_size, memory_efficient, spect_scaling, upsample_mode, WN_config, win_length, hop_length):
        super(WaveGlow, self).__init__()
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.win_size = win_length
        self.hop_length = hop_length
        self.n_mel_channels = n_mel_channels
        
        self.multispeaker = WN_config['speaker_embed_dim'] > 0
        
        if yoyo_WN:
            from efficient_modules import WN
        else:
            from glow import WN
        
        self.upsample_factor = hop_length // n_group
        sub_win_size = win_length // n_group
        # self.upsampler = nn.ConvTranspose1d(n_mel_channels, n_mel_channels, sub_win_size, self.upsample_factor,
        #                                    padding=sub_win_size // 2, bias=False)
        
        self.convinv = nn.ModuleList()
        self.WN = nn.ModuleList()
        
        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        self.z_split_sizes = []
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels -= n_early_size
                self.z_split_sizes.append(n_early_size)

            assert n_remaining_channels > 0 # no n_group remaining

            self.convinv.append(
                InvertibleConv1x1(n_remaining_channels, memory_efficient=memory_efficient))
            
            if yoyo_WN:
                self.WN.append(
                    AffineCouplingBlock(WN, memory_efficient=memory_efficient, in_channels=n_remaining_channels // 2,
                                    aux_channels=n_mel_channels, **WN_config))
            else:
                self.WN.append(
                    AffineCouplingBlock(WN, memory_efficient=memory_efficient,
                        n_in_channels=n_remaining_channels//2, n_mel_channels=n_mel_channels, **WN_config))
        self.z_split_sizes.append(n_remaining_channels)
    
    
    def forward(self, spect, audio, speaker_ids=None): # optional spect input
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        #  Upsample spectrogram to size of audio
        spect = self._upsample_mels(spect) # [B, mels, T//hop_length] -> [B, mels, T//n_group]
        
        batch_dim, n_mel_channels, group_steps = spect.shape
        audio = audio.view(batch_dim, -1, self.n_group).transpose(1, 2) # [B, H, T//n_group]
        
        assert audio.size(2) <= spect.size(2)
        spect = spect[..., :audio.size(2)] # [B, mels, T//hop_length*n_group]
        
        audio = audio.unsqueeze(1) # move channel dim to H
        # [B, 1, H, T//n_group]
        
        output_audio = []
        split_sections = [self.n_early_size, self.n_group]
        for k, (convinv, affine_coup) in enumerate(zip(self.convinv, self.WN)):
            if k % self.n_early_every == 0 and k > 0:
                split_sections[1] -= self.n_early_size
                early_output, audio = audio.split(split_sections, 2)
                # these 2 lines actually copy tensors, may need optimization in the future
                output_audio.append(early_output)
                audio = audio.clone()
            
            audio, log_det_W = convinv(audio) # Mix H info [B, 1, H, T//n_group] -> [B, 1, H, T//n_group]
            
            audio, log_s = affine_coup(audio, spect, speaker_ids=speaker_ids)
            if k:
                logdet += log_det_W + log_s.sum((1, 2))
            else:
                logdet = log_det_W + log_s.sum((1, 2))
        
        assert split_sections[1] == self.z_split_sizes[-1]
        output_audio.append(audio)
        return torch.cat(output_audio, 1).transpose(1, 2).contiguous().view(batch_dim, -1), logdet
    
    def _upsample_mels(self, spect):
        spect = F.pad(spect, (0, 1))
        return F.interpolate(spect, size=((spect.size(2) - 1) * self.upsample_factor + 1,), mode='linear')
        # return self.upsampler(spect)
    
    def inverse(self, z, spect, speaker_ids=None):
        spect = self._upsample_mels(spect)
        batch_dim, n_mel_channels, group_steps = spect.shape
        z = z.view(batch_dim, -1, self.n_group).transpose(1, 2)
        assert z.size(2) <= spect.size(2)
        spect = spect[..., :z.size(2)]
        
        remained_z = []
        for r in z.split(self.z_split_sizes, 1):
            remained_z.append(r.clone())
        *remained_z, z = remained_z
        
        for k, invconv, affine_coup in zip(range(self.n_flows - 1, -1, -1), self.convinv[::-1], self.WN[::-1]):
            
            z, log_s = affine_coup.inverse(z, spect, speaker_ids=speaker_ids)
            z, log_det_W = invconv.inverse(z)
            
            if k == self.n_flows - 1:
                logdet = log_det_W + log_s.sum((1, 2))
            else:
                logdet += log_det_W + log_s.sum((1, 2))
            
            if k % self.n_early_every == 0 and k:
                z = torch.cat((remained_z.pop(), z), 1)
        
        z = z.transpose(1, 2).contiguous().view(batch_dim, -1)
        return z, logdet
    
    @torch.no_grad()
    def infer(self, spect, speaker_ids=None, sigma=1.):
        if len(spect.shape) == 2:
            spect = spect[None, ...]
        
        batch_dim, n_mel_channels, steps = spect.shape
        samples = steps * self.hop_length
        
        z = spect.new_empty((batch_dim, samples)).normal_(std=sigma)
        # z = torch.randn(batch_dim, self.n_group, group_steps, dtype=spect.dtype, device=spect.device).mul_(sigma)
        audio, _ = self.inverse(z, spect, speaker_ids)
        return audio


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
