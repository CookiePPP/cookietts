import sys
sys.path.append('tacotron2')
import torch
from CookieTTS.utils.audio.stft import STFT


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, sampling_rate=48000, filter_length=None, hop_length=None,
                 win_length=None, n_mel_channels=160, n_frames=20, mu=0, var=0.01, wg_sigma=0.01, stft_device='cpu', speaker_dependant=False, speaker_id=0):
        super(Denoiser, self).__init__()
        self.stft_device = stft_device
        if filter_length is None:
            filter_length = sampling_rate//40
        if win_length is None:
            win_length = sampling_rate//40
        if hop_length is None:
            hop_length = sampling_rate//400
        self.stft = STFT(filter_length=filter_length,
                         hop_length=hop_length,
                         win_length=win_length).to(device=self.stft_device)
        
        mel_input = torch.randn(
            (1, n_mel_channels, n_frames),
            dtype=next(waveglow.parameters()).dtype,
            device=next(waveglow.parameters()).device)
        mel_input = mel_input * float(var) + float(mu)
        
        with torch.no_grad():
            if speaker_dependant:# calculate a seperate bias for each speaker
                if hasattr(waveglow, 'speaker_embed'):
                    n_speakers = waveglow.speaker_embed.num_embeddings
                elif hasattr(waveglow, 'WN') and hasattr(waveglow.WN[0].WN, 'speaker_embed'):
                    n_speakers = waveglow.WN[0].WN.speaker_embed.num_embeddings
                else:
                    n_speakers = 1
                speaker_id = torch.zeros(1, device=mel_input.device, dtype=torch.int64)
                bias_audio = waveglow.infer(mel_input, speaker_ids=speaker_id, sigma=wg_sigma)
                bias_audio = bias_audio.to(device=self.stft_device, dtype=torch.float).repeat(n_speakers, 1)# [1, T] -> [n_speakers, T]
                for speaker_id in range(1, n_speakers):
                    speaker_id = torch.tensor([speaker_id,], device=mel_input.device, dtype=torch.int64)
                    bias_audio_ = waveglow.infer(mel_input, speaker_ids=speaker_id, sigma=wg_sigma).to(device=self.stft_device, dtype=torch.float)
                    assert not torch.isinf(bias_audio_).any(), 'Inf elements found in Vocoder Output'
                    assert not torch.isnan(bias_audio_).any(), 'NaN elements found in Vocoder Output'
                    bias_audio[speaker_id] = bias_audio_
                bias_spec = self.stft.transform(bias_audio)[0]# -> [n_speakers, n_mel, dec_T]
            else:# use the same bias for each speaker
                speaker_id = torch.tensor([speaker_id,], device=mel_input.device, dtype=torch.int64)
                bias_audio = waveglow.infer(mel_input, speaker_ids=speaker_id, sigma=wg_sigma).to(device=self.stft_device, dtype=torch.float)
                assert not torch.isinf(bias_audio).any(), 'Inf elements found in Vocoder Output'
                assert not torch.isnan(bias_audio).any(), 'NaN elements found in Vocoder Output'
                bias_spec, _ = self.stft.transform(bias_audio)
            assert not torch.isinf(bias_spec).any(), 'Inf elements found in bias_spec'
            assert not torch.isnan(bias_spec).any(), 'NaN elements found in bias_spec'
        
        self.register_buffer('bias_spec', bias_spec.mean(dim=2, keepdim=True))# [n_speakers, n_mel, 1]
    
    def forward(self, wg_audio, speaker_ids=None, strength=0.1):
        with torch.no_grad():
            wg_audio_device = wg_audio.device
            audio = wg_audio.to(self.stft_device).float()
            audio_spec, audio_angles = self.stft.transform(audio)
            if speaker_ids is None or self.bias_spec.shape[0] == 1:
                audio_spec_denoised = audio_spec - self.bias_spec * strength
            else:
                audio_spec_denoised = audio_spec - self.bias_spec[speaker_ids] * strength
            audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
            audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
            if wg_audio_device != self.stft_device:
                audio_denoised.to(wg_audio)
            return audio_denoised