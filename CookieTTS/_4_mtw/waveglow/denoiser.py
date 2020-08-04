import sys
sys.path.append('tacotron2')
import torch
from layers import STFT


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length=2400, n_overlap=4,
                 win_length=2400, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length/n_overlap),
                         win_length=win_length).cuda()
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 160, 88),
                dtype=next(waveglow.parameters()).dtype,
                device=next(waveglow.parameters()).device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 160, 88),
                dtype=next(waveglow.parameters()).dtype,
                device=next(waveglow.parameters()).device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            speaker_id = torch.zeros(1, device=mel_input.device, dtype=torch.int64)
            bias_audio = waveglow.infer(mel_input, speaker_ids=speaker_id, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])
    
    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
