import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
try:
    import iso226
except Exception as ex:
    # if iso226 missing,
    print(ex)
    print("iso226 package missing, attempting install using python3 -m pip from 'https://github.com/jacobbaylesssmc/iso226'.")
    os.system("git clone https://github.com/jacobbaylesssmc/iso226; cd iso226; python3 -m pip install ./")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CookieTTS.utils.audio.stft import STFT

class ISO_226(torch.nn.Module):
    def __init__(self, sampling_rate=48000, filter_length=2400, hop_length=600, win_length=2400, stft_device='cpu'):
        super(ISO_226, self).__init__()
        self.stft_device = stft_device
        self.stft = STFT(filter_length=filter_length,
                         hop_length=hop_length,
                         win_length=win_length).to(device=self.stft_device)
        
        iso226_spl_from_freq = iso226.iso226_spl_itpl(L_N=60, hfe=True)# get InterpolatedUnivariateSpline for Perc Sound Pressure Level at Difference Frequencies with 60DB ref.
        self.freq_weights = torch.tensor([(10**(60./10))/(10**(iso226_spl_from_freq(freq)/10)) for freq in np.linspace(0, sampling_rate//2, (filter_length//2)+1)])
        self.freq_weights = self.freq_weights.to(self.stft_device)[None, :, None]# [B, n_mel, T]
        freq_weights = self.freq_weights.clone()
        freq_weights[freq_weights<0.008] = 1e5
        self.inv_freq_weights = 1/freq_weights
    
    def forward(self, in_audio):
        with torch.no_grad():
            in_audio_device = in_audio.device
            audio = in_audio.to(self.stft_device).float()
            audio_spec, audio_angles = self.stft.transform(audio)
            audio_spec *= self.freq_weights
            audio = self.stft.inverse(audio_spec, audio_angles).squeeze(1)
            if in_audio_device != self.stft_device:
                audio.to(in_audio)
            return audio
    
    def inverse(self, in_audio):
        with torch.no_grad():
            in_audio_device = in_audio.device
            audio = in_audio.to(self.stft_device).float()
            audio_spec, audio_angles = self.stft.transform(audio)
            audio_spec *= self.inv_freq_weights
            audio = self.stft.inverse(audio_spec, audio_angles).squeeze(1)
            if in_audio_device != self.stft_device:
                audio.to(in_audio)
            return audio