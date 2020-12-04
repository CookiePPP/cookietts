import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import torch

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


def get_spect(audio):
    if type(audio) == torch.Tensor:
        output_torch = True
        audio = audio.cpu().numpy()
    else:
        output_torch = False
    
    # Remove drifting noise
    audio = signal.filtfilt(b, a, audio)
    
    # Ddd a little random noise for model roubstness
    audio = audio * 0.96 + (np.random.rand(audio.shape[0])-0.5)*1e-06
    
    # Compute spect
    D = pySTFT(audio).T
    
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    melspect = np.clip((D_db + 100) / 100, 0, 1)    
    
    if output_torch:
        melspect = torch.from_numpy(melspect)
    return melspect