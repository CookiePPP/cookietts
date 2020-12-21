"""
BSD 3-Clause License

Copyright (c) 2017, Prem Seetharaman
All rights reserved.

* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
from CookieTTS.utils.audio.audio_processing import window_sumsquare, dynamic_range_compression, dynamic_range_decompression
from CookieTTS.utils.dataset.utils import load_wav_to_torch


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann', dtype=torch.float32):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])
        
        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window
        
        self.register_buffer('forward_basis', forward_basis.float().to(dtype))
        self.register_buffer('inverse_basis', inverse_basis.float().to(dtype))
    
    @torch.jit.script
    def transform_jit(input_data: torch.Tensor,
                      forward_basis: torch.Tensor,
                      filter_length: int,
                      hop_length: int,
                      return_phase: bool
                      ):
        # input audio samples [B, T]
        input_data_shape = input_data.size()
        num_batches = input_data_shape[0]
        num_samples = input_data_shape[1]
        
        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples) # [B, T] -> [B, 1, T]
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1), # [B, 1, 1, T]
            (int(filter_length / 2), int(filter_length / 2), 0, 0), # padding half filterlen to each side
            mode='reflect')
        input_data = input_data.squeeze(1) # [B, 1, 1, T] -> [B, 1, T+filter_length]
        
        forward_transform = torch.nn.functional.conv1d(
            input_data.to(forward_basis), # [B, 1, T+filter_length]
            forward_basis,
            stride=hop_length, # apply hann windowed conv
            padding=0) # [B, filter_length, T//hop_length + 1]
        
        cutoff = int((filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        
        magnitude = (real_part**2 + imag_part**2).sqrt()
        phase = torch.atan2(imag_part, real_part) if return_phase else None
        return magnitude, phase
    
    def transform(self, input_data, return_phase=True):
        magnitude, phase = self.transform_jit(input_data, self.forward_basis, self.filter_length, self.hop_length, return_phase)
        return magnitude, phase
    
    # below is a clone of transform_jit and transform
    # This is done because there seems to be a bug where if you call a torch.jit.script function with a GPU,
    # then it'll always try to initialize CUDA when that function is called,
    # even when a dataloader worker calls the same function from another device...
    #@torch.jit.script
    def transform_nonjit(self, input_data: torch.Tensor,
                      forward_basis: torch.Tensor,
                      filter_length: int,
                      hop_length: int,
                      return_phase: bool
                      ):
        # input audio samples [B, T]
        input_data_shape = input_data.size()
        num_batches = input_data_shape[0]
        num_samples = input_data_shape[1]
        
        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples) # [B, T] -> [B, 1, T]
        input_data = torch.nn.functional.pad(
            input_data.unsqueeze(1), # [B, 1, 1, T]
            (int(filter_length / 2), int(filter_length / 2), 0, 0), # padding half filterlen to each side
            mode='reflect')
        input_data = input_data.squeeze(1) # [B, 1, 1, T] -> [B, 1, T+filter_length]
        
        forward_transform = torch.nn.functional.conv1d(
            input_data.to(forward_basis), # [B, 1, T+filter_length]
            forward_basis,
            stride=hop_length, # apply hann windowed conv
            padding=0) # [B, filter_length, T//hop_length + 1]
        
        cutoff = int((filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        
        magnitude = (real_part**2 + imag_part**2).sqrt()
        phase = torch.atan2(imag_part, real_part) if return_phase else None
        return magnitude, phase
    
    def transform_gpu(self, input_data, return_phase=True):
        magnitude, phase = self.transform_nonjit(input_data, self.forward_basis, self.filter_length, self.hop_length, return_phase)
        return magnitude, phase
    
    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)
        
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)
        
        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            
            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length
        
        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform
    
    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0, clamp_val=1e-5, stft_dtype=torch.float32):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate  = sampling_rate
        self.mel_fmax = mel_fmax
        self.clip_val = clamp_val
        self.stft_fn  = STFT(filter_length, hop_length, win_length, dtype=stft_dtype)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
    
    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes, clip_val=self.clip_val)
        return output
    
    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes, clip_val=self.clip_val)
        return output
    
    def get_mel_from_path(self, audiopath):
        audio = load_wav_to_torch(audiopath, target_sr=self.sampling_rate)[0]
        return self.mel_spectrogram(audio.unsqueeze(0))
    
    @torch.no_grad()
    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y) >= -1.), f'Tensor.min() of {torch.min(y).item()} is less than -1.0'
        assert(torch.max(y) <=  1.), f'Tensor.max() of {torch.max(y).item()} is greater than 1.0'
        
        if y.device != 'cpu':# done for compatibility, if this jit function is called with CUDA (even from different Objects on different threads on difference devices!), it will attempt to initialize CUDA on every call to this function.
            magnitudes = self.stft_fn.transform_gpu(y, return_phase=False)[0] # get magnitudes at each (overlapped) window [B, T] ->  # [B, filter_len, T//hop_length+1]
        else:
            magnitudes = self.stft_fn.transform(y, return_phase=False)[0] # get magnitudes at each (overlapped) window [B, T] ->  # [B, filter_len, T//hop_length+1]
        
        mag_shape = magnitudes.shape# [B, n_mel, T//hop_length+1]
        if False:#mag_shape[0] == 1:# do sparse op if possible. Pytorch 1.6 required for sparse with batches.
            mel_basis = self.mel_basis.to_sparse()
            mel_output = torch.mm(mel_basis, magnitudes.squeeze(0)).unsqueeze(0)
                      # [n_mel, filter_len] @ [filter_len, T//hop_length+1] -> [1, n_mel, T//hop_length+1]
        else:
            mel_basis = self.mel_basis.unsqueeze(0).repeat(mag_shape[0], 1, 1)
            mel_output = torch.bmm(mel_basis, magnitudes)# [B, n_mel, filter_len] @ [B, filter_len, T//hop_length+1] -> [B, n_mel, T//hop_length+1]
                                                         # This op uses an emourmous amount of contiguous memory with 2400 filter len and 160 n_mel.
        
        mel_output = self.spectral_normalize(mel_output) # clamp min to 1e-5 and convert magnitudes to natural log scale
        return mel_output # [B, n_mel, T+//hop_length + 1]