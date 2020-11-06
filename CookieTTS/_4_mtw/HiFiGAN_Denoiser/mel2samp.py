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
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
import random
import argparse
import json
import torch
import torch.nn.functional as F
import torch.utils.data
import sys
import numpy as np
import librosa
from scipy.io.wavfile import read
from math import ceil, exp
from glob import glob

class FileNotSuitableException(Exception):
    """Custom Exception Class."""
    pass

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    max_value = np.iinfo(data.dtype).max
    return torch.from_numpy(data).float(), sampling_rate, max_value


def get_mel_from_file(mel_path):
    melspec = np.load(mel_path)
    melspec = torch.autograd.Variable(torch.from_numpy(melspec), requires_grad=False)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def GaussianBlur(inp, blur_strength=1.0):
    inp = inp.unsqueeze(1) # [B, height, width] -> [B, 1, height, width]
    var_ = blur_strength
    norm_dist = torch.distributions.normal.Normal(0, var_)
    conv_kernel = torch.stack([norm_dist.cdf(torch.tensor(i+0.5)) - norm_dist.cdf(torch.tensor(i-0.5)) for i in range(int(-var_*3),int(var_*3+1))], dim=0)[None, None, :, None]
    input_padding = (conv_kernel.shape[2]-1)//2
    out = F.conv2d(F.pad(inp, (0,0,input_padding,input_padding), mode='reflect'), conv_kernel).squeeze(1) # [B, 1, height, width] -> [B, height, width]
    return out


@torch.jit.script
def DTW(batch_pred, batch_target, scale_factor: int, range_: int):
    """
    Calcuates ideal time-warp for each frame to minimize L1 Error from target.
    Params:
        pred: [B, ?, T] FloatTensor
        target: [B, ?, T] FloatTensor
        scale_factor: Scale factor for linear interpolation.
                      Values greater than 1 allows blends neighbouring frames to be used.
        range_: Range around the target frame that predicted frames should be tested as possible candidates to output.
                If range is set to 1, then predicted frames with more than 0.5 distance cannot be used. (where 0.5 distance means blending the 2 frames together).
    Returns:
        pred_dtw: [B, ?, T] FloatTensor. Aligned copy of pred which should more closely match the target.
    """
    assert range_ % 2 == 1, 'range_ must be an odd integer.'
    assert batch_pred.shape == batch_target.shape, 'pred and target shapes must match.'
    
    batch_pred_dtw = batch_pred * 0.
    for i, (pred, target) in enumerate(zip(batch_pred, batch_target)):
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
        # shift pred into all aligned forms that might produce improved L1
        pred_pad = torch.nn.functional.pad(pred, (range_//2, range_//2))
        pred_expanded = torch.nn.functional.interpolate(pred_pad, scale_factor=float(scale_factor), mode='linear', align_corners=False)# [B, C, T] -> [B, C, T*s]
        
        p_shape = pred.shape
        pred_list = []
        for j in range(scale_factor*range_):
            pred_list.append(pred_expanded[:,:,j::scale_factor][:,:,:p_shape[2]])
        
        pred_dtw = pred.clone()
        for pred_interpolated in pred_list:
            new_l1 = torch.nn.functional.l1_loss(pred_interpolated, target, reduction='none').sum(dim=1, keepdim=True)
            old_l1 = torch.nn.functional.l1_loss(pred_dtw, target, reduction='none').sum(dim=1, keepdim=True)
            pred_dtw = torch.where(new_l1 < old_l1, pred_interpolated, pred_dtw)
        batch_pred_dtw[i:i+1] = pred_dtw
    return batch_pred_dtw


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, validation_files, noise_folders, segment_length, sampling_rate, check_files=False,
            min_log_std=-5.1,
            min_SNR=10, max_SNR=30,
            min_white_noise_log10_std=-4.0, max_white_noise_log10_std=-1.0,
            min_augmented_sample_rate=22050, max_augmented_sample_rate=48000
            ):
        self.noise_files = []
        for noise_folder in noise_folders:
            self.noise_files.extend( glob(os.path.join(noise_folder, '**','*.wav'), recursive=True) )
        
        self.audio_files = load_filepaths_and_text(training_files)
        if check_files:
            print("Files before checking: ", len(self.audio_files))
            if True: # list comp non-verbose # this path is significantly faster
                # filter audio files that don't exist
                self.audio_files = [x for x in self.audio_files if os.path.exists(x[0])]
                assert len(self.audio_files), "self.audio_files is empty"
                
                # filter audio files that are too short
                self.audio_files = [x for x in self.audio_files if (os.stat(x[0]).st_size//2) >= segment_length+600]
                assert len(self.audio_files), "self.audio_files is empty"
            else: # forloop with verbose support
                i = 0
                i_offset = 0
                for i_ in range(len(self.audio_files)):
                    i = i_ + i_offset
                    if i == len(self.audio_files): break
                    file = self.audio_files[i]
                    
                    if not os.path.exists(file[0]): # check if audio file exists
                        print(f"'{file[0]}' does not exist")
                        self.audio_files.remove(file); i_offset-=1; continue
                    
                    if 1:# performant mode if bitdepth is already known
                        bitdepth = 2
                        size = os.stat(file[0]).st_size
                        duration = size // bitdepth#duration in samples
                        if duration <= segment_length: # check if audio file is shorter than segment_length
                            #print(f"'{file[0]}' is too short")
                            self.audio_files.remove(file); i_offset-=1; continue
                    else:
                        audio_data, sample_r = load_wav_to_torch(file[0])
                        if audio_data.size(0) <= segment_length: # check if audio file is shorter than segment_length
                            print(f"'{file[0]}' is too short")
                            self.audio_files.remove(file); i_offset-=1; continue
            print("Files after checking: ", len(self.audio_files))
        
        self.speaker_ids = self.create_speaker_lookup_table(self.audio_files)
        
        # (optional) Apply weighting to MLP Datasets
        duplicated_audiopaths = [x for x in self.audio_files if "SlicedDialogue" in x[0]]
        for i in range(0):
            self.audio_files.extend(duplicated_audiopaths)
        
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.min_log_std = min_log_std
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR
        self.min_white_noise_log10_std = min_white_noise_log10_std
        self.max_white_noise_log10_std = max_white_noise_log10_std
        self.min_augmented_sample_rate = min_augmented_sample_rate
        self.max_augmented_sample_rate = max_augmented_sample_rate
    
    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d
    
    def get_speaker_id(self, speaker_id):
        """Convert external speaker_id to internel [0 to max_speakers] range speaker_id"""
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])
    
    def get_segment(self, audio, mel, segment_length, hop_length, n_channels=160):
        """get audio and mel segment from an already generated spectrogram and audio."""
        mel_segment_length = int(segment_length/hop_length)+1 # 8400/600 + 1 = 15
        if audio.size(0) >= segment_length:
            max_mel_start = int((audio.size(0)-segment_length)/hop_length) - 1 # mel.size(1) - mel_segment_length
            mel_start = random.randint(0, max_mel_start) if max_mel_start > 0 else 0
            audio_start = mel_start*hop_length
            audio = audio[audio_start:audio_start + segment_length]
            mel = mel[:,mel_start:mel_start + mel_segment_length]
        else:
            mel_start = 0
            len_pad = int((segment_length/ hop_length) - mel.shape[1])
            pad = np.ones((n_channels, len_pad), dtype=np.float32) * -11.512925
            mel =  np.append(mel, pad, axis=1)
            audio = torch.nn.functional.pad(audio, (0, segment_length - audio.size(0)), 'constant').data
        return audio, mel, mel_start, mel_start + mel_segment_length
    
    def noisify_audio(self, audio):
        noisy_audio = audio.clone()
        
        # Get audio file
        looking_for_long_enough_audio_file = True
        while looking_for_long_enough_audio_file:
            noise_path = random.sample(self.noise_files, 1)[0] # get random noisy audio file path
            noise_audio, noise_sr = load_wav_to_torch(noise_path) # load audio file
            noise_audio = torch.from_numpy(librosa.core.resample(noise_audio.numpy(), noise_sr, self.sampling_rate, type='scipy'))
            if noise_audio.shape[0] > audio.shape[0]: # if noisy audio is file is longer than clean audio, exit While loop.
                looking_for_long_enough_audio_file = False
        
        max_start = noise_audio.shape[0] - audio.shape[0]
        start = int(random.uniform(0, max_start))
        noise_audio = noise_audio[start:start+audio.shape[0]] # get random segment
        
        # adjust volume of noise to random levels between min and max SNR.
        SNRdb = random.uniform(self.min_SNR, self.max_SNR)
        target_SNR = 10.**(SNRdb/10.)
        noise_audio_RMS = (noise_audio - noise_audio.mean()).pow(2).sum().pow(0.5)
        audio_RMS = (audio - audio.mean()).pow(2).sum().pow(0.5)
        current_SNR = (audio_RMS**2)/(noise_audio_RMS**2)
        delta_SNR = current_SNR/target_SNR
        noise_audio *= (delta_SNR**0.5)
        
        # Lazy Low-Pass using Scipy/Librosa
        augmented_sample_rate = round(random.uniform(self.min_augmented_sample_rate, self.max_augmented_sample_rate))
        noisy_audio = librosa.core.resample(noisy_audio.numpy(), self.sampling_rate, augmented_sample_rate, res_type='kaiser_best')
        noisy_audio = torch.from_numpy(librosa.core.resample(noisy_audio, augmented_sample_rate, self.sampling_rate, res_type='kaiser_best'))[:audio.shape[0]]
        
        noisy_audio += torch.randn(*audio.shape)*(10**random.uniform(self.min_white_noise_log10_std, self.max_white_noise_log10_std))# (optional) Add white noise
        noisy_audio += noise_audio # (optional) Add Noise from "noise_folders"
        return noisy_audio.clamp(min=-1.0, max=1.0)
    
    def get_from_path(self, audiopath, segment_length=None, min_log_std=None):
        if min_log_std is None:
            min_log_std = self.min_log_std
        if segment_length is None:
            segment_length = self.segment_length
        audio, sampling_rate, max_value = load_wav_to_torch(audiopath)
        self.MAX_WAV_VALUE = max(max_value, audio.max().item(), -audio.min().item()) # I'm not sure how, but sometimes the magnitude of audio exceeds the max of the datatype used before casting.
        assert audio.shape[0], f"Audio has 0 length.\nFile: {audiopath}\nIndex: {index}"
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        
        max_audio_start = max(audio.size(0) - segment_length, 0)
        std = 9e9
        for i in range(3):
            audio_start = random.randint(0, max_audio_start) if max_audio_start else 0
            audio_segment = audio[audio_start:audio_start + segment_length]
            if torch.std(audio_segment) > (exp(min_log_std)*self.MAX_WAV_VALUE):
                break
        else:
            print("No Loud Sample Found, filename:", audiopath)
        audio = audio_segment
        assert audio.shape[0], f"Audio has 0 length.\nFile: {audiopath}\nIndex: {index}"
        
        # generate mel from audio segment
        audio = audio.clamp(min=-1.*self.MAX_WAV_VALUE, max=(1.*self.MAX_WAV_VALUE)-1.)
        
        # normalize audio [-1 to 1]
        audio /= self.MAX_WAV_VALUE
        
        noisy_audio = self.noisify_audio(audio)
        return noisy_audio, audio
    
    def get_item(self, index, min_log_std=None):
        filename = self.audio_files[index] # get Filelist line
        noisy_audio, audio = self.get_from_path(filename[0])# Read audio
        
        speaker_id = self.get_speaker_id(filename[2])
        noisy_audio, audio, speaker_id = noisy_audio.contiguous(), audio.contiguous(), speaker_id.contiguous()
        return (noisy_audio, audio, speaker_id) # (mel, audio, speaker_id)
    
    def __getitem__(self, index, min_log_std=None):
        return self.get_item(index, min_log_std)

    def __len__(self):
        return len(self.audio_files)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
