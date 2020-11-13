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
from scipy.io.wavfile import read
from math import ceil, exp
import CookieTTS.utils.audio.stft as STFT

# utils
from waveglow_utils import PreEmphasis, InversePreEmphasis
from CookieTTS.utils.audio.iso226 import ISO_226

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
    def __init__(self, training_files, validation_files, validation_windows, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, load_mel_from_disk, preempthasis,
                 iso226_empthasis=False, check_files=False, load_hidden_from_disk=False, mel_from_disk_dtw=True,
                 use_gaussian_blur=False, gaussian_blur_min=0.0, gaussian_blur_max=2.0, 
                 dtw_scale_factor=5, dtw_range=5, min_log_std=-5.1,
                 load_from_disk_max_l1_err=None, load_from_disk_max_mse_err=None,
                 n_mel_channels=160, use_logvar_channels=False, logvar_gt_scale=0.0,
                 blend_with_load_from_disk_start_mel=False, blend_with_load_from_disk_end_mel=False):
        assert segment_length % hop_length == 0, 'segment_length must be n times hop_length'
        
        self.audio_files = load_filepaths_and_text(training_files)
        if check_files:
            print("Files before checking: ", len(self.audio_files))
            if True: # list comp non-verbose # this path is significantly faster
                # filter audio files that don't exist
                self.audio_files = [x for x in self.audio_files if os.path.exists(x[0])]
                assert len(self.audio_files), "self.audio_files is empty"
                
                # filter spectrograms that don't exist
                if load_mel_from_disk > 0.0:
                    self.audio_files = [x for x in self.audio_files if os.path.exists(x[1])]
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
                    
                    if load_mel_from_disk > 0.0 and not os.path.exists(file[1]): # check if mel exists
                        print(f"'{file[1]}' does not exist")
                        self.audio_files.remove(file); i_offset-=1; continue
                    
                    if 1:# performant mode if bitdepth is already known
                        bitdepth = 2
                        size = os.stat(file[0]).st_size
                        duration = size // bitdepth#duration in samples
                        if duration <= segment_length: # check if audio file is shorter than segment_length
                            #print(f"'{file[0]}' is too short")
                            self.audio_files.remove(file); i_offset-=1; continue
                    else:
                        audio_data, sample_r, *_ = load_wav_to_torch(file[0])
                        if audio_data.size(0) <= segment_length: # check if audio file is shorter than segment_length
                            print(f"'{file[0]}' is too short")
                            self.audio_files.remove(file); i_offset-=1; continue
            print("Files after checking: ", len(self.audio_files))
        
        self.load_hidden_from_disk = load_hidden_from_disk
        self.load_mel_from_disk = load_mel_from_disk
        self.speaker_ids = self.create_speaker_lookup_table(self.audio_files)
        
        # (optional) Apply weighting to MLP Datasets
        duplicated_audiopaths = [x for x in self.audio_files if "SlicedDialogue" in x[0]]
        for i in range(0):
            self.audio_files.extend(duplicated_audiopaths)
        
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = STFT.TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 n_mel_channels=n_mel_channels,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax) if not load_hidden_from_disk else None
        if iso226_empthasis:
            self.iso226 = ISO_226(sampling_rate=sampling_rate,
                                  filter_length=sampling_rate//40,# params picked based on quick listening test
                                  hop_length=sampling_rate//400,
                                  win_length=sampling_rate//40,
                                  stft_device='cpu')
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        
        self.use_logvar = use_logvar_channels
        self.logvar_gt_scale = logvar_gt_scale
        self.min_log_std = min_log_std
        self.use_gaussian_blur = use_gaussian_blur
        self.gaussian_blur_min = gaussian_blur_min
        self.gaussian_blur_max = gaussian_blur_max
        
        self.load_from_disk_dtw= mel_from_disk_dtw
        self.dtw_scale_factor= dtw_scale_factor
        self.dtw_range= dtw_range
        self.max_l1_err = load_from_disk_max_l1_err
        self.max_mse_err = load_from_disk_max_mse_err
        self.blend_with_load_from_disk_start_mel = blend_with_load_from_disk_start_mel
        self.blend_with_load_from_disk_end_mel = blend_with_load_from_disk_end_mel
    
    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_ids = np.sort(np.unique([x[2] for x in audiopaths_and_text]))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d
    
    def get_speaker_id(self, speaker_id):
        """Convert external speaker_id to internel [0 to max_speakers] range speaker_id"""
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])
    
    def get_mel(self, audio):
        """Take audio, normalize [-1 to 1] and convert to spectrogram"""
        audio_norm = audio / self.MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm).squeeze(0)
        return melspec
    
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
    
    def get_item(self, index, min_log_std=None):
        if min_log_std is None:
            min_log_std = self.min_log_std
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate, max_value = load_wav_to_torch(filename[0])
        self.MAX_WAV_VALUE = max(max_value, audio.max().item(), -audio.min().item()) # I'm not sure how, but sometimes the magnitude of audio exceeds the max of the datatype used before casting.
        assert audio.shape[0], f"Audio has 0 length.\nFile: {filename[0]}\nIndex: {index}"
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        
        if self.load_hidden_from_disk:
            # load hidden state context from disk
            hdn = np.load(filename[3])# [dim, T//hop_length]
            
            # offset the audio if the GTA spectrogram uses an offset
            #if ".hdn.npy" in filename[3] or (".hdn" in filename[3] and ".npy" in filename[3] and filename[3].split(".hdn")[1].split(".npy")[0]):
            if ".hdn" in filename[3] and ".npy" in filename[3] and filename[3].split(".hdn")[1].split(".npy")[0]:
                offset = int(filename[3].split(".hdn")[1].split(".npy")[0])
                audio = audio[offset:]
            
            # Take segment
            for i in range(20):
                audio_segment, hdn_segment, start_step, stop_step = self.get_segment(audio, hdn, self.segment_length, self.hop_length, n_channels=hdn.shape[0]) # get random segment of audio file
                if torch.std(audio_segment) > (exp(min_log_std)*self.MAX_WAV_VALUE): # if sample is not silent, use sample.
                    break
            else:
                print("No loud segments found, filename:", filename[0])
            audio, hdn = audio_segment, hdn_segment
            cond = torch.from_numpy(hdn).float()
        else:
            if random.random() < self.load_mel_from_disk: # load_mel_from_disk is now a probability instead of bool.
                # load mel from disk
                mel = np.load(filename[1])
                
                # offset the audio if the GTA spectrogram uses an offset
                #if ".mel.npy" in filename[1] or (".mel" in filename[1] and ".npy" in filename[1] and filename[1].split(".mel")[1].split(".npy")[0]):
                if ".mel" in filename[1] and ".npy" in filename[1] and filename[1].split(".mel")[1].split(".npy")[0]:
                    offset = int(filename[1].split(".mel")[1].split(".npy")[0])
                    audio = audio[offset:]
                
                # Take segment
                for i in range(20):
                    audio_segment, mel_segment, start_step, stop_step = self.get_segment(audio, mel, self.segment_length, self.hop_length, n_channels=self.n_mel_channels) # get random segment of audio file
                    if torch.std(audio_segment) > (exp(min_log_std)*self.MAX_WAV_VALUE): # if sample is not silent, use sample.
                        break
                else:
                    print("No loud segments found, filename:", filename[0])
                audio, mel = audio_segment, mel_segment
                
                mel = torch.from_numpy(mel).float()
                
                if self.use_logvar:
                    mel, logvar_mel = mel.chunk(2, dim=0)# [n_mel*2, T] -> [n_mel, T], [n_mel, T]
                else:
                    if mel.shape[0] == self.n_mel_channels*2:
                        mel = mel.chunk(2, dim=0)[0]# [n_mel*2, T] -> [n_mel, T]
                
                if self.load_from_disk_dtw:
                    gt_mel = self.get_mel(audio)
                    if self.max_l1_err and torch.nn.functional.l1_loss(mel, gt_mel) > self.max_l1_err:
                        raise FileNotSuitableException
                    if self.max_mse_err and torch.nn.functional.mse_loss(mel, gt_mel) > self.max_mse_err:
                        raise FileNotSuitableException
                    target_mel = gt_mel.clone()
                    pred_mel = mel.clone()
                    if self.blend_with_load_from_disk_start_mel and self.blend_with_load_from_disk_end_mel:
                        target_mel[self.blend_with_load_from_disk_end_mel:, :] = 0.0
                        pred_mel[self.blend_with_load_from_disk_end_mel:, :] = 0.0
                    mel = DTW(pred_mel.unsqueeze(0), target_mel.unsqueeze(0), scale_factor=self.dtw_scale_factor, range_=self.dtw_range).squeeze(0)# Pred mel Time-Warping to align more accurately with target audio
                    if self.blend_with_load_from_disk_start_mel and self.blend_with_load_from_disk_end_mel:
                        assert self.blend_with_load_from_disk_start_mel < mel.shape[0], "'blend_with_load_from_disk_start_mel' is larger than n_mel_channels"
                        assert self.blend_with_load_from_disk_end_mel < mel.shape[0], "'blend_with_load_from_disk_end_mel' is larger than n_mel_channels"
                        assert self.blend_with_load_from_disk_start_mel != self.blend_with_load_from_disk_end_mel, "'blend_with_load_from_disk_start_mel' is equal to 'blend_with_load_from_disk_end_mel'"
                        # blend ground truth and GTA spectrograms together. Higher frequencies will use GT, Lower Frequencies will use Pred
                        # Mel below start will use 100% Pred, Mel above end will use 100% GT
                        # Mels between will use a linearly shifting blend of the 2.
                        # Will linearly blend from min mel to max mel
                        mel_scales = ((torch.arange(1, self.n_mel_channels+1).float()-self.blend_with_load_from_disk_start_mel).clamp(0)/(self.blend_with_load_from_disk_end_mel-self.blend_with_load_from_disk_start_mel)).clamp(max=1.0)# Tensor starting with zeros, then linearly change from 0.0 to 1.0 while between start and end, then 1.0's for the remainder.
                        gt_mel_scales = mel_scales.unsqueeze(1)# [n_mel] -> [n_mel, 1]
                        pred_mel_scales = 1-(mel_scales.unsqueeze(1))# [n_mel] -> [n_mel, 1]
                        mel = mel*pred_mel_scales + gt_mel*gt_mel_scales # blend Pred and GT
                
                if self.use_logvar:
                    if self.logvar_gt_scale:# mix the predicted error with the **actual** error.
                        gt_logvar_mel = (torch.nn.functional.l1_loss(mel, gt_mel, reduction='none').pow(2)+1e-7).log()
                        logvar_mel = logvar_mel*(1-self.logvar_gt_scale) + gt_logvar_mel*self.logvar_gt_scale
                    mel = torch.cat((mel, logvar_mel), dim=0)# [n_mel, T] -> [n_mel*2, T]
                cond = mel
            else:
                # Take segment
                if audio.size(0) >= self.segment_length:
                    max_audio_start = audio.size(0) - self.segment_length
                    std = 9e9
                    for i in range(20):
                        audio_start = random.randint(0, max_audio_start)
                        audio_segment = audio[audio_start:audio_start + self.segment_length]
                        if torch.std(audio_segment) > (exp(min_log_std)*self.MAX_WAV_VALUE):# if sample is not silent, use sample for WaveGlow.
                            break
                    else:
                        print("No Loud Sample Found, filename:",filename[0])
                    audio = audio_segment
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
                assert audio.shape[0], f"Audio has 0 length.\nFile: {filename[0]}\nIndex: {index}"
                # generate mel from audio segment
                audio = audio - audio.mean()
                audio = audio.clamp(min=-1.*self.MAX_WAV_VALUE, max=(1.*self.MAX_WAV_VALUE)-1.)
                mel = self.get_mel(audio)
                cond = mel
        
        if self.use_gaussian_blur:
            cond = GaussianBlur(cond.unsqueeze(0), random.random()*(self.gaussian_blur_max-self.gaussian_blur_min) + self.gaussian_blur_min).squeeze(0)
        
        # normalize audio [-1 to 1]
        audio = audio / self.MAX_WAV_VALUE
        
        if hasattr(self, 'iso226'): # (optional) apply frequency weighting
            audio = self.iso226(audio.unsqueeze(0)).squeeze(0)
        
        speaker_id = self.get_speaker_id(filename[2])
        cond, audio, speaker_id = cond.contiguous(), audio.contiguous(), speaker_id.contiguous()
        return (cond, audio, speaker_id) # (mel, audio, speaker_id)
    
    def __getitem__(self, index, min_log_std=None):
        file_acquired = False
        attempts = 0
        while not file_acquired:
            try:
                item = self.get_item(index, min_log_std)
                file_acquired = True
            except FileNotSuitableException as ex:
                index = int(random.random()*len(self.audio_files)-0.1)
                attempts+=1
                if attempts == 11:
                    print("More than 10 files with Error greater than threshold.")
        return item

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
