import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from CookieTTS._4_mtw.hifigan.nvSTFT import load_wav_to_torch
from CookieTTS._4_mtw.hifigan.nvSTFT import STFT as STFT_Class
from CookieTTS.utils.dataset.data_utils import DTW
from glob import glob
from tqdm import tqdm
try:
    import pyworld as pw
except:
    pw = None

def check_file_lengths(sampling_rate, segment_size, training_files_old):
    segment_size_s = segment_size/sampling_rate
    training_files_new = []
    for file in tqdm(training_files_old):
        audio, native_sr = load_wav_to_torch(file, min_sr=sampling_rate*0.9, target_sr=None, return_empty_on_exception=True)
        audio_s = len(audio) / native_sr
        if audio_s > segment_size_s:
            training_files_new.append(file)
    return training_files_new

def check_files(sampling_rate, segment_size, training_files):
    len_training_files = len(training_files)
    training_files = [x for x in training_files if os.path.exists(x)]
    if (len_training_files - len(training_files)) > 0:
        print(len_training_files - len(training_files), "Audio Files don't exist (and have been removed from training)")
    
    # check for fine-tuning audio files
    len_training_files = len(training_files)
    training_files = [x for x in training_files if os.path.exists(os.path.splitext(x)[0]+'.pm_audio.pt')]
    if (len_training_files - len(training_files)) > 0:
        print(len_training_files - len(training_files), "GTA Audio Files don't exist (and have been removed from training)")
    
    # check for fine-tuning postnet spectrograms.
    len_training_files = len(training_files)
    training_files = [x for x in training_files if os.path.exists(os.path.splitext(x)[0]+'.pred_mel.pt')]
    if (len_training_files - len(training_files)) > 0:
        print(len_training_files - len(training_files), "GTA Pred Spect Files don't exist (and have been removed from training)")
    
    # check audio files are sufficient length for Vocoder.
    len_training_files = len(training_files)
    training_files = check_file_lengths(sampling_rate, segment_size, training_files)
    if (len_training_files - len(training_files)) > 0:
        print(len_training_files - len(training_files), "Files are too short (and have been removed from training)")
    return training_files
    

def get_dataset_filelist(a, segment_size, sampling_rate):
    if a.input_wavs_dir is None:
        with open(a.input_training_file, 'r', encoding='utf-8') as fi:
            training_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

        with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
            validation_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    else:
        print("Searching for WAV files in '--input_wav_dir' arg...")
        wav_files = sorted(glob(os.path.join(a.input_wavs_dir, '**', '*.wav'), recursive=True))
        print(f"Found {len(wav_files)} WAV Files.")
        random.Random(1).shuffle(wav_files)
        
        training_files   = wav_files[:int(len(wav_files)*0.95) ]
        validation_files = wav_files[ int(len(wav_files)*0.95):]
    
    if not a.skip_file_checks:
        print("Checking files")
        training_files   = check_files(sampling_rate, segment_size, training_files)
        validation_files = check_files(sampling_rate, segment_size, validation_files)
    
    return training_files, validation_files

def get_nonzero_indexes(voiced):# get first and last zero index in array/1d tensor
    start_indx = 0
    for i in range(len(voiced)):
        if voiced[i] != 0:
            start_indx = i
            break
    end_indx = len(voiced)
    for i in reversed(range(len(voiced))):
        if voiced[i] != 0:
            end_indx = i
            break
    return start_indx, end_indx

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, trim_non_voiced=False):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.STFT = STFT_Class(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
        self.device = device
        self.fine_tuning = fine_tuning
        self.trim_non_voiced = trim_non_voiced

    def get_pitch(self, audio):
        # Extract Pitch/f0 from raw waveform using PyWORLD
        audio = audio.numpy().astype(np.float64)
        """
        f0_floor : float
            Lower F0 limit in Hz.
            Default: 71.0
        f0_ceil : float
            Upper F0 limit in Hz.
            Default: 800.0
        """
        f0, timeaxis = pw.dio(
            audio, self.sampling_rate,
            frame_period=(self.hop_size/self.sampling_rate)*1000.,
        )  # For hop size 256 frame period is 11.6 ms
        
        f0 = torch.from_numpy(f0).float().clamp(min=0.0, max=800)  # (Number of Frames) = (654,)
        voiced_mask = (f0>3)# voice / unvoiced flag
        if voiced_mask.sum() > 0:
            voiced_f0_mean = f0[voiced_mask].mean()
            f0[~voiced_mask] = voiced_f0_mean
        
        return f0, voiced_mask# [dec_T], [dec_T]
    
    def __getitem__(self, index):
        audiopath = self.audio_files[index]
        
        # load audio
        if not self.fine_tuning:
            audio, sampling_rate = load_wav_to_torch(audiopath, target_sr=self.sampling_rate)
            audio = audio - audio.mean()# remove DC offset
            audio = (audio / audio.abs().max()) * 0.95# and normalize volume        
            if self.trim_non_voiced:# trim out non-voiced segments
                assert len(audio.shape) == 1# [B]
                f0, voiced = self.get_pitch(audio)
                start_indx, end_indx = get_nonzero_indexes(voiced)
                audio = audio[start_indx*self.hop_size:end_indx*self.hop_size]
        else:
            pm_audio_path = os.path.splitext(audiopath)[0]+'.pm_audio.pt'# predicted mel audio
            audio = torch.load(pm_audio_path).float()# [T]
        
        audio = audio.unsqueeze(0)# [B] -> [1, B]
        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            
            gt_mel = self.STFT.get_mel(audio)
            mel = gt_mel# 'mel' is the input to the vocoder, gt_mel is the original mel that will be used as a target by the model.
        else:
            pred_mel_path = os.path.splitext(audiopath)[0]+'.pred_mel.pt'
            mel = torch.load(pred_mel_path).float()
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)# [n_mel, mel_T] -> [1, n_mel, mel_T]
            
            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)
                
                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            gt_mel = self.STFT.get_mel(audio)
            min_mel_len = min(gt_mel.shape[-1], mel.shape[-1])
            mel    = mel[:, :, :min_mel_len]
            gt_mel = gt_mel[:, :, :min_mel_len]
            mel = DTW(mel, gt_mel, 5, 3)
        
        return (mel.squeeze(), audio.squeeze(0), audiopath, gt_mel.squeeze())

    def __len__(self):
        return len(self.audio_files)
