import sys
import os
import numpy as np
import random
import time
from datetime import datetime

import argparse
import torch
import matplotlib.pyplot as plt
import json
from shutil import copyfile
from glob import glob

from CookieTTS.utils.audio.stft import TacotronSTFT
from CookieTTS._2_ttm.tacotron2_tm.model import Tacotron2, load_model
from CookieTTS._4_mtw.waveglow.denoiser import Denoiser
from CookieTTS.utils.text import text_to_sequence
from CookieTTS.utils.dataset.utils import load_filepaths_and_text
from CookieTTS.utils.model.utils import alignment_metric

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict


def plot_spect(spect, path, range=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spect, cmap='inferno', aspect="auto", origin="lower",
                   interpolation='none')
    if range is not None:
        assert len(range) == 2, 'range params should be a 2 element List of [Min, Max].'
        assert range[1] > range[0], 'Max (element 1) must be greater than Min (element 0).'
        im.set_clim(range[0], range[1])
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    
    fig.canvas.draw()
    plt.savefig(path)
    plt.close()

class T2S:
    def __init__(self, conf):
        self.conf = conf
        torch.set_grad_enabled(False)
        
        self.possible_speakers = None
        self.possible_audiopaths_dict  = {}
        self.possible_transcripts_dict = {}
        
        filter_length  =  2048
        hop_length     =   512
        win_length     =  2048
        n_mel_channels =   192
        sampling_rate  = 44100.0
        mel_fmin       =    60.0
        mel_fmax       = 16000.0
        stft_clamp_val =     1e-4
        self.stft = TacotronSTFT(
            filter_length,  hop_length,    win_length,
            n_mel_channels, sampling_rate, mel_fmin,
            mel_fmax, clamp_val=stft_clamp_val).cuda()
        
        # load HiFi-GAN
        self.MTW_current = self.conf['MTW']['default_model']
        assert self.MTW_current in self.conf['MTW']['models'].keys(), "HiFi-GAN default model not found in config models"
        vocoder_path = self.conf['MTW']['models'][self.MTW_current]['modelpath']
        self.vocoder, self.MTW_conf = self.load_hifigan(vocoder_path)
        
        print("T2S Initialized and Ready!")
    
    def load_hifigan(self, vocoder_path):
        print("Loading HiFi-GAN...")
        from CookieTTS._4_mtw.hifigan.models import load_model as load_hifigan_model
        vocoder, vocoder_config = load_hifigan_model(vocoder_path)
        vocoder.half()
        print("Done!")
        
        print("Clearing CUDA Cache... ", end='')
        torch.cuda.empty_cache()
        print("Done!")
        
        print('\n'*10)
        import gc # prints currently alive Tensors and Variables  # - And fixes the memory leak? I guess poking the leak with a stick is the answer for now.
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    pass#print(type(obj), obj.size())
            except:
                pass
        print('\n'*10)
        
        return vocoder, vocoder_config
    
    def update_hifigan(self, vocoder_name):
        print(f"Changing HiFi-GAN to {vocoder_name}")
        self.MTW_current = vocoder_name
        assert self.MTW_current in self.conf['MTW']['models'].keys(), f"HiFi-GAN model '{vocoder_name}' not found in config models"
        vocoder_path = self.conf['MTW']['models'][self.MTW_current]['modelpath']
        self.vocoder, self.MTW_conf = self.load_hifigan(vocoder_path)
    
    def write_best_audio(self, audiopaths, best_index):
        text_path = os.path.join(self.conf['output_directory'], 'dump.txt')
        
        first_transcript = open(os.path.splitext(audiopaths[0])[0]+'.txt', 'r', encoding='utf8').read()
        same_transcript = all(open(os.path.splitext(audiopath)[0]+'.txt', 'r', encoding='utf8').read()==first_transcript for audiopath in audiopaths[1:])
        
        first_speaker = os.path.dirname(os.path.realpath(audiopaths[0]))
        same_speaker  = all(os.path.dirname(os.path.realpath(audiopath)) == first_speaker for audiopath in audiopaths)
        with open(text_path, 'a') as f:
            best_audiopath = audiopaths.pop(best_index)
            f.write(f'{best_audiopath}|{audiopaths[0]}|{same_transcript}|{same_speaker}|{"|".join(audiopaths[1:])}\n')
        
        for audiopath in ([best_audiopath,]+audiopaths):# delete symlinked audiopaths in the flask output folder
            os.remove(os.path.join(self.conf['output_directory'], os.path.split(audiopath)[1]))
            os.remove(os.path.join(self.conf['output_directory'], os.path.splitext(os.path.split(audiopath)[1])[0]+'.mel.png'))
    
    @torch.no_grad()
    def get_samples(self, ttsdict):
        os.makedirs(self.conf['output_directory'], exist_ok=True)
        
        speaker                 = ttsdict.get('speaker', None)
        min_samples_per_speaker = ttsdict.get('min_samples_per_speaker', ttsdict['samples_to_compare'])
        whitelist_speakers = ttsdict.get('whitelist_speakers', [])
        
        if self.possible_speakers is None:
            # get list of possible speaker options.
            possible_speakers = sorted([x for x in os.listdir(self.conf['datasets'][0]) if os.path.isdir(os.path.join(self.conf['datasets'][0], x)) and 'meta' not in x])
            for dataset in self.conf['datasets'][1:]:
                new_speakers = sorted([x for x in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, x)) and 'meta' not in x])
                possible_speakers = [x for x in possible_speakers if x in new_speakers]# use only speakers that appear in both dataset folders
            
            # filter out speakers with too few files in the 'main' dataset
            possible_speakers = [x for x in possible_speakers if len(glob(os.path.join(self.conf['datasets'][0], x, '*.wav'))) > min_samples_per_speaker]
            
            self.possible_speakers = possible_speakers
        else:
            possible_speakers = self.possible_speakers
        
        # apply whitelist if needed
        whitelisted_possible_speakers = possible_speakers
        if len(whitelist_speakers) > 0:
            whitelisted_possible_speakers = [x for x in possible_speakers if x in whitelist_speakers]
        
        audiopaths  = []
        transcripts = []
        speakers    = []
        for i in range(ttsdict['samples_to_compare']):
            if speaker is None or (not ttsdict['same_speaker']):
                speaker = random.choice(whitelisted_possible_speakers)
            
            if speaker in self.possible_audiopaths_dict:
                possible_audiopaths  = self.possible_audiopaths_dict[speaker]
                possible_transcripts = self.possible_transcripts_dict[speaker]
            else:
                possible_audiopaths = [x for dataset in self.conf['datasets'] for x in glob(os.path.join(dataset, speaker, '*.wav'))]
                possible_audiopaths = [x for x in possible_audiopaths if os.path.exists(os.path.splitext(x)[0]+'.txt')]
                self.possible_audiopaths_dict[speaker] = possible_audiopaths
                
                possible_transcripts = [open(os.path.splitext(x)[0]+'.txt', 'r', encoding='utf8').read() for x in possible_audiopaths]
                self.possible_transcripts_dict[speaker] = possible_transcripts
            assert len(possible_audiopaths), f"couldn't find any audiopaths for speaker {speaker}."
            assert len(possible_audiopaths) == len(possible_transcripts)
            
            if len(audiopaths):# if an audio file has already been used, remove it's path and transcript from the possible options
                bad_indexes = set([i for i, x in enumerate(possible_audiopaths) if x in audiopaths])
                possible_audiopaths  = [x for i, x in enumerate(possible_audiopaths ) if i not in bad_indexes]
                possible_transcripts = [x for i, x in enumerate(possible_transcripts) if i not in bad_indexes]
                assert len(possible_audiopaths), f"couldn't find any remaining audiopaths/transcripts for speaker {speaker}."
                assert len(possible_audiopaths) == len(possible_transcripts)
                
            if ttsdict['same_transcript']:
                if len(transcripts):
                    possible_audiopaths = [x for i, x in enumerate(possible_audiopaths) if possible_transcripts[i] in transcripts]
                else:
                    possible_audiopaths = [x for i, x in enumerate(possible_audiopaths) if possible_transcripts.count(possible_transcripts[i]) >= ttsdict['samples_to_compare']]
                    assert len(possible_audiopaths), f"couldn't find any audiopaths with enough repeated transcripts for speaker {speaker}."
                assert len(possible_audiopaths), f"couldn't find any audiopaths remaining transcripts for speaker {speaker}."
            
            audiopath = random.choice(possible_audiopaths)
            assert audiopath not in audiopaths
            audiopaths.append(audiopath)
            
            textpath = os.path.splitext(audiopath)[0]+'.txt'
            transcripts.append(open(textpath, 'r', encoding='utf8').read())
        
        # get spectrograms
        spectpaths = []
        for audiopath in audiopaths:
            spectpath = os.path.join(self.conf['output_directory'], os.path.splitext(os.path.split(audiopath)[1])[0]+'.mel.png')
            spectpaths.append(spectpath)
            if not os.path.exists(spectpath):
                spect = self.stft.get_mel_from_path(audiopath)
                plot_spect(spect[0].data.cpu().numpy(), spectpath, range=[-9.2103, 2.0])
        
        audionames = []
        spectnames = []
        for audiopath, spectpath in zip(audiopaths, spectpaths):
            out_audiopath = os.path.join(self.conf['output_directory'], os.path.split(audiopath)[1])
            try:
                os.symlink(audiopath, out_audiopath)
            except:
                copyfile(audiopath, out_audiopath)
            
            audionames.append(os.path.split(audiopath)[1])
            spectnames.append(os.path.split(audiopath)[1])
        
        # return reconstructed audiopaths and mel-spectrograms
        out = {
        'audiopaths': audionames,
     'absaudiopaths': audiopaths,
        'spectpaths': spectnames,
       'transcripts': transcripts,
          'speakers': speakers  ,
 'possible_speakers': possible_speakers,
        }
        return out