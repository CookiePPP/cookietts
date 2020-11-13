import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import librosa
import syllables

import CookieTTS.utils.audio.stft as STFT
from CookieTTS.utils.dataset.utils import load_wav_to_torch, load_filepaths_and_text
from CookieTTS.utils.text import text_to_sequence
from CookieTTS.utils.text.ARPA import ARPA

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, check_files=True, TBPTT=True, shuffle=False, speaker_ids=None, audio_offset=0, verbose=False):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.segment_length = hparams.segment_length
        if speaker_ids is None:
            if hasattr(hparams, 'raw_speaker_ids') and hparams.raw_speaker_ids:
                self.speaker_ids = {k:k for k in range(hparams.n_speakers)} # map IDs in files directly to internal IDs
            else:
                self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text, numeric_sort=hparams.numeric_speaker_ids)
        else:
            self.speaker_ids = speaker_ids
        
        # Shuffle Audiopaths
        random.seed(hparams.seed)
        self.random_seed = hparams.seed
        random.shuffle(self.audiopaths_and_text)
        
        if check_files:
            self.checkdataset(show_warnings=True, show_info=verbose)
        
        # Silence Padding
        #self.silence_value = hparams.silence_value
        #self.silence_pad_start = hparams.silence_pad_start# frames to pad the start of each clip
        #self.silence_pad_end = hparams.silence_pad_end  # frames to pad the end of each clip
        
        self.len = len(self.audiopaths_and_text)
    
    def create_speaker_lookup_table(self, audiopaths_and_text, numeric_sort=False):
        """
        if numeric_sort:
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        if numeric_sort:
            speaker_ids_in_filelist = [int(x[2]) for x in audiopaths_and_text]
        else:
            speaker_ids_in_filelist = [str(x[2]) for x in audiopaths_and_text]
        speaker_ids = np.sort(np.unique(speaker_ids_in_filelist))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d
    
    def checkdataset(self, show_warnings=True, show_info=False, valid_exts=['.wav','.flac']): # TODO, change for list comprehension which is a few magnitudes faster.
        print("Checking dataset files...")
        audiopaths_length = len(self.audiopaths_and_text)
        banned_paths = []
        
        # remove exts not in valid_exts
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if any(x[0].endswith(y) for y in valid_exts)]
        assert len(self.audiopaths_and_text)
        
        # remove items with banned filepaths
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if not any(i in x[0] for i in banned_paths)]
        assert len(self.audiopaths_and_text)
        
        # remove files that don't exist
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if os.path.exists(x[0])]
        assert len(self.audiopaths_and_text)
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if os.path.exists(x[1])]
        assert len(self.audiopaths_and_text)
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if os.path.exists(x[4])]
        assert len(self.audiopaths_and_text)
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if os.path.exists(x[5])]
        assert len(self.audiopaths_and_text)
        
        # check wav filesizes are 2x segment_length (to remove files that are obviously too short without having to load them all)
        self.audiopaths_and_text = [x for x in self.audiopaths_and_text if (not x[0].endswith('.wav')) or os.stat(x[0]).st_size//2 > self.segment_length]
        assert len(self.audiopaths_and_text)
        
        # check audio files for anything that is too short.
        #self.audiopaths_and_text = [x for x in self.audiopaths_and_text if load_wav_to_torch(x[0])[0].shape[0] >= self.segment_length]
        #assert len(self.audiopaths_and_text)
        
        #i = 0
        #i_offset = 0
        #for i_ in range(len(self.audiopaths_and_text)):
        #    i = i_ + i_offset # iterating on an array you're also updating will cause some indexes to be skipped.
        #    if i == len(self.audiopaths_and_text): break
        #    file = self.audiopaths_and_text[i]
        #    if not any(file[0].endswith(x) for x in valid_exts):
        #        if show_warnings:
        #            print("|".join(file), "\n[warning] in filelist while expecting '.wav' or other audio format. Being Ignored.")
        #        self.audiopaths_and_text.remove(file)
        #        i_offset-=1
        #        continue
        #    if not os.path.exists(file[0]) or not os.path.exists(file[1]):
        #        if show_warnings:
        #            print("|".join(file), "\n[warning] does not exist and has been ignored")
        #        self.audiopaths_and_text.remove(file)
        #        i_offset-=1
        #        continue
        #    if (file[0].endswith('.wav') and os.stat(file[0]).st_size//2 < self.segment_length) or load_wav_to_torch(file[0])[0].shape[0] < self.segment_length:
        #        if show_warnings:
        #            print("|".join(file), "\n[warning] is too short and has been ignored")
        #        self.audiopaths_and_text.remove(file)
        #        i_offset-=1
        #        continue
        #    if not os.path.exists(file[4]):
        #        if show_warnings:
        #            print("|".join(file), "\n[warning] encoder durations does not exist")
        #        self.audiopaths_and_text.remove(file)
        #        i_offset-=1
        #        continue
        #    if not os.path.exists(file[5]):
        #        if show_warnings:
        #            print("|".join(file), "\n[warning] encoder outputs does not exist")
        #        self.audiopaths_and_text.remove(file)
        #        i_offset-=1
        #        continue
        #    if any(i in file[0] for i in banned_paths):
        #        if show_warnings and show_info:
        #            print("|".join(file), "\n[info] is in banned paths and has been ignored.")
        #        self.audiopaths_and_text.remove(file)
        #        i_offset-=1
        #        continue
        print("Done")
        print(audiopaths_length, "items in metadata file")
        print(len(self.audiopaths_and_text), "validated and being used.")
    
    def get_contexts(self, start_index, encoder_outputs, durations):
        enc_T, enc_dim = encoder_outputs.shape
        dec_T = int(durations.sum().item())
        
        attention = torch.zeros(dec_T, enc_T) # [dec_T, enc_T]
        attention_pos = torch.arange(dec_T).float()
        pos = torch.tensor(0.0)
        for enc_ind, dur in enumerate(durations):
            end_pos = pos+dur # []
            attention[:, enc_ind][(attention_pos >= pos) & (attention_pos < end_pos)] = 1.0
            pos+=dur # []
        attention = attention[start_index//self.hop_length:(start_index+self.segment_length)//self.hop_length]
        return attention.matmul(encoder_outputs) # [dec_T, enc_T] @ [enc_T, enc_dim] -> [dec_T, enc_dim]
    
    def get_audio_text_pair(self, index):
        audiopath, _, speaker_id, _, duration_path, enc_out_path = self.audiopaths_and_text[index]
        encoder_outputs = torch.from_numpy(np.load(enc_out_path)).float()# [enc_T, enc_dim]
        durations = torch.from_numpy(np.load(duration_path)).float()     # [enc_T]
        audio, sampling_rate = load_wav_to_torch(audiopath)   # [T]
        
        max_audio_start = audio.size(0) - self.segment_length
        audio_start = random.randint(0, max_audio_start//self.hop_length)*self.hop_length
        audio_segment = audio[audio_start:audio_start+self.segment_length]
        attention_contexts = self.get_contexts(audio_start, encoder_outputs, durations)# [dec_T, enc_dim]
        return (audio_segment, attention_contexts, encoder_outputs, durations)
    
    def __getitem__(self, index):
        loaded = False
        i = 0
        while not loaded:
            try:
                data = self.get_audio_text_pair(index)
                loaded = True
            except Exception as ex:
                i+=1
                if i > 10:
                    raise ex
        return data
    
    def __len__(self):
        return self.len


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        pass
    
    def __call__(self, batch):
        """Collate's training batch from audio and encoder outputs
        """
        B = len(batch)
        # include mel padded, gate padded and speaker ids
        audio = torch.stack([x[0] for x in batch], dim=0)
        attention_contexts = torch.stack([x[1] for x in batch], dim=0)# -> [B, dec_T, enc_dim]
        
        text_lengths = torch.tensor([batch[i][2].shape[0] for i in range(B)]).long() # [seq, emb] -> [B, seq, emb]
        encoder_outputs = torch.zeros(len(batch), text_lengths.max().item(), batch[0][1].shape[1]).float()
        durations = torch.zeros(len(batch), text_lengths.max().item()).float()
        for i in range(B):
            enc_out = batch[i][2]
            encoder_outputs[i, :enc_out.shape[0], :] = enc_out.float()
            dur = batch[i][3]
            durations[i, :dur.shape[0]] = dur.float()
        
        model_inputs = (audio, attention_contexts, encoder_outputs, text_lengths, durations)
        return model_inputs