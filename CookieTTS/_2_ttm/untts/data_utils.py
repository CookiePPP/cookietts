import random
import os
import re
import numpy as np
import torch
import torch.utils.data
import librosa
import syllables
import pyworld as pw
import pyloudnorm as pyln

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
        self.arpa = ARPA(hparams.dict_path)
        self.p_arpabet = hparams.p_arpabet
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.truncated_length = hparams.truncated_length
        self.batch_size = hparams.batch_size
        self.speaker_ids = speaker_ids
        self.audio_offset = audio_offset
        self.shuffle = shuffle
        if speaker_ids is None:
            if hasattr(hparams, 'raw_speaker_ids') and hparams.raw_speaker_ids:
                self.speaker_ids = {k:k for k in range(hparams.n_speakers)} # map IDs in files directly to internal IDs
            else:
                self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text, numeric_sort=hparams.numeric_speaker_ids)
        
        # ---------- CHECK FILES --------------
        self.start_token = hparams.start_token
        self.stop_token = hparams.stop_token
        if check_files:
            self.checkdataset(show_warnings=True, show_info=verbose)
        # -------------- CHECK FILES --------------
        
        self.stft = STFT.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        
        if False:# Apply weighting to MLP Datasets
            duplicated_audiopaths = [x for x in self.audiopaths_and_text if "SlicedDialogue" in x[0]]
            for i in range(3):
                self.audiopaths_and_text.extend(duplicated_audiopaths)
        
        # Shuffle Audiopaths
        random.seed(hparams.seed)
        self.random_seed = hparams.seed
        random.shuffle(self.audiopaths_and_text)
        
        # Silence Padding
        self.silence_value = hparams.spect_padding_value
        self.silence_pad_start = hparams.silence_pad_start# frames to pad the start of each clip
        self.silence_pad_end = hparams.silence_pad_end  # frames to pad the end of each clip
        
        # -------------- PREDICT LENGTH (TBPTT) --------------
        self.batch_size = hparams.batch_size if speaker_ids is None else hparams.val_batch_size
        n_gpus = hparams.n_gpus
        self.rank = hparams.rank
        self.total_batch_size = self.batch_size * n_gpus # number of audio files being processed together
        self.truncated_length = hparams.truncated_length # frames
        
        if hparams.use_TBPTT and TBPTT:
            print('Calculating audio lengths of all files...')
            self.audio_lengths = torch.tensor([self.get_mel(x[0]).shape[1]+self.silence_pad_start+self.silence_pad_end for x in self.audiopaths_and_text]) # get the length of every file (the long way)
            print('Done.')
        else:
            self.audio_lengths = torch.tensor([self.truncated_length-1 for x in self.audiopaths_and_text]) # use dummy lengths
        self.update_dataloader_indexes()
        # -------------- PREDICT LENGTH (TBPTT) --------------
    
    def shuffle_dataset(self):
        print("Shuffling Dataset")
        
        # shuffle filelist and audio lengths (they're shuffled together so they still line up) (note, shuffle uses a new Random() instance with same seed so should perform the same shuffle operation in a distributed environment)
        zipped = list(zip(self.audiopaths_and_text, self.audio_lengths.numpy()))
        self.audiopaths_and_text, self.audio_lengths = zip(*random.Random(self.random_seed).sample(zipped, len(zipped)))
        self.audio_lengths = torch.tensor(self.audio_lengths)
        
        # regen the order to load truncated files/which states to preserve
        self.update_dataloader_indexes()
    
    def update_dataloader_indexes(self):
        """Simulate the entire epoch and plan in advance which spectrograms will be processed by what when."""
        self.dataloader_indexes = []
        
        batch_remaining_lengths = self.audio_lengths[:self.total_batch_size]
        batch_frame_offset = torch.zeros(self.total_batch_size)
        batch_indexes = torch.tensor(list(range(self.total_batch_size)))
        processed = 0
        currently_empty_lengths = 0
        
        while self.audio_lengths.shape[0]+1 > processed+self.total_batch_size+currently_empty_lengths:
            # replace empty lengths
            currently_empty_lengths = (batch_remaining_lengths<1).sum().item()
            # update batch_indexes
            batch_indexes[batch_remaining_lengths<1] = torch.arange(processed+self.total_batch_size, processed+self.total_batch_size+currently_empty_lengths)
            # update batch_frame_offset
            batch_frame_offset[batch_remaining_lengths<1] = 0
            # update batch_remaining_lengths
            try:
                batch_remaining_lengths[batch_remaining_lengths<1] = self.audio_lengths[processed+self.total_batch_size:processed+self.total_batch_size+currently_empty_lengths]
            except RuntimeError: # RuntimeError typically occurs when there are no remaining files (so this is technically droplast=True).
                break
            
            # update how many audiofiles have been fully used
            processed+=currently_empty_lengths
            
            self.dataloader_indexes.extend(list(zip(batch_indexes.numpy(), batch_frame_offset.numpy())))
            
            batch_remaining_lengths = batch_remaining_lengths - self.truncated_length # truncate batch
            batch_frame_offset = batch_frame_offset + self.truncated_length
        
        self.len = len(self.dataloader_indexes)
    
    def checkdataset(self, show_warnings=False, show_info=False, max_frames_per_char=80): # TODO, change for list comprehension which is a few magnitudes faster.
        print("Checking dataset files...")
        audiopaths_length = len(self.audiopaths_and_text)
        filtered_chars = ["☺","␤"]
        banned_strings = ["[","]"]
        banned_paths = []
        music_stuff = True
        start_token = self.start_token
        stop_token = self.stop_token
        for index, file in enumerate(self.audiopaths_and_text): # index must use seperate iterations from remove
            if music_stuff and r"Songs/" in file[0]:
                self.audiopaths_and_text[index][1] = "♫" + self.audiopaths_and_text[index][1] + "♫"
            for filtered_char in filtered_chars:
                self.audiopaths_and_text[index][1] = self.audiopaths_and_text[index][1].replace(filtered_char,"")
            self.audiopaths_and_text[index][1] = start_token + self.audiopaths_and_text[index][1] + stop_token
        i = 0
        i_offset = 0
        for i_ in range(len(self.audiopaths_and_text)):
            i = i_ + i_offset # iterating on an array you're also updating will cause some indexes to be skipped.
            if i == len(self.audiopaths_and_text): break
            file = self.audiopaths_and_text[i]
            if self.load_mel_from_disk and '.wav' in file[0]:
                if show_warnings:
                    print("|".join(file), "\n[warning] in filelist while expecting '.npy' . Being Ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            elif not self.load_mel_from_disk and '.npy' in file[0]:
                if show_warnings:
                    print("|".join(file), "\n[warning] in filelist while expecting '.wav' . Being Ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if not os.path.exists(file[0]):
                if show_warnings:
                    print("|".join(file), "\n[warning] does not exist and has been ignored")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            
            path = os.path.splitext(file[0])[0]+'_palign_out.npy'
            if not os.path.exists(path):
                if show_warnings:
                    print(path, "\n[warning] does not exist and has been ignored")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            path = os.path.splitext(file[0])[0]+'_galign_out.npy'
            if not os.path.exists(path):
                if show_warnings:
                    print(path, "\n[warning] does not exist and has been ignored")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            
            if not len(file[1]):
                if show_warnings:
                    print("|".join(file), "\n[warning] has no text and has been ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if len(file[1]) < 3:
                if show_warnings and show_info:
                    print("|".join(file), "\n[info] has no/very little text.")
            if not ((file[1].strip())[-1] in r"!?,.;:♫␤"):
                if show_warnings and show_info:
                    print("|".join(file), "\n[info] has no ending punctuation.")
            if self.load_mel_from_disk:
                melspec = torch.from_numpy(np.load(file[0], allow_pickle=True))
                mel_length = melspec.shape[1]
                if mel_length == 0:
                    print("|".join(file), "\n[warning] has 0 duration and has been ignored")
                    self.audiopaths_and_text.remove(file)
                    i_offset-=1
                    continue
                if mel_length > 1000: # over 12.5s
                    print("|".join(file), "\n[warning] is over 1000 frames long and has been ignored.")
                    self.audiopaths_and_text.remove(file)
                    i_offset-=1
                    continue
                if (mel_length / len(file[1])) > max_frames_per_char:
                    print("|".join(file), f"\n[warning] has more than {max_frames_per_char} frames per char. ({(mel_length / len(file[1])):.4})")
                    self.audiopaths_and_text.remove(file)
                    i_offset-=1
                    continue
            if any(i in file[1] for i in banned_strings):
                if show_warnings and show_info:
                    print("|".join(file), "\n[info] is in banned strings and has been ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
            if any(i in file[0] for i in banned_paths):
                if show_warnings and show_info:
                    print("|".join(file), "\n[info] is in banned paths and has been ignored.")
                self.audiopaths_and_text.remove(file)
                i_offset-=1
                continue
        print("Done")
        print(audiopaths_length, "items in metadata file")
        print(len(self.audiopaths_and_text), "validated and being used.")
    
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
    
    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename.replace('.npy','.wav'))
        if self.audio_offset: # used for extreme GTA'ing
            audio = audio[self.audio_offset:]
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        
        if not self.load_mel_from_disk:
            melspec = self.stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0)
        else:
            melspec = torch.from_numpy(np.load(filename, allow_pickle=True)).float()
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        return melspec, audio_norm, sampling_rate
    
    def get_mel_text_pair(self, index):
        filelist_index, spectrogram_offset = self.dataloader_indexes[index]
        next_filelist_index, next_spectrogram_offset = self.dataloader_indexes[index+self.total_batch_size] if index+self.total_batch_size < self.len else (None, None)
        
        audiopath, text, speaker = self.audiopaths_and_text[filelist_index]
        self.audiopath = audiopath
        
        # get mel-spect from audio
        mel, audio, sampling_rate = self.get_mel(audiopath) # get mel/AEF
        
        # syllables per second
        sylps = self.get_syllables_per_second(text, (mel.shape[-1]*self.hop_length)/self.sampling_rate)# [] FloatTensor
        
        # (randomly) convert to phoneme input
        use_phones = random.random() < self.p_arpabet
        text = self.arpa.get(text) if use_phones else text
        
        text = self.get_text(text) # convert text into tensor representation
        
        # add silence
        mel = torch.cat((
            torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
            mel,# get mel-spec as tensor from audiofile.
            torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
            ), dim=1)# arr -> [n_mel, dec_T]
        
        speaker_id = self.get_speaker_id(speaker) # get speaker_id as tensor normalized [ 0 -> len(speaker_ids) ]
        
        alignment = self.get_alignments(audiopath, arpa=use_phones)
        
        torchmoji = self.get_torchmoji_hidden(audiopath)
        
        perc_loudness = self.get_perc_loudness(audio, sampling_rate)
        f0, voiced_mask = self.get_pitch(audio, self.sampling_rate, self.hop_length)
        energy = self.get_energy(mel)
        
        f0          = f0[int(spectrogram_offset):int(spectrogram_offset+self.truncated_length)]
        voiced_mask = voiced_mask[int(spectrogram_offset):int(spectrogram_offset+self.truncated_length)]
        energy      = energy[int(spectrogram_offset):int(spectrogram_offset+self.truncated_length)]
        alignment   = alignment[int(spectrogram_offset):int(spectrogram_offset+self.truncated_length), :]
        mel         = mel[..., int(spectrogram_offset):int(spectrogram_offset+self.truncated_length)]
        
        char_f0          = self.get_charavg_from_frames(f0                 , alignment)# [enc_T]
        char_voiced_mask = self.get_charavg_from_frames(voiced_mask.float(), alignment)# [enc_T]
        char_energy      = self.get_charavg_from_frames(energy             , alignment)# [enc_T]
        
        return (text       , mel      , speaker_id   ,#     ([0], [1], [2],
                alignment  , torchmoji, perc_loudness,#      [3], [4], [5],
                f0         , energy   , sylps        ,#      [6], [7], [8],
                voiced_mask,                          #      [9],
                char_f0    , char_voiced_mask, char_energy)# [10],[11],[12])
    
    def get_alignments(self, audiopath, arpa=False):
        if arpa:
            alignpath = os.path.splitext(audiopath)[0]+'_palign_out.npy'
        else:
            alignpath = os.path.splitext(audiopath)[0]+'_galign_out.npy'
        alignment = np.load(alignpath)
        return torch.from_numpy(alignment).float()
    
    def get_perc_loudness(self, audio, sampling_rate):
        meter = pyln.Meter(sampling_rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        perc_loudness = torch.tensor(loudness)
        return perc_loudness# []
    
    def get_charavg_from_frames(self, x, alignment):# [dec_T], [dec_T, enc_T]
        norm_alignment    =      alignment / alignment.sum(dim=0, keepdim=True).clamp(min=0.01)
        # [dec_T, enc_T] <- [dec_T, enc_T] / [dec_T, 1]
        
        x.float().unsqueeze(0)# [dec_T] -> [1, dec_T]
        y = x @ norm_alignment# [1, dec_T] @ [dec_T, enc_T] -> [1, enc_T]
        
        assert not (torch.isinf(y) | torch.isnan(y)).any()
        return y.squeeze(0)# [enc_T]
    
    def get_pitch(self, audio, sampling_rate, hop_length):
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
            audio, sampling_rate,
            frame_period=(hop_length/sampling_rate)*1000.,
        )  # For hop size 256 frame period is 11.6 ms
        
        f0 = torch.from_numpy(f0).float().clamp(min=0.0, max=800)  # (Number of Frames) = (654,)
        voiced_mask = (f0>3)# voice / unvoiced flag
        if voiced_mask.sum() > 0:
            voiced_f0_mean = f0[voiced_mask].mean()
            f0[~voiced_mask] = voiced_f0_mean
        
        assert not (torch.isinf(f0) | torch.isnan(f0)).any(), f"f0 from pyworld is NaN. Info below\nlen(audio) = {len(audio)}\nf0 = {f0}\naudiopath = {self.audiopath.replace('.npy','.wav')}\nsampling_rate = {sampling_rate}"
        return f0, voiced_mask# [dec_T], [dec_T]
    
    def get_energy(self, spect):
        # Extract energy
        energy = torch.sqrt(torch.sum(spect[4:]**2, dim=0))# [n_mel, dec_T] -> [dec_T]
        return energy# [dec_T]
    
    def get_torchmoji_hidden(self, audiopath):
        audiopath_without_ext = ".".join(audiopath.split(".")[:-1])
        path_path_len = min(len(audiopath_without_ext), 999)
        file_path_safe = audiopath_without_ext[0:path_path_len]
        hidden_state = np.load(file_path_safe + "_.npy")
        return torch.from_numpy(hidden_state).float()
    
    def get_syllables_per_second(self, text, duration):
        n_syl = syllables.estimate(text)
        sylps = n_syl/duration
        return torch.tensor(sylps) # []
    
    def get_speaker_id(self, speaker_id):
        return torch.IntTensor([self.speaker_ids[int(speaker_id)]])
    
    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm
    
    def __getitem__(self, index):
        if self.shuffle and index == self.rank*self.batch_size: # [0,1,2,3],[4,5,6,7],[8,9,10,11] # shuffle_dataset if first item of this GPU of this epoch
           self.shuffle_dataset()
        return self.get_mel_text_pair(index)
    
    def __len__(self):
        return self.len


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        self.pad_value = hparams.spect_padding_value
    
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_ids, mel, speaker_id, alignment, torchmoji, perc_loudness, f0, energy, sylps], [text, ...], ... ]
                [   0    ,  1 ,     2     ,     3    ,     4    ,       5      , 6 ,   7   ,   8  ]
        """
        B = len(batch)
        
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(B, max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
        
        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        
        # include mel padded, gate padded and speaker ids
        mel_padded       = torch.ones(B, num_mels, max_target_len) * self.pad_value
        output_lengths   = torch.LongTensor(B)
        speaker_ids      = torch.LongTensor(B)
        alignments       = torch.zeros(B, max_target_len, max_input_len)# [B, dec_T, enc_T]
        torchmoji_hidden = torch.FloatTensor(B, batch[0][4].shape[0]) if (batch[0][3] is not None) else None
        perc_loudnesss   = torch.zeros(B)
        sylps            = torch.zeros(B)
        f0s              = torch.zeros(B, max_target_len)
        energys          = torch.zeros(B, max_target_len)
        voiced_mask      = torch.zeros(B, max_target_len)
        char_f0          = torch.zeros(B, max_input_len)
        char_voiced_mask = torch.zeros(B, max_input_len)
        char_energy      = torch.zeros(B, max_input_len)
        
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            
            output_lengths[i] = mel.size(1)
            
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            
            alignment = batch[ids_sorted_decreasing[i]][3]
            alignments[i, :alignment.shape[0], :alignment.shape[1]] = alignment
            
            if torchmoji_hidden is not None:
                torchmoji_hidden[i] = batch[ids_sorted_decreasing[i]][4]
            
            perc_loudnesss[i] = batch[ids_sorted_decreasing[i]][5]
            
            f0 = batch[ids_sorted_decreasing[i]][6]
            f0s[i, :f0.shape[0]] = f0
            
            energy = batch[ids_sorted_decreasing[i]][7]
            energys[i, :energy.shape[0]] = energy
            
            sylps[i] = batch[ids_sorted_decreasing[i]][8]
            
            vmask = batch[ids_sorted_decreasing[i]][9]
            voiced_mask[i, :vmask.shape[0]] = vmask
            
            f0 = batch[ids_sorted_decreasing[i]][10]
            char_f0[i, :f0.shape[0]] = f0
            
            vmask = batch[ids_sorted_decreasing[i]][11]
            char_voiced_mask[i, :vmask.shape[0]] = vmask
            
            energy = batch[ids_sorted_decreasing[i]][12]
            char_energy[i, :energy.shape[0]] = energy
        
        model_inputs = (text_padded, mel_padded, speaker_ids, input_lengths, output_lengths,
                        alignments, torchmoji_hidden, perc_loudnesss, f0s, energys, sylps,
                        voiced_mask, char_f0, char_voiced_mask, char_energy)
        return model_inputs
