import random
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
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
        self.arpa = ARPA(hparams.dict_path)
        self.p_arpabet = hparams.p_arpabet
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.truncated_length = hparams.truncated_length
        self.context_frames = hparams.context_frames
        self.batch_size = hparams.batch_size
        self.speaker_ids = speaker_ids
        self.emotion_classes = hparams.emotion_classes
        self.n_classes = len(hparams.emotion_classes)
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
        self.silence_value = hparams.silence_value
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
                    if show_warnings:
                        print("|".join(file), "\n[warning] has 0 duration and has been ignored")
                    self.audiopaths_and_text.remove(file)
                    i_offset-=1
                    continue
                if False and mel_length > self.truncated_length:# Disabled, need to be added to config
                    if show_warnings:
                        print("|".join(file), f"\n[warning] is over {self.truncated_length} frames long and has been ignored.")
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
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if self.audio_offset: # used for extreme GTA'ing
                audio = audio[self.audio_offset:]
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
            melspec = self.stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0)
        else:
            melspec = torch.from_numpy(np.load(filename, allow_pickle=True)).float()
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        return melspec
    
    def one_hot_embedding(self, labels, num_classes=None):
        """Embedding labels to one-hot form.
        
        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.
        
        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        if num_classes is None:
            num_classes = self.n_classes
        y = torch.eye(num_classes)
        return y[labels]
    
    def get_mel_text_pair(self, index,
                              ignore_text=False, ignore_emotion=False, ignore_speaker=False, ignore_torchmoji=False, ignore_sylps=False, ignore_mel=False):
        text = emotion_id = emotion_onehot = speaker_id = torchmoji = sylps = init_mel = mel = None
        filelist_index, spectrogram_offset = self.dataloader_indexes[index]
        prev_filelist_index, prev_spectrogram_offset = self.dataloader_indexes[max(0, index-self.total_batch_size)]
        is_not_last_iter = index+self.total_batch_size < self.len
        next_filelist_index, next_spectrogram_offset = self.dataloader_indexes[index+self.total_batch_size] if is_not_last_iter else (None, None)
        preserve_decoder_state = torch.tensor(True if (filelist_index == prev_filelist_index) else False)# preserve model state if this iteration is continuing the file from the last iteration.
        continued_next_iter = torch.tensor(True if (filelist_index == next_filelist_index) else False)# whether this file continued into the next iteration
        
        #audiopath, gtext, ptext, speaker_id, *_ = self.audiopaths_and_text[filelist_index]
        audiopath, text, speaker_id, *_ = self.audiopaths_and_text[filelist_index]
        
        if not ignore_mel:
            # get mel
            mel = self.get_mel(audiopath)
            # add silence
            mel = torch.cat((
                torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
                mel,# get mel-spec as tensor from audiofile.
                torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
                ), dim=1)# arr -> [n_mel, dec_T]
            
            init_mel = F.pad(mel, (self.context_frames, 0))[:, int(spectrogram_offset):int(spectrogram_offset)+self.context_frames]
            # initial input to the decoder. zeros if this is first segment of this file, else last frame of prev segment.
            
            # take a segment.
            mel = mel[: , int(spectrogram_offset):int(spectrogram_offset+self.truncated_length)]
        
        if (not ignore_sylps) and (not ignore_mel) and (not ignore_text):
            sylps = self.get_syllables_per_second(text, (mel.shape[-1]*self.hop_length)/self.sampling_rate)# [] FloatTensor
        
        if not ignore_text:
            if random.random() < self.p_arpabet:# (randomly) convert to phonemes
                text = self.arpa.get(text)
            #text = ptext if random.random() < self.p_arpabet else gtext
            
            text = self.get_text(text)# convert text into tensor representation
        
        if not ignore_speaker:
            speaker_id = self.get_speaker_id(speaker_id)# get speaker_id as tensor normalized [ 0 -> len(speaker_ids) ]
        
        if not ignore_emotion:
            emotion_id = self.get_emotion_id(audiopath)# [1] IntTensor
            emotion_onehot = self.one_hot_embedding(emotion_id, num_classes=self.n_classes+1).squeeze(0)[:-1]# [n_classes]
        
        if not ignore_torchmoji:
            torchmoji = self.get_torchmoji_hidden(audiopath)
        
        return (text, mel, speaker_id, torchmoji, preserve_decoder_state, continued_next_iter, init_mel, sylps, emotion_id, emotion_onehot, index)
             # ( 0  ,  1 ,     2     ,     3    ,             4         ,            5       ,     6   ,   7  ,     8     ,        9      ,  10  )
    
    def get_torchmoji_hidden(self, audiopath):
        audiopath_without_ext = ".".join(audiopath.split(".")[:-1])
        path_path_len = min(len(audiopath_without_ext), 999)
        file_path_safe = audiopath_without_ext[0:path_path_len]
        hidden_state = np.load(file_path_safe + "_.npy")
        #hidden_state = np.load(audiopath.replace('.wav','_tm.npy'))
        return torch.from_numpy(hidden_state).float()
    
    def get_emotion_id(self, audiopath):
        emotion_id = self.n_classes # int
        if len(audiopath.split("_")) >= 6:
            emotions = audiopath.split("_")[4].lower().split(" ") # e.g: ["neutral",]
            for emotion in reversed(emotions):
                try:
                    emotion_id = self.emotion_classes.index(emotion) # INT in set {0, 1, ... n_classes-1}
                except:
                    pass
        return torch.LongTensor([emotion_id,]) # [1]
    
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
        if self.shuffle and index == self.rank: # [0,3,6,9],[1,4,7,10],[2,5,8,11] # shuffle_dataset if first item of this GPU of this epoch
           self.shuffle_dataset()
        return self.get_mel_text_pair(index)
    
    def __len__(self):
        return self.len


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        self.n_frames_per_step = hparams.n_frames_per_step
        self.n_classes = len(hparams.emotion_classes)
        self.context_frames = hparams.context_frames
    
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
        
        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        
        # include mel padded, gate padded and speaker ids
        mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
        gate_padded = torch.zeros(len(batch), max_target_len)
        output_lengths = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        torchmoji_hidden = torch.FloatTensor(len(batch), batch[0][3].shape[0])
        preserve_decoder_states = torch.FloatTensor(len(batch))
        sylps = torch.FloatTensor(len(batch))
        emotion_id = torch.FloatTensor(len(batch))
        emotion_onehot = torch.FloatTensor(len(batch), self.n_classes)
        init_mel = torch.FloatTensor(len(batch), num_mels, self.context_frames)
        
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-(~batch[ids_sorted_decreasing[i]][5]).long():] = 1# set positive gate if this file isn't going to be continued next iter. (i.e: if this is the last segment of the file.)
            output_lengths[i] = mel.size(1)
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][2]
            if torchmoji_hidden is not None:
                torchmoji_hidden[i] = batch[ids_sorted_decreasing[i]][3]
            preserve_decoder_states[i] = batch[ids_sorted_decreasing[i]][4]
            init_mel[i] = batch[ids_sorted_decreasing[i]][6]
            sylps[i] = batch[ids_sorted_decreasing[i]][7]
            emotion_id[i] = batch[ids_sorted_decreasing[i]][8]
            emotion_onehot[i] = batch[ids_sorted_decreasing[i]][9]
        
        model_inputs = (text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_ids,
                         torchmoji_hidden, preserve_decoder_states, init_mel, sylps, emotion_id, emotion_onehot)
        return model_inputs
