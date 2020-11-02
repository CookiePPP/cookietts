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

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, filelist, hparams, check_files=True, TBPTT=True, shuffle=False, speaker_ids=None, audio_offset=0, verbose=False):
        self.filelist = load_filepaths_and_text(filelist)
        
        #####################
        ## Text / Phonemes ##
        #####################
        self.text_cleaners = hparams.text_cleaners
        self.arpa = ARPA(hparams.dict_path)
        self.p_arpabet     = hparams.p_arpabet
        
        self.emotion_classes    = getattr(hparams, "emotion_classes", list())
        self.n_classes          = len(self.emotion_classes)
        self.audio_offset       = audio_offset
        
        #################
        ## Speaker IDs ##
        #################
        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            if hasattr(hparams, 'raw_speaker_ids') and hparams.raw_speaker_ids:
                self.speaker_ids = {k:k for k in range(hparams.n_speakers)} # map IDs in files directly to internal IDs
            else:
                self.speaker_ids = self.create_speaker_lookup_table(self.filelist, numeric_sort=hparams.numeric_speaker_ids)
        
        ###################
        ## File Checking ##
        ###################
        self.start_token = hparams.start_token
        self.stop_token  = hparams.stop_token
        if check_files:
            self.checkdataset(show_warnings=True, show_info=verbose)
        
        ###############################
        ## Mel-Spectrogram Generator ##
        ###############################
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length    = hparams.hop_length
        self.mel_fmax      = hparams.mel_fmax
        self.stft = STFT.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        
        # Silence Padding
        self.silence_value     = hparams.silence_value
        self.silence_pad_start = hparams.silence_pad_start# frames to pad the start of each clip
        self.silence_pad_end   = hparams.silence_pad_end  # frames to pad the end of each clip
        
        ################################################
        ## (Optional) Apply weighting to MLP Datasets ##
        ################################################
        if False:
            duplicated_audiopaths = [x for x in self.filelist if "SlicedDialogue" in x[0]]
            for i in range(3):
                self.filelist.extend(duplicated_audiopaths)
        
        # Shuffle Audiopaths
        random.seed(hparams.seed)
        self.random_seed = hparams.seed
        random.shuffle(self.filelist)
        
        #####################################################################
        ## PREDICT LENGTH (TBPTT) - Truncated Backpropagation through time ##
        #####################################################################
        # simulate the entire epoch so the decoder can be linked together between iters.
        self.shuffle    = shuffle
        
        self.batch_size = hparams.batch_size
        self.rank       = hparams.rank
        self.total_batch_size = self.batch_size * hparams.n_gpus # number of audio files being processed together
        self.max_segment_length = hparams.max_segment_length # frames
        
        if hparams.use_TBPTT and TBPTT:
            print('Calculating audio lengths of all files...')
            self.audio_lengths = torch.tensor([self.get_mel(x[0]).shape[1]+self.silence_pad_start+self.silence_pad_end for x in self.filelist]) # get the length of every file (the long way)
            print('Done.')
        else:
            self.audio_lengths = torch.tensor([self.max_segment_length-1 for x in self.filelist]) # use dummy lengths
        self.update_dataloader_indexes()
    
    def shuffle_dataset(self):
        print("Shuffling Dataset")
        
        # shuffle filelist and audio lengths (they're shuffled together so they still line up) (note, shuffle uses a new Random() instance with same seed so should perform the same shuffle operation in a distributed environment)
        zipped = list(zip(self.filelist, self.audio_lengths.numpy()))
        self.filelist, self.audio_lengths = zip(*random.Random(self.random_seed).sample(zipped, len(zipped)))
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
            
            batch_remaining_lengths = batch_remaining_lengths - self.max_segment_length # truncate batch
            batch_frame_offset = batch_frame_offset + self.max_segment_length
        
        self.len = len(self.dataloader_indexes)
    
    def checkdataset(self, show_warnings=False, show_info=False, max_frames_per_char=80): # TODO, change for list comprehension which is a few magnitudes faster.
        print("Checking dataset files...")
        audiopaths_length = len(self.filelist)
        filtered_chars = ["☺","␤"]
        banned_strings = ["[","]"]
        banned_paths = []
        music_stuff = True
        start_token = self.start_token
        stop_token = self.stop_token
        for index, file in enumerate(self.filelist): # index must use seperate iterations from remove
            if music_stuff and r"Songs/" in file[0]:
                self.filelist[index][1] = "♫" + self.filelist[index][1] + "♫"
            for filtered_char in filtered_chars:
                self.filelist[index][1] = self.filelist[index][1].replace(filtered_char,"")
            self.filelist[index][1] = start_token + self.filelist[index][1] + stop_token
        i = 0
        i_offset = 0
        for i_ in range(len(self.filelist)):
            i = i_ + i_offset # iterating on an array you're also updating will cause some indexes to be skipped.
            if i == len(self.filelist): break
            file = self.filelist[i]
            if self.load_mel_from_disk and '.wav' in file[0]:
                if show_warnings:
                    print("|".join(file), "\n[warning] in filelist while expecting '.npy' . Being Ignored.")
                self.filelist.remove(file)
                i_offset-=1
                continue
            elif not self.load_mel_from_disk and '.npy' in file[0]:
                if show_warnings:
                    print("|".join(file), "\n[warning] in filelist while expecting '.wav' . Being Ignored.")
                self.filelist.remove(file)
                i_offset-=1
                continue
            if not os.path.exists(file[0]):
                if show_warnings:
                    print("|".join(file), "\n[warning] does not exist and has been ignored")
                self.filelist.remove(file)
                i_offset-=1
                continue
            if not len(file[1]):
                if show_warnings:
                    print("|".join(file), "\n[warning] has no text and has been ignored.")
                self.filelist.remove(file)
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
                    self.filelist.remove(file)
                    i_offset-=1
                    continue
                if False and mel_length > self.max_segment_length:# Disabled, need to be added to config
                    if show_warnings:
                        print("|".join(file), f"\n[warning] is over {self.max_segment_length} frames long and has been ignored.")
                    self.filelist.remove(file)
                    i_offset-=1
                    continue
                if (mel_length / len(file[1])) > max_frames_per_char:
                    print("|".join(file), f"\n[warning] has more than {max_frames_per_char} frames per char. ({(mel_length / len(file[1])):.4})")
                    self.filelist.remove(file)
                    i_offset-=1
                    continue
            if any(i in file[1] for i in banned_strings):
                if show_warnings and show_info:
                    print("|".join(file), "\n[info] is in banned strings and has been ignored.")
                self.filelist.remove(file)
                i_offset-=1
                continue
            if any(i in file[0] for i in banned_paths):
                if show_warnings and show_info:
                    print("|".join(file), "\n[info] is in banned paths and has been ignored.")
                self.filelist.remove(file)
                i_offset-=1
                continue
        print("Done")
        print(audiopaths_length, "items in metadata file")
        print(len(self.filelist), "validated and being used.")
    
    def create_speaker_lookup_table(self, filelist, numeric_sort=True):
        """
        if numeric_sort:
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        else:
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
        """
        if numeric_sort:
            speaker_ids_in_filelist = [int(x[2]) for x in filelist]
        else:
            speaker_ids_in_filelist = [str(x[2]) for x in filelist]
        speaker_ids = np.sort(np.unique(speaker_ids_in_filelist))
        d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
        return d
    
    def get_audio(self, filepath):
        audio, sampling_rate = load_wav_to_torch(filepath, min_sr=self.mel_fmax*2)
        return audio, sampling_rate
    
    def get_mel_from_audio(self, audio, sampling_rate):
        if self.audio_offset: # used for extreme GTA'ing
            audio = audio[self.audio_offset:]
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
        melspec = self.stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0)
        return melspec
    
    def get_mel_from_npfile(self, filepath):
        melspec = torch.from_numpy(np.load(filepath)).float()
        assert melspec.size(0) == self.stft.n_mel_channels, (f'Mel dimension mismatch: given {melspec.size(0)}, expected {self.stft.n_mel_channels}')
        return melspec
    
    def get_alignment_from_npfile(self, filepath, text_length, spec_length):
        melspec = torch.from_numpy(np.load(filepath)).float()
        assert melspec.shape[0] == spec_length, "Saved Alignment has wrong decoder length"
        assert melspec.shape[1] == text_length, "Saved Alignment has wrong encoder length"
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
    
    def get_mel_text_pair(self, index, args):# get args using get_args() from CookieTTS.utils
        filelist_index, spectrogram_offset = self.dataloader_indexes[index]
        prev_filelist_index, prev_spectrogram_offset = self.dataloader_indexes[max(0, index-self.total_batch_size)]
        is_not_last_iter = index+self.total_batch_size < self.len
        next_filelist_index, next_spectrogram_offset = self.dataloader_indexes[index+self.total_batch_size] if is_not_last_iter else (None, None)
        
        output['preserve_prev_state'] = torch.tensor(True if (filelist_index == prev_filelist_index) else False)# preserve model state if this iteration is continuing the file from the last iteration.
        output['continued_next_iter'] = torch.tensor(True if (filelist_index == next_filelist_index) else False)# whether this file continued into the next iteration
        
        #audiopath, gtext, ptext, speaker_id, *_ = self.filelist[filelist_index]
        audiopath, text, speaker_id, *_ = self.filelist[filelist_index]
        
        if True or any(arg in ['gt_audio','gt_mel','frame_f0','frame_energy','frame_voiced','frame_voiced'] for arg in args):
            audio, sampling_rate = self.get_audio(audiopath)
            if 'gt_audio' in args:
                output['gt_audio'] = audio
            if 'sampling_rate' in args:
                output['sampling_rate'] = torch.tensor(sampling_rate)
        
        if 'gt_mel' in args:
            # get mel
            mel = self.get_mel_from_audio(audio, sampling_rate)
            
            # add silence
            mel = torch.cat((
                torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
                mel,# get mel-spec as tensor from audiofile.
                torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
                ), dim=1)# arr -> [n_mel, dec_T]
            
            init_mel = F.pad(mel, (self.context_frames, 0))[:, int(spectrogram_offset):int(spectrogram_offset)+self.context_frames]
            # initial input to the decoder. zeros if this is first segment of this file, else last frame of prev segment.
            
            # take a segment.
            output['gt_mel'] = mel
        
        if any([arg in ('pred_mel','dtw_pred_mel') for arg in args]):
            if os.path.exists( os.path.splitext(audiopath)[0]+'_pred.npy' ):
                pred_mel = self.get_mel_from_npfile( os.path.splitext(audiopath)[0]+'_pred.npy' )
                
                # add silence
                pred_mel = torch.cat((
                    torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
                    pred_mel,# get mel-spec as tensor from audiofile.
                    torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
                    ), dim=1)# arr -> [n_mel, dec_T]
                
                output['pred_mel'] = pred_mel
        
                if 'dtw_pred_mel' in args:
                    output['dtw_pred_mel'] = 
        
        if 'gt_sylps':
            sylps = self.get_syllables_per_second(text, len(audio)/sampling_rate)# [] FloatTensor
            output['sylps'] = sylps
        
        if any([arg in ('text', 'alignment','char_f0','char_voiced','char_energy') for arg in args]):
            output['gtext_str'] = text
            output['ptext_str'] = self.arpa.get(text)
            
            use_phones = random.random() < self.p_arpabet
            output['text_str'] = output['ptext_str'] if use_phones else output['gtext_str']
            if 'text' in args:
                output['text'] = output['ptext_str'] if use_phones else output['gtext_str']# (randomly) convert to phonemes
                output['text'] = self.get_text(output['text'])# convert text into tensor representation
        
        if 'speaker_id' in args:
            speaker_id = self.get_speaker_id(speaker_id)# get speaker_id as tensor normalized [ 0 -> len(speaker_ids) ]
            output['speaker_id'] = speaker_id
        
        if any([arg in ['emotion_id','emotion_onehot'] for arg in args]):
            emotion_id = self.get_emotion_id(audiopath)# [1] IntTensor
            output['emotion_id']     = emotion_id
            emotion_onehot = self.one_hot_embedding(emotion_id, num_classes=self.n_classes+1).squeeze(0)[:-1]# [n_classes]
            output['emotion_onehot'] = emotion_onehot
        
        if 'torchmoij_hdn' in args:
            torchmoji = self.get_torchmoji_hidden( os.path.splitext(audiopath)[0]+'_tm.npy' )
            output['torchmoij_hdn'] = torchmoji
        
        if 'perc_loudness' in args:
            output['perc_loudness'] = self.get_perc_loudness(audio, sampling_rate)
        
        if any([arg in ('frame_f0','frame_voiced','char_f0','char_voiced') for arg in args]):
            f0, voiced_mask = self.get_pitch(audio, self.sampling_rate, self.hop_length)
            output['frame_f0']     = f0
            output['frame_voiced'] = voiced_mask
        
        if any([arg in ('frame_energy','char_energy') for arg in args]):
            output['frame_energy'] = self.get_energy(mel)
        
        if any([arg in ('alignment','char_f0','char_voiced','char_energy') for arg in args]):
            output['alignment'] = alignment = self.get_alignments(audiopath, arpa=use_phones)
            if 'char_f0' in args:
                output['char_f0']     = self.get_charavg_from_frames(f0                 , alignment)# [enc_T]
            if 'char_voiced' in args:
                output['char_voiced'] = self.get_charavg_from_frames(voiced_mask.float(), alignment)# [enc_T]
            if 'char_energy' in args:
                output['char_energy'] = self.get_charavg_from_frames(energy             , alignment)# [enc_T]
        
        ########################
        ## Trim into Segments ##
        ########################
        if self.random_segments:
            max_start = mel.shape[-1] - self.max_segment_length
            spectrogram_offset = random.randint(0, max_start) if max_start > 0 else 0
        
        if 'frame_f0' in output:
            output['frame_f0']     = output['frame_f0'][int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
            output['frame_voiced'] = output['frame_voiced'][int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'frame_energy' in output:
            output['frame_energy'] = output['frame_energy'][int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'alignment' in output:
            output['alignment'] = output['alignment'][int(spectrogram_offset):int(spectrogram_offset+self.truncated_length), :]
        
        if 'gt_mel' in output:
            output['gt_mel'] = output['gt_mel'][: , int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'pred_mel' in output:
            output['pred_mel'] = output['pred_mel'][: , int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        return output
        
        return (text, mel, speaker_id, torchmoji, preserve_decoder_state, continued_next_iter, init_mel, sylps, emotion_id, emotion_onehot, index)
             # ( 0  ,  1 ,     2     ,     3    ,             4         ,            5       ,     6   ,   7  ,     8     ,        9      ,  10  )
    
    def get_torchmoji_hidden(self, fpath):
        hidden_state = np.load(fpath)
        return torch.from_numpy(hidden_state).float()
    
    def get_alignments(self, audiopath, arpa=False):
        if arpa:
            alignpath = os.path.splitext(audiopath)[0]+'_palign.npy'
        else:
            alignpath = os.path.splitext(audiopath)[0]+'_galign.npy'
        alignment = np.load(alignpath)
        return torch.from_numpy(alignment).float()
    
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
    
    def get_syllables_per_second(self, text, dur_s):
        n_syl = syllables.estimate(text)
        sylps = n_syl/dur_s
        return torch.tensor(sylps) # []
    
    def get_perc_loudness(self, audio, sampling_rate):
        meter = pyln.Meter(sampling_rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        perc_loudness = torch.tensor(loudness)
        return perc_loudness# []
    
    def get_speaker_id(self, speaker_id):
        """Convert external speaker_id to internel [0 to max_speakers] range speaker_id"""
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
