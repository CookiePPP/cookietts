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
        
        output['pres_prev_state'] = torch.tensor(True if (filelist_index == prev_filelist_index) else False)# preserve model state if this iteration is continuing the file from the last iteration.
        output['cont_next_iter'] = torch.tensor(True if (filelist_index == next_filelist_index) else False)# whether this file continued into the next iteration
        
        #audiopath, gtext, ptext, speaker_id, *_ = self.filelist[filelist_index]
        audiopath, text, speaker_id, *_ = self.filelist[filelist_index]
        output['audiopath'] = audiopath
        
        if True or any(arg in ['gt_audio','gt_mel','gt_frame_f0','gt_frame_energy','gt_frame_voiced','gt_char_f0','gt_char_energy','gt_char_voiced'] for arg in args):
            audio, sampling_rate = self.get_audio(audiopath)
            if 'gt_audio' in args:
                output['gt_audio'] = audio
            if 'sampling_rate' in args:
                output['sampling_rate'] = torch.tensor(float(sampling_rate))
        
        if any([arg in ('gt_mel','dtw_pred_mel') for arg in args]):
            # get mel
            mel = self.get_mel_from_audio(audio, sampling_rate)
            
            # add silence
            mel = torch.cat((
                torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
                mel,# get mel-spec as tensor from audiofile.
                torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
                ), dim=1)# arr -> [n_mel, mel_T]
            
            init_mel = F.pad(mel, (self.context_frames, 0))[:, int(spectrogram_offset):int(spectrogram_offset)+self.context_frames]
            # initial input to the decoder. zeros if this is first segment of this file, else last frame of prev segment.
            
            output['gt_mel']   = mel
            output['init_mel'] = init_mel
            del mel, init_mel
        
        if any([arg in ('pred_mel','dtw_pred_mel') for arg in args]):
            if os.path.exists( os.path.splitext(audiopath)[0]+'_pred.npy' ):
                pred_mel = self.get_mel_from_npfile( os.path.splitext(audiopath)[0]+'_pred.npy' )
                
                # add silence
                pred_mel = torch.cat((
                    torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
                    pred_mel,# get mel-spec as tensor from audiofile.
                    torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
                    ), dim=1)# arr -> [n_mel, mel_T]
                
                output['pred_mel'] = pred_mel
                
                if 'dtw_pred_mel' in args:# Pred mel Time-Warping to align more accurately with target mel
                    output['dtw_pred_mel'] = DTW(output['pred_mel'].unsqueeze(0), output['gt_mel'].unsqueeze(0),
                             scale_factor=self.dtw_scale_factor, range_=self.dtw_range).squeeze(0)
                del pred_mel
        
        if 'gt_sylps' in args:
            gt_sylps = self.get_syllables_per_second(text, len(audio)/sampling_rate)# [] FloatTensor
            output['gt_sylps'] = gt_sylps
            del gt_sylps
        
        if any([arg in ('text', 'gt_sylps', 'alignment','gt_char_f0','gt_char_voiced','gt_char_energy') for arg in args]):
            output['gtext_str'] = text
            output['ptext_str'] = self.arpa.get(text)
            del text
            
            use_phones = random.random() < self.p_arpabet
            output['text_str'] = output['ptext_str'] if use_phones else output['gtext_str']
            if 'text' in args:
                output['text'] = output['ptext_str'] if use_phones else output['gtext_str']# (randomly) convert to phonemes
                output['text'] = self.get_text(output['text'])# convert text into tensor representation
        
        if any(arg in ('speaker_id','speaker_id_ext') for arg in args):
            output['speaker_id_ext'] = speaker_id_ext
            output['speaker_id'] = self.get_speaker_id(speaker_id_ext)# get speaker_id as tensor normalized [ 0 -> len(speaker_ids) ]
        
        if any([arg in ['gt_emotion_id','gt_emotion_onehot'] for arg in args]):
            output['gt_emotion_id'] = self.get_emotion_id(audiopath)# [1] IntTensor
            gt_emotion_onehot = self.one_hot_embedding(gt_emotion_id, num_classes=self.n_classes+1).squeeze(0)[:-1]# [n_classes]
            output['gt_emotion_onehot'] = gt_emotion_onehot
        
        if 'torchmoij_hdn' in args:
            torchmoji = self.get_torchmoji_hidden( os.path.splitext(audiopath)[0]+'_tm.npy' )
            output['torchmoij_hdn'] = torchmoji
        
        if 'gt_perc_loudness' in args:
            output['gt_perc_loudness'] = self.get_perc_loudness(audio, sampling_rate)
        
        if any([arg in ('gt_frame_f0','gt_frame_voiced','gt_char_f0','gt_char_voiced') for arg in args]):
            f0, voiced_mask = self.get_pitch(audio, self.sampling_rate, self.hop_length)
            output['gt_frame_f0']     = f0
            output['gt_frame_voiced'] = voiced_mask
        
        if any([arg in ('gt_frame_energy','gt_char_energy') for arg in args]):
            output['gt_frame_energy'] = self.get_energy(mel)
        
        if any([arg in ('alignment','gt_char_f0','gt_char_voiced','gt_char_energy') for arg in args]):
            output['alignment'] = alignment = self.get_alignments(audiopath, arpa=use_phones)
            if 'gt_char_f0' in args:
                output['gt_char_f0']     = self.get_charavg_from_frames(f0                 , alignment)# [txt_T]
            if 'gt_char_voiced' in args:
                output['gt_char_voiced'] = self.get_charavg_from_frames(voiced_mask.float(), alignment)# [txt_T]
            if 'gt_char_energy' in args:
                output['gt_char_energy'] = self.get_charavg_from_frames(output['gt_frame_energy'], alignment)# [txt_T]
            if 'gt_char_dur' in args:
                output['gt_char_dur'] = alignment.sum(0)# [mel_T, txt_T] -> [txt_T]
        
        ########################
        ## Trim into Segments ##
        ########################
        if self.random_segments:
            max_start = mel.shape[-1] - self.max_segment_length
            spectrogram_offset = random.randint(0, max_start) if max_start > 0 else 0
        
        if 'gt_frame_f0' in output:
            output['gt_frame_f0']     = output['gt_frame_f0'][int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
            output['gt_frame_voiced'] = output['gt_frame_voiced'][int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'gt_frame_energy' in output:
            output['gt_frame_energy'] = output['gt_frame_energy'][int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'alignment' in output:
            output['alignment'] = output['alignment'][int(spectrogram_offset):int(spectrogram_offset+self.truncated_length), :]
        
        if 'gt_mel' in output:
            output['gt_mel'] = output['gt_mel'][: , int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'pred_mel' in output:
            output['pred_mel'] = output['pred_mel'][: , int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        if 'dtw_pred_mel' in output:
            output['dtw_pred_mel'] = output['dtw_pred_mel'][: , int(spectrogram_offset):int(spectrogram_offset+self.max_segment_length)]
        
        return output
    
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
    
    def get_perc_loudness(self, audio, sampling_rate):
        meter = pyln.Meter(sampling_rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        gt_perc_loudness = torch.tensor(loudness)
        return gt_perc_loudness# []
    
    def get_charavg_from_frames(self, x, alignment):# [mel_T], [mel_T, txt_T]
        norm_alignment   =      alignment / alignment.sum(dim=0, keepdim=True).clamp(min=0.01)
        # [mel_T, txt_T] <- [mel_T, txt_T] / [mel_T, 1]
        
        x.float().unsqueeze(0)# [mel_T] -> [1, mel_T]
        y = x @ norm_alignment# [1, mel_T] @ [mel_T, txt_T] -> [1, txt_T]
        
        assert not (torch.isinf(y) | torch.isnan(y)).any()
        return y.squeeze(0)# [txt_T]
    
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
        
        assert not (torch.isinf(f0) | torch.isnan(f0)).any(), f"f0 from pyworld is NaN. Info below\nlen(audio) = {len(audio)}\nf0 = {f0}\nsampling_rate = {sampling_rate}"
        return f0, voiced_mask# [mel_T], [mel_T]
    
    def get_energy(self, spect):
        # Extract energy
        energy = torch.sqrt(torch.sum(spect[4:]**2, dim=0))# [n_mel, mel_T] -> [mel_T]
        return energy# [mel_T]
    
    def get_emotion_id(self, audiopath):
        gt_emotion_id = self.n_classes # int
        if len(audiopath.split("_")) >= 6:
            emotions = audiopath.split("_")[4].lower().split(" ") # e.g: ["neutral",]
            for emotion in reversed(emotions):
                try:
                    gt_emotion_id = self.emotion_classes.index(emotion) # INT in set {0, 1, ... n_classes-1}
                except:
                    pass
        return torch.LongTensor([gt_emotion_id,]) # [1]
    
    def get_syllables_per_second(self, text, dur_s):
        n_syl = syllables.estimate(text)
        sylps = n_syl/dur_s
        return torch.tensor(sylps) # []
    
    def get_gt_perc_loudness(self, audio, sampling_rate):
        meter = pyln.Meter(sampling_rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        gt_perc_loudness = torch.tensor(loudness)
        return gt_perc_loudness# []
    
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
        self.n_classes = len(getattr(hparams, "emotion_classes", list())
        self.context_frames = getattr(hparams, "context_frames", 1)
        self.sort_text_len_decending = True
    
    def collate_left(self, tensor_arr, max_len=None, dtype=None, device=None, index_lookup=None, pad_val=0.0, check_const_channels=True):
        """
        Take array of tensors and spit out a merged version. (left aligned, fill any empty areas with pad_val)
        Optional Index Lookup.
        PARAMS:
            max_len: Output Tensor Length
            dtype: Output Tensor Datatype
            index_lookup: A dict of indexes incase there is a specific order the items should be ordered
            pad_val: Value used to fill empty space in the output Tensor
            check_const_channels: Enable/Disable Assertion for constant channel dim in inputs.
        INPUT:
            [[C, T?], [C, T?], [C, T?], [C, T?], ...]
            or
            [[T?], [T?], [T?], [T?], [T?], ...]
            or
            [[C], [C], [C], [C], [C], ...]
            or
            [[1], [1], [1], [1], ...]
            or
            [[], [], [], [], ...]
        OUTPUT:
            [B, C, T]
            or
            [B, T]
            or
            [B, C]
            or
            [B]
        """
        B = len(tensor_arr)
        max_len = max(x.shape[-1] for x in tensor_arr) if max_len is None else max_len
        dtype = tensor_arr[0].dtype if dtype is None else dtype
        device = tensor_arr[0].device if device is None else device
        if index_lookup is None:
            index_lookup = {i:i for i in range(B)}
        else:
            assert len(index_lookup.keys()) == B, f'lookup has {len(index_lookup.keys())} keys, expected {B}.'
        
        if all(len(tensor_arr[i].view(-1)) == 1 for i in range(B)):
            output = torch.zeros(B, device=device, dtype=dtype)
            for i, _ in enumerate(tensor_arr):
                output[i] = tensor_arr[index_lookup[i]].to(device, dtype)
        elif len(tensor_arr[0].shape) == 1:
            output = torch.ones(B, max_len, device=device, dtype=dtype)
            if pad_val:
                output *= pad_val
            for i, _ in enumerate(tensor_arr):
                item = tensor_arr[index_lookup[i]].to(device, dtype)
                output[i, :item.shape[1]] = item
        elif len(tensor_arr[0].shape) == 2:
            C = max(tensor_arr[i].shape[1] for i in range(B))
            if check_const_channels:
                assert all(C == item.shape[1] for item in tensor_arr), f'an item in input has channel_dim != channel_dim of the first item.\n{"\n".join(["Shape "+str(i)+" = "+str(item.shape) for i, item in enumerate(tensor_arr)])}'
            output = torch.ones(B, C, max_len, device=device, dtype=dtype)
            if pad_val:
                output *= pad_val
            for i, _ in enumerate(tensor_arr):
                item = tensor_arr[index_lookup[i]].to(device, dtype)
                output[i, :item.shape[0], :item.shape[1]] = item
        else:
            raise Exception(f"Unexpected input shape, got {len(tensor_arr[0].shape)} dims and expected 1 or 2 dims.")
        return output
    
    def collatea(self, *args, check_const_channels=False, **kwargs):
        return self.collatek(*args, check_const_channels=check_const_channels, **kwargs)
    
    def collatek(self, batch, key, index_lookup, dtype=None, pad_val=0.0, ignore_missing_key=True, check_const_channels=True):
        if ignore_missing_key and not any(key in item for item in batch):
            return None
        else:
            assert all(key in item for item in batch), f'item in batch is missing key "{key}"'
        
        if all(type(item[key]) == torch.Tensor for key in batch.keys()):
            return self.collate_left([item[key] for item in batch], dtype=dtype, index_lookup=index_lookup, pad_val=pad_val, check_const_channels=check_const_channels)
        elif not any(type(item[key]) == torch.Tensor for key in batch.keys()):
            assert dtype is None, f'dtype specified as "{dtype}" but input has no Tensors.'
            arr = (item[key] for item in batch)
            return [arr[index_lookup[i]] for i, _ in enumerate(arr)]
        else:
            raise Exception(f"Mixed types found in batch items, key is '{key}'")
    
    def __call__(self, batch):
        """
        Collate's training batch from __getitem__ input features
        (this the merging all the features loaded from each audio file into the same tensors so the model can process together)
        PARAMS
            batch: [{"text": text[txt_T], "gt_mel": gt_mel[n_mel, mel_T]}, {"text": text, "gt_mel": gt_mel}, ...]
        ------
        RETURNS
            out: {"text": text_batch, "gt_mel": gt_mel_batch}
        """
        B = len(batch)# Batch Size
        
        if self.sort_text_len_decending and all("text" in item for item in batch):# if text, reorder entire batch to go from longest text -> shortest text.
            input_lengths, ids_sorted = torch.sort(
                torch.LongTensor([len(item['text']) for item in batch]), dim=0, descending=True)
        else:
            ids_sorted = {x:x for x in range(B)}# elif no text, batch can be whatever order it's loaded in
        
        out['text']              = self.collatek(batch, 'text',              ids_sorted, dtype=torch.long )# [B, txt_T]
        out['gtext_str']         = self.collatek(batch, 'gtext_str',         ids_sorted, dtype=None       )# [str, ...]
        out['ptext_str']         = self.collatek(batch, 'ptext_str',         ids_sorted, dtype=None       )# [str, ...]
        out['text_str']          = self.collatek(batch, 'text_str',          ids_sorted, dtype=None       )# [str, ...]
        out['audiopath']         = self.collatek(batch, 'audiopath',         ids_sorted, dtype=None       )# [str, ...]
        
        out['alignment']         = self.collatea(batch, 'alignment',         ids_sorted, dtype=torch.float)# [B, mel_T, txt_T]
        
        out['gt_sylps']          = self.collatek(batch, 'gt_sylps',          ids_sorted, dtype=torch.float)# [B]
        out['torchmoij_hdn']     = self.collatek(batch, 'torchmoij_hdn',     ids_sorted, dtype=torch.float)# [B, C]
        
        out['speaker_id']        = self.collatek(batch, 'speaker_id',        ids_sorted, dtype=torch.long )# [B]
        out['speaker_id_ext']    = self.collatek(batch, 'speaker_id_ext',    ids_sorted, dtype=None       )# [int, ...]
        
        out['gt_emotion_id']     = self.collatek(batch, 'gt_emotion_id',     ids_sorted, dtype=torch.long )# [B]
        out['gt_emotion_onehot'] = self.collatek(batch, 'gt_emotion_onehot', ids_sorted, dtype=torch.long )# [B, n_emotions]
        
        out['init_mel']          = self.collatek(batch, 'init_mel',          ids_sorted, dtype=torch.float)# [B, C, context]
        out['pres_prev_state']   = self.collatek(batch, 'pres_prev_state',   ids_sorted, dtype=torch.bool )# [B]
        out['cont_next_iter']    = self.collatek(batch, 'cont_next_iter',    ids_sorted, dtype=torch.bool )# [B]
        
        out['gt_mel']            = self.collatek(batch, 'gt_mel',            ids_sorted, dtype=torch.float)# [B, C, mel_T]
        out['pred_mel']          = self.collatek(batch, 'pred_mel',          ids_sorted, dtype=torch.float)# [B, C, mel_T]
        out['dtw_pred_mel']      = self.collatek(batch, 'dtw_pred_mel',      ids_sorted, dtype=torch.float)# [B, C, mel_T]
        
        out['gt_frame_f0']       = self.collatek(batch, 'gt_frame_f0',       ids_sorted, dtype=torch.float)# [B, mel_T]
        out['gt_frame_energy']   = self.collatek(batch, 'gt_frame_energy',   ids_sorted, dtype=torch.float)# [B, mel_T]
        out['gt_frame_voiced']   = self.collatek(batch, 'gt_frame_voiced',   ids_sorted, dtype=torch.float)# [B, mel_T]
        
        out['gt_char_f0']        = self.collatek(batch, 'gt_char_f0',        ids_sorted, dtype=torch.float)# [B, txt_T]
        out['gt_char_energy']    = self.collatek(batch, 'gt_char_energy',    ids_sorted, dtype=torch.float)# [B, txt_T]
        out['gt_char_voiced']    = self.collatek(batch, 'gt_frame_voiced',   ids_sorted, dtype=torch.float)# [B, txt_T]
        out['gt_char_dur']       = self.collatek(batch, 'gt_char_dur',       ids_sorted, dtype=torch.float)# [B, txt_T]
        
        out['gt_perc_loudness']  = self.collatek(batch, 'gt_perc_loudness',  ids_sorted, dtype=torch.float)# [B]
        
        out['gt_audio']          = self.collatek(batch, 'gt_audio',          ids_sorted, dtype=torch.float)# [B, wav_T]
        out['sampling_rate']     = self.collatek(batch, 'sampling_rate',     ids_sorted, dtype=torch.float)# [B]
        
        if all('gt_mel' in item for item in batch):
            mel_T = max(item['gt_mel'].shape[-1] for item in batch)
            out['gt_gate_logits'] = torch.zeros(B, mel_T, dtype=torch.float)
            for i in range(B):
                out['gt_gate_logits'][i, mel.size(1)-(~batch[ids_sorted[i]]['cont_next_iter']).long():] = 1
                # set positive gate if this file isn't going to be continued next iter.
                # (i.e: if this is the last segment of the file.)
        
        out = {k:v for k,v in out.items() if v is not None} # remove any entries with "None" values.
        
        return out