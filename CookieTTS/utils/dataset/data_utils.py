import random
import time
import pickle
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
import re
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

# meta processing
from CookieTTS.utils.dataset.metadata import get_dataset_meta
from tqdm import tqdm

# audio processing
import librosa
import pyworld as pw
import pyloudnorm as pyln
import CookieTTS.utils.audio.stft as STFT
from CookieTTS.utils.dataset.utils import load_wav_to_torch, load_filepaths_and_text

# text processing
from CookieTTS.utils.text import text_to_sequence
from CookieTTS.utils.text.ARPA import ARPA

# torchmoji text processing
import json
from CookieTTS.utils.torchmoji.sentence_tokenizer import SentenceTokenizer
from CookieTTS.utils.torchmoji.model_def import torchmoji_feature_encoding
from CookieTTS.utils.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

# misc
import syllables


def generate_filelist_from_datasets(DATASET_FOLDER,
        DATASET_CONF_FOLDER=None,
        AUDIO_FILTER=['*.wav'],
        AUDIO_REJECTS=['*_Noisy_*','*_Very Noisy_*'],
        MIN_DURATION=0.73,
        MIN_SPEAKER_DURATION_SECONDS=20, rank=0):
    if DATASET_CONF_FOLDER is None:
        DATASET_CONF_FOLDER = os.path.join(DATASET_FOLDER, 'meta')
    # define meta dict (this is where all the data is collected)
    meta = {}
    
    # list of datasets that will be processed over the next while.
    datasets = sorted([x for x in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, x)) and 'meta' not in x])
    
    # check default configs exist (and prompt user for datasets without premade configs)
    defaults_fpath = os.path.join(DATASET_CONF_FOLDER, 'defaults.pkl')
    os.makedirs(os.path.split(defaults_fpath)[0], exist_ok=True)
    if os.path.exists(defaults_fpath):
        try:
            defaults = pickle.load(open(defaults_fpath, "rb"))
        except:
            defaults = {}
    else:
        if rank > 0:
            while not os.path.exists(defaults_fpath):
                time.sleep(1.0)
            time.sleep(1.0)
            defaults = pickle.load(open(defaults_fpath, "rb"))
        else:
            defaults = {}
    
    print(f'Checking Defaults for Datasets...')
    speaker_default = None
    for dataset in datasets:
        if dataset not in defaults:
            defaults[dataset] = {}
        
        if 'speaker' not in defaults[dataset]:
            print(f'default speaker for "{dataset}" dataset is missing.\nPlease enter the name of the default speaker\nExamples: "Nancy", "Littlepip", "Steven"')
            print(f'Press Enter with no input to use "{dataset}"')
            usr_inp = input('> ')
            if len(usr_inp.strip()) == 0:
                usr_inp = dataset
            defaults[dataset]['speaker'] = usr_inp
            print("")
    
    if any(any(x not in ds_defaults for x in ['emotion','noise_level','source','source_type']) for ds_defaults in defaults.values()):# if any Default is missing from any dataset
        emotion_default = noise_level_default = source_default = source_type_default = None
        for dataset in datasets:
            if 'emotion' not in defaults[dataset]:
                print(f'default emotion for "{dataset}" dataset is missing.\nPlease enter the default emotion\nExamples: "Neutral", "Bored", "Audiobook"')
                if emotion_default is not None:
                    print(f'Press Enter with no input to use "{emotion_default}"')
                usr_inp = input('> ')
                assert len(usr_inp.strip()) > 0 or emotion_default, 'No input given!'
                if len(usr_inp.strip()) > 0:
                    emotion_default = usr_inp.strip()
                defaults[dataset]['emotion'] = emotion_default if len(usr_inp.strip()) == 0 else usr_inp
                print("")
            
            if 'noise_level' not in defaults[dataset]:
                print(f'default noise level for "{dataset}" dataset is missing.\nPlease enter the default noise level\nExamples: "Clean", "Noisy", "Very Noisy"')
                if noise_level_default is not None:
                    print(f'Press Enter with no input to use "{noise_level_default}"')
                usr_inp = input('> ')
                assert len(usr_inp.strip()) > 0 or noise_level_default, 'No input given!'
                if len(usr_inp.strip()) > 0:
                    noise_level_default = usr_inp.strip()
                defaults[dataset]['noise_level'] = noise_level_default if len(usr_inp.strip()) == 0 else usr_inp
                print("")
            
            if 'source' not in defaults[dataset]:
                print(f'default source for "{dataset}" dataset is missing.\nPlease enter the default source\nExamples: "My Little Pony", "Team Fortress 2", "University of Edinburgh"')
                if source_default is not None:
                    print(f'Press Enter with no input to use "{source_default}"')
                usr_inp = input('> ')
                assert len(usr_inp.strip()) > 0 or source_default, 'No input given!'
                if len(usr_inp.strip()) > 0:
                    source_default = usr_inp.strip()
                defaults[dataset]['source'] = source_default if len(usr_inp.strip()) == 0 else usr_inp
                print("")
            
            if 'source_type' not in defaults[dataset]:
                print(f'default source type for "{dataset}" dataset is missing.\nPlease enter the default source type\nExamples: "TV Show", "Audiobook", "Audiodrama", "Newspaper Extracts"')
                if source_type_default is not None:
                    print(f'Press Enter with no input to use "{source_type_default}"')
                usr_inp = input('> ')
                assert len(usr_inp.strip()) > 0 or source_type_default, 'No input given!'
                if len(usr_inp.strip()) > 0:
                    source_type_default = usr_inp.strip()
                defaults[dataset]['source_type'] = source_type_default if len(usr_inp.strip()) == 0 else usr_inp
                print("")
        del emotion_default, noise_level_default, source_default, source_type_default, usr_inp
    
    with open(defaults_fpath, 'wb') as pickle_file:
        pickle.dump(defaults, pickle_file, pickle.HIGHEST_PROTOCOL)
    print('Done!')
    
    # add paths, transcripts, speaker names, emotions, noise levels to meta object
    print(f'Adding paths, transcripts, speaker names, emotions, noise levels from Datasets to meta...')
    for dataset in datasets:
        default_speaker     = defaults[dataset]['speaker']
        default_emotion     = defaults[dataset]['emotion']
        default_noise_level = defaults[dataset]['noise_level']
        default_source      = defaults[dataset]['source']
        default_source_type = defaults[dataset]['source_type']
        dataset_dir = os.path.join(DATASET_FOLDER, dataset)
        meta_local = get_dataset_meta(dataset_dir, audio_ext=AUDIO_FILTER, audio_rejects=AUDIO_REJECTS, default_speaker=default_speaker, default_emotion=default_emotion, default_noise_level=default_noise_level, default_source=default_source, default_source_type=default_source_type)
        meta[dataset] = meta_local
        del dataset_dir, default_speaker, default_emotion, default_noise_level, default_source, default_source_type
    print('Done!')
    
    # Assign speaker ids to speaker names
    # Write 'speaker_dataset|speaker_name|speaker_id|speaker_audio_duration' lookup table to txt file
    print(f'Loading speaker information + durations and assigning IDs...')
    #print('Ctrl + C to skip, this will remove Duration information from the speaker list')
    #try:
    import soundfile as sf
    speaker_durations = {}
    dataset_lookup = {}
    bad_paths = {}
    for dataset in tqdm(datasets):
        bad_paths[dataset] = []
        prev_wd = os.getcwd()
        os.chdir(os.path.join(DATASET_FOLDER, dataset))
        for i, clip in enumerate(meta[dataset]):
            speaker = clip['speaker']
            
            # get duration of file
            try:
                audio, sampling_rate = load_wav_to_torch(clip['path'], return_empty_on_exception=True)
                clip_duration = len(audio)/sampling_rate
                if clip_duration < MIN_DURATION:
                    bad_paths[dataset].append(clip['path'])
                    continue
            except Exception as ex:
                print('PATH:', clip['path'],"\nfailed to read.")
                bad_paths[dataset].append(clip['path'])
                continue
                #if input("Delete item? (y/n)\n> ").lower() in ['yes','y','1']:
                #    os.unlink(clip['path'])
                #    del meta[dataset][i]
                #    continue
                #else:
                #    raise Exception(ex)
            if speaker not in speaker_durations.keys():
                speaker_durations[speaker] = 0
                dataset_lookup[speaker] = dataset
            speaker_durations[speaker]+=clip_duration
        os.chdir(prev_wd)
    #except KeyboardInterrupt:
    #    print('Skipping speaker Durations')
    
    for key, bad_list in bad_paths.items():
        meta[key] = [clip for clip in meta[key] if not clip['path'] in bad_list]
    
    for dataset in datasets:
        speaker_set = set(list(speaker_durations.keys()))
        meta[dataset] = [x for x in meta[dataset] if x['speaker'] in speaker_set and speaker_durations[x['speaker']] > MIN_SPEAKER_DURATION_SECONDS]
    
    # Write speaker info to txt file
    fpath = os.path.join(DATASET_CONF_FOLDER, 'speaker_info.txt')
    with open(fpath, "w") as f:
        lines = []
        lines.append(f';{"speaker_id":<9}|{"speaker_name":<32}|{"dataset":<24}|{"source":<24}|{"source_type":<20}|duration_hrs\n;')
        for speaker_id, (speaker_name, duration) in tqdm(enumerate(speaker_durations.items())):
            dataset = dataset_lookup[speaker_name]
            if duration < MIN_SPEAKER_DURATION_SECONDS:
                continue
            try:
                clip = next(y for x in list(meta.values()) for y in x if y['speaker'] == speaker_name) # get the first clip with that speaker...
            except StopIteration as ex:
                print(speaker_name, duration, speaker_id)
                raise ex
            source = clip['source'] or "Unknown" # and pick up the source...
            source_type = clip['source_type'] or "Unknown" # and source type that speaker uses.
            assert source, 'Recieved no dataset source'
            assert source_type, 'Recieved no dataset source type'
            assert speaker_name, 'Recieved no speaker name.'
            assert duration, f'Recieved speaker "{speaker_name}" with 0 duration.'
            lines.append(f'{speaker_id:<10}|{speaker_name:<32}|{dataset:<24}|{source:<24}|{source_type:<20}|{duration/3600:>8.4f}')
        f.write('\n'.join(lines))
    print('Done!')
    
    # Unpack meta into filelist
    # filelist = [["path","quote","speaker_id"], ["path","quote","speaker_id"], ...]
    filelist = []
    for dataset, clips in meta.items():
        speaker_lookup = {speaker: index for index, speaker in enumerate(list(speaker_durations.keys()))}
        for i, clip in enumerate(clips):
            audiopath  = clip["path"]
            quote      = clip["quote"]
            speaker_id = speaker_lookup[clip['speaker']]
            if len(clip["quote"]) < 4:
                continue
            filelist.append([audiopath, quote, speaker_id])
    
    # speakerlist = [["name","id","dataset","source","source_type","duration"], ...]
    speakerlist = []
    for speaker_id, (speaker_name, duration) in enumerate(speaker_durations.items()):
        dataset = dataset_lookup[speaker_name]
        if duration < MIN_SPEAKER_DURATION_SECONDS:
            continue
        try:
            clip = next(y for x in list(meta.values()) for y in x if y['speaker'] == speaker_name) # get the first clip with that speaker...
        except StopIteration as ex:
            print(speaker_name, duration, speaker_id)
            raise ex
        source = clip['source'] or "Unknown" # and pick up the source...
        source_type = clip['source_type'] or "Unknown" # and source type that speaker uses.
        speakerlist.append((speaker_name, speaker_id, dataset, source, source_type, duration/3600.))
    
    outputs = {
        "filelist": filelist,
     "speakerlist": speakerlist,
     "speaker_ids": {i:i for i in range(len(dataset_lookup.keys()))},
    }
    return outputs

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

def tryattr(obj, attr, default=None):
    if type(attr) is str:
        attr = (attr,)
    val = None
    for attr_ in attr:
        val = val or getattr(obj, attr_, '')
    if default is None:
        assert val is not '', f'attr {attr} not found in obj'
    elif val is '':
        return default
    return val

class config:
    def __init__(self, hparams, dict_path, speaker_ids):
        hparams = AttrDict(hparams) if type(hparams) == dict else hparams
        self.get_params(hparams, dict_path, speaker_ids)
    
    def get_params(hparams, dict_path, speaker_ids):
        """Try a buncha names for params in config file. Easier than manually making each repo's config match."""
        self.win_length = tryattr(hparams, ("window_length","window_len","win_len","win_size","window_size"))
        self.hop_length = tryattr(hparams, ("hop_length","hop_len","hop_size"))
        self.fft_length = tryattr(hparams, ("filter_length","fft_length","fft_len","filter_len","filter_size","fft_size"))
        self.mel_fmin   = tryattr(hparams, ("mel_fmin","fmin","freq_min",))
        self.mel_fmax   = tryattr(hparams, ("mel_fmax","fmax","freq_max",))
        self.n_mel      = tryattr(hparams, ("n_mel","n_mel_channels","num_mels",))
        self.sampling_rate = tryattr(hparams, ("sampling_rate","sample_rate","samplerate",))
        
        self.p_arpabet     = tryattr(hparams, ("p_arpabet","use_arpabet", "arpabet", "cmudict", "use_cmudict"), 1.0)
        self.dict_path     = dict_path or tryattr(hparams, ("dict_path",))
        self.text_cleaners = tryattr(hparams, ("text_cleaners",))
        self.start_token   = tryattr(hparams, ("start_token",), '')
        self.stop_token    = tryattr(hparams, ("stop_token" ,), '')


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, filelist, hparams, args, check_files=True, TBPTT=True, shuffle=False, speaker_ids=None, audio_offset=0, verbose=False):
        self.filelist = filelist
        self.args = args
        self.force_load = hparams.force_load
        
        #####################
        ## Text / Phonemes ##
        #####################
        self.text_cleaners = hparams.text_cleaners
        self.arpa = ARPA(hparams.dict_path)
        self.p_arpabet     = hparams.p_arpabet
        
        self.emotion_classes = getattr(hparams, "emotion_classes", list())
        self.n_classes       = len(self.emotion_classes)
        self.audio_offset    = audio_offset
        
        #####################
        ## TorchMoji Embed ##
        #####################
        print(f'Tokenizing using dictionary from {VOCAB_PATH}')
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        
        self.torchmoji_tokenizer = SentenceTokenizer(vocabulary, fixed_length=120)
        
        print(f'Loading model from {PRETRAINED_PATH}.')
        self.torchmoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)
        
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
            self.check_dataset()
        
        ###############################
        ## Audio Trimming / Loudness ##
        ###############################
        self.trim_margin_left   = getattr(hparams, 'trim_margin_left'  , 0.0125                    )
        self.trim_margin_right  = getattr(hparams, 'trim_margin_right' , 0.0125                    )
        self.trim_top_db        = getattr(hparams, 'trim_top_db'       , [46  ,46  ,46  ,46  ,46  ])
        self.trim_window_length = getattr(hparams, 'trim_window_length', [8192,4096,2048,1024,512 ])
        self.trim_hop_length    = getattr(hparams, 'trim_hop_length'   , [1024,512 ,256 ,128 ,128 ])
        self.trim_ref           = getattr(hparams, 'trim_ref'          , [np.amax]*5               )
        self.trim_emphasis_str  = getattr(hparams, 'trim_emphasis_str' , [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ])
        self.trim_cache_audio   = getattr(hparams, 'trim_cache_audio'  , False)
        self.trim_enable        = getattr(hparams, 'trim_enable'      , True)
        
        self.target_lufs = getattr(hparams, 'target_lufs' , None)
        ###############################
        ## Mel-Spectrogram Generator ##
        ###############################
        self.cache_mel = getattr(hparams, "cache_mel", False)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length    = hparams.hop_length
        self.mel_fmax      = hparams.mel_fmax
        self.stft = STFT.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax, clamp_val=hparams.stft_clamp_val)
        
        # Silence Padding
        self.silence_value     = hparams.silence_value
        self.silence_pad_start = hparams.silence_pad_start# frames to pad the start of each clip
        self.silence_pad_end   = hparams.silence_pad_end  # frames to pad the end of each clip
        self.context_frames    = hparams.context_frames
        
        self.random_segments = getattr(hparams, 'random_segments', False)
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
    
    def check_dataset(self):
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
        
        print(f"{len([x for x in self.filelist if not os.path.exists(x[0])])} Files missing")
        self.filelist = [x for x in self.filelist if os.path.exists(x[0])]
        
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
        audio, sampling_rate = load_wav_to_torch(filepath, min_sr=self.mel_fmax*2, target_sr=self.sampling_rate)
        return audio, sampling_rate
    
    def trim_audio(self, audio, sr, file_path=''):
        audio = audio.numpy()
        
        trim_margin_left   = self.trim_margin_left
        trim_margin_right  = self.trim_margin_right
        trim_top_db        = self.trim_top_db
        trim_window_length = self.trim_window_length
        trim_hop_length    = self.trim_hop_length
        trim_ref           = self.trim_ref
        trim_emphasis_str  = self.trim_emphasis_str
        
        # apply audio trimming
        for i, (margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_, preemphasis_strength_) in enumerate(zip(trim_margin_left, trim_margin_right, trim_top_db, trim_window_length, trim_hop_length, trim_ref, trim_emphasis_str)):
            if type(ref_) == str:
                ref_ = getattr(np, ref_, np.amax)
            
            if preemphasis_strength_:
                sound_filt = librosa.effects.preemphasis(audio, coef=preemphasis_strength_)
                _, index = librosa.effects.trim(sound_filt, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            else:
                _, index = librosa.effects.trim(audio, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            try:
                audio = audio[int(max(index[0]-margin_left_, 0)):int(index[1]+margin_right_)]
            except TypeError:
                print(f'Slice Left:\n{max(index[0]-margin_left_, 0)}\nSlice Right:\n{index[1]+margin_right_}')
            assert len(audio), f"Audio trimmed to 0 length by pass {i+1}\nconfig = {[margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_]}\nFile_Path = '{file_path}'"
        
        return torch.from_numpy(audio)
    
    def get_mel_from_audio(self, audio, sampling_rate):
        if self.audio_offset: # used for extreme GTA'ing
            audio = audio[self.audio_offset:]
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
        melspec = self.stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0)
        return melspec
    
    def get_mel_from_ptfile(self, filepath):
        melspec = torch.load(filepath).float()
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
    
    def get_data_from_inputs(self, audiopath, text, speaker_id_ext, mel_offset=0):# get args using get_args() from CookieTTS.utils
        args = self.args
        output = {}
        
        output['audiopath'] = audiopath
        
        if True or any(arg in ['gt_audio','gt_mel','gt_frame_f0','gt_frame_energy','gt_frame_voiced','gt_char_f0','gt_char_energy','gt_char_voiced','gt_perc_loudness'] for arg in args):
            
            trimmed_audiopath = f'{os.path.splitext(audiopath)[0]}_trimaudio.pt'
            if self.trim_enable and self.trim_cache_audio and os.path.exists(trimmed_audiopath):
                audio, sampling_rate = torch.load(trimmed_audiopath), self.sampling_rate
            else:
                audio, sampling_rate = self.get_audio(audiopath)
                audio_duration = len(audio)/sampling_rate
                if self.trim_enable:
                    audio = self.trim_audio(audio, sampling_rate, file_path=audiopath)
                    if self.trim_cache_audio:
                        torch.save(audio, trimmed_audiopath)
            
            if 'gt_audio' in args:
                output['gt_audio'] = audio
            if 'sampling_rate' in args:
                output['sampling_rate'] = torch.tensor(float(sampling_rate))
        
        if 'gt_perc_loudness' in args or (self.target_lufs is not None):
            output['gt_perc_loudness'] = self.get_perc_loudness(audio, sampling_rate, audiopath, audio_duration)
        
        if self.target_lufs is not None:
            output['gt_audio'] = self.update_loudness(audio, sampling_rate, self.target_lufs,
                                                                  output['gt_perc_loudness'])
            if output['gt_audio'].abs().max() > 1.0:
                output['gt_audio'] = output['gt_audio'] / output['gt_audio'].abs().max()
            output['gt_perc_loudness'] = torch.tensor(self.target_lufs)
        
        if any([arg in ('gt_mel','dtw_pred_mel') for arg in args]):
            
            # get mel
            mel_path = os.path.splitext(audiopath)[0]+'.gt_mel.pt'
            mel = None
            if self.cache_mel and os.path.exists(mel_path):
                try:
                    mel = torch.load(mel_path)
                    mel = None if mel.shape[0] != self.stft.n_mel_channels else mel
                except:
                    mel = None
            
            if mel is None:
                mel = self.get_mel_from_audio(output['gt_audio'], sampling_rate)
                if self.cache_mel:
                    torch.save(mel, mel_path)
            
            # add silence
            mel = torch.cat((
                torch.ones(self.stft.n_mel_channels, self.silence_pad_start)*self.silence_value, # add silence to start of file
                mel,# get mel-spec as tensor from audiofile.
                torch.ones(self.stft.n_mel_channels, self.silence_pad_end)*self.silence_value, # add silence to end of file
            ), dim=1)# arr -> [n_mel, mel_T]
            
            init_mel = F.pad(mel, (self.context_frames, 0))[:, int(mel_offset):int(mel_offset)+self.context_frames]
            # initial input to the decoder. zeros if this is first segment of this file, else last frame of prev segment.
            
            output['gt_mel']   = mel
            output['init_mel'] = init_mel
            del mel, init_mel
        
        if any([arg in ('pred_mel','dtw_pred_mel') for arg in args]):
            pred_mel_path = os.path.splitext(audiopath)[0]+'.pred_mel.pt'
            if os.path.exists( pred_mel_path ):
                pred_mel = self.get_mel_from_ptfile( pred_mel_path )
                
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
        
        if 'torchmoji_hdn' in args:
            tm_path = os.path.splitext(audiopath)[0]+'_tm.pt'
            if os.path.exists(tm_path):
                torchmoji = self.get_torchmoji_hidden_from_file(tm_path)
            else:
                torchmoji = self.get_torchmoji_hidden_from_text(output['gtext_str'])
                torch.save(torchmoji, tm_path)
            output['torchmoji_hdn'] = torchmoji# [Embed]
        
        if any([arg in ('gt_frame_f0','gt_frame_voiced','gt_char_f0','gt_char_voiced') for arg in args]):
            f0, voiced_mask = self.get_pitch(output['gt_audio'], self.sampling_rate, self.hop_length)
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
            mel_offset = random.randint(0, max_start) if max_start > 0 else 0
        
        if 'gt_frame_f0' in output:
            output['gt_frame_f0']     = output['gt_frame_f0'][int(mel_offset):int(mel_offset+self.max_segment_length)]
            output['gt_frame_voiced'] = output['gt_frame_voiced'][int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'gt_frame_energy' in output:
            output['gt_frame_energy'] = output['gt_frame_energy'][int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'alignment' in output:
            output['alignment'] = output['alignment'][int(mel_offset):int(mel_offset+self.truncated_length), :]
        
        if 'gt_mel' in output:
            output['gt_mel'] = output['gt_mel'][: , int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'pred_mel' in output:
            output['pred_mel'] = output['pred_mel'][: , int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'dtw_pred_mel' in output:
            output['dtw_pred_mel'] = output['dtw_pred_mel'][: , int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        return output
    
    def get_torchmoji_hidden_from_file(self, fpath):
        return torch.load(fpath).float()
    
    def get_torchmoji_hidden_from_text(self, text):
        with torch.no_grad():
            tokenized, _, _ = self.torchmoji_tokenizer.tokenize_sentences([text,])
            embed = self.torchmoji_model(tokenized)
        return torch.from_numpy(embed).squeeze(0).float()# [Embed]
    
    def get_alignments(self, audiopath, arpa=False):
        if arpa:
            alignpath = os.path.splitext(audiopath)[0]+'_palign.npy'
        else:
            alignpath = os.path.splitext(audiopath)[0]+'_galign.npy'
        alignment = np.load(alignpath)
        return torch.from_numpy(alignment).float()
    
    def get_perc_loudness(self, audio, sampling_rate, audiopath, audio_duration):
        meter = pyln.Meter(sampling_rate) # create BS.1770 meter
        try:
            loudness = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        except Exception as ex:
            print(audio_duration, audiopath)
            raise ex
        gt_perc_loudness = torch.tensor(loudness)
        return gt_perc_loudness# []
    
    def update_loudness(self, audio, sampling_rate, target_lufs, original_lufs=None):
        if original_lufs is None:
            meter = pyln.Meter(sampling_rate) # create BS.1770 meter
            original_lufs = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        
        ddb = target_lufs-original_lufs
        audio *= ((10**(ddb*0.1))**0.5)
        return audio
    
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
    
    def get_item_from_fileline(self, index, audiopath, text, speaker_id_ext):
        filelist_index, mel_offset = self.dataloader_indexes[index]
        
        output = self.get_data_from_inputs(audiopath, text, speaker_id_ext, mel_offset=mel_offset)
        
        prev_filelist_index, prev_spectrogram_offset = self.dataloader_indexes[max(0, index-self.total_batch_size)]
        output['pres_prev_state'] = torch.tensor(True if (filelist_index == prev_filelist_index) else False)# preserve model state if this iteration is continuing the file from the last iteration.
        
        is_not_last_iter = index+self.total_batch_size < self.len
        next_filelist_index, next_spectrogram_offset = self.dataloader_indexes[index+self.total_batch_size] if is_not_last_iter else (None, None)
        output['cont_next_iter'] = torch.tensor(True if (filelist_index == next_filelist_index) else False)# whether this file continued into the next iteration
        return output
    
    def __getitem__(self, index):
        if self.shuffle and index == self.rank: # [0,3,6,9],[1,4,7,10],[2,5,8,11] # shuffle_dataset if first item of this GPU of this epoch
           self.shuffle_dataset()
        
        output = None
        if self.force_load:
            while output is None:
                try:
                    audiopath, text, speaker_id_ext, *_ = self.filelist[index]
                    output = self.get_item_from_fileline(index, audiopath, text, speaker_id_ext)
                except Exception as ex:
                    print(f"Failed to load '{audiopath}'")
                    print(ex)
                    index = random.randint(0, self.len-1)# change the audio file being loaded if this one fails to load.
        else:
            output = self.get_item_from_fileline(index, *self.filelist[index][:3])
        
        return output
    
    def __len__(self):
        return self.len


class Collate():
    def __init__(self, hparams):
        self.n_frames_per_step = hparams.n_frames_per_step
        self.n_classes = len(getattr(hparams, "emotion_classes", list()))
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
        max_len = max(x.shape[-1] if len(x.shape) else 1 for x in tensor_arr) if max_len is None else max_len
        dtype = tensor_arr[0].dtype if dtype is None else dtype
        device = tensor_arr[0].device if device is None else device
        if index_lookup is None:
            index_lookup = list(range(B))
        else:
            assert len(index_lookup) == B, f'lookup has {len(index_lookup.keys())} keys, expected {B}.'
        
        if all(not any(dim > 1 for dim in tensor_arr[i].shape) for i in range(B)):
            output = torch.zeros(B, device=device, dtype=dtype)
            for i, _ in enumerate(tensor_arr):
                output[i] = tensor_arr[index_lookup[i]].to(device, dtype)
        elif len(tensor_arr[0].shape) == 1:
            output = torch.ones(B, max_len, device=device, dtype=dtype)
            if pad_val:
                output *= pad_val
            for i, _ in enumerate(tensor_arr):
                item = tensor_arr[index_lookup[i]].to(device, dtype)
                output[i, :item.shape[0]] = item
        elif len(tensor_arr[0].shape) == 2:
            C = max(tensor_arr[i].shape[0] for i in range(B))
            if check_const_channels:
                assert all(C == item.shape[0] for item in tensor_arr), f'an item in input has channel_dim != channel_dim of the first item.\n{"nl".join(["Shape "+str(i)+" = "+str(item.shape) for i, item in enumerate(tensor_arr)])}'
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
        
        if all(type(item[key]) == torch.Tensor for item in batch):
            return self.collate_left([item[key] for item in batch], dtype=dtype, index_lookup=index_lookup, pad_val=pad_val, check_const_channels=check_const_channels)
        elif not any(type(item[key]) == torch.Tensor for item in batch):
            assert dtype is None, f'dtype specified as "{dtype}" but input has no Tensors.'
            arr = [item[key] for item in batch]
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
        out = {}
        B = len(batch)# Batch Size
        
        if self.sort_text_len_decending and all("text" in item for item in batch):# if text, reorder entire batch to go from longest text -> shortest text.
            text_lengths, ids_sorted = torch.sort(
                torch.LongTensor([len(item['text']) for item in batch]), dim=0, descending=True)
            out['text_lengths'] = text_lengths
        else:
            ids_sorted = list(range(B))# elif no text, batch can be whatever order it's loaded in
        
        if all("gt_mel" in item for item in batch):
            out['mel_lengths'] = torch.tensor([batch[ids_sorted[i]]['gt_mel'].shape[-1] for i in range(B)])
        elif all("pred_mel" in item for item in batch):
            out['mel_lengths'] = torch.tensor([batch[ids_sorted[i]]['pred_mel'].shape[-1] for i in range(B)])
        elif all("gt_frame_f0" in item for item in batch):
            out['mel_lengths'] = torch.tensor([batch[ids_sorted[i]]['gt_frame_f0'].shape[-1] for i in range(B)])
        
        out['text']              = self.collatek(batch, 'text',              ids_sorted, dtype=torch.long )# [B, txt_T]
        out['gtext_str']         = self.collatek(batch, 'gtext_str',         ids_sorted, dtype=None       )# [str, ...]
        out['ptext_str']         = self.collatek(batch, 'ptext_str',         ids_sorted, dtype=None       )# [str, ...]
        out['text_str']          = self.collatek(batch, 'text_str',          ids_sorted, dtype=None       )# [str, ...]
        out['audiopath']         = self.collatek(batch, 'audiopath',         ids_sorted, dtype=None       )# [str, ...]
        
        out['alignments']        = self.collatea(batch, 'alignments',        ids_sorted, dtype=torch.float)# [B, mel_T, txt_T]
        
        out['gt_sylps']          = self.collatek(batch, 'gt_sylps',          ids_sorted, dtype=torch.float)# [B]
        out['torchmoji_hdn']     = self.collatek(batch, 'torchmoji_hdn',     ids_sorted, dtype=torch.float)# [B, C]
        
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
        
       #out['gt_gate_logits']    = \
        if all('gt_mel' in item for item in batch):
            mel_T = max(item['gt_mel'].shape[-1] for item in batch)
            out['gt_gate_logits'] = torch.zeros(B, mel_T, dtype=torch.float)
            for i in range(B):
                out['gt_gate_logits'][i, out['mel_lengths'][i]-(~batch[ids_sorted[i]]['cont_next_iter']).long():] = 1
                # set positive gate if this file isn't going to be continued next iter.
                # (i.e: if this is the last segment of the file.)
        
        out = {k:v for k,v in out.items() if v is not None} # remove any entries with "None" values.
        
        return out
