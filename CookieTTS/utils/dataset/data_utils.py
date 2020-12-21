import random
import traceback
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
from scipy.signal import butter, sosfilt
try:
    import pyworld as pw
except Exception as ex:
    print(ex)
    print("Warning! pyworld not imported.")
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

# speaker encoder
#from CookieTTS.utils.dataset.autovc_speaker_encoder.make_metadata import get_speaker_encoder
from CookieTTS.utils.dataset.resem.voice_encoder import VoiceEncoder
# misc
import syllables


def latest_modified_date(directory):
    return max(os.stat(root).st_mtime for root,_,_ in os.walk(directory))

# https://stackoverflow.com/a/43116588
def latest_modified_date_filtered(directory, exts=['.wav','.flac','.txt']):
    def filepaths(directory):
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                yield os.path.join(root, filename)
    files = (filepath for filepath in filepaths(directory) if any(filepath.lower().endswith(x) for x in exts))
    latest = max(files, key=os.path.getmtime)
    return os.stat(latest).st_mtime

def generate_filelist_from_datasets(DATASET_FOLDER,
        DATASET_CONF_FOLDER=None,
        AUDIO_FILTER  = ['*.wav'],
        AUDIO_REJECTS = ['*_Noisy_*','*_Very Noisy_*'],
        MIN_DURATION  =   0.73,
        MAX_DURATION  =  30.00,
        MIN_CHAR_LEN  =   7,
        MAX_CHAR_LEN  = 160,
        MIN_SPEAKER_DURATION=10.0,
        valid_time =4*3600.0,# set to 4 hours (which should be enough time to complete 1 epoch hopefully) # time_after_last_check_before_modifications_invalidate_the_dataset
        rank=0):
    if DATASET_CONF_FOLDER is None:
        DATASET_CONF_FOLDER = os.path.join(DATASET_FOLDER, 'meta')
    
    # define meta dict (this is where all the data is collected)
    should_reload_data = True
    if os.path.exists(os.path.join(DATASET_CONF_FOLDER, 'metadata.pkl')) and os.path.exists(os.path.join(DATASET_CONF_FOLDER, 'metadata_lm_dates.pkl')):
        dict_ = pickle.load(open(os.path.join(DATASET_CONF_FOLDER, 'metadata.pkl'), "rb"))
        should_reload_data = (
          dict_.get("DATASET_FOLDER", None)       != DATASET_FOLDER or
          dict_.get("DATASET_CONF_FOLDER", None)  != DATASET_CONF_FOLDER or
          dict_.get("AUDIO_FILTER", None)         != AUDIO_FILTER or
          dict_.get("AUDIO_REJECTS", None)        != AUDIO_REJECTS or
          dict_.get("MIN_DURATION", 0.73)         != MIN_DURATION or
          dict_.get("MAX_DURATION", 30.0)         != MAX_DURATION or
          dict_.get("MIN_CHAR_LEN",    7)         != MIN_CHAR_LEN or
          dict_.get("MAX_CHAR_LEN",  160)         != MAX_CHAR_LEN or
          dict_.get("MIN_SPEAKER_DURATION", None) != MIN_SPEAKER_DURATION
        )
        if not should_reload_data:
            meta              = dict_["meta"]
            speaker_durations = dict_["speaker_durations"]
            dataset_lookup    = dict_["dataset_lookup"]
            bad_paths         = dict_["bad_paths"]
            meta_lm = pickle.load(open(os.path.join(DATASET_CONF_FOLDER, 'metadata_lm_dates.pkl'), "rb"))
            for dataset, last_dataset_check in meta_lm.items():
                if os.path.exists(os.path.join(DATASET_FOLDER, dataset)):
                    last_modified = latest_modified_date_filtered(os.path.join(DATASET_FOLDER, dataset))
                    if dataset in meta and last_modified > valid_time+last_dataset_check:
                        print(f"Dataset {dataset} being reloaded.")
                        del meta[dataset]
                        del bad_paths[dataset]
                        for speaker in [x for x in speaker_durations.keys() if dataset_lookup[x] == dataset]:
                            del speaker_durations[speaker]
                            del dataset_lookup[speaker]
                else:
                    del meta[dataset]
                    del bad_paths[dataset]
                    for speaker in [x for x in speaker_durations.keys() if dataset_lookup[x] == dataset]:
                        del speaker_durations[speaker]
                        del dataset_lookup[speaker]
        del dict_
    
    if should_reload_data:
        meta = {}
        meta_lm = None
        speaker_durations = {}
        dataset_lookup    = {}
        bad_paths         = {}
    
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
    
    # list of datasets that will be processed over the next while.
    datasets = sorted([x for x in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, x)) and 'meta' not in x])
    
    if meta_lm is None:
        unchecked_datasets = datasets
    else:
        unchecked_datasets = [dataset for dataset in datasets if dataset not in meta]
    
    print(f'Checking Defaults for Datasets...')
    speaker_default = None
    for dataset in datasets:
        if dataset not in defaults:
            defaults[dataset] = {}
        
        # load defaults from .txt files if exists.
        if 'speaker' not in defaults[dataset] and os.path.exists(os.path.join(DATASET_FOLDER, dataset, 'default_speaker.txt')):
            defaults[dataset]['speaker'    ] = open(os.path.join(DATASET_FOLDER, dataset, 'default_speaker.txt'), 'r', encoding='utf8').read()
        if 'emotion' not in defaults[dataset] and os.path.exists(os.path.join(DATASET_FOLDER, dataset, 'default_emotion.txt')):
            defaults[dataset]['emotion'    ] = open(os.path.join(DATASET_FOLDER, dataset, 'default_emotion.txt'), 'r', encoding='utf8').read()
        if 'noise_level' not in defaults[dataset] and os.path.exists(os.path.join(DATASET_FOLDER, dataset, 'default_noise_level.txt')):
            defaults[dataset]['noise_level'] = open(os.path.join(DATASET_FOLDER, dataset, 'default_noise_level.txt'), 'r', encoding='utf8').read()
        if 'source' not in defaults[dataset] and os.path.exists(os.path.join(DATASET_FOLDER, dataset, 'default_source.txt')):
            defaults[dataset]['source'     ] = open(os.path.join(DATASET_FOLDER, dataset, 'default_source.txt'), 'r', encoding='utf8').read()
        if 'source_type' not in defaults[dataset] and os.path.exists(os.path.join(DATASET_FOLDER, dataset, 'default_source_type.txt')):
            defaults[dataset]['source_type'] = open(os.path.join(DATASET_FOLDER, dataset, 'default_source_type.txt'), 'r', encoding='utf8').read()
        
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
                print(f'default source type for "{dataset}" dataset is missing.\nPlease enter the default source type\nExamples: "Show", "Audiobook", "Audiodrama", "Game", "Newspaper Extracts"')
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
    print('Done!\n')
    
    # add paths, transcripts, speaker names, emotions, noise levels to meta object
    print(f'Adding paths, transcripts, speaker names, emotions, noise levels from Datasets to meta...')
    for dataset in unchecked_datasets:
        default_speaker     = defaults[dataset]['speaker']
        default_emotion     = defaults[dataset]['emotion']
        default_noise_level = defaults[dataset]['noise_level']
        default_source      = defaults[dataset]['source']
        default_source_type = defaults[dataset]['source_type']
        dataset_dir = os.path.join(DATASET_FOLDER, dataset)
        meta_local = get_dataset_meta(dataset_dir, audio_ext=AUDIO_FILTER, audio_rejects=AUDIO_REJECTS, default_speaker=default_speaker, default_emotion=default_emotion, default_noise_level=default_noise_level, default_source=default_source, default_source_type=default_source_type)
        meta[dataset] = meta_local
        print("\n\n")
        del dataset_dir, default_speaker, default_emotion, default_noise_level, default_source, default_source_type
    print('Done!\n')
    
    # Assign speaker ids to speaker names
    # Write 'speaker_dataset|speaker_name|speaker_id|speaker_audio_duration' lookup table to txt file
    print(f'Loading speaker information + durations, assigning IDs and writing speaker_info.txt to datasets meta/config folder...')
    import soundfile as sf
    speaker_durations = speaker_durations if speaker_durations else {}
    dataset_lookup    = dataset_lookup    if dataset_lookup    else {}
    bad_paths         = bad_paths         if bad_paths         else {}
    
    total_files = sum(len(clips) for clips in [v for k,v in meta.items() if k in unchecked_datasets])
    #for k, v in meta.items():
    #    if k in unchecked_datasets:
    #        for clip in v:
    #            out.append(clip)
    files = [x['path'] for k,v in meta.items() if k in unchecked_datasets for x in v]
    def get_duration(fpath):
        audio, sampling_rate = load_wav_to_torch(fpath, min_sr=22049.0, return_empty_on_exception=True)
        clip_duration = len(audio)/sampling_rate
        return clip_duration
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    clip_durations = {}
    with tqdm(total=total_files) as pbar:
        with ThreadPoolExecutor(max_workers=32) as ex:
            futures = [ex.submit(get_duration, path) for path in files]
            for i, future in enumerate(futures):
                path = files[i]
                try:
                    clip_duration = future.result()
                    clip_durations[path] = clip_duration
                except Exception as exc:
                    print('PATH:', path, "\nfailed to read.")
                    print(traceback.format_exc())
                    #bad_paths[dataset].append(path)
                    #if input("Delete item? (y/n)\n> ").lower() in ['yes','y','1']:
                    #    os.unlink(path)
                    #    del meta[dataset][i]
                    #    continue
                    #else:
                    #    raise Exception(exc)
                pbar.update(1)

    with tqdm(total=total_files) as pbar:
        for dataset in unchecked_datasets:
            bad_paths[dataset] = []
            prev_wd = os.getcwd()
            os.chdir(os.path.join(DATASET_FOLDER, dataset))
            for i, clip in enumerate(meta[dataset]):
                speaker = clip['speaker']
                pbar.update()
                
                # get duration of file
                clip_duration = clip_durations[clip['path']]
                if clip_duration < MIN_DURATION or clip_duration > MAX_DURATION:
                    bad_paths[dataset].append(clip['path'])
                    continue
                if speaker not in speaker_durations.keys():
                    speaker_durations[speaker] = 0.
                    dataset_lookup[speaker] = dataset
                speaker_durations[speaker]+=clip_duration
            os.chdir(prev_wd)
    del total_files
    
    time.sleep(0.1)
    
    total_files = sum(len(clips) for clips in meta.values())
    for key, bad_list in bad_paths.items():
        bad_list_set = set(bad_list)
        meta[key] = [clip for clip in meta[key] if not clip['path'] in bad_list_set]
    new_total_files = sum(len(clips) for clips in meta.values())
    if total_files-new_total_files > 0:
        print(f"Removed {total_files-new_total_files} Short or Corrupted Audio Files.\n{new_total_files} File Remain.")
    
    total_files = new_total_files
    for dataset in datasets:
        speaker_set = set(list(speaker_durations.keys()))
        meta[dataset] = [x for x in meta[dataset] if x['speaker'] in speaker_set and speaker_durations[x['speaker']] > MIN_SPEAKER_DURATION]
    new_total_files = sum(len(clips) for clips in meta.values())
    if total_files-new_total_files > 0:
        print(f"Removed {total_files-new_total_files} Files from speakers that have too little data.\n{new_total_files} File Remain.")
    
    ####################################################
    ## Save meta so it doesn't have to be regenerated ##
    ####################################################
    print("Saving 'metadata.plk' and 'metadata_lm_dates.pkl' to save all the new data")
    meta_fpath = os.path.join(DATASET_CONF_FOLDER, 'metadata.pkl')
    with open(meta_fpath, 'wb') as pickle_file:
        out = {
                  "meta": meta,
     "speaker_durations": speaker_durations,
        "dataset_lookup": dataset_lookup,
             "bad_paths": bad_paths,
        "DATASET_FOLDER": DATASET_FOLDER,
   "DATASET_CONF_FOLDER": DATASET_CONF_FOLDER,
          "AUDIO_FILTER": AUDIO_FILTER,
         "AUDIO_REJECTS": AUDIO_REJECTS,
          "MIN_DURATION": MIN_DURATION,
          "MAX_DURATION": MAX_DURATION,
          "MIN_CHAR_LEN": MIN_CHAR_LEN,
          "MAX_CHAR_LEN": MAX_CHAR_LEN,
  "MIN_SPEAKER_DURATION": MIN_SPEAKER_DURATION,
        }
        pickle.dump(out, pickle_file, pickle.HIGHEST_PROTOCOL)
    
    meta_lm = {}
    for dataset in datasets:
        meta_lm[dataset] = latest_modified_date_filtered(os.path.join(DATASET_FOLDER, dataset))
    meta_lm_fpath = os.path.join(DATASET_CONF_FOLDER, 'metadata_lm_dates.pkl')
    with open(meta_lm_fpath, 'wb') as pickle_file:
        pickle.dump(meta_lm, pickle_file, pickle.HIGHEST_PROTOCOL)
    print('Done!\n')
    
    #################################
    ## Start dumping meta to files ##
    #################################
    # Write speaker info to txt file
    fpath = os.path.join(DATASET_CONF_FOLDER, 'speaker_info.txt')
    with open(fpath, "w") as f:
        lines = []
        lines.append(f';{"dataset":<23}|{"speaker_name":<32}|{"speaker_id":<10}|{"source":<24}|{"source_type":<20}|duration_hrs\n;')
        for speaker_id, speaker_name in enumerate(sorted(speaker_durations.keys())):
            duration = speaker_durations[speaker_name]
            dataset = dataset_lookup[speaker_name]
            if duration < MIN_SPEAKER_DURATION:
                continue
            try:
                clip = next(y for x in list(meta.values()) for y in x if y['speaker'] == speaker_name) # get the first clip with that speaker...
            except StopIteration as ex:
                print(speaker_name, duration, speaker_id)
                raise ex
            source      = clip['source'     ].replace('\n','') or "Unknown" # and pick up the source...
            source_type = clip['source_type'].replace('\n','') or "Unknown" # and source type that speaker uses.
            assert source, 'Recieved no dataset source'
            assert source_type, 'Recieved no dataset source type'
            assert speaker_name, 'Recieved no speaker name.'
            assert duration, f'Recieved speaker "{speaker_name}" with 0 duration.'
            lines.append(f'{dataset:<24}|{speaker_name:<32}|{speaker_id:<10}|{source:<24}|{source_type:<20}|{duration/3600:>8.4f}')
        f.write('\n'.join(lines))
    print('Done!\n')
    
    # Unpack meta into filelist
    # filelist = [["path","quote","speaker_id"], ["path","quote","speaker_id"], ...]
    speaker_lookup = {speaker: index for index, speaker in enumerate(sorted(speaker_durations.keys()))}
    filelist = []
    for dataset, clips in meta.items():
        for i, clip in enumerate(clips):
            audiopath  = clip["path"]
            quote      = clip["quote"]
            speaker_id = speaker_lookup[clip['speaker']]
            
            if len(clip["quote"]) < MIN_CHAR_LEN or len(clip["quote"]) > MAX_CHAR_LEN:
                continue
            if speaker_durations[clip['speaker']] < MIN_SPEAKER_DURATION:
                continue
            filelist.append([audiopath, quote, speaker_id])
    
    # speakerlist = [["name","id","dataset","source","source_type","duration"], ...]
    speakerlist = []
    for speaker_id, speaker_name in enumerate(sorted(speaker_durations.keys())):
        duration = speaker_durations[speaker_name]
        dataset = dataset_lookup[speaker_name]
        if duration < MIN_SPEAKER_DURATION:
            continue
        try:
            clip = next(y for x in list(meta.values()) for y in x if y['speaker'] == speaker_name) # get the first clip with that speaker...
        except StopIteration as ex:
            print(speaker_name, duration, speaker_id)
            raise ex
        source      = clip['source'] or "Unknown" # and pick up the source...
        source_type = clip['source_type'] or "Unknown" # and source type that speaker uses.
        speakerlist.append((dataset, str(speaker_name), speaker_id, source, source_type, duration/3600.))
    
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
        assert val != '', f'attr {attr} not found in obj'
    elif val == '':
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
    def __init__(self, filelist, hparams, args, check_files=True, TBPTT=True, shuffle=False, deterministic_arpabet=False, speaker_ids=None, audio_offset=0, verbose=False):
        self.filelist = filelist
        self.args = args
        self.force_load = hparams.force_load
        
        #####################
        ## Text / Phonemes ##
        #####################
        if any([arg in ('text', 'gt_sylps', 'alignment','gt_char_f0','gt_char_voiced','gt_char_energy','torchmoji_hdn') for arg in args]):
            self.text_cleaners = hparams.text_cleaners
            self.arpa = ARPA(hparams.dict_path)
            self.p_arpabet     = hparams.p_arpabet
            self.start_token = hparams.start_token
            self.stop_token  = hparams.stop_token
            self.max_chars_length = hparams.max_chars_length# chars # doesn't get used unless TBPTT and random segments are disabled.
            
            if 'torchmoji_hdn' in args:
                #####################
                ## TorchMoji Embed ##
                #####################
                print(f'Tokenizing using dictionary from {VOCAB_PATH}')
                with open(VOCAB_PATH, 'r') as f:
                    vocabulary = json.load(f)
                
                self.torchmoji_tokenizer = SentenceTokenizer(vocabulary, fixed_length=120)
                
                print(f'Loading model from {PRETRAINED_PATH}.')
                self.torchmoji_model = torchmoji_feature_encoding(PRETRAINED_PATH)
        
        self.emotion_classes = getattr(hparams, "emotion_classes", list())
        self.n_classes       = len(self.emotion_classes)
        self.audio_offset    = audio_offset
        
        #################
        ## Speaker IDs ##
        #################
        self.n_speakers = hparams.n_speakers
        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            if hasattr(hparams, 'raw_speaker_ids') and hparams.raw_speaker_ids:
                self.speaker_ids = {k:k for k in range(self.n_speakers)} # map IDs in files directly to internal IDs
            else:
                self.speaker_ids = self.create_speaker_lookup_table(self.filelist, numeric_sort=hparams.numeric_speaker_ids)
        assert len(self.speaker_ids.values()) <= self.n_speakers, f'More speakers found in dataset(s) than set in hparams.py n_speakers\nFound {len(self.speaker_ids.values())} Speakers, Expected {hparams.n_speakers} or Less.'
        
        ###################
        ## File Checking ##
        ###################
        if check_files:
            self.check_dataset()
        
        #####################
        ## Speaker Encoder ##
        #####################
        if any(arg in ('parallel_speaker_embed','non_parallel_speaker_embed') for arg in args):
            self.speaker_encoder = VoiceEncoder()
            
            if hparams.rank == 0:# generate speaker embeddings in advance
                self.speaker_encoder.cuda()
                self.pregenerate_speaker_embeddings()
                self.speaker_encoder.cpu()
        
        ###############################
        ## Audio Trimming / Loudness ##
        ###############################
        self.filt_min_freq = getattr(hparams, 'filt_min_freq' ,    60)
        self.filt_max_freq = getattr(hparams, 'filt_max_freq' , 18000)
        self.filt_order    = getattr(hparams, 'filt_order'    ,     6)
        self.trim_margin_left   = getattr(hparams, 'trim_margin_left'  , 0.0125                    )
        self.trim_margin_right  = getattr(hparams, 'trim_margin_right' , 0.0125                    )
        self.trim_top_db        = getattr(hparams, 'trim_top_db'       , [46  ,46  ,46  ,46  ,46  ])
        self.trim_window_length = getattr(hparams, 'trim_window_length', [8192,4096,2048,1024,512 ])
        self.trim_hop_length    = getattr(hparams, 'trim_hop_length'   , [1024,512 ,256 ,128 ,128 ])
        self.trim_ref           = getattr(hparams, 'trim_ref'          , ['amax']*5               )
        self.trim_emphasis_str  = getattr(hparams, 'trim_emphasis_str' , [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ])
        self.trim_cache_audio   = getattr(hparams, 'trim_cache_audio'  , False)
        self.trim_enable        = getattr(hparams, 'trim_enable'       , True)
        
        self.target_lufs = getattr(hparams, 'target_lufs' , None)
        ###############################
        ## Mel-Spectrogram Generator ##
        ###############################
        self.cache_mel = False if audio_offset > 0 else getattr(hparams, "cache_mel", False)
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length    = hparams.hop_length
        self.mel_fmax      = hparams.mel_fmax
        self.stft = STFT.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax, clamp_val=hparams.stft_clamp_val)
        
        # Silence Padding
        self.silence_value     = getattr(hparams, 'silence_value'    , -11.5129)
        self.silence_pad_start = getattr(hparams, 'silence_pad_start', 0)# frames to pad the start of each clip
        self.silence_pad_end   = getattr(hparams, 'silence_pad_end'  , 0)# frames to pad the  end  of each clip
        self.context_frames    = getattr(hparams, 'context_frames'   , 1)
        
        self.random_segments = getattr(hparams, 'random_segments', False)
        ################################################
        ## (Optional) Apply weighting to MLP Datasets ##
        ################################################
        if False:
            duplicated_audiopaths = [x for x in self.filelist if "SlicedDialogue" in x[0]]
            for i in range(3):
                self.filelist.extend(duplicated_audiopaths)
        
        # Shuffle Audiopaths
        self.random_seed = hparams.seed
        random.seed(hparams.seed)
        random.shuffle(self.filelist)
        
        self.deterministic_arpabet = deterministic_arpabet
        #####################################################################
        ## PREDICT LENGTH (TBPTT) - Truncated Backpropagation through time ##
        #####################################################################
        # simulate the entire epoch so the decoder can be linked together between iters.
        self.shuffle    = shuffle
        self.TBPTT      = TBPTT
        
        self.use_TBPTT  = hparams.use_TBPTT
        self.batch_size = hparams.batch_size
        self.rank       = hparams.rank
        self.total_batch_size   = hparams.batch_size * hparams.n_gpus # number of audio files being processed together
        self.max_segment_length = hparams.max_segment_length # frames
        
        self.update_filelist(self.filelist)
    
    def update_filelist(self, filelist):
        self.filelist = filelist
        
        if self.use_TBPTT and self.TBPTT:
            loaded_durs = False
            
            trim_config = [self.filt_min_freq, self.filt_max_freq, self.filt_order, self.trim_margin_left, self.trim_margin_right, self.trim_top_db, self.trim_window_length, self.trim_hop_length, self.trim_ref, self.trim_emphasis_str]
            durpath = f'audio_lengths.pt'
            if (not loaded_durs) and os.path.exists(durpath):
                try:
                    dict_=torch.load(durpath)
                    if dict_['trim_config'] == trim_config:
                        clip_durations = dict_['clip_durations']
                        loaded_durs = True
                    del dict_
                except:
                    print(traceback.format_exc())
            
            if not loaded_durs:
                clip_durations = {}
            
            print('Calculating audio lengths of all files...')
            
            def get_duration(fpath):
                audio, sampling_rate = self.get_trimmed_audio(fpath)
                clip_duration = len(audio)/sampling_rate
                return clip_duration, fpath
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            _ = get_duration(self.filelist[0][0])
            
            with tqdm(total=len(self.filelist)) as pbar:
                with ThreadPoolExecutor(max_workers=16) as ex:
                    missing_paths = [x[0] for x in self.filelist if not x[0] in clip_durations]
                    futures = [ex.submit(get_duration, x) for x in missing_paths]
                    for future in as_completed(futures):
                        dur, path = future.result()
                        clip_durations[path] = dur
                        pbar.update(1)
            self.audio_lengths = torch.tensor([(round(clip_durations[x[0]]*self.sampling_rate)//self.hop_length)+1+self.silence_pad_start+self.silence_pad_end for x in tqdm(self.filelist)], device='cpu') # get the length of every file (the long way)
            if self.rank == 0:
                torch.save({"clip_durations": clip_durations, "trim_config": trim_config}, durpath)
            print('Done.')
        else:
            self.audio_lengths = torch.tensor([self.max_segment_length-1 for x in self.filelist], device='cpu') # use dummy lengths
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
        # prep outputs/inputs
        dataloader_indexes = []
        
        total_batch_size   = self.total_batch_size
        max_segment_length = self.max_segment_length
        audio_lengths = self.audio_lengths.cpu()
        audio_indexes = torch.arange(len(audio_lengths))
        n_audio_lengths = len(audio_lengths)
        
        # init first iter
        batch_indexes = audio_indexes[:total_batch_size].clone()
        batch_lengths = audio_lengths[:total_batch_size].clone()
        batch_lengths_remaining = batch_lengths         .clone()
        activated_files = len(batch_indexes)# number of files either in processing or already processed
        
        # start loop
        while True:
            # add item to this iter's batch
            batch_frame_offset = batch_lengths-batch_lengths_remaining
            dataloader_indexes.extend(list(zip(batch_indexes.numpy(), batch_frame_offset.numpy())))
            
            # simulate processing iter
            batch_lengths_remaining-=max_segment_length
            
            # replace items in batch if they have reached the end of their respective audio sequences
            b_finished_items = batch_lengths_remaining<=0   # finished item mask
            n_finished_items = b_finished_items.sum().item()# number of items finished this pass
            if activated_files+n_finished_items>n_audio_lengths:# if there are not enough items left to keep the next batch filled, stop processing
                break
            batch_indexes          [b_finished_items] = audio_indexes[activated_files:activated_files+n_finished_items]
            batch_lengths          [b_finished_items] = audio_lengths[activated_files:activated_files+n_finished_items]
            batch_lengths_remaining[b_finished_items] = audio_lengths[activated_files:activated_files+n_finished_items]
            activated_files += n_finished_items# number of items started so far (from all passes)
        
        self.dataloader_indexes = dataloader_indexes
        self.len = len(self.dataloader_indexes)
    
    def check_dataset(self):
        print("Checking dataset files...")
        audiopaths_length = len(self.filelist)
        banned_paths = []
        music_stuff = True
        
        # if the dataloader will load text, do this preprocessing stuff too
        if any([arg in ('text', 'gt_sylps', 'alignment','gt_char_f0','gt_char_voiced','gt_char_energy','torchmoji_hdn') for arg in self.args]) or hasattr(self, 'start_token') or hasattr(self, 'stop_token'):
            filtered_chars = ["☺","␤"]
            banned_strings = ["[","]"]
            start_token = self.start_token
            stop_token  = self.stop_token
            #for index, file in enumerate(self.filelist): # index must use seperate iterations from remove
            #    if music_stuff and r"Songs/" in file[0]:
            #        self.filelist[index][1] = "♫" + self.filelist[index][1] + "♫"
            #    for filtered_char in filtered_chars:
            #        self.filelist[index][1] = self.filelist[index][1].replace(filtered_char,"")
            #    self.filelist[index][1] = start_token + self.filelist[index][1] + stop_token
            
            def filter_multi(inp, filtered_chars):
                out = inp
                for filtered_char in filtered_chars:
                    out = out.replace(filtered_char,"")
                return out
            
            self.filelist = [[line[0],f'♫{line[1]}♫' if r"Songs/" in line[0] else line[1], *line[2:]] for index, line in enumerate(self.filelist)]
            
            self.filelist = [[line[0],f'{start_token}{filter_multi(line[1], filtered_chars)}{stop_token}', *line[2:]] for index, line in enumerate(self.filelist)]
        
        len_data = len(self.filelist)
        self.filelist = [x for x in self.filelist if os.path.exists(x[0])]
        print(f"{len_data-len(self.filelist)} Files not found and being ignored.")
        
        print("Done checking files!")
        print(audiopaths_length, "items in metadata file")
        print(len(self.filelist), "validated and being used for training.")
    
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
        audio, sampling_rate = load_wav_to_torch(filepath, min_sr=(self.mel_fmax*2.)-1., target_sr=self.sampling_rate)
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        return audio, sampling_rate
    
    def get_trimmed_audio(self, audiopath):
        audio_loaded = False
        
        trim_config = [self.filt_min_freq, self.filt_max_freq, self.filt_order, self.trim_margin_left, self.trim_margin_right, self.trim_top_db, self.trim_window_length, self.trim_hop_length, self.trim_ref, self.trim_emphasis_str]
        trimmed_audiopath = f'{os.path.splitext(audiopath)[0]}_trimaudio.pt'
        if self.trim_enable and self.trim_cache_audio and os.path.exists(trimmed_audiopath):
            try:
                dict_ = torch.load(trimmed_audiopath)
                if dict_['trim_config'] == trim_config:
                    audio_loaded = True
                    audio         = dict_['audio']
                    sampling_rate = dict_['sampling_rate']
                del dict_
            except:
                print(traceback.format_exc())
        
        if not audio_loaded:
            audio, sampling_rate = self.get_audio(audiopath)
            if self.trim_enable:
                audio = self.trim_audio(audio, sampling_rate, file_path=audiopath)
                if self.trim_cache_audio and self.rank == 0:
                    torch.save({'audio': audio.data, 'sampling_rate': sampling_rate, 'trim_config': trim_config}, trimmed_audiopath)
            audio_loaded = True
        return audio, sampling_rate
    
    def trim_audio(self, audio, sr, file_path=''):
        audio = audio.numpy()
        
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            sos = butter(order, [low, high], analog=False, btype='band', output='sos')
            return sos

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, sos=None):
            if sos is None:
                sos = butter_bandpass(lowcut, highcut, fs, order=order)
            y = sosfilt(sos, data)
            return y
        
        filt_min_freq = self.filt_min_freq
        filt_max_freq = self.filt_max_freq
        filt_order    = self.filt_order
        trim_margin_left   = self.trim_margin_left
        trim_margin_right  = self.trim_margin_right
        trim_top_db        = self.trim_top_db
        trim_window_length = self.trim_window_length
        trim_hop_length    = self.trim_hop_length
        trim_ref           = self.trim_ref
        trim_emphasis_str  = self.trim_emphasis_str
        
        if not hasattr(self, 'filt_sos'):
            self.filt_sos = butter_bandpass(filt_min_freq, filt_max_freq, self.sampling_rate, order=filt_order)
        
        # bandpass filter audio to decrease energy under/above 40Hz and 18000Hz
        if all(x is not None for x in [filt_min_freq, filt_max_freq, filt_order]) and filt_order > 0:
            audio = butter_bandpass_filter(audio.astype('float64'), filt_min_freq, filt_max_freq, self.sampling_rate, order=filt_order, sos=self.filt_sos)
        
        audio = np.clip(audio, -1.0, 1.0)
        
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
        
        return torch.from_numpy(audio).clamp(max=1.0, min=-1.0).float()
    
    def get_mel_from_audio(self, audio, sampling_rate):
        if self.audio_offset: # used for extreme GTA'ing
            audio = audio[self.audio_offset:]
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
        melspec = self.stft.mel_spectrogram(audio.detach().cpu().unsqueeze(0)).squeeze(0)
        return melspec
    
    def get_mel_from_ptfile(self, filepath):
        melspec = torch.load(filepath).float()
        assert melspec.size(0) == self.stft.n_mel_channels, (f'Mel dimension mismatch: given {melspec.size(0)}, expected {self.stft.n_mel_channels}')
        return melspec
    
    def get_mel_from_audiopath(self, audiopath):
        audio, sampling_rate = self.get_trimmed_audio(audiopath)
        melspec = self.get_mel_from_audio(audio, sampling_rate)
        return melspec
    
    def get_alignment_from_npfile(self, filepath, text_length, spec_length):
        melspec = torch.from_numpy(np.load(filepath)).float()
        assert melspec.shape[0] == spec_length, f"Saved Alignment has wrong decoder length, got {melspec.shape[0]}, expected {spec_length}"
        assert melspec.shape[1] == text_length, f"Saved Alignment has wrong encoder length, got {melspec.shape[1]}, expected {text_length}"
        return melspec
    
    def indexes_to_one_hot(self, indexes, num_classes=None):
        """Converts a vector of indexes to a batch of one-hot vectors. """
        indexes = indexes.type(torch.int64).view(-1, 1)
        num_classes = num_classes if num_classes is not None else int(torch.max(indexes)) + 1
        one_hots = torch.zeros(indexes.size()[0], num_classes).scatter_(1, indexes, 1)
        one_hots = one_hots.view(*indexes.shape, -1)
        return one_hots
    
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
    
    def get_rand_audiopath_from_speaker(self, speaker_id_ext):
        audiopaths = [x[0] for x in self.filelist if x[2] == speaker_id_ext]
        new_audiopath = random.choice(audiopaths)
        return new_audiopath
    
    def get_speaker_encoder_embed(self, audiopath):
        assert hasattr(self, 'speaker_encoder')
        try:
            if os.path.exists(os.path.splitext(audiopath)[0]+'_spkemb.pt'):
                speaker_embed = torch.load(os.path.splitext(audiopath)[0]+'_spkemb.pt').data.float()
            else:
                with torch.no_grad():
                    speaker_embed = self.speaker_encoder.get_embed_from_path(audiopath).data.float()
                torch.save(speaker_embed, os.path.splitext(audiopath)[0]+'_spkemb.pt')
        except Exception as ex:
            print(traceback.format_exc())
            speaker_embed = torch.zeros(768).float()
        return speaker_embed
    
    def pregenerate_speaker_embeddings(self):
        for audiopath, quote, speaker_id_ext, *_ in tqdm(self.filelist):
            try:
                self.get_speaker_encoder_embed(audiopath)
            except Exception as ex:
                print(f"Failed to load '{audiopath}'")
                print(traceback.format_exc())
    
    def get_data_from_inputs(self, audiopath, text, speaker_id_ext, pred_mel_T:int=0, mel_offset=0):# get args using get_args() from CookieTTS.utils
        args = self.args
        output = {}
        
        output['audiopath'     ] = audiopath
        output['gtext_str'     ] = text
        output['speaker_id_ext'] = speaker_id_ext
        
        if True or any(arg in ['gt_audio','gt_mel','gt_frame_f0','gt_frame_energy','gt_frame_voiced','gt_char_f0','gt_char_energy','gt_char_voiced','gt_perc_loudness'] for arg in args):
            audio, sampling_rate = self.get_trimmed_audio(audiopath)
            audio_duration = len(audio)/sampling_rate
            if 'gt_audio' in args:
                output['gt_audio'] = audio
            if 'sampling_rate' in args:
                output['sampling_rate'] = torch.tensor(float(load_wav_to_torch(audiopath)[1]))
        
        if 'gt_perc_loudness' in args or (self.target_lufs is not None):
            output['gt_perc_loudness'] = self.get_perc_loudness(audio, sampling_rate, audiopath, audio_duration)
        
        if self.target_lufs is not None:
            output['gt_audio'] = self.update_loudness(audio, sampling_rate, self.target_lufs, output['gt_perc_loudness'])
            output['gt_perc_loudness'] = self.get_perc_loudness(audio, sampling_rate, audiopath, audio_duration)
        
        if any([arg in ('gt_mel','dtw_pred_mel') for arg in args]):
            
            # get mel
            mel_path = os.path.splitext(audiopath)[0]+'.gt_mel.pt'
            mel = None
            if self.cache_mel and os.path.exists(mel_path):
                try:
                    mel = torch.load(mel_path).cpu().float()
                    assert mel.shape[0] == self.stft.n_mel_channels
                except:
                    mel = None
            
            if mel is None:
                mel = self.get_mel_from_audio(output['gt_audio'], sampling_rate)
                if self.cache_mel:
                    torch.save(mel, mel_path)
            
            if (self.use_TBPTT and self.TBPTT) and pred_mel_T and pred_mel_T != mel.shape[-1]:
                print(f"TBPTT calcuated wrong mel_T, got {pred_mel_T}, expected {mel.shape[-1]}!")
            
            # add silence
            mel = torch.cat((
                torch.ones(mel.shape[0], self.silence_pad_start)*self.silence_value,# add silence to start of file
                mel,# get mel-spec as tensor from audiofile.
                torch.ones(mel.shape[0], self.silence_pad_end  )*self.silence_value,# add silence to end of file
                ), dim=1)# arr -> [n_mel, mel_T]
            
            output['gt_mel'] = mel
            output['remaining_mel_length'] = mel.shape[-1]# this key gets updated again later if truncation/slicing is used
            
            if self.context_frames:# initial input to the decoder. zeros if this is first segment of this file, else last frame of prev segment.
                output['init_mel'] = F.pad(mel, (self.context_frames, 0))
        
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
            del pred_mel_path
        
        if 'gt_sylps' in args:
            gt_sylps = self.get_syllables_per_second(text, len(audio)/sampling_rate)# [] FloatTensor
            output['gt_sylps'] = gt_sylps.float()
            del gt_sylps
        
        if any([arg in ('text', 'gt_sylps', 'alignment','gt_char_f0','gt_char_voiced','gt_char_energy','torchmoji_hdn') for arg in args]):
            output['ptext_str'] = self.arpa.get(text)
            del text
            
            use_phones = random.Random(audiopath).random() < self.p_arpabet if self.deterministic_arpabet else random.random() < self.p_arpabet
            output['arpa'] = use_phones
            output['text_str'] = output['ptext_str'] if use_phones else output['gtext_str']
            if 'text' in args:
                output['text'] = output['ptext_str'] if use_phones else output['gtext_str']# (randomly) convert to phonemes
                output['text'] = self.get_text(output['text'])# convert text into tensor representation
                if self.random_segments is False and (self.use_TBPTT and self.TBPTT) is False:# if not using TBPTT or Random Segments:
                    output['text'] = output['text'][:self.max_chars_length]                   #   cut of the excess text that 99% won't be used in the first segment of audio.
        
        if any(arg in ('speaker_id','speaker_id_onehot') for arg in args):
            output['speaker_id'] = self.get_speaker_id(speaker_id_ext)# get speaker_id as tensor normalized [ 0 -> len(speaker_ids) ]
            output['speaker_id_onehot'] = self.indexes_to_one_hot(output['speaker_id'], num_classes=self.n_speakers).squeeze().float()# [n_speakers]
        
        if any(arg in ('parallel_speaker_embed',) for arg in args):
            output['parallel_speaker_embed'] = self.get_speaker_encoder_embed(audiopath)
        
        if any(arg in ('non_parallel_speaker_embed',) for arg in args):
            # get another audiopath for this speaker
            other_audiopath = self.get_rand_audiopath_from_speaker(speaker_id_ext)
            output['non_parallel_speaker_embed'] = self.get_speaker_encoder_embed(other_audiopath)
        
        if any([arg in ['gt_emotion_id','gt_emotion_onehot'] for arg in args]):
            output['gt_emotion_id'] = self.get_emotion_id(audiopath)# [1] IntTensor
            gt_emotion_onehot = self.indexes_to_one_hot(gt_emotion_id, num_classes=self.n_classes+1).squeeze(0)[:-1]# [n_classes]
            output['gt_emotion_onehot'] = gt_emotion_onehot.float()
        
        if 'torchmoji_hdn' in args:
            tm_path = os.path.splitext(audiopath)[0]+'_tm.pt'
            if os.path.exists(tm_path):
                torchmoji = self.get_torchmoji_hidden_from_file(tm_path).float()
            else:
                torchmoji = self.get_torchmoji_hidden_from_text(output['gtext_str']).float()
                if not os.path.exists(tm_path):
                    torch.save(torchmoji, tm_path)
            output['torchmoji_hdn'] = torchmoji# [Embed]
        
        if any([arg in ('gt_frame_f0','gt_frame_voiced','gt_char_f0','gt_char_voiced') for arg in args]):
            f0, voiced_mask = self.get_pitch(output['gt_audio'], self.sampling_rate, self.hop_length)
            output['gt_frame_f0']     = f0.float()
            output['gt_frame_voiced'] = voiced_mask
        
        if any([arg in ('gt_frame_energy','gt_char_energy') for arg in args]):
            output['gt_frame_energy'] = self.get_energy(mel).float()
        
        if any([arg in ('alignment','gt_char_f0','gt_char_voiced','gt_char_energy') for arg in args]):
            output['alignment'] = alignment = self.get_alignments(audiopath, arpa=use_phones)
            if 'gt_char_f0' in args:
                output['gt_char_f0']     = self.get_charavg_from_frames(f0                 , alignment).float()# [txt_T]
            if 'gt_char_voiced' in args:
                output['gt_char_voiced'] = self.get_charavg_from_frames(voiced_mask.float(), alignment)# [txt_T]
            if 'gt_char_energy' in args:
                output['gt_char_energy'] = self.get_charavg_from_frames(output['gt_frame_energy'], alignment).float()# [txt_T]
            if 'gt_char_dur' in args:
                output['gt_char_dur'] = alignment.sum(0).float()# [mel_T, txt_T] -> [txt_T]
        
        if 'diagonality' in args:
            diagonality_path = f'{os.path.splitext(audiopath)[0]}_diag.pt'
            if os.path.exists(diagonality_path):
                output['diagonality'] = torch.load(diagonality_path).float()
            else:
                output['diagonality'] = torch.tensor(1.08).float()
        
        if 'avg_prob' in args:
            avg_prob_path = f'{os.path.splitext(audiopath)[0]}_avgp.pt'
            if os.path.exists(avg_prob_path):
                output['avg_prob'] = torch.load(avg_prob_path).float()
            else:
                output['avg_prob'] = torch.tensor(0.6).float()
        
        ########################
        ## Trim into Segments ##
        ########################
        if self.random_segments is True:
            max_start = mel.shape[-1] - self.max_segment_length
            mel_offset = random.randint(0, max_start) if max_start > 0 else 0
        
        if 'gt_frame_f0' in output:
            output['gt_frame_f0']     = output['gt_frame_f0'][int(mel_offset):int(mel_offset+self.max_segment_length)]
            output['gt_frame_voiced'] = output['gt_frame_voiced'][int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'gt_frame_energy' in output:
            output['gt_frame_energy'] = output['gt_frame_energy'][int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'alignment' in output:
            output['alignment'] = output['alignment'][int(mel_offset):int(mel_offset+self.max_segment_length), :]
        
        if 'init_mel' in output:
            output['init_mel'] = output['init_mel'][:, int(mel_offset):int(mel_offset)+self.context_frames]
        
        if 'gt_mel' in output:
            assert int(mel_offset) < output['gt_mel'].shape[-1], f"TBPTT calcuated mel_T or mel_offset is invalid! got index of {mel_offset} while mel is {output['gt_mel'].shape[-1]} frames long."
            output[  'gt_mel'] = output[  'gt_mel'][: , int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'pred_mel' in output:
            output['pred_mel'] = output['pred_mel'][: , int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'dtw_pred_mel' in output:
            output['dtw_pred_mel'] = output['dtw_pred_mel'][: , int(mel_offset):int(mel_offset+self.max_segment_length)]
        
        if 'gt_audio' in output:
            output['gt_audio'] = output['gt_audio'][int(mel_offset)*self.hop_length:int(mel_offset+self.max_segment_length)*self.hop_length]
        
        if 'remaining_mel_length' in output:
            output['remaining_mel_length'] -= int(mel_offset)
        
        ##################
        ##   Clean up   ##
        ##################
        if not any(arg in ['gt_audio',] for arg in args):
            del output['gt_audio']
        
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
            alignpath = os.path.splitext(audiopath)[0]+'_palign.pt'
        else:
            alignpath = os.path.splitext(audiopath)[0]+'_galign.pt'
        return torch.load(alignpath).float()
    
    def get_perc_loudness(self, audio, sampling_rate, audiopath, audio_duration):
        meter = pyln.Meter(sampling_rate) # create BS.1770 meter
        try:
            loudness = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        except Exception as ex:
            print(audio_duration, audiopath)
            raise ex
        gt_perc_loudness = torch.tensor(loudness).float()
        return gt_perc_loudness# []
    
    def update_loudness(self, audio, sampling_rate, target_lufs, original_lufs=None):
        if original_lufs is None:
            meter = pyln.Meter(sampling_rate) # create BS.1770 meter
            original_lufs = meter.integrated_loudness(audio.numpy()) # measure loudness (in dB)
        
        if type(original_lufs) == torch.Tensor:
            original_lufs = original_lufs.to(audio)
        ddb = target_lufs-original_lufs
        audio = audio*((10**(ddb*0.1))**0.5)
        if audio.abs().max():
            audio /= audio.abs().max()
        return audio
    
    def get_charavg_from_frames(self, x, alignment):# [mel_T], [mel_T, txt_T]
        norm_alignment = alignment / alignment.sum(dim=0, keepdim=True).clamp(min=0.01)
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
    
    def get_item_from_fileline(self, index, filelist_index, mel_offset, audiopath, text, speaker_id_ext):
        pred_mel_T = self.audio_lengths[filelist_index] if hasattr(self, 'audio_lengths') else None
        
        output = self.get_data_from_inputs(audiopath, text, speaker_id_ext, pred_mel_T, mel_offset=mel_offset)
        
        prev_filelist_index, prev_spectrogram_offset = self.dataloader_indexes[max(0, index-self.total_batch_size)]
        output['pres_prev_state'] = torch.tensor(bool(filelist_index == prev_filelist_index))# preserve model state if this iteration is continuing the file from the last iteration.
        
        is_not_last_iter = index+self.total_batch_size < self.len
        next_filelist_index, next_spectrogram_offset = self.dataloader_indexes[index+self.total_batch_size] if is_not_last_iter else (None, None)
        output['cont_next_iter'] = torch.tensor(bool(filelist_index == next_filelist_index))# whether this file continued into the next iteration
        return output
    
    def __getitem__(self, index):
        with torch.no_grad():
            return self.getitem(index)
    
    def getitem(self, index):
        output = None
        n_fails= 0
        if self.force_load:
            while output is None:
                try:
                    start_time = time.time()
                    filelist_index, mel_offset = self.dataloader_indexes[index]
                    audiopath, text, speaker_id_ext, *_ = self.filelist[filelist_index]
                    output = self.get_item_from_fileline(index, filelist_index, mel_offset, audiopath, text, speaker_id_ext)
                    elapsed_time = time.time()-start_time
                    
                    warning_load_time = 12.0 # if file takes longer than 12 seconds to load, print the audiopath for the user to inspect.
                    if elapsed_time > warning_load_time:
                        print(f"File took {elapsed_time:.1f}s to load!\n'{audiopath}'")
                except Exception as ex:
                    print(f"Failed to load '{audiopath}'")
                    print(traceback.format_exc())
                    index = random.randint(0, self.len-1)# change the audio file being loaded if this one fails to load.
                    n_fails+=1
                    if n_fails > 20:
                        break
        else:
            filelist_index, mel_offset = self.dataloader_indexes[index]
            audiopath, text, speaker_id_ext, *_ = self.filelist[filelist_index]
            output = self.get_item_from_fileline(index, filelist_index, mel_offset, audiopath, text, speaker_id_ext)
        
        return output
    
    def __len__(self):
        return self.len


class Collate():
    def __init__(self, hparams):
        self.sort_text_len_decending = getattr(hparams, 'sort_text_len_decending', False)
        self.random_segments = getattr(hparams, 'random_segments', False)
        self.segment_length  = getattr(hparams, 'max_segment_length', 8192)
        self.hop_length      = getattr(hparams, 'hop_length', 1)
    
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
            output = torch.ones(B, max_len, device=device, dtype=dtype).fill_(pad_val)
            for i, _ in enumerate(tensor_arr):
                item = tensor_arr[index_lookup[i]].to(device, dtype)
                output[i, :item.shape[0]] = item
        elif len(tensor_arr[0].shape) == 2:
            C = max(tensor_arr[i].shape[0] for i in range(B))
            if check_const_channels:
                assert all(C == item.shape[0] for item in tensor_arr), f'an item in input has channel_dim != channel_dim of the first item.\n{"nl".join(["Shape "+str(i)+" = "+str(item.shape) for i, item in enumerate(tensor_arr)])}'
            output = torch.ones(B, C, max_len, device=device, dtype=dtype).fill_(pad_val)
            for i, _ in enumerate(tensor_arr):
                item = tensor_arr[index_lookup[i]].to(device, dtype)
                output[i, :item.shape[0], :item.shape[1]] = item
        else:
            raise Exception(f"Unexpected input shape, got {len(tensor_arr[0].shape)} dims and expected 1 or 2 dims.")
        assert not (torch.isnan(output) | torch.isinf(output)).any(), 'NaN or Inf value found in computation'
        return output
    
    def collatea(self, *args, check_const_channels=False, **kwargs):
        return self.collatek(*args, check_const_channels=check_const_channels, **kwargs)
    
    def collatek(self, batch, key, index_lookup, dtype=None, pad_val=0.0, ignore_missing_key=True, check_const_channels=True):
        if ignore_missing_key and not any(key in item for item in batch):
            return None
        else:
            assert all(key in item for item in batch), f'item in batch is missing key "{key}"'
        
        max_len = None
        if self.random_segments:
            if key in ['gt_mel','pred_mel','dtw_pred_mel','gt_frame_f0','gt_frame_energy','gt_frame_voiced',]:
                max_len = self.segment_length
            elif key in ['gt_audio',]:
                max_len = self.segment_length * self.hop_length
        
        if all(type(item[key]) == torch.Tensor for item in batch):
            return self.collate_left([item[key] for item in batch], dtype=dtype, index_lookup=index_lookup, max_len=max_len, pad_val=pad_val, check_const_channels=check_const_channels)
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
        
        if all("text" in item for item in batch):
            out['text_lengths'] = torch.LongTensor([len(item['text']) for item in batch])
            if self.sort_text_len_decending:# if text, reorder entire batch to go from longest text -> shortest text.
                text_lengths, ids_sorted = torch.sort(out['text_lengths'], dim=0, descending=True)
                out['text_lengths'] = text_lengths
            else:
                ids_sorted = list(range(B))# elif len_decending sorting is disabled
        else:   ids_sorted = list(range(B))# else no text, batch can be whatever order it's loaded in
        
        if all("gt_mel" in item for item in batch):
            out['mel_lengths'] = torch.tensor([batch[ids_sorted[i]]['gt_mel'].shape[-1] for i in range(B)])
        elif all("pred_mel" in item for item in batch):
            out['mel_lengths'] = torch.tensor([batch[ids_sorted[i]]['pred_mel'].shape[-1] for i in range(B)])
        elif all("gt_frame_f0" in item for item in batch):
            out['mel_lengths'] = torch.tensor([batch[ids_sorted[i]]['gt_frame_f0'].shape[-1] for i in range(B)])
        
        if all("gt_audio" in item for item in batch):
            out['audio_lengths'] = torch.tensor([batch[ids_sorted[i]]['gt_audio'].shape[-1] for i in range(B)])
        
        out['text']              = self.collatek(batch, 'text',              ids_sorted, dtype=torch.long )# [B, txt_T]
        out['arpa']              = self.collatek(batch, 'arpa',              ids_sorted, dtype=None       )# [bool, ...]
        out['gtext_str']         = self.collatek(batch, 'gtext_str',         ids_sorted, dtype=None       )# [str, ...]
        out['ptext_str']         = self.collatek(batch, 'ptext_str',         ids_sorted, dtype=None       )# [str, ...]
        out['text_str']          = self.collatek(batch, 'text_str',          ids_sorted, dtype=None       )# [str, ...]
        out['audiopath']         = self.collatek(batch, 'audiopath',         ids_sorted, dtype=None       )# [str, ...]
        
        out['alignments']        = self.collatea(batch, 'alignments',        ids_sorted, dtype=torch.float)# [B, mel_T, txt_T]
        
        out['gt_sylps']          = self.collatek(batch, 'gt_sylps',          ids_sorted, dtype=torch.float)# [B]
        out['diagonality']       = self.collatek(batch, 'diagonality',       ids_sorted, dtype=torch.float)# [B]
        out['avg_prob']          = self.collatek(batch, 'avg_prob',          ids_sorted, dtype=torch.float)# [B]
        
        out['torchmoji_hdn']     = self.collatek(batch, 'torchmoji_hdn',     ids_sorted, dtype=torch.float)# [B, C]
        
        out['speaker_id_ext']    = self.collatek(batch, 'speaker_id_ext',    ids_sorted, dtype=None       )# [int, ...]
        out['speaker_id']        = self.collatek(batch, 'speaker_id',        ids_sorted, dtype=torch.long )# [B]
        out['speaker_id_onehot'] = self.collatek(batch, 'speaker_id_onehot', ids_sorted, dtype=torch.long )# [B]
        
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
        
        out['parallel_speaker_embed']     = self.collatek(batch, 'parallel_speaker_embed',     ids_sorted, dtype=torch.float)# [B, C]
        out['non_parallel_speaker_embed'] = self.collatek(batch, 'non_parallel_speaker_embed', ids_sorted, dtype=torch.float)# [B, C]
        
        out['remaining_mel_lengths']      = self.collatek(batch, 'remaining_mel_length',       ids_sorted, dtype=None       )# [int, ...]
        
       #out['gt_gate_logits']    = \
        if all('gt_mel' in item for item in batch):
            out['gt_gate_logits'] = torch.zeros(B, out['gt_mel'].shape[-1], dtype=torch.float)
            for i in range(B):
                out['gt_gate_logits'][i, out['remaining_mel_lengths'][i]-1:] = 1.
                # set positive gate if this file isn't going to be continued next iter (i.e: if this is the last segment of the file.)
        
        out = {k:v for k,v in out.items() if v is not None} # remove any entries with "None" values.
        
        return out
