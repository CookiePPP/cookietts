import os

# save preprocessing directory for later
preprocess_dir = os.path.abspath(os.path.split(__file__)[0])
os.chdir(preprocess_dir)
print(f"preprocess_dir = '{preprocess_dir}'")

# load preprocessing config file
import json
with open("config.json") as f:
    conf = json.loads(f.read())

DATASET_FOLDER = os.path.abspath(conf['DATASET_FOLDER'])
DATASET_CONF_FOLDER = os.path.abspath(conf['DATASET_CONF_FOLDER'])
THREADS = conf['THREADS']
OUTPUT_SAMPLE_RATE = conf['OUTPUT_SAMPLE_RATE']
OUTPUT_BIT_DEPTH = conf['OUTPUT_BIT_DEPTH']
DICT_PATH = conf['DICT_PATH']
MIN_SPEAKER_DURATION_SECONDS = conf['MIN_SPEAKER_DURATION_SECONDS']
REGENERATE_EMOTION_INFO = conf['REGENERATE_EMOTION_INFO']
TRAIN_PERCENT = conf['TRAIN_PERCENT']

from CookieTTS.utils.text.ARPA import ARPA
arpa = ARPA(DICT_PATH)

# load downloads config file
with open("../_0_download/config.json") as f:
    dconf = json.loads(f.read())

# define recursive blob, this is just searches a path recursively for files.
from glob import glob
def globr(*args, **kwargs):
    """recurive glob"""
    return glob(*args, recursive=True, **kwargs)

def step_2_1():
    """extract compressed files from downloads into preprocessed folder for processing."""
    # chdir to downloads dir
    os.chdir(os.path.join("../_0_download", dconf['downloads_folder']))
    
    def clone_directory_structure(inputpath, outputpath):
        """copy folder structure of inputpath to outputpath"""
        assert inputpath != outputpath
        import os
        for dirpath, dirnames, filenames in os.walk(inputpath):
            structure = os.path.join(outputpath, dirpath[len(inputpath)+1:])
            if not os.path.isdir(structure):
                os.mkdir(structure)
    
    # copy folders without content from downloads folder into `1_preprocess/datasets` folder.
    clone_directory_structure(os.getcwd(), DATASET_FOLDER)
    
    # symlink all compressed files to `1_preprocess/datasets`, extract, then unlink the compressed ones.
    from shutil import copyfile
    from CookieTTS.utils.dataset.extract_unknown import extract
    compressed_files = [*globr("**/*.7z"), *globr("**/*.zip"), *globr("**/*.rar"), *globr("**/*.tar"), *globr("**/*.tar.bz2")]
    for compressed_file in compressed_files: # for compressed file in compressed files, extract to datasets folder.
        symlink_source = os.path.abspath(compressed_file)
        symlink_dest = os.path.join(DATASET_FOLDER, compressed_file)
        if not os.path.exists(symlink_dest):
            print(f'"{symlink_source}" ---> "{symlink_dest}"')
            try:
                os.symlink(symlink_source, symlink_dest)
            except:
                copyfile(symlink_source, symlink_dest)
        extract(symlink_dest)
        os.unlink(symlink_dest)
    
    # and extract compressed files inside the compressed files.
    compressed_files = [*globr("**/*.7z"), *globr("**/*.zip"), *globr("**/*.rar"), *globr("**/*.tar"), *globr("**/*.tar.bz2")]
    while len(compressed_files):
        extract(compressed_files[0])
        os.unlink(compressed_files[0])
        compressed_files = [*globr("**/*.7z"), *globr("**/*.zip"), *globr("**/*.rar"), *globr("**/*.tar"), *globr("**/*.tar.bz2")]
    
    # move back to preprocess directorys
    os.chdir(preprocess_dir)
    
def step_2_2():
    """move remaining folders/files to preprocessed datasets folder."""
    # chdir to downloads dir
    os.chdir(os.path.join("../_0_download", dconf['downloads_folder']))
    
    # copy all remaining files to `1_preprocess/datasets`
    from shutil import copyfile
    from glob import glob
    
    # get all files not in compressed files
    all_files = globr("**/*.*")
    compressed_files = [*globr("**/*.7z"), *globr("**/*.zip"), *globr("**/*.rar"), *globr("**/*.tar"), *globr("**/*.tar.bz2")]
    remaining_files = [x for x in all_files if x not in compressed_files]
    
    # copy every remaining file into datasets directory
    for file in remaining_files:
        if os.path.isdir(file):
            continue
        source = os.path.abspath(file)
        dest = os.path.join(DATASET_FOLDER, file)
        if not os.path.exists(dest):
            print(f'"{source}" ---> "{dest}"')
            try: # try hardlink...
                os.link(source, dest)
            except OSError: # if different partition...
                copyfile(source, dest) # then copy file instead
            except Exception as ex:
                print(ex)
                raise Exception(ex)
    
    # move back to preprocess directorys
    os.chdir(preprocess_dir)

# extract compressed files from downloads into preprocessed folder for processing.
if 0:
    step_2_1()

# move remaining folders/files to preprocessed datasets folder
if 0:
    step_2_2()

#################################################################################
### for Blizzard2011                                                          ###
###   Slice clips from original Studio files                                  ###
#################################################################################
dataset = 'Blizzard2011'
if dconf['Blizzard2011']['download']:
    dataset_dir = os.path.join(DATASET_FOLDER, dataset)
    #os.chdir(dataset_dir)
    from slice_bliazzard2011 import NancySplitRawIntoClips, NancyWriteTranscripts
    NancySplitRawIntoClips(dataset_dir)
    NancyWriteTranscripts(dataset_dir)
    del dataset_dir
del dataset


#################################################################################
### for VCTK                                                                  ###
###   delete/ignore microphone not in use.                                    ###
###   High-pass filter                                                        ###
###   Low-Pass filter                                                         ###
###   Resample to target sample rate. # note, need to save sr info somewhere. ###
###   Aggressive Trimming                                                     ###
###   Save the ['path','sample_rate'] pairs                                   ###
#################################################################################
path_sample_rates = {}

dataset = 'VCTK'
if dconf['VCTK']['download']:
    import numpy as np
    from scripts.audio_preprocessing import multiprocess_directory, process_audio_multiprocess
    dataset_dir = os.path.join(DATASET_FOLDER, dataset)
    print("Filtering and Trimming VCTK Dataset")
    #        Pass |  1  |  2  |  3  |
    type        = ['hp' ,'hp' ,'lp' ]
    cutoff_freq = [150  ,40   ,18000]
    order       = [4    ,9    ,9    ]
    
    #    Pass |  1  |  2  |  3  |  4  |  5  |  6  |
    ref_db   =[60   ,44   ,40   ,40   ,42   ,44   ]
    win_len  =[16000,9600 ,4800 ,2400 ,1200 ,600  ]
    hop_len  =[1800 ,1200 ,600  ,300  ,150  ,150  ]
    ref_f    =[np.amax]*6
    empth    =[0.0  ,0.0  ,0.0  ,0.0  ,0.0  ,0.0  ]
    margin_l =[0    ,0    ,0    ,0    ,0    ,0    ]
    margin_r =[0    ,0    ,0    ,0    ,0    ,0    ]
    def func(array):
        return process_audio_multiprocess(array,
            filt_type=type,
            filt_cutoff_freq=cutoff_freq,
            filt_order=order,
            trim_margin_left=margin_l,
            trim_margin_right=margin_r,
            trim_top_db=ref_db,
            trim_window_length=win_len,
            trim_hop_length=hop_len,
            trim_ref=ref_f,
            trim_preemphasis_strength=empth,
            SAMPLE_RATE=OUTPUT_SAMPLE_RATE,
            ignore_dirs=[],
        )
    print(f'Currently Processing folder: "{dataset_dir}"')
    multiproc_path_srs = multiprocess_directory(func, dataset_dir, threads=THREADS)
    merged_path_srs = {k:v for list_item in multiproc_path_srs for (k,v) in list_item.items()}
    path_sample_rates = {**path_sample_rates, **merged_path_srs}
    print('Done!')
    del dataset_dir, type, cutoff_freq, order, ref_db, win_len, hop_len, ref_f, empth, margin_l, margin_r, multiproc_path_srs, merged_path_srs
del dataset

##################################################################################
### for all audio                                                              ###
###   High-pass filter                                                         ###
###   Low-Pass filter                                                          ###
###   Resample to target sample rate. # note, need to save sr info somewhere.  ###
###   Trim                                                                     ###
##################################################################################
if 0:
    import numpy as np
    from scripts.audio_preprocessing import multiprocess_directory, process_audio_multiprocess
    # lp/hp filter and trim global
    # Filter Pass |  1  |  2  |  3  |
    type        = ['hp' ,'hp' ,'lp' ]
    cutoff_freq = [150  ,40   ,18000]
    order       = [4    ,9    ,9    ]
    
    #Trim Pass |  1 |  2 |  3 |  4 |  5 |
    ref_db   = [46  ,46  ,46  ,46  ,46  ]
    win_len  = [9600,4800,2400,1200,600 ]
    hop_len  = [1200,600 ,300 ,150 ,150 ]
    ref_f    = [np.amax]*5
    empth    = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ]
    margin_l = [0   ,0   ,0   ,0   ,0   ]
    margin_r = [0   ,0   ,0   ,0   ,0   ]
    print("Filtering and Trimming All Datasets (other than VCTK)")
    for i in range(len(type)):
        print(f"Filter Pass {i+1}, Global {cutoff_freq[i]}Hz {'highpass' if type[i]=='hp' else 'lowpass'} filter.")
    for i in range(len(win_len)):
        print(f"Trimming Pass {i+1}:\n\tWindow Length = {win_len[i]},\n\tHop Length = {hop_len[i]},\n\tReference db = {ref_db[i]},\n\tMargin Left = {margin_l[i]}")
    
    def func(array):
        return process_audio_multiprocess(array,
            filt_type=type,
            filt_cutoff_freq=cutoff_freq,
            filt_order=order,
            trim_margin_left=margin_l,
            trim_margin_right=margin_r,
            trim_top_db=ref_db,
            trim_window_length=win_len,
            trim_hop_length=hop_len,
            trim_ref=ref_f,
            trim_preemphasis_strength=empth,
            SAMPLE_RATE=OUTPUT_SAMPLE_RATE,
            ignore_dirs=["VCTK",],#ignore VCTK
        )
    print(f'Currently Processing folder: "{DATASET_FOLDER}"')
    multiproc_path_srs = multiprocess_directory(func, DATASET_FOLDER, threads=THREADS)
    merged_path_srs = {k:v for list_item in multiproc_path_srs for (k,v) in list_item.items()}
    path_sample_rates = {**path_sample_rates, **merged_path_srs}
    print('Done!')
    del type, cutoff_freq, order, ref_db, win_len, hop_len, ref_f, empth, margin_l, margin_r, multiproc_path_srs, merged_path_srs

##################################################################################
### for all audio                                                              ###
###   Equalize volumes/amplitudes                                              ###
##################################################################################
if 0:
    from scripts.audio_preprocessing import normalize_volumes_mixmode
    normalize_volumes_mixmode(DATASET_FOLDER, amplitude=0.08, ext='.wav')


##################################################
### Collect all metadata into central object   ###
###  - audio paths                             ###
###  - transcripts                             ###
###  - speaker names                           ###
###  - emotions                                ###
###  - noise levels                            ###
###  - source                                  ###
###  - source type                             ###
###  - native sample rate                      ###
##################################################
if True:
    # define meta dict (this is where all the data is collected)
    meta = {}
    
    # list of datasets that will be processed over the next while.
    datasets = [x for x in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, x))]
    
    # check default configs exist (and prompt user for datasets without premade configs)
    for dataset in datasets:
        dataset_conf_dir = os.path.join(DATASET_CONF_FOLDER, dataset)
        if not os.path.exists(dataset_conf_dir):
            os.makedirs(dataset_conf_dir)
        if not os.path.exists(os.path.join(dataset_conf_dir, 'default_speaker.txt')):
            with open(os.path.join(dataset_conf_dir, 'default_speaker.txt'), 'w') as f:
                f.write( input(f'default speaker for "{dataset}" dataset is missing.\nPlease enter the name of the default speaker\nExamples: "Nancy", "Littlepip"\n> ') )
                print('')
        if not os.path.exists(os.path.join(dataset_conf_dir, 'default_emotion.txt')):
            with open(os.path.join(dataset_conf_dir, 'default_emotion.txt'), 'w') as f:
                f.write( input(f'default emotion for "{dataset}" dataset is missing.\nPlease enter the default emotion\nExamples: "Neutral", "Bored", "Audiobook"\n> ') )
                print('')
        if not os.path.exists(os.path.join(dataset_conf_dir, 'default_noise_level.txt')):
            with open(os.path.join(dataset_conf_dir, 'default_noise_level.txt'), 'w') as f:
                f.write( input(f'default noise level for "{dataset}" dataset is missing.\nPlease enter the default noise level\nExamples: "Clean", "Noisy", "Very Noisy"\n> ') )
                print('')
        if not os.path.exists(os.path.join(dataset_conf_dir, 'default_source.txt')):
            with open(os.path.join(dataset_conf_dir, 'default_source.txt'), 'w') as f:
                f.write( input(f'default source for "{dataset}" dataset is missing.\nPlease enter the default source\nExamples: "My Little Pony", "Team Fortress 2", "University of Edinburgh"\n> ') )
                print('')
        if not os.path.exists(os.path.join(dataset_conf_dir, 'default_source_type.txt')):
            with open(os.path.join(dataset_conf_dir, 'default_source_type.txt'), 'w') as f:
                f.write( input(f'default source type for "{dataset}" dataset is missing.\nPlease enter the default source type\nExamples: "TV Show", "Audiobook", "Audiodrama", "Newspaper Extracts", "Game"\n> ') )
                print('')
        del dataset_conf_dir
    
    # add paths, transcripts, speaker names, emotions, noise levels to meta object
    from scripts.metadata import get_dataset_meta
    
    for dataset in datasets:
        dataset_dir = os.path.join(DATASET_FOLDER, dataset)
        dataset_conf_dir = os.path.join(DATASET_CONF_FOLDER, dataset)
        default_speaker = open(os.path.join(dataset_conf_dir, 'default_speaker.txt'), 'r').read().strip().strip('"')
        assert default_speaker, f'default_speaker from\n"{os.path.join(dataset_conf_dir, "default_speaker.txt")}"\nis invalid.'
        default_emotion = open(os.path.join(dataset_conf_dir, 'default_emotion.txt'), 'r').read().strip().strip('"')
        assert default_emotion, f'default_emotion from\n"{os.path.join(dataset_conf_dir, "default_emotion.txt")}"\nis invalid.'
        default_noise_level = open(os.path.join(dataset_conf_dir, 'default_noise_level.txt'), 'r').read().strip().strip('"')
        assert default_noise_level, f'default_noise_level from\n"{os.path.join(dataset_conf_dir, "default_noise_level.txt")}"\nis invalid.'
        default_source = open(os.path.join(dataset_conf_dir, 'default_source.txt'), 'r').read().strip().strip('"')
        assert default_source, f'default_source from\n"{os.path.join(dataset_conf_dir, "default_source.txt")}"\nis invalid.'
        default_source_type = open(os.path.join(dataset_conf_dir, 'default_source_type.txt'), 'r').read().strip().strip('"')
        assert default_source_type, f'default_source_type from\n"{os.path.join(dataset_conf_dir, "default_source_type.txt")}"\nis invalid.'
        meta_local = get_dataset_meta(dataset_dir, default_speaker=default_speaker, default_emotion=default_emotion, default_noise_level=default_noise_level, default_source=default_source, default_source_type=default_source_type)
        meta[dataset] = meta_local
        del dataset_dir, dataset_conf_dir, default_speaker, default_emotion, default_noise_level, default_source, default_source_type
    
    # add native sample rates to the meta object
    for dataset in datasets:
        for clip in meta[dataset]:
            try:
                meta[dataset][clip]['sample_rate'] = path_sample_rates[os.path.abspath(clip['path'])]
            except KeyError:
                pass
    
    # Assign speaker ids to speaker names
    # Write 'speaker_dataset|speaker_name|speaker_id|speaker_audio_duration' lookup table to txt file
    import soundfile as sf
    speaker_durations = {}
    dataset_lookup = {}
    for dataset in meta.keys():
        prev_wd = os.getcwd()
        os.chdir(os.path.join(DATASET_FOLDER, dataset))
        for i, clip in enumerate(meta[dataset]):
            speaker = clip['speaker']
            if speaker not in speaker_durations.keys():
                speaker_durations[speaker] = 0
                dataset_lookup[speaker] = dataset
            
            # get duration of file
            if '.wav' in clip['path']:
                clip_duration = os.stat(clip['path']).st_size / (OUTPUT_SAMPLE_RATE*OUTPUT_BIT_DEPTH)
            else:
                try:
                    clip_duration = len(sf.read(clip['path'], always_2d=True)[:0]) / OUTPUT_SAMPLE_RATE
                except Exception as ex:
                    print('PATH:', clip['path'],"\nfailed to read.")
                    if input("Delete item? (y/n)\n> ").lower() in ['yes','y','1']:
                        os.unlink(clip['path'])
                        del meta[dataset][i]
                        continue
                    else:
                        raise Exception(ex)
            speaker_durations[speaker]+=clip_duration
        os.chdir(prev_wd)
    
    # Write speaker info to txt file
    with open('speaker_info.txt', "w") as f:
        lines = []
        lines.append(f';{"dataset":<23}|{"source":<24}|{"source_type":<20}|{"speaker_name":<32}|{"speaker_id":<10}|duration_hrs\n;')
        for speaker_id, (speaker_name, duration) in enumerate(speaker_durations.items()):
            dataset = dataset_lookup[speaker_name]
            clip = next(x for x in meta[dataset] if x['speaker'] == speaker_name) # get the first clip with that speaker...
            source = clip['source'] or "Unknown" # and pick up the source...
            source_type = clip['source_type'] or "Unknown" # and source type that speaker uses.
            assert source, 'Recieved no dataset source'
            assert source_type, 'Recieved no dataset source type'
            assert speaker_name, 'Recieved no speaker name.'
            assert duration, f'Recieved speaker "{speaker_name}" with 0 duration.'
            if duration < MIN_SPEAKER_DURATION_SECONDS:
                continue
            lines.append(f'{dataset:<24}|{source:<24}|{source_type:<20}|{speaker_name:<32}|{speaker_id:<10}|{duration/3600:>8.4f}')
        f.write('\n'.join(lines))
    
    
    # ( because of SEMI-SUPERVISED... CONTROLLABLE SPEECH SYNTHESIS https://arxiv.org/pdf/1910.01709.pdf )
    # Assign emotion_ids and approximate 'Arousal' and 'Valence' values (this will require a manually written lookup table)
    # Write 'emotion|emotion_id|emotion_latent_0|emotion_latent_1' lookup table to txt file
    # note - I'm not sure how to set the latents so this will just generate an blakn table for the user to fill-in.
    if not os.path.exists('emotion_info.txt') or REGENERATE_EMOTION_INFO:
        emotions = list(set([j for i in [y['emotions'] for z in meta.values() for y in z] for j in i])) # find all emotions in all clips in all datasets
        with open('emotion_info.txt', "w") as f:
            lines = []
            lines.append(f';{"emotion":<23}|{"emotion_id":<12}|{"arousal":<10}|{"valence":<10}\n;')
            for emotion_id, emotion in enumerate(emotions):
                lines.append(f'{emotion:<24}|{0:<12}|{0.0:<10}|{0.0:<10}')
            f.write('\n'.join(lines))
        print("blank emotion_info.txt written! Please modify before usage!")
    
    # Noise level baseline will be 0 for Clean, 1 for Noisy, 2 for Very Noisy.
    # "Clean" vs "Other", and voices like the TF2 Announcer/Administrator will need to be figured out.
    ## this is being skipped for now.
    
    # get phoneme_transcript for every clip.
    # I normally use a lookup table and simply ignore any words not in the file, however there's now 2 more options I can see.
    # additional option 1: g2p neural network for predicting phonemes from graphemes.
    # additional option 2: force aligner system that uses the audio file and grapheme_transcript to produce the phoneme_transcript.
    use_g2p = False
    use_forced_aligner = True
    if use_forced_aligner: # MFA has unique needs in that it will need to be ran seperately for each speaker.
        from CookieTTS.utils.dataset import MFA
        working_directory = os.path.abspath(os.path.join(os.path.dirname(DATASET_FOLDER), 'MFA_tmp')) # something safe that can be deleted
        
        # get clips sorted by speaker
        speaker_paths = {}
        for dataset in meta.keys():
            for i, clip in enumerate(meta[dataset]):
                speaker = clip['speaker']
                if speaker not in speaker_paths.keys():
                    speaker_paths[speaker] = []
                speaker_paths[speaker].append( [clip['path'], clip['quote']] )
        
        # run aligner over every audio file (one speaker at a time)
        path_data_pairs = {}
        for speaker, path_quotes in speaker_paths.items():
            data = MFA.force_align_path_quote_pairs(path_quotes, working_directory, DICT_PATH)
            paths = [x[0] for x in path_quotes]
            for path, data in zip(paths, data):
                path_data_pairs[path] = data
            del data, paths
        
        # merge info back into meta
        for dataset in meta.keys():
            for i, clip in enumerate(meta[dataset]):
                data = path_data_pairs[clip['path']]
                meta[dataset][i]['phoneme_transcript'] = data['arpabet_quote']
                meta[dataset][i]['clip_start'] = data['clip_start']
                meta[dataset][i]['clip_end'] = data['clip_end']
                meta[dataset][i]['words_start'] = data['words_start']
                meta[dataset][i]['words_end'] = data['words_end']
                meta[dataset][i]['phone_start'] = data['phone_start']
                meta[dataset][i]['phone_end'] = data['phone_end']
                meta[dataset][i]['words'] = data['words']
                meta[dataset][i]['phones'] = data['phones']
                del data
        # Montreal Force Aligner done.
    else:
        if use_g2p:
            from g2p_en import G2p
            g2p = G2p()
    
        use_forced_aligner = False
        for dataset in meta.keys():
            prev_wd = os.getcwd()
            os.chdir(os.path.join(DATASET_FOLDER, dataset))
            if use_forced_aligner:
                MFA.force_align(dataset, DICT_PATH)
            for i, clip in enumerate(meta[dataset]):
                grapheme_transcript = meta[dataset][i]['quote']
                if use_g2p:
                    # TODO: convert g2p arrout into phoneme transcript
                    phoneme_transcript = g2p(grapheme_transcript)
                    #phoneme_transcript =  
                    pass
                else: # lookup
                    phoneme_transcript = arpa.get(grapheme_transcript)
                meta[dataset][i]['phoneme_transcript'] = phoneme_transcript
            os.chdir(prev_wd)
    
    
    # (if using FastSpeech or PAG-Tacotron) Generate and save phoneme timing information and/or ground truth alignment graphs
    
    
    # (if using FastPitch or FastSpeech2) Generate and save pitch information for every clip
    
    
    # (if using FastSpeech2) Generate and save energy information for every clip
    
    
    # (if using torchMoji) Generate and save torchMoji hidden states for every clip
    
    
    # (if using SV2TTS) Generate and save speaker embeddings for every clip.
    # Note - not working on this till I have too many speakers to deal with
    
    # Collect "audio_path|grapheme_transcript|phoneme_transcript|speaker_id|sample_rate|emotion_id|noise_level" octuplets for every clip.
    # Write the data into SEPERATED txt's for every dataset.
    # e.g: if using Clipper and VCTK datasets, there should 4 files;
    #     VCTK/filelist_train.txt
    #     VCTK/filelist_validation.txt
    #     Clipper_MLP/filelist_train.txt
    #     Clipper_MLP/filelist_validation.txt
    import random
    for dataset, clips in meta.items():
        train_path = os.path.join(DATASET_FOLDER, dataset, 'filelist_train.txt')
        val_path = os.path.join(DATASET_FOLDER, dataset, 'filelist_validation.txt')
        with open(train_path, 'w') as ft, open(val_path, 'w') as fv:
            ft.write(f';audio_path|grapheme_transcript|phoneme_transcript|speaker_id|sample_rate|emotion_id|noise_level\n; dataset: "{os.path.split(dataset)[-1]}"\n;\n')
            fv.write(f';audio_path|grapheme_transcript|phoneme_transcript|speaker_id|sample_rate|emotion_id|noise_level\n; dataset: "{os.path.split(dataset)[-1]}"\n;\n')
            train_lines = []
            val_lines = []
            for i, clip in enumerate(clips):
                if speaker_durations[clip['speaker']] < MIN_SPEAKER_DURATION_SECONDS:
                    continue
                speaker_id = list(speaker_durations.keys()).index(clip['speaker'])
                emotions = clip["emotions"]
                emotion_ids = [emotions.index(x) for x in clip["emotions"]]
                sample_rate = clip["sample_rate"] if "sample_rate" in clip.keys() else ''
                
                write_line = f'{clip["path"]}|{clip["quote"]}|{clip["phoneme_transcript"]}|{speaker_id}|{emotions}|{sample_rate}|{emotion_ids}|{clip["noise"]}'
                if random.Random(i).random() < TRAIN_PERCENT:
                    train_lines.append(write_line)
                else:
                    val_lines.append(write_line)
                del speaker_id, emotions, emotion_ids, sample_rate, write_line
            
            ft.write('\n'.join(train_lines))
            fv.write('\n'.join(val_lines))
    
    # and a full meta dump of everything
    import json
    meta_dump_path = os.path.join(DATASET_FOLDER, 'meta_dump.txt')
    with open(meta_dump_path, 'w') as outfile:
        json.dump(meta, outfile)