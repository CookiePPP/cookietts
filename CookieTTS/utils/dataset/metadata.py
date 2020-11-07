import os
from glob import glob


def identify_transcript_storage(directory, audio_files, audio_ext, audio_basename_lookup, txt_files, txt_names, txt_name_lookup):
    """figure out where/how the transcripts are stored"""
    transcript = None
    
    # 2.1.1 test if txts use Clipper Format
    # check how many of the first 20 audio files have a matching txt (in the same dir)
    set_txt_files = set(txt_files)
    files_with_txts = 0
    for i, audio_file in enumerate(audio_files):
        if os.path.splitext(audio_file)[0]+'.txt' in set_txt_files:
            files_with_txts += 1
    #print(f'Found {files_with_txts} audio files with matching text files (of {len(audio_files)} total audio files).')
    if files_with_txts >= len(audio_files)*0.9: # if atleast 90% of audio files have a matching txt
        return ["clipper",]
    del files_with_txts, set_txt_files
    
    # look for txt or csv with name "*_master_dataset.txt"
    # this comes up for Persona Nerd datasets. I don't know which ones specifically.
    n_valid_txts = 0
    valid_txts = list()
    for txt_file in txt_files:
        if os.stat(txt_file).st_size > 4 and txt_file.endswith("_master_dataset.txt"):
            valid_txts.append(txt_file)
            n_valid_txts += 1
    if n_valid_txts == 1:
        return "tacotron", valid_txts
    del n_valid_txts, valid_txts
    
    # 2.1.2 test if txts use Tacotron (or LJSpeech) Style Format
    #look for txt or csv file with more than 3 lines and containing '|' chars.
    n_valid_txts = 0
    valid_txts = list()
    for txt_file in txt_files:
        if os.stat(txt_file).st_size > 80: # if txt_file has a reasonable size
            text = open(txt_file, "r").read()
            n_pipes = text.count('|') # get number of pipe symbols
            n_nl = text.count('\n') # get number of newline symbols
            if n_pipes > 2 and n_nl > 0: # if the text file has more than 2 pipes and a newline symbol
                prev_wd_ = os.getcwd()
                if os.path.split(txt_file)[0]:# move into txt dir (in-case the audio paths are relative)
                    os.chdir(os.path.split(txt_file)[0])
                paths = [x.split("|")[0] for x in text.split("\n") if len(x.strip())] # get paths
                #n_exists = sum([os.path.exists(x) for x in paths]) # check how many paths exist
                n_exists = sum([os.path.splitext(os.path.split(x)[1])[0] in audio_basename_lookup.keys() for x in paths]) # check how many names exist
                if n_exists/len(paths) > 0.95: # if more than 95% of the paths in the left-most section contain existing files
                    n_valid_txts += 1 # add it as a valid txt file
                    valid_txts.append(txt_file) # and save the txt files path (relative to the dataset root)
                os.chdir(prev_wd_)
                del n_exists, prev_wd_
            del text, n_pipes, n_nl
    if n_valid_txts == 1:
        return "ljspeech", valid_txts
    elif n_valid_txts > 1:
        return "tacotron", valid_txts
    del n_valid_txts, valid_txts
    
    # 2.1.3 test if txts use VCTK Style Format
    # for each audio file, check if a text file exists of the same name, but in another directory.
    n_audio_files_with_txt = 0
    txt_basenames = [os.path.splitext(os.path.split(txt_file)[-1])[0] for txt_file in txt_files]
    for audio_file in audio_files:
        audio_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
        if audio_basename in txt_basenames:
            n_audio_files_with_txt+=1
    
    if n_audio_files_with_txt/len(audio_files) > 0.9: # if more than 90% of audio files have a txt file with the same name, but in different directories
        return ["vctk",] # return vctk
    
    raise NotImplementedError(f'Could not identify transcript type for the "{directory}" dataset')


def filelist_get_transcript(audio_file, filelist, filelist_paths, filelist_names, filelist_basenames):
    """get transcript from central filelists array."""
    try: # try identify transcript from audiopath
        path_index = filelist_paths.index(audio_file.replace("\\","/"))
    except ValueError as ex:
        try: # try identify transcript from audio filename
            path_index = filelist_names.index(os.path.split(audio_file)[1])
        except ValueError as ex:
            try: # try identify transcript from audio basename
                path_index = filelist_basenames.index(  os.path.splitext(os.path.split(audio_file)[1])[0]  )
            except ValueError as ex:
                print(f'"{audio_file}" not found in filelists')
                raise FileNotFoundError(ex)
    transcript = filelist[path_index][1]
    return transcript.strip()


def clipper_get_transcript(audio_file):
    """get transcript by loading the text file with same name as audio file from same dir."""
    audio_file_directory = os.path.split(audio_file)[0]
    audio_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
    text_file = os.path.join(audio_file_directory, audio_file_basename+".txt")
    if not os.path.exists(text_file):
        raise FileNotFoundError(f'audio file at "{audio_file}" has no matching txt file.')
    try:
        transcript = open(text_file, "r", encoding="utf-8").read()
    except UnicodeDecodeError:
        transcript = open(text_file, "r", encoding="latin-1").read()
    return transcript.strip()


def vctk_get_transcript(audio_file, txt_lookup):
    """
    get transcript from text file (with same name) in another subdirectory.
    
    PARAMS:
        audio_file: path to audio file of interest
        txt_lookup: a dict with `text_filename` keys and `text_path` values.
    
    RETURNS:
        transcript matching the input audio_file.
    """
    audio_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
    text_filename = audio_file_basename+".txt"
    text_path = txt_lookup[text_filename]
    transcript = open(text_path, "r").read()
    return transcript.strip()


def clipper_naming_exceptions(audio_file, source, source_type, voice):
    audio_file = audio_file.replace("\\","/")
    
    # any naming exceptions
    if "Other/Star Trek (John de Lancie, Discord)" in audio_file:
        source_type, source, voice = "Show", "Star Trek", "Q"
    elif "Other/Eli, Elite Dangerous (John de Lancie, Discord)" in audio_file:
        source_type, source, voice = "Game", "Elite Dangerous", "Eli"
    elif "Other/A Little Bit Wicked (Kristin Chenoworth, Skystar)" in audio_file:
        source_type, source, voice = "Audiobook", "A Little Bit Wicked", voice
    elif "Other/Sum - Tales From the Afterlives (Emily Blunt, Tempest)" in audio_file:
        source_type, source, voice = "Audiobook", "Sum - Tales From the Afterlives", voice
    elif "Other/Dr. Who" in audio_file:
        source_type, source, voice = "Audiobook", "Dr. Who", voice
    elif "Other/Dan vs" in audio_file:
        source_type, source, voice = "Show", "Dan vs", voice
    elif "Other/TFH" in audio_file:
        source_type, source, voice = "Game", "Them's Fightin' Herds", voice
    elif "Other/CGP Grey" in audio_file:
        source_type, source, voice = "Show", "CGP Grey", voice
    elif "Other/ATHF" in audio_file:
        source_type, source, voice = "Show", "Aqua Teen Hunger Force", voice
    elif "/Songs" in audio_file:
        source_type, source, voice = "Music", "My Little Pony", voice
    
    return voice, source, source_type


def get_timestamp(splitted):
    timestamp = "_".join(splitted[0:3]) # e.g: ["00","00","00"] -> "00_00_05"
    try:
        for s in splitted[0:3]:
            _ = int(s) # check that first underscores are numbers only
    except ValueError as ex:
        print(ex)
        raise ValueError('"'+"_".join(splitted)+f'" has 6 or more underscores but does not follow clipper naming scheme.')
    return timestamp


def remove_ending_periods(directory):
    """
    Remove ending periods not part of extension.
    e.g:
    "00_00_49_Celestia_Neutral_Very Noisy_girls, thank you so much for coming..wav"
     to
    "00_00_49_Celestia_Neutral_Very Noisy_girls, thank you so much for coming.wav"
    """
    files_arr = sorted([os.path.abspath(x) for x in glob(os.path.join(directory,"**/*.*"), recursive=True)])
    assert len(files_arr), f'no audio files found for {directory} dataset.'
    
    file_dict = {x: (os.path.splitext(x)[0].rstrip('.')+os.path.splitext(x)[-1]) for x in files_arr if x != (os.path.splitext(x)[0].rstrip('.')+os.path.splitext(x)[-1])}
    for src, dst in file_dict.items():
        os.rename(src, dst)


def get_dataset_meta(directory, meta=None, default_speaker=None, default_emotion=None, default_noise_level=None, default_source=None, default_source_type='audiobook', audio_ext=["*.wav",], audio_rejects=[], naming_system=None):
    """
    Looks for
     - audio paths
     - transcripts
     - speaker names
     - emotions
     - noise levels
     - source
     - source_type
    inside dataset folder and returns all info found as array.
    
    PARAMS:
        directory: the root directory of the dataset
        meta: (optional) the array that the dataset info is appended to.
             Use this to chain different metas together or just ignore it.
        default_speaker: speaker to use if none can be identified
        default_emotion: emotion to use if none can be identified
        default_noise_level: noise_level to use if none can be identified
        default_source: e.g: "My Little Pony", "Dan vs", "Them's Fightin' Herds"
        default_source_type: e.g: "game","music","audiobook","tv show"
        naming_system: (TODO)
            options:
                "clipper": get names from string between 3rd and 4th underscore
                      e.g: "audio_folder/00_00_00_Mrs. Cake_Sad_Noisy_Transcript 3_.wav" -> "Mrs. Cake"
                      e.g: 'audio_0.wav' -> default_speaker
                "vctk": get names from the name of the parent folder
                      e.g: "p234/audio_0.wav" -> "p234"
                "p4g": get names from the name of the filelist
                      e.g: audio files from "Twilight.txt" have the speaker of "Twilight"
                      e.g: audio files from "train.txt" have the speaker of "train"
                None: use default_speaker. (clipper filenames can still override)
                      e.g: 'audio_0.wav' -> default_speaker
                      e.g: '00_00_00_Mrs. Cake_Sad_Noisy_Transcript 3_.wav' -> 'Mrs. Cake'
    
    RETURNS:
        meta: an array of metadata dicts
    """
    if meta is None:
        meta = []
    if default_emotion is None:
        default_emotion = 'unknown'
    assert default_speaker, f'default speaker required for dataset "{directory}".'
    prev_wd = os.getcwd() # save location
    directory = os.path.abspath(directory)
    os.chdir(directory) # and move into dataset directory
    
    # 0 - fix inconsistent naming in clipper dataset
    remove_ending_periods(os.path.abspath(directory))
    
    # 1 - get audiopaths
    audio_files = []
    for ext in audio_ext:
        audio_files.extend([os.path.abspath(x) for x in glob(f"**/{ext}", recursive=True)])
    banned_files = []
    for rjct in audio_rejects:
        banned_files.extend([os.path.abspath(x) for x in glob(f"**/{rjct}", recursive=True)])
    banned_files = set(banned_files)
    
    audio_files = sorted([x for x in list(set(audio_files)) if not x in banned_files])
    assert len(audio_files), f'no audio files found for "{directory}" dataset.'
    print(f'Found {len(audio_files)} audio files.')
    
    audio_basename_lookup = {os.path.splitext(os.path.split(x)[1])[0]: os.path.abspath(x) for x in audio_files}
    txt_files = sorted([os.path.abspath(x) for x in [*glob("**/*.txt", recursive=True), *glob("**/*.csv", recursive=True)] if os.path.exists(x)])
    assert all([os.path.exists(x) for x in txt_files])
    assert len(txt_files), f'no text files found for "{directory}" dataset.'
    print(f'Found {len(txt_files)} text files.')
    txt_names = [os.path.split(x)[-1] for x in txt_files]
    txt_name_lookup = {name: path for name, path in zip(txt_names, txt_files)}
    
    # 2 - get transcripts
    # 2.1 - figure out where/how the transcripts are stored
    print(f'Identifying "{directory}" dataset type.')
    dataset_style, *_ = identify_transcript_storage(directory, audio_files, audio_ext, audio_basename_lookup, txt_files, txt_names, txt_name_lookup)
    print(f'"{directory}" identified as {dataset_style} style dataset.')
    
    # 2.2 - Collect the metadata
    if (dataset_style == "ljspeech") or (dataset_style == "tacotron"): # collect central filelist
        valid_txts = _[0]
        filelist = list()
        for txt in valid_txts:
            text = open(txt, "r").read()
            text = [x.strip().split("|") for x in text.split("\n") if len(x.strip()) and not '{' in x] # `and not '{' in x` <- ignoring provided ARPAbet.
            filelist.extend(text)
        filelist_paths = [x[0].replace(".npy",".wav").replace("\\","/") for x in filelist]
        filelist_names = [os.path.split(x)[-1] for x in filelist_paths]
        filelist_basenames = [os.path.splitext(x)[0] for x in filelist_names]
    
    if dataset_style == "vctk":
        pass
    
    files_added=0
    files_skipped=0
    for audio_file in audio_files:
        audio_name = os.path.split(audio_file)[-1]
        audio_basename = os.path.splitext(audio_name)[0]
        
        # 2.2.1 - get transcript
        try:
            if dataset_style == "clipper":
                transcript = clipper_get_transcript(audio_file)
            elif (dataset_style == "ljspeech") or (dataset_style == "tacotron"):
                transcript = filelist_get_transcript(audio_file, filelist, filelist_paths, filelist_names, filelist_basenames)
            elif dataset_style == "vctk":
                transcript = vctk_get_transcript(audio_file, txt_name_lookup)
            else:
                raise NotImplementedError
        except FileNotFoundError as ex:
            print(ex, f'Skipping file: "{audio_file}"', sep='\n')
            files_skipped+=1; continue
        except KeyError as ex:
            print(ex, f'Skipping file: "{audio_file}"', sep='\n')
            files_skipped+=1; continue
        if len(transcript) < 2:
            print(f'Skipping file: "{audio_file}"', sep='\n')
            files_skipped+=1; continue
        
        # 2.2.2 - get speaker name, emotion(s), noise level, source, source_type
        voice = default_speaker           # defaults
        emotions = [default_emotion,]     # defaults
        noise_level = default_noise_level # defaults
        source = default_source           # defaults
        source_type = default_source_type # defaults
        
        if len(audio_basename.split("_")) >= 6: # eg.: "00_00_00_Mrs. Cake_Sad_Noisy_Transcript 3_.wav"
            splitted = audio_basename.split("_") # e.g: ["00","00","00","Mrs. Cake","Sad","Noisy","Transcript 3",""]
            
            timestamp = get_timestamp(splitted) # e.g: ["00","00","00",...] -> "00_00_05"
            voice = splitted[3].title() # e.g: "Mrs. Cake"
            emotions = splitted[4].lower().split(" ")   # e.g: ["neutral",]
            noise_level = splitted[5].lower()           # e.g: "" = clean, "noisy" = Noisy, "very noisy" = Very Noisy
            filename_transcript = "?".join(splitted[6:]) # underscores are unknown symbols, normally question marks but not 100% of the time.
            
            if "Sliced Dialogue" in audio_file: # overrides for clipper dataset
                voice, source, source_type = clipper_naming_exceptions(audio_file, source, source_type, voice)
        
        # 2.2.3 - add metadata to list
        meta.append({
            'path': audio_file,
            'quote': transcript,
            'speaker': voice,
            'emotions': emotions,
            'noise': noise_level,
            'source': source,
            'source_type': source_type,
        })
        files_added+=1
    
    print(f'Total: {len(audio_files)} Files\nAdded: {files_added} Files\nSkipped: {files_skipped} Files')
    os.chdir(prev_wd) # move back to original location
    return meta

if __name__ == "__main__":
    
    # testing code below
    
    os.chdir("../tests/fake_datasets")
    print(f'\nCURRENT_WORKING_DIR: "{os.getcwd()}"')
    
    print("\n")
    dataset_meta = get_dataset_meta("LJSpeech_Style1", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    correct_answer = [{'path': 'audio\\audio_0.wav', 'quote': 'Transcript 0.', 'speaker': 'speaker 0', 'emotions': ['unknown'], 'noise': 'clean', 'source': 'test', 'source_type': 'audiobook'}, {'path': 'audio\\audio_1.wav', 'quote': 'Transcript 1.', 'speaker': 'speaker 0', 'emotions': ['unknown'], 'noise': 'clean', 'source': 'test', 'source_type': 'audiobook'}, {'path': 'audio\\audio_2.wav', 'quote': 'Transcript 2.', 'speaker': 'speaker 0', 'emotions': ['unknown'], 'noise': 'clean', 'source': 'test', 'source_type': 'audiobook'}]
    #assert dataset_meta == correct_answer, f'"LJSpeech_Style1" has incorrect output'
    
    print("\n")
    dataset_meta = get_dataset_meta("LJSpeech_Style2", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("LJSpeech_Style3", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("Clipper_Style1", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("Clipper_Style2", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("Clipper_Style3", default_speaker='unknown', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    correct_answer = [{'path': 'Sliced Dialogue\\00_00_00_Mrs. Cake_Sad_Noisy_Transcript 3_.wav', 'quote': 'Transcript 3?', 'speaker': 'Mrs. Cake', 'emotions': ['sad'], 'noise': 'noisy', 'source': 'test', 'source_type': 'audiobook'}, {'path': 'Sliced Dialogue\\00_00_00_Princess Luna_Canterlot Voice Angry_Very Noisy_Transcript 4_!.wav', 'quote': 'Transcript 4?!', 'speaker': 'Princess Luna', 'emotions': ['canterlot', 'voice', 'angry'], 'noise': 'very noisy', 'source': 'test', 'source_type': 'audiobook'}, {'path': 'Sliced Dialogue\\00_00_00_Rainbow Dash_Happy_Very Noisy_Transcript 2!.wav', 'quote': 'Transcript 2!', 'speaker': 'Rainbow Dash', 'emotions': ['happy'], 'noise': 'very noisy', 'source': 'test', 'source_type': 'audiobook'}, {'path': 'Sliced Dialogue\\00_00_00_Twilight_Neutral__Transcript 1..wav', 'quote': 'Transcript 1.', 'speaker': 'Twilight', 'emotions': ['neutral'], 'noise': '', 'source': 'test', 'source_type': 'audiobook'}]
    #assert dataset_meta == correct_answer, f'"Clipper_Style3" has incorrect output'
    
    print("\n")
    dataset_meta = get_dataset_meta("Tacotron_Style1", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("Tacotron_Style2", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("VCTK_Style1", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    print("\n")
    dataset_meta = get_dataset_meta("VCTK_Style2", default_speaker='speaker 0', default_emotion='unknown', default_noise_level='clean', default_source='test')
    print("\n".join([str(x) for x in dataset_meta]))
    
    
    
    #print("\n")                                       # cannot identify text-audio pairs when audio has "_mic" appended.
    #dataset_meta = get_dataset_meta("VCTK_Style3")    # cannot identify text-audio pairs when audio has "_mic" appended.
    #print("\n".join([str(x) for x in dataset_meta]))  # cannot identify text-audio pairs when audio has "_mic" appended.
    
    
    print("\n\nTests Completed!\nmetadata.py should be working correctly.")