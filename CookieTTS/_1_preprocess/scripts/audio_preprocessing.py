import os
import numpy as np
import librosa
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

def even_split(a, n):
    """split array `a` into `n` seperate evenly sized chunks"""
    n = min(n, len(a)) # if less elements in array than chunks to output, change chunks to array length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def multiprocess_directory(function, directory, regex="**/*.flac", threads=16):
    """
    Calls `function` on all files in directory that match 'regex'
    
    PARAMS:
        function: the function to be called by each thread
        directory: the directory to search for files
        regex: query for files. '**' indicates recursive search, '*' is a standard wildcard.
        threads: number of threads to spawn.
    RETURNS:
        None
    
    Note - Despite being called multiprocess, this is actually using multithreading.
    """
    from random import shuffle
    p = Pool(threads)
    file_paths = glob(os.path.join(directory, regex), recursive=True)
    shuffle(file_paths)
    split_file_paths = list(even_split(file_paths,threads))
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    print(executor.map(function, split_file_paths))
    return p.map(function, split_file_paths)


def multiprocess_filearray(function, file_paths, threads=16):
    """
    Splits given filepaths into 'threads' number of even sized chunks, and calls function in a seperate thread on each one.
    
    PARAMS:
        function: the function to be called by each thread
        file_path: the files paths that will be split into chunks and fed into each thread
        threads: number of threads to spawn.
    RETURNS:
        None
    
    Note - Despite being called multiprocess, this is actually using multithreading.
    """
    p = Pool(threads)
    split_file_paths = list(even_split(file_paths,threads))
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    print(executor.map(function, split_file_paths))
    return p.map(function, split_file_paths)


def normalize_volumes_mixmode(directory, amplitude=0.08, ext='.wav'):
    """
    Calls 'normalize-audio' for every subdirectory in directory.
    
    Will calcuate the mean RMS amplitude of all files in each subdirectory, average them, get difference from amplitude param, and apply a constant amplitude modifier to every audio file in the folder.
    This means audio files in the same folder will all get louder or quieter by the same around, but different folders will be adjusted by different amounts so each folder has roughly the same volume.
    
    PARAMS:
        directory: the root directory, where every directory under this root will be updated.
        amplitude: the target average amplitude for every subdirectory
        ext: the extension of the audio files to be analysed and updated.
    RETURNS:
        None
    """
    subdirectories = [x[0] for x in os.walk(directory)]
    for subdirectory in subdirectories:
        os.system(f"normalize-audio -w 16 -a {amplitude} -b '{subdirectory}/'*{ext}")


def process_audio_multiprocess(file_paths_arr,
        filt_type, filt_cutoff_freq, filt_order,
        trim_margin_left, trim_margin_right, trim_top_db, trim_window_length, trim_hop_length, trim_ref, trim_preemphasis_strength,
        SAMPLE_RATE=48000, MIN_SAMPLE_RATE=15999, BIT_DEPTH=2,
        ignore_dirs=["Noise samples","_Noisy_","_Very Noisy_"], skip_existing=False,
        in_ext_=None, out_ext=".wav", use_tqdm=True, dump_sample_rates=True
    ):
    """
    Take an array of audio file paths. Apply processing and trimming and save the output.
    PARAMS:
        file_paths_arr: Array of audio paths, FLAC's or WAV's recommended.
        
        filt_type: options of 'hp','lp' which is a 'high-pass' and 'low-pass' filter respectively.
        filt_cutoff_freq: threshold frequency for the filter.
        filt_order: similar to the strength of the filter, also effects processing time.
        
        trim_margin_left: save samples to the left of silence.
        trim_margin_right: save samples to the right of silence.
        trim_top_db: decibelles under reference db that is considered silence.
        trim_window_length: number of samples to average over.
        trim_hop_length: number of samples to shift the window each time.
        trim_ref: reference db, typical functions are np.amax and np.mean.
        trim_preemphasis_strength: empthasis filter which can be used to make trimming more sensitive to higher frequencies.
                                   The empthasised audio is only used to identify trimming locations, the original audio
                                   will still be output.
        
        SAMPLE_RATE: the output sample rate of the processed audio files.
        MIN_SAMPLE_RATE: minimum sample rate for an audio file to be processed.
        BIT_DEPTH: doesn't do anything right now. At some point will be used to pick the bit-depth of output audio files.
        
        ignore_dirs: skip audio files where a str from ignore_dirs is found in the filepath.
        skip_existing: skip files that would overwrite an existing file.
        
        in_ext_: ...
        out_ext: the output extension of the audio files. Anything supported by soundfile should work however only FLAC and WAV have been tested by me.
        use_tqdm: add progress bar
        dump_sample_rates: return samples_rates.
    RETURNS:
        samples_rates: an dict of output files and their sample rates before being processed.
                       e.g: {
                                path 0: sample_rate 0,
                                path 1: sample_rate 1,
                                path 2: sample_rate 2, ...
                            }
    
    Note - filt params are zipped together so must be lists of the same length.
    Note - trim params are zipped together so must be lists of the same length.
    Note - This uses file_paths_arr as input because it is intended to be used in a multiprocessing environment where
                                    a host will split a directories audio files into chunks before calling this func.
    """
    import soundfile as sf
    import scipy
    from scipy import signal
    
    if dump_sample_rates:
        sample_rates = {} # array of dicts. e.g: [{path 0: sample_rate 0}, {path 1: sample_rate 1}, {path 2: sample_rate 2}, ...]
    
    skip = 0
    prev_sr = 0
    iterator = tqdm(file_paths_arr, smoothing=0.0) if use_tqdm else file_paths_arr
    for file_path in iterator: # recursive directory search
        in_ext = in_ext_ if (in_ext_ is not None) else os.path.splitext(os.path.split(file_path)[-1])[-1] # get ext from file_path or use override.
        out_path = file_path.replace(in_ext,out_ext)
        if skip_existing and os.path.exists(out_path):
            continue
        if any([filter_dir in file_path for filter_dir in ignore_dirs]):
            continue
        
        # VCTK cleanup
        #if file_path.endswith(f"_mic1{in_ext}"):
        #    os.rename(file_path, file_path.replace(f"_mic1{in_ext}",in_ext))
        #if file_path.endswith(f"_mic2{in_ext}"):
        #    continue
        try:
            native_sound, native_SR = sf.read(file_path, always_2d=True)
        except RuntimeError as ex:
            print(f'"{os.path.split(file_path)[-1]}" failed to load and has been deleted.\nDELETED PATH: "{file_path}"')
            os.unlink(file_path)
            #raise RuntimeError(ex)
        native_sound = native_sound[:,0]# take first channel (either mono or left audio channel)
        native_sound = np.asfortranarray(native_sound).astype('float64') # and ensure the audio is contiguous
        
        if native_SR < MIN_SAMPLE_RATE: # skip any files with native_SR below the minimum
            continue
        if native_SR != SAMPLE_RATE: # ensure all audio is same Sample Rate
            try:
                sound = librosa.core.resample(native_sound, native_SR, SAMPLE_RATE)
            except ValueError as ex:
                print(ex, file_path, native_SR, len(native_sound), sep="\n")
                raise ValueError(ex)
        else:
            sound = native_sound
        
        if dump_sample_rates:
            sample_rates[os.path.abspath(out_path)] = native_SR
        
        # 24 bit -> 16 bit, 32 bit -> 16 bit
        if max(np.amax(native_sound), -np.amin(native_sound)) > (2**23): # if samples exceed values possible at 24 bit
            sound = (sound / 2**(31-15))#.astype('int16') # change bit depth from 32 bit to 16 bit
        elif max(np.amax(native_sound), -np.amin(native_sound)) > (2**15): # if samples exceed values possible at 16 bit
            sound = (sound / 2**(23-15))#.astype('int16') # change bit depth from 24 bit to 16 bit
        
        # apply audio filters
        for type_, freq_, order_ in zip(filt_type, filt_cutoff_freq, filt_order): # eg[ ['lp'], [40], [10] ] # i.e [type, freq, strength]
            sos = signal.butter(order_, freq_, type_, fs=SAMPLE_RATE, output='sos') # calcuate filter somethings
            sound = signal.sosfilt(sos, sound) # apply filter
        
        # apply audio trimming
        for i, (margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_, preemphasis_strength_) in enumerate(zip(trim_margin_left, trim_margin_right, trim_top_db, trim_window_length, trim_hop_length, trim_ref, trim_preemphasis_strength)):
            if preemphasis_strength_:
                sound_filt = librosa.effects.preemphasis(sound, coef=preemphasis_strength_)
                _, index = librosa.effects.trim(sound_filt, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            else:
                _, index = librosa.effects.trim(sound, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            try:
                sound = sound[int(max(index[0]-margin_left_, 0)):int(index[1]+margin_right_)]
            except TypeError:
                print(f'Slice Left:\n{max(index[0]-margin_left_, 0)}\nSlice Right:\n{index[1]+margin_right_}')
            assert len(sound), f"Audio trimmed to 0 length by pass {i+1}\nconfig = {[margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_]}\nFile_Path = '{file_path}'"
        
        # write updated audio to file
        if os.path.exists(out_path):
            os.unlink(out_path) # using unlink incase the out_path object is a symlink
        sf.write(out_path, sound, SAMPLE_RATE)
    
    if dump_sample_rates:
        return sample_rates