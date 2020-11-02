import os
import time
import argparse
import math
import numpy as np
import torch
import layers
from torch.utils.data import DataLoader
from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from tqdm import tqdm
from scipy.io.wavfile import read
from CookieTTS.utils.dataset import load_wav_to_torch, load_filepaths_and_text
from multiprocessing import Pool


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # list(chunks([0,1,2,3,4,5,6,7,8,9],2)) -> [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def even_split(a, n):
    """Split a into n seperate chunks of roughly even length."""
    n = min(n, len(a)) # if less elements in array than chunks to output, change chunks to array length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def multiprocess_arr(function, file_paths, threads=16):
    p = Pool(threads)
    split_file_paths = list(even_split(file_paths, threads))
    result = p.map(function, split_file_paths)
    for output in result:
        if output:
            print(output)


def multiprocess_gen_mels(audiopaths_internal):
    import layers
    stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)
    return_string = ""
    total = len(audiopaths_internal)
    for index, path in enumerate(audiopaths_internal):
        if index < 0: continue
        #try:
        file = path.replace(".npy",".wav")
        audio, sampling_rate = load_wav_to_torch(file)
        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(file, 
                sampling_rate, stft.sampling_rate))
        melspec = stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0).cpu().numpy()
        np.save(file.replace('.wav', ''), melspec)
        if not index % 1000:
            print(total-index)
        #except Exception as ex:
        #    return_string+=(path+" failed to process\nException: "+str(ex)+"\n")
    if not return_string:
        return_string = "No Errors on this process."
    return return_string


def create_mels(training_filelist, validation_filelist, threads):
    import glob
    audiopaths = []
    audiopaths.extend(
        list(set([x[0] for x in load_filepaths_and_text(training_filelist) ]))
    ) # add all unique audio paths for training data
    audiopaths.extend(
        list(set([x[0] for x in load_filepaths_and_text(validation_filelist) ]))
    ) # add all unique audio paths for validation data
    print(str(len(audiopaths))+" files being converted to mels")
    multiprocess_arr(multiprocess_gen_mels, audiopaths, threads=threads)

if __name__ == '__main__':
    hparams = create_hparams()
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    threads = 1 # Uses about 46GB of RAM each

    print("Generating Mels")
    create_mels(hparams.training_files, hparams.validation_files, threads)
    print("Finished Generating Mels")
