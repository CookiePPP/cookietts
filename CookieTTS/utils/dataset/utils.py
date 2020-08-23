import numpy as np
import torch
from scipy.io.wavfile import read
import soundfile as sf

def load_wav_to_torch(full_path):
    if full_path.endswith('wav'):
        sampling_rate, data = read(full_path) # scipy only supports .wav but reads faster...
    else:
        data, sampling_rate = sf.read(full_path, always_2d=True)[:,0] # than soundfile.
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = 2**31 if max_mag > (2**15) else (2**15 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate, max_mag


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line_strip.split(split) for line_strip in (line.strip() for line in f) if line_strip and line_strip[0] is not ";"]
    return filepaths_and_text


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files