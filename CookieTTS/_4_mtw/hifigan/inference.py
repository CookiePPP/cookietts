from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from CookieTTS._4_mtw.hifigan.env import AttrDict
from nvSTFT import load_wav_to_torch
from nvSTFT import STFT as STFT_Class
from CookieTTS._4_mtw.hifigan.models import Generator
from librosa.util import normalize

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, STFT):
    generator = Generator(h).to(device)
    
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    
    filelist = os.listdir(a.input_wavs_dir)
    
    os.makedirs(a.output_dir, exist_ok=True)
    
    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            gt_audio, sampling_rate = load_wav_to_torch(os.path.join(a.input_wavs_dir, filname), target_sr=STFT.target_sr)
            #gt_audio = gt_audio - gt_audio.mean()
            gt_audio = gt_audio/gt_audio.abs().max() * 0.95
            x = STFT.get_mel(gt_audio.unsqueeze(0).to(device))# [1, n_mel, dec_T]
            
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * (2**15)
            audio = audio.cpu().numpy().astype('int16')
            
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0][:20] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0][:20] + '_original.wav')
            gt_audio *= (2**15)
            write(output_file, h.sampling_rate, gt_audio.squeeze().cpu().numpy().astype('int16'))
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir',     default='test_files_generated')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    global STFT
    STFT = STFT_Class(h.sampling_rate, h.num_mels, h.n_fft, h.win_size, h.hop_size, h.fmin, h.fmax)
    
    inference(a, STFT)


if __name__ == '__main__':
    main()

