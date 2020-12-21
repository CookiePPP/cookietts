import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
import time
import argparse
import math
import numpy as np

import itertools
from glob import glob

import torch
from model import AutoVC, load_model
from hparams import create_hparams
from CookieTTS.utils.model.GPU import to_gpu

import json
from CookieTTS._4_mtw.hifigan.env import AttrDict
from CookieTTS._4_mtw.hifigan.models import Generator

from CookieTTS.utils.dataset.utils import load_filepaths_and_text
import CookieTTS.utils.audio.stft as STFT
from scipy.io.wavfile import write

from tqdm import tqdm
import os.path

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['state_dict'])# load model weights
    elif 'model' in checkpoint_dict:
        model.load_state_dict(checkpoint_dict['model'])# load model weights
    
    iteration = checkpoint_dict.get('iteration', 0)
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, iteration


# speaker encoder
from CookieTTS.utils.dataset.autovc_speaker_encoder.make_metadata import get_speaker_encoder    
speaker_encoder = get_speaker_encoder()

def get_speaker_embed(path):
    if path.endswith('.pt'):# if pytorch file
        return torch.load(path).float()
    elif path.endswith('.npy'):# if numpy file
        return torch.from_numpy(np.load(path)).float()
    else:
        return speaker_encoder.get_embed_from_path(path).float()

def get_hifi_gan(checkpoint_path):
    assert os.path.exists(checkpoint_path), f'HiFi-GAN Checkpoint at "{checkpoint_path}" does not exist!'
    
    config_file = os.path.join(os.path.split(checkpoint_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    generator = Generator(h).to('cuda')
    generator.load_state_dict(checkpoint_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del checkpoint_dict
    return generator, h

def get_hparams_from_cp(path):
    assert os.path.isfile(path)
    checkpoint_dict = torch.load(path, map_location='cpu')
    hparams = checkpoint_dict['hparams']
    del checkpoint_dict
    return hparams

def main(args, rank=0, n_gpus=1):
    # get hparams from checkpoint file
    hparams = get_hparams_from_cp(args.checkpoint_path)
    
    torch.backends.cudnn.enabled   = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    # initialize blank model
    print('Initializing AutoVC...')
    model = load_model(hparams)
    print('Done')
    model.eval()
    
    # (optional) show the names of each layer in model, mainly makes it easier to copy/paste what you want to adjust
    if hparams.print_layer_names_during_startup:
        print(*[f"Layer{i} = "+str(x[0])+" "+str(x[1].shape) for i,x in enumerate(list(model.named_parameters()))], sep="\n")
    
    if True and rank == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("{:,} total parameters in model".format(pytorch_total_params))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("{:,} trainable parameters.".format(pytorch_total_params))
    
    print("Initializing AMP Model(s) / Optimzier(s)")
    if hparams.fp16_run:
        model.half()
    
    if args.checkpoint_path is not None:
        model, iteration = load_checkpoint(args.checkpoint_path, model)
        print('Checkpoint Loaded')
    
    print("Initializing STFT Module")
    # STFT / Mel Func
    stft = STFT.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax, clamp_val=hparams.stft_clamp_val)
    
    print("Loading HiFi-GAN Checkpoint")
    vocoder, h = get_hifi_gan(args.vocoder_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if os.path.exists(args.test_list):
        filelist = load_filepaths_and_text(args.test_list)
    elif os.path.exists(args.test_dir):
        audio_files = glob(os.path.join(args.test_dir, '*.*'))
        filelist = itertools.combinations(audio_files, 2)
    else:
        raise Exception # this code should never reach this point!
    
    for source_path, target_path, *_ in filelist:
        # load data into 'batch'
        source_mel = stft.get_mel_from_path(source_path)  # [1, n_mel, mel_T]
        c_org = get_speaker_embed(source_path).unsqueeze(0)# [1, Embed]
        c_trg = get_speaker_embed(target_path).unsqueeze(0)# [1, Embed]
        batch = {'gt_mel': source_mel, 'c_org': c_org, 'c_trg': c_trg,}
        
        batch_gpu = model.parse_batch(batch) # move batch to GPU (async)
        pred = model(**batch_gpu)# run batch through model and produce voice-converted spectrogram
        
        pred_audio = vocoder(pred['pred_mel_postnet'])# convert pred_spect into audio
        
        # write pred_audio to file
        output_path = os.path.join(args.output_dir, os.path.splitext(os.path.split(source_path)[-1][:20])[0]+'_spoken_by_'+os.path.splitext(os.path.split(target_path)[-1][:20])[0]+'.wav')
        print(f"Writing {os.path.split(output_path)[-1]}")
        write(output_path, h.sampling_rate, pred_audio.float().squeeze().mul_(2**15).cpu().numpy().astype('int16'))

if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_list', type=str, required=False, default='test_filelist.txt', 
            help='txt filelist with | seperated source|target pairs. The outputs will be the text of the source audio spoken using the speaker from the target.')
    parser.add_argument('-d', '--test_dir', type=str, required=False, default='mix_mode', 
            help='directory of audio files to mix and match.')
    parser.add_argument('-o', '--output_dir',      type=str, required=False, default='test_audio_generated',
            help='directory to dump generated audios.')
    parser.add_argument('-c', '--checkpoint_path', type=str, required= True,
            help='checkpoint path')
    parser.add_argument('-v', '--vocoder_path',    type=str, required= True,
            help='vocoder path - path to HiFi-GAN generator e.g; g00010500')
    
   
    args = parser.parse_args()
    global hparams
    hparams = create_hparams()
    
    assert os.path.exists(args.test_list) or os.path.exists(args.test_dir), '--text_list or --text_dir must be specified and exist!'
    
    main(args)