import os
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
from distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from fp16_optimizer import FP16_Optimizer

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from train import init_distributed

import time


class StreamingMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0
    
    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed")
    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # Initialize distributed communication
    torch.distributed.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    print("Done initializing distributed")


def prepare_dataloaders(hparams, audio_offset=0):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams, TBPTT=False, check_files=False, verbose=True, audio_offset=audio_offset)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        shuffle = False

    train_loader = DataLoader(trainset, num_workers=0, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False, # default pin_memory=False, True should allow async memory transfers # Causes very random CUDA errors (after like 4+ hours)
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, None, collate_fn, train_sampler, trainset


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model = batchnorm_to_float(model.half())
        model.decoder.attention_layer.score_mask_value = float(finfo('float16').min)
    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def get_global_mean(data_loader, global_mean_npy, hparams):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return (torch.tensor(global_mean).half()).cuda() if hparams.fp16_run else (torch.tensor(global_mean).float()).cuda()
    else:
        raise Exception("No global_mean.npy found while in training_mode.")
    return global_mean


@torch.no_grad()
def GTA_Synthesis(output_directory, checkpoint_path, n_gpus,
          rank, group_name, hparams, training_mode, verify_outputs, use_val_files, fp16_save, extra_info='', audio_offset=0):
    """Generate Ground-Truth-Aligned Spectrograms for Training WaveGlow."""
    if audio_offset:
        hparams.load_mel_from_disk = False
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if use_val_files:
        filelisttype = "val"
        hparams.training_files = hparams.validation_files
    else:
        filelisttype = "train"
    
    train_loader, _, collate_fn, train_sampler, train_set = prepare_dataloaders(hparams, audio_offset=audio_offset)
    
    if training_mode and hparams.drop_frame_rate > 0.:
        if rank != 0: # if global_mean not yet calcuated, wait for main thread to do it
            while not os.path.exists(hparams.global_mean_npy): time.sleep(1)
        global_mean = get_global_mean(train_loader, hparams.global_mean_npy, hparams)
        hparams.global_mean = global_mean
    
    model = load_model(hparams)
    # Load checkpoint if one exists
    assert checkpoint_path is not None
    if checkpoint_path is not None:
        model = warm_start_model(checkpoint_path, model)
    
    if training_mode:
        model.train()
    else:
        model.eval()
    
    if hparams.distributed_run or torch.cuda.device_count() > 1:
        batch_parser = model.parse_batch
    else:
        batch_parser = model.parse_batch
    # ================ MAIN TRAINNIG LOOP! ===================
    os.makedirs(os.path.join(output_directory), exist_ok=True)
    f = open(os.path.join(output_directory, f'map_{filelisttype}_{rank}.txt'),'a', encoding='utf-8')
    os.makedirs(os.path.join(output_directory,'mels'), exist_ok=True)
    
    total_number_of_data = len(train_set.audiopaths_and_text)
    max_itter = int(total_number_of_data/hparams.batch_size)
    remainder_size = total_number_of_data % hparams.batch_size
    
    duration = time.time()
    total = len(train_loader)
    rolling_sum = StreamingMovingAverage(100)
    for i, batch in enumerate(train_loader):
        batch_size = hparams.batch_size if i is not max_itter else remainder_size
        
        # get wavefile path
        audiopaths_and_text = train_set.audiopaths_and_text[i*hparams.batch_size:i*hparams.batch_size + batch_size]
        audiopaths = [x[0] for x in audiopaths_and_text] # file name list
        orig_speaker_ids = [x[2] for x in audiopaths_and_text] # file name list
        
        # get len texts
        indx_list = np.arange(i*hparams.batch_size, i*hparams.batch_size + batch_size).tolist()
        len_text_list = []
        for batch_index in indx_list:
            text, *_ = train_set.__getitem__(batch_index)
            len_text_list.append(text.size(0))
        
        _, input_lengths, _, _, output_lengths, speaker_id, _, _ = batch # output_lengths: original mel length
        input_lengths_, ids_sorted_decreasing = torch.sort(torch.LongTensor(len_text_list), dim=0, descending=True)
        ids_sorted_decreasing = ids_sorted_decreasing.numpy() # ids_sorted_decreasing, original index
        
        org_audiopaths = [] # original_file_name
        mel_paths = []
        speaker_ids = []
        for k in range(batch_size):
            d = audiopaths[ids_sorted_decreasing[k]]
            org_audiopaths.append(d)
            mel_paths.append(d.replace(".npy",".mel").replace('.wav','.mel'))
            speaker_ids.append(orig_speaker_ids[ids_sorted_decreasing[k]])
        
        x, _ = batch_parser(batch)
        _, mel_outputs_postnet, _, _ = model(x, teacher_force_till=9999, p_teacher_forcing=1.0)
        mel_outputs_postnet = mel_outputs_postnet.data.cpu().numpy()
        
        for k in range(batch_size):
            wav_path = org_audiopaths[k].replace(".npy",".wav")
            offset_append = '' if audio_offset == 0 else str(audio_offset)
            mel_path = mel_paths[k]+offset_append+'.npy' # ext = '.mel.npy' or '.mel1.npy' ... '.mel599.npy'
            speaker_id = speaker_ids[k]
            map = "{}|{}|{}\n".format(wav_path,mel_path,speaker_id)
            f.write(map)
            
            mel = mel_outputs_postnet[k,:,:output_lengths[k]]
            print(wav_path, input_lengths[k], output_lengths[k], mel_outputs_postnet.shape, mel.shape, speaker_id)
            if fp16_save:
                mel = mel.astype(np.float16)
            np.save(mel_path, mel)
            if verify_outputs:
                orig_shape = train_set.get_mel(wav_path).shape
                assert orig_shape == mel.shape, f"Target shape {orig_shape} does not match generated mel shape {mel.shape}.\nFilepath: '{wav_path}'" # check mel from wav_path has same shape as mel just saved
        duration = time.time() - duration
        avg_duration = rolling_sum.process(duration)
        time_left = round(((total-i) * avg_duration)/3600, 2)
        print(f'{extra_info}{i}/{total} compute and save GTA melspectrograms in {i}th batch, {duration}s, {time_left}hrs left')
        duration = time.time()
    f.close()


if __name__ == '__main__':
    """Example:
    CUDA_VISIBLE_DEVICES=3 python3 GTA.py -o "GTA_flist" -c "outdir_truncated1/checkpoint_194000" --extremeGTA 100 --fp16_save
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--extremeGTA', type=int, default=0, required=False,
                        help='Generate a Ground Truth Aligned output every interval specified. This will run tacotron hop_length//interval times per file and use thousands of GBs of storage. Caution is advised')
    parser.add_argument('--use_training_mode', action='store_true',
                        help='Use model.train() while generating alignments. Will increase both variablility and inaccuracy.')
    parser.add_argument('--verify_outputs', action='store_true',
                        help='Check output length matches length of original wav input.')
    parser.add_argument('--use_validation_files', action='store_true',
                        help='Ground Truth Align validation files instead of training files.')
    parser.add_argument('--fp16_save', action='store_true',
                        help='Save spectrograms using np.float16.')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.n_gpus = args.n_gpus
    hparams.rank = args.rank
    hparams.use_TBPTT = False # remove limit
    hparams.truncated_length = 2**15 # remove limit
    hparams.check_files=False # disable checks
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    print("Rank:", args.rank)
    
    # cookie stuff
    hparams.load_mel_from_disk = False
    hparams.training_files = hparams.training_files.replace("mel_train","train").replace("_merged.txt",".txt")
    hparams.validation_files = hparams.validation_files.replace("mel_val","val").replace("_merged.txt",".txt")
    
    hparams.batch_size = hparams.batch_size * 6 # no gradients stored so batch size can go up a bunch
    
    torch.autograd.set_grad_enabled(False)
    
    if args.extremeGTA:
        for ind, ioffset in enumerate(range(0, hparams.hop_length, args.extremeGTA)): # generate aligned spectrograms for all audio samples
            if ind < 0: continue
            GTA_Synthesis(args.output_directory, args.checkpoint_path, args.n_gpus, args.rank, args.group_name, hparams, args.use_training_mode, args.verify_outputs, args.use_validation_files, args.fp16_save, audio_offset=ioffset, extra_info=f"{ind}/{hparams.hop_length//args.extremeGTA} ")
    else:
        GTA_Synthesis(args.output_directory, args.checkpoint_path, args.n_gpus, args.rank, args.group_name, hparams, args.use_training_mode, args.verify_outputs, args.use_validation_files, args.fp16_save)
    print("GTA Done!")
