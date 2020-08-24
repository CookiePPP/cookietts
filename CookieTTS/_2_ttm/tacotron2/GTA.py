import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
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

from model import load_model
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
    collate_fn = TextMelCollate(hparams)
    
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset, shuffle=False)
    else:
        train_sampler = None
    
    train_loader = DataLoader(trainset, num_workers=hparams.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False, # default pin_memory=False, True should allow async memory transfers # Causes very random CUDA errors (after like 4+ hours)
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, None, collate_fn, train_sampler, trainset


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
    
    if hparams.fp16_run:
        model = amp.initialize(model, opt_level='O2')
    
    model.decoder.dump_attention_weights = True # hidden param to dump attention weights
    
    # ================ MAIN TRAINNIG LOOP! ===================
    os.makedirs(os.path.join(output_directory), exist_ok=True)
    f = open(os.path.join(output_directory, f'map_{filelisttype}_{rank}.txt'),'a', encoding='utf-8')
    os.makedirs(os.path.join(output_directory,'mels'), exist_ok=True)
    
    total_number_of_data = len(train_set.audiopaths_and_text)
    max_itter = total_number_of_data//(hparams.batch_size*n_gpus)
    remainder_size = (total_number_of_data % (hparams.batch_size*n_gpus))
    remainder_size = min(remainder_size-(rank*hparams.batch_size), hparams.batch_size)
    
    processed_files = 0
    failed_files = 0
    duration = time.time()
    total = len(train_loader)
    rolling_sum = StreamingMovingAverage(100)
    for i, batch in enumerate(train_loader):
        last_batch = i == max_itter
        batch_size = remainder_size if last_batch else hparams.batch_size
        
        #print( [batch[-1] for j in range(len(batch))] )
        
        # get wavefile path
        batch_start = (i*hparams.batch_size*n_gpus) + rank
        batch_end   = ((i+1)*hparams.batch_size*n_gpus) + rank
        audiopaths_and_text = train_set.audiopaths_and_text[batch_start:batch_end][::n_gpus]
        audiopaths  = [x[0] for x in audiopaths_and_text] # filelist
        speaker_ids = [x[2] for x in audiopaths_and_text] # filelist
        
        # get len texts
        indx_list = np.arange(batch_start, batch_end, n_gpus).tolist()
        len_text_list = []
        for batch_index in indx_list:
            text, *_ = train_set.get_mel_text_pair(batch_index,
                                              ignore_emotion=1, ignore_speaker=1, ignore_torchmoji=1, ignore_sylps=1, ignore_mel=1)
            len_text_list.append(text.size(0))
        
        _, input_lengths, _, _, output_lengths, *_, index = batch # output_lengths: original mel length
        input_lengths_, ids_sorted_decreasing = torch.sort(torch.LongTensor(len_text_list), dim=0, descending=True)
        assert (input_lengths_ == input_lengths).all().item(), 'Error loading text lengths! Text Lengths from Dataloader do not match Text Lengths from GTA.py'
        ids_sorted_decreasing = ids_sorted_decreasing.numpy() # ids_sorted_decreasing, original index
        
        sorted_audiopaths = [] # original_file_name
        sorted_mel_paths = []
        sorted_speaker_ids = []
        for k in range(batch_size):
            sorted_audiopath = audiopaths[ids_sorted_decreasing[k]]
            sorted_audiopaths.append(sorted_audiopath)
            sorted_mel_paths.append( sorted_audiopath.replace(".npy",".mel").replace('.wav','.mel'))
            
            sorted_speaker_id = speaker_ids[ids_sorted_decreasing[k]]
            sorted_speaker_ids.append(sorted_speaker_id)
        
        x, _ = model.parse_batch(batch)
        _, mel_outputs_postnet, _, *_ = model(x, teacher_force_till=9999, p_teacher_forcing=1.0, drop_frame_rate=0.0)
        mel_outputs_postnet = mel_outputs_postnet.data.cpu()
        
        for k in range(batch_size):
            wav_path = sorted_audiopaths[k].replace(".npy",".wav")
            mel = mel_outputs_postnet[k,:,:output_lengths[k]]
            mel_path = sorted_mel_paths[k]
            speaker_id = sorted_speaker_ids[k]
            
            offset_append = '' if audio_offset == 0 else str(audio_offset)
            save_path = mel_path+offset_append+'.npy' # ext = '.mel.npy' or '.mel1.npy' ... '.mel599.npy'
            map = f"{wav_path}|{save_path}|{speaker_id}\n"
            f.write(map)
            
            mel_shape = list(mel[:model.n_mel_channels, :].shape)
            
            if verify_outputs:
                gt_mel = train_set.get_mel(wav_path)
                orig_shape = list(gt_mel.shape)
                MAE = torch.nn.functional.l1_loss(mel[:model.n_mel_channels, :], gt_mel).item()
                MSE = torch.nn.functional.mse_loss(mel[:model.n_mel_channels, :], gt_mel).item()
                # check mel from wav_path has same shape as mel just saved
            else:
                MSE = MAE = orig_shape = 'N/A'
            
            print(f"PATH: '{wav_path}'\nText Length: {input_lengths[k].item()}\nMel Shape:{mel_shape}\nSpeaker_ID: {speaker_id}\nTarget_Size: {orig_shape}\nMSE: {MSE}\nMAE: {MAE}\n")
            
            if orig_shape == 'N/A' or orig_shape == mel_shape:
                processed_files+=1
            else:
                failed_files+=1
                print(f"Target shape {orig_shape} does not match generated mel shape {mel_shape}.\nFilepath: '{wav_path}'")
                continue
            
            mel = mel.numpy()
            if fp16_save:
                mel = mel.astype(np.float16)
            np.save(save_path, mel)
        
        duration = time.time() - duration
        avg_duration = rolling_sum.process(duration)
        time_left = round(((total-i) * avg_duration)/3600, 2)
        print(f'{extra_info}{i}/{total} compute and save GTA melspectrograms in {i}th batch, {duration}s, {time_left}hrs left. {processed_files} processed, {failed_files} failed.')
        duration = time.time()
    f.close()
    
    # merge all generated filelists from every GPU
    filenames = [f'map_{filelisttype}_{j}.txt' for j in range(n_gpus)]
    with open(os.path.join(output_directory, f'map_{filelisttype}.txt'), 'w') as outfile:
        for fname in filenames:
            with open(os.path.join(output_directory, fname)) as infile:
                for line in infile:
                    if len(line.strip()):
                        outfile.write(line)


if __name__ == '__main__':
    """
    This script will run Tacotron2 over the hparams filelist(s), and save ground truth aligned spectrograms for each file.
    In the output_directory will be a filelist that can be used to train WaveGlow/WaveFlow on the aligned tacotron outputs, which will increase audio quality when generating new text.
    
    Example:
    CUDA_VISIBLE_DEVICES=0,1,2 python3 -m multiproc GTA.py -o "GTA_flist" -c "outdir/checkpoint_300000" --extremeGTA 100 --verify_outputs --fp16_save
    CUDA_VISIBLE_DEVICES=0,1,2 python3 -m multiproc GTA.py -o "GTA_flist" -c "outdir/checkpoint_300000" --extremeGTA 100 --verify_outputs --fp16_save --use_validation_files
    
     - In this example, CUDA_VISIBLE_DEVICES selects the 1st, 2nd and 3rd GPUs
     - '-o GTA_flist' is the location that the new filelist(s) will be saved
     - '-c ...' is the Tacotron2 checkpoint that will be used.
     - There are 2 commands here because both the training and validation_files are being generated.
    
    Params:
    --output_directory:
        Where to save the new filelist that is generated by this process.
        -o
    --checkpoint_path:
        Where to save the new filelist that is generated by this process.
        -c
    --extremeGTA: INT
        Example: 'python3 GTA.py -o outdir -c checkpoint_10000 --extremeGTA 100'
        Align to same file multiple times with a time offset.
        Will save `hop_length // extremeGTA` aligned copies for each file.
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
    parser.add_argument('--num_workers', type=int, default=8,
                        required=False, help='how many processes are used to load data for each GPU. Each process will use a chunk of RAM. 5 Workers on a Threadripper 2950X is enough to feed one RTX 2080 Ti in fp16 mode.')
    parser.add_argument('--extremeGTA', type=int, default=0, required=False,
                        help='Generate a Ground Truth Aligned output every interval specified. This will run tacotron hop_length//interval times per file and can use thousands of GBs of storage. Caution is advised')
    parser.add_argument('--use_training_mode', action='store_true',
                        help='Use model.train() while generating alignments. Will increase both variablility and inaccuracy.')
    parser.add_argument('--verify_outputs', action='store_true',
                        help='Debug Option. Checks output file and input file match.')
    parser.add_argument('--use_validation_files', action='store_true',
                        help='Ground Truth Align validation files instead of training files.')
    parser.add_argument('--fp16_save', action='store_true',
                        help='Save spectrograms using np.float16 aka Half Precision. Will reduce the storage space required.')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.n_gpus = args.n_gpus
    hparams.rank = args.rank
    hparams.num_workers = args.num_workers
    hparams.use_TBPTT = False # remove limit
    hparams.truncated_length = 2**15 # remove limit
    hparams.check_files=False # disable checks
    hparams.p_arpabet = 0.0
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    print("FP16 Run:", hparams.fp16_run)
    print("Distributed Run:", hparams.distributed_run)
    print("Rank:", args.rank)
    
    if hparams.fp16_run:
        from apex import amp
    
    # cookie stuff
    hparams.load_mel_from_disk = False
    hparams.training_files = hparams.training_files.replace("mel_train","train").replace("_merged.txt",".txt")
    hparams.validation_files = hparams.validation_files.replace("mel_val","val").replace("_merged.txt",".txt")
    
    if not args.use_validation_files:
        hparams.batch_size = hparams.batch_size * 12 # no gradients stored so batch size can go up a bunch
    
    torch.autograd.set_grad_enabled(False)
    
    if hparams.distributed_run:
        init_distributed(hparams, args.n_gpus, args.rank, args.group_name)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.extremeGTA:
        for ind, ioffset in enumerate(range(0, hparams.hop_length, args.extremeGTA)): # generate aligned spectrograms for all audio samples
            if ind < 0:
                continue
            GTA_Synthesis(args.output_directory, args.checkpoint_path, args.n_gpus, args.rank, args.group_name, hparams, args.use_training_mode, args.verify_outputs, args.use_validation_files, args.fp16_save, audio_offset=ioffset, extra_info=f"{ind}/{hparams.hop_length//args.extremeGTA} ")
    else:
        GTA_Synthesis(args.output_directory, args.checkpoint_path, args.n_gpus, args.rank, args.group_name, hparams, args.use_training_mode, args.verify_outputs, args.use_validation_files, args.fp16_save)
    print("GTA Done!")
