import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
import time
import argparse
import math
from numpy import finfo
import numpy as np

import torch
import torch.nn.functional as F
from distributed import DistributedDataParallel
import torch.distributed as dist
from torch.nn import DataParallel

from model import Tacotron2, load_model
from CookieTTS.utils import get_args, force
from hparams import create_hparams
from train import init_distributed, prepare_dataloaders, init_distributed, StreamingMovingAverage


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = checkpoint_dict['state_dict']
    filtered_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict_missing = {k: v for k,v in pretrained_dict.items() if k not in model_dict}
    print("model_dict_missing.keys() =", model_dict_missing.keys())
    model.load_state_dict(filtered_dict)
    return model


def get_global_mean(data_loader, global_mean_npy, hparams):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return (torch.tensor(global_mean).half()).cuda() if hparams.fp16_run else (torch.tensor(global_mean).float()).cuda()
    else:
        raise Exception("No global_mean.npy found while in training_mode.")
    return global_mean


def get_durations(alignments, output_lengths, input_lengths):
    batch_durations = []
    for alignment, output_length, input_length in zip(alignments, output_lengths, input_lengths):
        alignment = alignment[:output_length, :input_length]
        dur_frames = torch.histc(torch.argmax(alignment, dim=1).float(), min=0, max=input_length-1, bins=input_length)# number of frames each letter taken the maximum focus of the model.
        assert dur_frames.sum().item() == output_length, f'{dur_frames.sum().item()} != {output_length}'
        batch_durations.append(dur_frames)
    return batch_durations# [[enc_T], [enc_T], [enc_T], ...]


def get_alignments(alignments, output_lengths, input_lengths):
    alignments_arr = []
    for alignment, output_length, input_length in zip(alignments, output_lengths, input_lengths):
        alignment = alignment[:output_length, :input_length]
        alignments_arr.append(alignment)
    return alignments_arr# [[dec_T, enc_T], [dec_T, enc_T], [dec_T, enc_T], ...]


@torch.no_grad()
def GTA_Synthesis(hparams, args, extra_info='', audio_offset=0):
    """Generate Ground-Truth-Aligned Spectrograms for Training WaveGlow."""
    rank   = args.rank
    n_gpus = args.n_gpus
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.use_validation_files:
        filelisttype = "val"
        hparams.training_files = hparams.validation_files
    else:
        filelisttype = "train"
    
    # initialize blank model
    print('Initializing Tacotron2...')
    model = load_model(hparams)
    print('Done')
    global model_args
    model_args = get_args(model.forward)
    model.eval()
    
    # Load checkpoint
    assert args.checkpoint_path is not None
    print('Loading Tacotron2 Checkpoint...')
    model = warm_start_model(args.checkpoint_path, model)
    print('Done')
    
    _ = model.train() if args.use_training_mode else model.eval()# set model to either train() or eval() mode. (controls dropout + DFR)
    
    print("Initializing AMP Model")
    if hparams.fp16_run:
        model = amp.initialize(model, opt_level='O2')
    print('Done')
    
    # define datasets/dataloaders
    train_loader, valset, collate_fn, train_sampler, trainset, *_ = prepare_dataloaders(hparams, model_args, args, None, audio_offset=audio_offset)
    
    # load and/or generate global_mean
    if args.use_training_mode and hparams.drop_frame_rate > 0.:
        if rank != 0: # if global_mean not yet calcuated, wait for main thread to do it
            while not os.path.exists(hparams.global_mean_npy): time.sleep(1)
        hparams.global_mean = get_global_mean(train_loader, hparams.global_mean_npy, hparams)
    
    # ================ MAIN TRAINNIG LOOP! ===================
    os.makedirs(os.path.join(args.output_directory), exist_ok=True)
    f = open(os.path.join(args.output_directory, f'map_{filelisttype}_gpu{rank}.txt'),'w', encoding='utf-8')
    
    processed_files = 0
    failed_files = 0
    duration = time.time()
    total = len(train_loader)
    rolling_sum = StreamingMovingAverage(100)
    for i, y in enumerate(train_loader):
        y_gpu = model.parse_batch(y) # move batch to GPU
        
        y_pred_gpu = force(model, valid_kwargs=model_args, **{**y_gpu, "teacher_force_till": 0, "p_teacher_forcing": 1.0, "drop_frame_rate": 0.0})
        y_pred = {k: v.cpu() for k,v in y_pred_gpu.items() if v is not None}# move model outputs to CPU
        if args.fp16_save:
            y_pred = {k: v.half() for k,v in y_pred.items()}# convert model outputs to fp16
        
        if args.save_letter_alignments or args.save_phone_alignments:
            alignments = get_alignments(y_pred['alignments'], y['mel_lengths'], y['text_lengths'])# [B, mel_T, txt_T] -> [[B, mel_T, txt_T], [B, mel_T, txt_T], ...]
        
        offset_append = '' if audio_offset == 0 else str(audio_offset)
        for j in range(len(y['gt_mel'])):
            gt_mel   = y['gt_mel'  ][j, :, :y['mel_lengths'][j]]
            pred_mel = y_pred['pred_mel_postnet'][j, :, :y['mel_lengths'][j]]
            
            audiopath      = y['audiopath'][j]
            speaker_id_ext = y['speaker_id_ext'][j]
            
            if True or (args.max_mse or args.max_mae):
                MAE = F. l1_loss(pred_mel, gt_mel).item()
                MSE = F.mse_loss(pred_mel, gt_mel).item()
                if args.max_mse and MSE > args.max_mse:
                    print(f"MSE ({MSE}) is greater than max MSE ({args.max_mse}).\nFilepath: '{audiopath}'\n")
                    failed_files+=1; continue
                if args.max_mae and MAE > args.max_mae:
                    print(f"MAE ({MAE}) is greater than max MAE ({args.max_mae}).\nFilepath: '{audiopath}'\n")
                    failed_files+=1; continue
            else:
                MAE = MSE = 'N/A'
            
            print(f"PATH: '{audiopath}'\nMel Shape:{list(gt_mel.shape)}\nSpeaker_ID: {speaker_id_ext}\nMSE: {MSE}\nMAE: {MAE}")
            if not args.do_not_save_mel:
                pred_mel_path = os.path.splitext(audiopath)[0]+'.pred_mel.pt'
                torch.save(pred_mel.clone(), pred_mel_path)
                pm_audio_path = os.path.splitext(audiopath)[0]+'.pm_audio.pt'# predicted mel audio
                torch.save(y['gt_audio'][j, :y['audio_lengths'][j]].clone(), pm_audio_path)
            if args.save_letter_alignments and hparams.p_arpabet == 0.:
                save_path_align_out = os.path.splitext(audiopath)[0]+'_galign.pt'
                np.save(alignments[j].clone(), save_path_align_out)
            if args.save_phone_alignments and hparams.p_arpabet == 1.:
                save_path_align_out = os.path.splitext(audiopath)[0]+'_palign.pt'
                np.save(alignments[j].clone(), save_path_align_out)
            map = f"{audiopath}|{y['gtext_str'][j]}|{speaker_id_ext}|\n"
            
            f.write(map)# write paths to text file
            processed_files+=1
            print("")
        
        duration = time.time() - duration
        avg_duration = rolling_sum.process(duration)
        time_left = round(((total-i) * avg_duration)/3600, 2)
        print(f'{extra_info}{i}/{total} compute and save GTA melspectrograms in {i}th batch, {duration}s, {time_left}hrs left. {processed_files} processed, {failed_files} failed.')
        duration = time.time()
    f.close()
    
    if n_gpus > 1:
        torch.distributed.barrier()# wait till all graphics cards reach this point.
    
    # merge all generated filelists from every GPU
    filenames = [f'map_{filelisttype}_gpu{j}.txt' for j in range(n_gpus)]
    if rank == 0:
        with open(os.path.join(args.output_directory, f'map_{filelisttype}.txt'), 'w') as outfile:
            for fname in filenames:
                with open(os.path.join(args.output_directory, fname)) as infile:
                    for line in infile:
                        if len(line.strip()):
                            outfile.write(line)


if __name__ == '__main__':
    """
    This script will run Tacotron2 over the hparams filelist(s), and save ground truth aligned spectrograms for each file.
    In the output_directory will be a filelist that can be used to train WaveGlow/WaveFlow on the aligned tacotron outputs, which will increase audio quality when generating new text.
    
    Example:
    CUDA_VISIBLE_DEVICES=0,1,2 python3 -m multiproc GTA.py -o "GTA_flist" -c "outdir/checkpoint_300000" --fp16_save --max_mse 0.40
    CUDA_VISIBLE_DEVICES=0,1,2 python3 -m multiproc GTA.py -o "GTA_flist" -c "outdir/checkpoint_300000" --fp16_save --max_mse 0.40 --use_validation_files
    
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
                        required=False, help='how many processes (workers) are used to load data for each GPU. Each process will use a chunk of RAM. 24 Workers on a Threadripper 2950X is enough to feed three RTX 2080 Ti\'s in fp16 mode.')
    parser.add_argument('--extremeGTA', type=int, default=0, required=False,
                        help='Generate a Ground Truth Aligned output every interval specified. This will run tacotron hop_length//interval times per file and can use thousands of GBs of storage. Caution is advised')
    parser.add_argument('--max_mse', default=None, required=False,
                        help='Maximum MSE from Ground Truth to be valid for saving. (Anything above this value will be discarded)')
    parser.add_argument('--max_mae', default=None, required=False,
                        help='Maximum MAE from Ground Truth to be valid for saving. (Anything above this value will be discarded)')
    parser.add_argument('--use_training_mode', action='store_true',
                        help='Use model.train() while generating alignments. Will increase both variablility and inaccuracy.')
    parser.add_argument('--use_validation_files', action='store_true',
                        help='Ground Truth Align validation files instead of training files.')
    parser.add_argument('--save_letter_alignments', action='store_true',
                        help='Save alignments of each grapheme in the input.')
    parser.add_argument('--save_phone_alignments', action='store_true',
                        help='Save alignments of each phoneme in the input.')
    parser.add_argument('--do_not_save_mel', action='store_true',
                        help='Do not save predicted mel-spectrograms / AEFs.')
    parser.add_argument('--fp16_save', action='store_true',
                        help='Save spectrograms using np.float16 aka Half Precision. Will reduce the storage space required.')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()
    
    hparams = create_hparams(args.hparams)
    hparams.n_gpus = args.n_gpus
    hparams.rank   = args.rank
    hparams.num_workers = args.num_workers
    hparams.use_TBPTT = False # remove limit
    hparams.truncated_length = 2**15 # remove limit
    hparams.p_arpabet = float(round(hparams.p_arpabet))
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    print("FP16 Run:", hparams.fp16_run)
    print("Distributed Run:", hparams.distributed_run)
    print("Rank:", args.rank)
    
    if hparams.fp16_run:
        from apex import amp
    
    if not args.use_validation_files:
        hparams.batch_size = hparams.batch_size * 6 # no gradients stored so batch size can go up a bunch
    
    torch.autograd.set_grad_enabled(False)
    
    if hparams.distributed_run:
        init_distributed(hparams, args.n_gpus, args.rank, args.group_name)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.extremeGTA:
        for ind, ioffset in enumerate(range(0, hparams.hop_length, args.extremeGTA)): # generate aligned spectrograms for all audio samples
            if ind < 0:
                continue
            GTA_Synthesis(hparams, args, audio_offset=ioffset, extra_info=f"{ind+1}/{hparams.hop_length//args.extremeGTA} ")
    elif args.save_letter_alignments and args.save_phone_alignments:
        hparams.p_arpabet = 0.0
        GTA_Synthesis(hparams, args, extra_info="1/2 ")
        hparams.p_arpabet = 1.0
        GTA_Synthesis(hparams, args, extra_info="2/2 ")
    else:
        GTA_Synthesis(hparams, args)
    print("GTA Done!")
