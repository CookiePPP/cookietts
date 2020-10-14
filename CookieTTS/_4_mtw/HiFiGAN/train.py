# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
os.environ["LRU_CACHE_CAPACITY"] = "3"
import torch
import os.path
import sys
import time
from math import ceil, e, exp
import math

import numpy as np
import soundfile as sf

save_file_check_path = "save"

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import broadcast
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from mel2samp import Mel2Samp
from tqdm import tqdm

from mel2samp import load_wav_to_torch
from scipy import signal

class LossExplosion(Exception):
    """Custom Exception Class. If Loss Explosion, raise Error and automatically restart training script from previous best_val_model checkpoint."""
    pass

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

def load_checkpoint(checkpoint_paths, model, optimizer, criterion, optimizer_d, scheduler, fp16_run, stage, warm_start=False, nvidia_checkpoint=False):
    checkpoint_paths = [checkpoint_paths,] if type(checkpoint_paths) == str else checkpoint_paths
    
    model_dict = model.state_dict() # just initialized state_dict
    
    for checkpoint_path in checkpoint_paths:
        assert os.path.isfile(checkpoint_path), f'"{checkpoint_path}" does not exist'
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if 'iteration' in checkpoint_dict:
            iteration = checkpoint_dict['iteration']
        
        if not warm_start:
            if 'optimizer' in checkpoint_dict.keys():
                optimizer.load_state_dict(checkpoint_dict['optimizer'])
            if fp16_run and 'amp' in checkpoint_dict.keys():
                amp.load_state_dict(checkpoint_dict['amp'])
            if scheduler and 'scheduler' in checkpoint_dict.keys():
                scheduler.load_state_dict(checkpoint_dict['scheduler'])
        
        checkpoint_model_dict = checkpoint_dict['model']
        if (str(type(checkpoint_model_dict)) != "<class 'collections.OrderedDict'>"):
            checkpoint_model_dict = checkpoint_model_dict.state_dict()
        
        if nvidia_checkpoint:# (optional) convert Nvidia/WaveGlow into my format
            checkpoint_model_dict = {k.replace(".in_layers",".WN.in_layers").replace(".res_skip_layers",".WN.res_skip_layers").replace(".start",".WN.start").replace(".end",".WN.end").replace(".conv.weight",".weight"): v for k, v in checkpoint_model_dict.items()}
        
        if warm_start:
            model_dict_missing = {k: v for k,v in checkpoint_model_dict.items() if k not in model_dict}
            if model_dict_missing:
                print(f"## {len(model_dict_missing.keys())} keys found that do not exist in the current model ##")
                print( '\n'.join([str(x) for x in model_dict_missing.keys()]) + '\n' )
            
            model_dict_mismatching = {k: v for k,v in checkpoint_model_dict.items() if k in model_dict and checkpoint_model_dict[k].shape != model_dict[k].shape}
            if model_dict_mismatching:
                print(f"## {len(model_dict_mismatching.keys())} keys found that do not match the shape in the current model ##")
                print( '\n'.join([str(x) for x in model_dict_mismatching.keys()]) + '\n' )
            
            pretrained_missing = {k: v for k,v in model_dict.items() if k not in checkpoint_model_dict}
            if pretrained_missing:
                print(f"## {len(pretrained_missing.keys())} keys in the current model that do not have a matching key in the checkpoint ##")
                print( '\n'.join([str(x) for x in pretrained_missing.keys()]) + '\n' )
            
            # Fiter out unneccessary keys
            filtered_dict = {k: v for k,v in checkpoint_model_dict.items() if k in model_dict and checkpoint_model_dict[k].shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
        else:
            model_dict = {k.replace("invconv1x1","convinv").replace(".F.",".WN.").replace("WNs.","WN."): v for k, v in checkpoint_model_dict.items()}
        
        if stage >= 2:
            if 'criterion' in checkpoint_dict and checkpoint_dict['hifigan_config']['stage'] == 2:
                criterion.load_state_dict(checkpoint_dict['criterion'])
            if 'optimizer_d' in checkpoint_dict:
                optimizer_d.load_state_dict(checkpoint_dict['optimizer_d'])
        print(f"Loaded checkpoint path '{checkpoint_path}'")
    
    model.load_state_dict(model_dict)
    print(f"New state_dict loaded! (iteration {iteration})")
    return model, optimizer, criterion, optimizer_d, iteration, scheduler

def save_checkpoint(model, optimizer, criterion, optimizer_d, learning_rate, iteration, amp, scheduler, speaker_lookup, filepath):
    tqdm.write("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    
    state_dict = model.state_dict()
    saving_dict = {'model': state_dict,
        'iteration': iteration,
        'scheduler': scheduler.state_dict(),
        'learning_rate': learning_rate,
        'speaker_lookup': speaker_lookup,
        'hifigan_config': hifigan_config,
        }
    if criterion is not None:
        saving_dict['criterion'] = criterion.state_dict()
    if optimizer_d is not None:
        saving_dict['optimizer_d'] = optimizer_d.state_dict()
    if optimizer is not None:
        saving_dict['optimizer'] = optimizer.state_dict()
    if amp:
        saving_dict['amp'] = amp.state_dict()
    torch.save(saving_dict, filepath)
    tqdm.write("Model Saved")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
def plot_spectrogram_to_numpy(spectrogram, range=None):
    fig, ax = plt.subplots(figsize=(16, 3))
    im = ax.imshow(spectrogram, cmap='inferno', aspect="auto", origin="lower",
                   interpolation='none')
    if range is not None:
        assert len(range) == 2, 'range params should be a 2 element List of [Min, Max].'
        assert range[1] > range[0], 'Max (element 1) must be greater than Min (element 0).'
        im.set_clim(range[0], range[1])
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def validate(model, trainset, logger, iteration, validation_files, speaker_lookup, output_directory, data_config, save_audio=True, max_length_s= 5., files_to_process=9):
    print("Validating... ", end="")
    
    model.eval()
    val_start_time = time.time()
    total = STFT_elapsed = samples_processed = 0
    with torch.no_grad():
        with torch.random.fork_rng(devices=[0,]):
            torch.random.manual_seed(0)# use same Z / random seed during validation so results are more consistent and comparable.
            
            with open(validation_files, encoding='utf-8') as f:
                audiopaths_and_melpaths = [line.strip().split('|') for line in f]
            
            model_type = "half" if next(model.parameters()).type() == "torch.cuda.HalfTensor" else "float"
            
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            files_processed = 0
            for i, (audiopath, melpath, *remaining) in enumerate(audiopaths_and_melpaths):
                if files_processed >= files_to_process: # number of validation files to run.
                    break
                noisy_audio, audio = trainset.get_from_path(audiopath, segment_length=720000) # load audio from wav file to tensor
                if audio.shape[0] > (data_config['sampling_rate']*max_length_s):
                    continue # ignore audio over max_length_seconds
                
                if hasattr(model, 'multispeaker') and model.multispeaker == True:
                    assert len(remaining), f"Speaker ID missing while multispeaker == True.\nLine: {i}\n'{'|'.join([autiopath, melpath])}'"
                    speaker_id = remaining[0]
                    speaker_id = torch.IntTensor([speaker_lookup[int(speaker_id)],])
                    speaker_id = speaker_id.cuda(non_blocking=True).long()
                else:
                    speaker_id = None
                
                noisy_audio = noisy_audio.half() if model_type == "half" else noisy_audio # for fp16 training
                
                pred_audio = model(noisy_audio.cuda().unsqueeze(0)).cpu().float().clamp(min=-0.999, max=0.999).squeeze(0)
                
                if save_audio:
                    audio_path = os.path.join(output_directory, "samples", str(iteration)+"-"+timestr, os.path.basename(audiopath)) # Write audio to checkpoint_directory/iteration/audiofilename.wav
                    os.makedirs(os.path.join(output_directory, "samples", str(iteration)+"-"+timestr), exist_ok=True)
                    sf.write(audio_path, pred_audio.squeeze().cpu().numpy(), data_config['sampling_rate'], "PCM_16") # save waveglow sample
                    
                    audio_path = os.path.join(output_directory, "samples", "Ground Truth", os.path.basename(audiopath))
                    if not os.path.exists(audio_path):# save ground truth
                        os.makedirs(os.path.join(output_directory, "samples", "Ground Truth"), exist_ok=True)
                        sf.write(audio_path, audio.squeeze().cpu().numpy(), data_config['sampling_rate'], "PCM_16")
                    
                    audio_path = os.path.join(output_directory, "samples", str(iteration)+"-"+timestr+"_NT", os.path.basename(audiopath))
                    if not os.path.exists(audio_path):# save noisy ground truth
                        os.makedirs(os.path.join(output_directory, "samples", str(iteration)+"-"+timestr+"_NT"), exist_ok=True)
                        sf.write(audio_path, noisy_audio.squeeze().cpu().numpy(), data_config['sampling_rate'], "PCM_16")
                files_processed+=1
                samples_processed+=pred_audio.shape[-1]
    
    if total:
        average_MSE = total_MSE/total
        average_MAE = total_MAE/total
        logger.add_scalar('val_MSE', average_MSE, iteration)
        logger.add_scalar('val_MAE', average_MAE, iteration)
    else:
        average_MSE = 1e3
        average_MAE = 1e3
        print("Average MSE: N/A", "Average MAE: N/A")
    
    time_elapsed = time.time()-val_start_time
    time_elapsed_without_stft = time_elapsed-STFT_elapsed
    samples_per_second = samples_processed/time_elapsed_without_stft
    print(f"[Avg MSE: {average_MSE:.6f} MAE: {average_MAE:.6f}]",
        f"[{time_elapsed_without_stft:.3f}s]",
        f"[{time_elapsed:.3f}s_stft]",
        f"[{time_elapsed_without_stft/files_processed:.3f}s/file]",
        f"[{samples_per_second/data_config['sampling_rate']:.3f}rtf]",
        f"[{samples_per_second:,.0f}samples/s]"
      )
    logger.add_scalar('val_rtf', samples_per_second/data_config['sampling_rate'], iteration)
    
    mel = speaker_id = None # del GPU based tensors.
    torch.cuda.empty_cache() # clear cache for next training
    model.train()
    
    return average_MSE, average_MAE

def get_optimizer(model, optimizer: str, fp16_run=True, optimizer_fused=True, learning_rate=1e-4, max_grad_norm=200):
    optimizer = optimizer.lower()
    if optimizer_fused:
        from apex import optimizers as apexopt
        if optimizer == "adam":
            optimizer = apexopt.FusedAdam(model.parameters(), lr=learning_rate)
        elif optimizer == "lamb":
            optimizer = apexopt.FusedLAMB(model.parameters(), lr=learning_rate, max_grad_norm=max_grad_norm)
    else:
        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "lamb":
            from lamb import Lamb as optLAMB
            optimizer = optLAMB(model.parameters(), lr=learning_rate)
    
    global amp
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else:
        amp = None
    return model, optimizer


def print_params(model, name='model'):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"{pytorch_total_params:,} total parameters in {name}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{pytorch_total_params:,} trainable parameters.")


def train(num_gpus, rank, group_name, stage, output_directory, epochs, learning_rate, sigma, iters_per_checkpoint, batch_size, seed, fp16_run, checkpoint_path, with_tensorboard, logdirname, datedlogdir, warm_start=False, optimizer='ADAM', start_zero=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    from model import HiFiGAN, HiFiGANLoss
    criterion = HiFiGANLoss(**hifigan_config).cuda()
    model = HiFiGAN(**hifigan_config).cuda()
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
        if stage >= 2:
            criterion = apply_gradient_allreduce(criterion)
    #=====END:   ADDED FOR DISTRIBUTED======
    
    criterion, optimizer_d = get_optimizer(criterion, optimizer, fp16_run, optimizer_fused=True) if stage >= 2 else (criterion, None)
    model, optimizer = get_optimizer(model, optimizer, fp16_run, optimizer_fused=True)
    
    ## LEARNING RATE SCHEDULER
    if True:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        min_lr = 1e-8
        factor = 0.1**(1/5) # amount to scale the LR by on Validation Loss plateau
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=20, cooldown=2, min_lr=min_lr, verbose=True, threshold=0.0001, threshold_mode='abs')
        print("ReduceLROnPlateau used as Learning Rate Scheduler.")
    else: scheduler=False
    
    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, criterion, optimizer_d, iteration, scheduler = load_checkpoint(checkpoint_path, model,
                                                      optimizer, criterion, optimizer_d, scheduler, fp16_run, stage, warm_start=warm_start)
        iteration += 1  # next iteration is iteration + 1
    if start_zero:
        iteration = 0
    
    trainset = Mel2Samp(**data_config, check_files=True)
    speaker_lookup = trainset.speaker_ids
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        train_sampler = DistributedSampler(trainset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=3, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    
    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)
    
    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        if datedlogdir:
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
            log_directory = os.path.join(output_directory, logdirname, timestr)
        else:
            log_directory = os.path.join(output_directory, logdirname)
        logger = SummaryWriter(log_directory)
    
    moving_average = int(min(len(train_loader), 200)) # average loss over entire Epoch
    rolling_sum = StreamingMovingAverage(moving_average)
    start_time = time.time()
    start_time_iter = time.time()
    start_time_dekaiter = time.time()
    model.train()
    
    # best (averaged) training loss
    if os.path.exists(os.path.join(output_directory, "best_model")+".txt"):
        best_model_loss = float(str(open(os.path.join(output_directory, "best_model")+".txt", "r", encoding="utf-8").read()).split("\n")[0])
    else:
        best_model_loss = 9e9
    
    # best (validation) MSE on inferred spectrogram.
    if os.path.exists(os.path.join(output_directory, "best_val_model")+".txt"):
        best_MSE = float(str(open(os.path.join(output_directory, "best_val_model")+".txt", "r", encoding="utf-8").read()).split("\n")[0])
    else:
        best_MSE = 9e9
    
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    print_params(model, name='generator')
    
    print(f"Segment Length: {data_config['segment_length']:,}\nBatch Size: {batch_size:,}\nNumber of GPUs: {num_gpus:,}\nSamples/Iter: {data_config['segment_length']*batch_size*num_gpus:,}")
    
    training = True
    while training:
        try:
            if rank == 0:
                epochs_iterator = tqdm(range(epoch_offset, epochs), initial=epoch_offset, total=epochs, smoothing=0.01, desc="Epoch", position=1, unit="epoch")
            else:
                epochs_iterator = range(epoch_offset, epochs)
            # ================ MAIN TRAINING LOOP! ===================
            for epoch in epochs_iterator:
                print(f"Epoch: {epoch}")
                if num_gpus > 1:
                    train_sampler.set_epoch(epoch)
                
                if rank == 0:
                    iters_iterator = tqdm(enumerate(train_loader), desc=" Iter", smoothing=0, total=len(train_loader), position=0, unit="iter", leave=True)
                else:
                    iters_iterator = enumerate(train_loader)
                for i, batch in iters_iterator:
                    # run external code every iter, allows the run to be adjusted without restarts
                    if (i==0 or iteration % param_interval == 0):
                        try:
                            with open("run_every_epoch.py") as f:
                                internal_text = str(f.read())
                                if len(internal_text) > 0:
                                    #code = compile(internal_text, "run_every_epoch.py", 'exec')
                                    ldict = {'iteration': iteration, 'seconds_elapsed': time.time()-start_time}
                                    exec(internal_text, globals(), ldict)
                                else:
                                    print("No Custom code found, continuing without changes.")
                        except Exception as ex:
                            print(f"Custom code FAILED to run!\n{ex}")
                        globals().update(ldict)
                        locals().update(ldict)
                        if show_live_params:
                            print(internal_text)
                    # Learning Rate Schedule
                    if custom_lr:
                        old_lr = learning_rate
                        if iteration < warmup_start:
                            learning_rate = warmup_start_lr
                        elif iteration < warmup_end:
                            learning_rate = (iteration-warmup_start)*((A_+C_)-warmup_start_lr)/(warmup_end-warmup_start) + warmup_start_lr # learning rate increases from warmup_start_lr to A_ linearly over (warmup_end-warmup_start) iterations.
                        else:
                            if iteration < decay_start:
                                learning_rate = A_ + C_
                            else:
                                iteration_adjusted = iteration - decay_start
                                learning_rate = (A_*(e**(-iteration_adjusted/B_))) + C_
                        assert learning_rate > -1e-8, "Negative Learning Rate."
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                        if optimizer_d is not None:
                            for param_group in optimizer_d.param_groups:
                                param_group['lr'] = learning_rate*d_lr_scale
                    else:
                        scheduler.patience = scheduler_patience
                        scheduler.cooldown = scheduler_cooldown
                        if override_scheduler_last_lr:
                            scheduler._last_lr = override_scheduler_last_lr
                        if override_scheduler_best:
                            scheduler.best = override_scheduler_best
                        if override_scheduler_last_lr or override_scheduler_best:
                            print(f"scheduler._last_lr = {scheduler._last_lr} scheduler.best = {scheduler.best}  |", end='')
                    model.zero_grad()
                    noisy_audio, gt_audio, speaker_ids = batch
                    noisy_audio = torch.autograd.Variable(noisy_audio.cuda(non_blocking=True))
                    gt_audio = torch.autograd.Variable(gt_audio.cuda(non_blocking=True))
                    speaker_ids = speaker_ids.cuda(non_blocking=True).long().squeeze(1)
                    pred_audio = model(noisy_audio)#, speaker_ids)
                    
                    metrics = criterion(pred_audio, gt_audio, amp, model, optimizer, optimizer_d, num_gpus, use_grad_clip, grad_clip_thresh)
                    
                    if not metrics['is_overflow'] and rank == 0:
                        # get current Loss Scale of first optimizer
                        loss_scale = amp._amp_state.loss_scalers[0]._loss_scale if fp16_run else 32768
                        
                        if with_tensorboard:
                            if (iteration % 100000 == 0):
                                # plot distribution of parameters
                                for tag, value in model.named_parameters():
                                    tag = tag.replace('.', '/')
                                    logger.add_histogram(tag, value.data.cpu().numpy(), iteration)
                            for key, value in metrics.items():
                                if key not in ['is_overflow',]:
                                    logger.add_scalar(key, value, iteration)
                            if (iteration % 20 == 0):
                                logger.add_scalar('learning.rate', learning_rate, iteration)
                            if (iteration % 10 == 0):
                                logger.add_scalar('duration', ((time.time() - start_time_dekaiter)/10), iteration)
                        
                        logged_loss = metrics['g_train_loss'] if stage >= 2 else metrics['train_loss']
                        grad_norm = metrics['grad_norm']
                        average_loss = rolling_sum.process(logged_loss)
                        if (iteration % 10 == 0):
                            tqdm.write("{} {}:  {:.3f} {:.3f}  {:.3f} {:08.3F} {:.8f}LR ({:.8f} Effective)  {:.2f}s/iter {:.4f}s/item".format(time.strftime("%H:%M:%S"), iteration, logged_loss, average_loss, best_MSE, round(grad_norm,3), learning_rate, min((grad_clip_thresh/grad_norm)*learning_rate,learning_rate), (time.time() - start_time_dekaiter)/10, ((time.time() - start_time_dekaiter)/10)/(batch_size*num_gpus)))
                            start_time_dekaiter = time.time()
                        else:
                            tqdm.write("{} {}:  {:.3f} {:.3f}  {:.3f} {:08.3F} {:.8f}LR ({:.8f} Effective) {}LS".format(time.strftime("%H:%M:%S"), iteration, logged_loss, average_loss, best_MSE, round(grad_norm,3), learning_rate, min((grad_clip_thresh/grad_norm)*learning_rate,learning_rate), loss_scale))
                        start_time_iter = time.time()
                    
                    if rank == 0 and (len(rolling_sum.values) > moving_average-2):
                        if (average_loss+best_model_margin) < best_model_loss:
                            checkpoint_path = os.path.join(output_directory, "best_model")
                            try:
                                save_checkpoint(model, optimizer, criterion, optimizer_d, learning_rate, iteration, amp, scheduler, speaker_lookup, checkpoint_path)
                            except KeyboardInterrupt: # Avoid corrupting the model.
                                save_checkpoint(model, optimizer, criterion, optimizer_d, learning_rate, iteration, amp, scheduler, speaker_lookup, checkpoint_path)
                            text_file = open((f"{checkpoint_path}.txt"), "w", encoding="utf-8")
                            text_file.write(str(average_loss)+"\n"+str(iteration))
                            text_file.close()
                            best_model_loss = average_loss #Only save the model if X better than the current loss.
                    if rank == 0 and iteration > 0 and ((iteration % iters_per_checkpoint == 0) or (os.path.exists(save_file_check_path))):
                        checkpoint_path = f"{output_directory}/waveglow_{iteration}"
                        save_checkpoint(model, optimizer, criterion, optimizer_d, learning_rate, iteration, amp, scheduler, speaker_lookup, checkpoint_path)
                        if (os.path.exists(save_file_check_path)):
                            os.remove(save_file_check_path)
                    
                    if iteration%validation_interval == 0:
                        if rank == 0:
                            MSE, MAE = validate(model, trainset, logger, iteration, data_config['validation_files'], speaker_lookup, output_directory, data_config)
                            if scheduler:
                                MSE = torch.tensor(MSE, device='cuda')
                                if num_gpus > 1:
                                    broadcast(MSE, 0)
                                scheduler.step(MSE.item())
                                if MSE < best_MSE:
                                    checkpoint_path = os.path.join(output_directory, "best_val_model")
                                    try:
                                        save_checkpoint(model, optimizer, criterion, optimizer_d, learning_rate, iteration, amp, scheduler, speaker_lookup, checkpoint_path)
                                    except KeyboardInterrupt: # Avoid corrupting the model.
                                        save_checkpoint(model, optimizer, learning_rate, iteration, amp, scheduler, speaker_lookup, checkpoint_path)
                                    text_file = open((f"{checkpoint_path}.txt"), "w", encoding="utf-8")
                                    text_file.write(str(MSE.item())+"\n"+str(iteration))
                                    text_file.close()
                                    best_MSE = MSE.item()
                        else:
                            if scheduler:
                                MSE = torch.zeros(1, device='cuda')
                                broadcast(MSE, 0)
                                scheduler.step(MSE.item())
                    iteration += 1
            training = False # exit the training While loop
        
        except LossExplosion as ex: # print Exception and continue from checkpoint. (turns out it takes < 4 seconds to restart like this, fucking awesome)
            print(ex) # print Loss
            checkpoint_path = os.path.join(output_directory, "best_model")
            assert os.path.exists(checkpoint_path), "best_model must exist for automatic restarts"
            
            # clearing VRAM for load checkpoint
            audio = mel = speaker_ids = loss = None
            torch.cuda.empty_cache()
            
            model.eval()
            model, optimizer, iteration, scheduler = load_checkpoint(checkpoint_path, model, optimizer, scheduler, fp16_run)
            learning_rate = optimizer.param_groups[0]['lr']
            epoch_offset = max(0, int(iteration / len(train_loader)))
            model.train()
            iteration += 1
            pass # and continue training.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='detects NaN/Infs in autograd backward pass and gives additional debug info.')
    args = parser.parse_args()
    
    if args.detect_anomaly: # checks backprop for NaN/Infs and outputs very useful stack-trace. Runs slowly while enabled.
        torch.autograd.set_detect_anomaly(True)
        print("Autograd Anomaly Detection Enabled!\n(Code will run slower but backward pass will output useful info if crashing or NaN/inf values)")
    
    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global hifigan_config
    hifigan_config = config["hifigan_config"]
    print(hifigan_config)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1
    
    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, hifigan_config['stage'], **train_config)
