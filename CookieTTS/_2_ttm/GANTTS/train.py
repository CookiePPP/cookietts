import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
import time
import argparse
import math
import numpy as np
from numpy import finfo

import torch
import torch.nn as nn
import torch.nn.functional as F
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import GANTTS, load_model, load_model_d
from data_utils import TextMelLoader, TextMelCollate
from logger import Tacotron2Logger
from hparams import create_hparams
from math import e
from tqdm import tqdm
from CookieTTS.utils.dataset.utils import load_wav_to_torch, load_filepaths_and_text

save_file_check_path = "save"

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


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")
    
    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Initialize distributed communication
    dist.init_process_group(backend=hparams.dist_backend, init_method=hparams.dist_url,
                                   world_size=n_gpus, rank=rank, group_name=group_name)
    
    print("Done initializing distributed")


def prepare_dataloaders(hparams, saved_lookup):
    # Get data, data loaders and collate function ready
    speaker_ids = saved_lookup if hparams.use_saved_speakers else None
    trainset = TextMelLoader(hparams.training_files, hparams, check_files=hparams.check_files, shuffle=False,
                           speaker_ids=speaker_ids)
    valset = TextMelLoader(hparams.validation_files, hparams, check_files=hparams.check_files, shuffle=False,
                           speaker_ids=trainset.speaker_ids)
    collate_fn = TextMelCollate(hparams)
    
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset,shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=2, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler, trainset


def prepare_directories_and_logger(output_directory, log_directory, rank, sampling_rate):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory), sampling_rate)
    else:
        logger = None
    return logger


def warm_start_force_model(checkpoint_path, model, from_zero=False):
    assert os.path.isfile(checkpoint_path)
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    pretrained_dict = checkpoint_dict['state_dict']
    model_dict = model.state_dict()
    # Fiter out unneccessary keys
    filtered_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict_missing = {k: v for k,v in pretrained_dict.items() if k not in model_dict}
    model_dict_mismatching = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape}
    pretrained_missing = {k: v for k,v in model_dict.items() if k not in pretrained_dict}
    if model_dict_missing: print(list(model_dict_missing.keys()),'does not exist in the current model and is being ignored')
    if model_dict_mismatching: print(list(model_dict_mismatching.keys()),"is the wrong shape and has been reset")
    if pretrained_missing: print(list(pretrained_missing.keys()),"doesn't have pretrained weights and has been reset")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    iteration = 0 if from_zero else (checkpoint_dict['iteration'] if 'iteration' in checkpoint_dict.keys() else 0)
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def warm_start_model(checkpoint_path, model, ignore_layers, from_zero=False):
    assert os.path.isfile(checkpoint_path)
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    iteration = 0 if from_zero else (checkpoint_dict['iteration'] if 'iteration' in checkpoint_dict.keys() else 0)
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def load_checkpoint(checkpoint_path, model, optimizer, discriminator, optimizer_d, from_zero=False):
    assert os.path.isfile(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    
    if 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if 'amp' in checkpoint_dict.keys():
        amp.load_state_dict(checkpoint_dict['amp'])
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    #if 'hparams' in checkpoint_dict.keys():
    #    hparams = checkpoint_dict['hparams']
    if 'average_loss' in checkpoint_dict.keys():
        average_loss = checkpoint_dict['average_loss']
    
    iteration = 0 if from_zero else checkpoint_dict['iteration']
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    
    # discriminator
    checkpoint_dict = torch.load(f'{checkpoint_path}_d', map_location='cpu')
    discriminator.load_state_dict(checkpoint_dict['state_dict'])
    if 'optimizer' in checkpoint_dict.keys():
        optimizer_d.load_state_dict(checkpoint_dict['optimizer'])
    
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, discriminator, optimizer_d, iteration, saved_lookup


def save_checkpoint(model, optimizer, model_d, optimizer_d, learning_rate, iteration, hparams, speaker_id_lookup, filepath):
    from CookieTTS.utils.dataset.utils import load_filepaths_and_text
    tqdm.write(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    
    # get speaker names to ID
    speakerlist = load_filepaths_and_text(hparams.speakerlist)
    speaker_name_lookup = {x[1]: speaker_id_lookup[x[2]] for x in speakerlist if x[2] in speaker_id_lookup.keys()}

    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,
                #'amp': amp.state_dict(),
                'hparams': hparams,
                'speaker_id_lookup': speaker_id_lookup,
                'speaker_name_lookup': speaker_name_lookup,
                }, filepath)
    
    torch.save({'iteration': iteration,
                'state_dict': model_d.state_dict(),
                'optimizer': optimizer_d.state_dict(),
                }, f'{filepath}_d')
    tqdm.write("Saving Complete")


def validate(model, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=2,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, drop_last=True, collate_fn=collate_fn)
        val_loss = 0.0
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            x = model.parse_batch(batch)
            with torch.random.fork_rng(devices=[0,]):
                torch.random.manual_seed(0)# use same seed during validation so results are more consistent and comparable.
                pred_audio, pred_durations = model(x)
            #if distributed_run:
            #    reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            #else:
            #    reduced_val_loss = loss.item()
            #val_loss += reduced_val_loss
            # end forloop
        #val_loss = val_loss / (i + 1)
        # end torch.no_grad()
    model.train()
    if rank == 0:
        tqdm.write(f"Validation loss {iteration}: {val_loss:9f}")
        logger.log_validation(val_loss, model, x, pred_audio, iteration)
    return val_loss


def train(output_directory, log_directory, checkpoint_path, warm_start, warm_start_force, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    # setup distributed
    hparams.n_gpus = n_gpus
    hparams.rank = rank
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)
    
    # reproducablilty stuffs
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    # initialize blank model
    model = load_model(hparams)
    model.eval()
    
    # initialize blank discriminator
    discriminator = load_model_d(hparams)
    discriminator.eval()
    
    # (optional) show the names of each layer in model, mainly makes it easier to copy/paste what you want to adjust
    if hparams.print_layer_names_during_startup:
        print(*[f"Layer{i} = "+str(x[0])+" "+str(x[1].shape) for i,x in enumerate(list(model.named_parameters()))], sep="\n")
    
    # (optional) Freeze layers by disabling grads
    if len(hparams.frozen_modules):
        for layer, params in list(model.named_parameters()):
            if any(layer.startswith(module) for module in hparams.frozen_modules):
                params.requires_grad = False
                print(f"Layer: {layer} has been frozen")
    
    # define optimizer (any params without requires_grad are ignored)
    if True:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=0.0, weight_decay=hparams.weight_decay)
        discriminator_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                             lr=0.0, weight_decay=hparams.weight_decay)
    else:
        optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=0.0, weight_decay=hparams.weight_decay)
        discriminator_optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, discriminator.parameters()),
                                             lr=0.0, weight_decay=hparams.weight_decay)
    
    if hparams.fp16_run:
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        discriminator, discriminator_optimizer = amp.initialize(discriminator, discriminator_optimizer, opt_level=opt_level)
    
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
        discriminator = apply_gradient_allreduce(discriminator)
    
    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank, hparams.sampling_rate)
    
    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    saved_lookup = None
    if checkpoint_path is not None:
        if warm_start:
            model, iteration, saved_lookup = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        elif warm_start_force:
            model, iteration, saved_lookup = warm_start_force_model(checkpoint_path, model)
        else:
            model, optimizer, discriminator, discriminator_optimizer, iteration, saved_lookup = load_checkpoint(
                                                                       checkpoint_path, model, optimizer, discriminator, discriminator_optimizer)
            iteration += 1  # next iteration is iteration + 1
        print('Model Loaded')
    
    # define datasets/dataloaders
    train_loader, valset, collate_fn, train_sampler, trainset = prepare_dataloaders(hparams, saved_lookup)
    epoch_offset = max(0, int(iteration / len(train_loader)))
    speaker_lookup = trainset.speaker_ids
    
    model.train()
    discriminator.train()
    is_overflow = False
    rolling_loss = StreamingMovingAverage(min(int(len(train_loader)), 200))
    rolling_d_loss = StreamingMovingAverage(min(int(len(train_loader)), 200))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in tqdm(range(epoch_offset, hparams.epochs), initial=epoch_offset, total=hparams.epochs, desc="Epoch:", position=1, unit="epoch"):
        tqdm.write("Epoch:{}".format(epoch))

        if hparams.distributed_run: # shuffles the train_loader when doing multi-gpu training
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        # start iterating through the epoch
        for i, batch in tqdm(enumerate(train_loader), desc="Iter:  ", smoothing=0, total=len(train_loader), position=0, unit="iter"):
            ###################################
            ### Live Learning Rate & Params ###
            ###################################
            if (iteration % 10 == 0 or i==0):
                try:
                    with open("run_every_epoch.py") as f:
                        internal_text = str(f.read())
                        if len(internal_text) > 0:
                            ldict = {'iteration': iteration}
                            exec(internal_text, globals(), ldict)
                        else:
                            print("[info] tried to execute 'run_every_epoch.py' but it is empty")
                except Exception as ex:
                    print(f"[warning] 'run_every_epoch.py' FAILED to execute!\nException:\n{ex}")
                globals().update(ldict)
                locals().update(ldict)
                if iteration < decay_start:
                    learning_rate = A_ + C_
                else:
                    iteration_adjusted = iteration - decay_start
                    learning_rate = (A_*(e**(-iteration_adjusted/B_))) + C_
                learning_rate = max(min_learning_rate, learning_rate)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                for param_group in discriminator_optimizer.param_groups:
                    param_group['lr'] = learning_rate * descriminator_loss_scale
            # /run external code every epoch, allows the run to be adjusting without restarts/
            
            #########################
            ###    Model Stuff    ###
            #########################
            model.zero_grad()
            
            x = model.parse_batch(batch) # move batch to GPU (async)
            true_labels = torch.zeros(hparams.batch_size, device=x[0].device, dtype=x[0].dtype)# [B]
            fake_labels = torch.ones( hparams.batch_size, device=x[0].device, dtype=x[0].dtype)# [B]
            
            pred_audio, pred_durations = model(x)
            
            model_fakeness = discriminator(pred_audio, x[1]) # [B] -> [] predict fakeness of generated samples
            model_loss = nn.BCELoss()(model_fakeness, true_labels) # calc loss to decrease fakeness of model
            reduced_model_loss = reduce_tensor(model_loss.data, n_gpus).item() if hparams.distributed_run else model_loss.item()
            
            if hparams.fp16_run:
                with amp.scale_loss(model_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                model_loss.backward()
            
            if grad_clip_thresh > 0.0:
                if hparams.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), grad_clip_thresh)
                    is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_thresh)
            
            optimizer.step()
            
            #############################
            ###  Discriminator Stuff  ###
            #############################
            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            true_fakeness = discriminator(x[0], x[1])# predicted fakeness of the actual audio sample.
            true_discriminator_loss = nn.BCELoss()(true_fakeness, true_labels)# loss for predicted fakeness of actual real audio.
            
            # add .detach() here think about this
            model_fakeness = discriminator(pred_audio.detach(), x[1])
            fake_discriminator_loss = nn.BCELoss()(model_fakeness, fake_labels)# calc loss to increase fakeness of discriminator when it sees these samples
            discriminator_loss = (true_discriminator_loss + fake_discriminator_loss) / 2
            reduced_discriminator_loss = reduce_tensor(discriminator_loss.data, n_gpus).item() if hparams.distributed_run else discriminator_loss.item()
            
            if hparams.fp16_run:
                with amp.scale_loss(discriminator_loss, discriminator_optimizer) as scaled_d_loss:
                    scaled_d_loss.backward()
            else:
                discriminator_loss.backward()
            
            if grad_clip_thresh > 0.0:
                if hparams.fp16_run:
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(discriminator_optimizer), grad_clip_thresh)
                    is_overflow = math.isinf(grad_norm_d) or math.isnan(grad_norm_d)
                else:
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), grad_clip_thresh)
            
            discriminator_optimizer.step()
            
            #########################
            ###  Logging Metrics  ###
            #########################
            if not is_overflow and rank == 0:
                duration = time.time() - start_time
                average_loss = rolling_loss.process(reduced_model_loss)
                average_d_loss = rolling_d_loss.process(reduced_discriminator_loss)
                tqdm.write("{} [Train_loss {:.4f} Avg {:.4f}] [Descrim_loss {:.4f} Avg {:.4f}] [Grad Norm {:.4f} D {:.4f}] [{:.2f}s/it] [{:.3f}s/file] [{:.7f} LR]".format(
                    iteration, reduced_model_loss, average_loss, reduced_discriminator_loss, average_d_loss, grad_norm, grad_norm_d, duration, (duration/(hparams.batch_size*n_gpus)), learning_rate) )
                logger.log_training(iteration, reduced_model_loss, reduced_discriminator_loss, grad_norm, grad_norm_d, learning_rate, duration)
                start_time = time.time()
            elif is_overflow and rank == 0:
                tqdm.write("Gradient Overflow! Skipping Step")
            
            #########################
            ### Save Checkpoints? ###
            #########################
            if not is_overflow and (iteration%hparams.iters_per_checkpoint == 0 or os.path.exists(save_file_check_path)):
                # save model checkpoint like normal
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, discriminator, discriminator_optimizer, learning_rate, iteration, hparams, speaker_lookup, checkpoint_path)
                if rank == 0 and os.path.exists(save_file_check_path):
                    os.remove(save_file_check_path)
            
            ################################################
            ###  Valiation (Pred Spectrograms / MOSNet)  ###
            ################################################
            if not is_overflow and (iteration%hparams.iters_per_validation==0 or (iteration < 1000 and iteration%250==0)):
                # perform validation and save "best_val_model" depending on validation loss
                val_loss = validate(model, valset, iteration, hparams.val_batch_size, n_gpus, collate_fn, logger, hparams.distributed_run, rank)
                #if rank == 0 and val_loss < best_validation_loss:
                #    checkpoint_path = os.path.join(output_directory, "best_val_model")
                #    save_checkpoint(model, optimizer, discriminator, discriminator_optimizer, learning_rate, iteration, hparams, speaker_lookup, checkpoint_path)
                #best_validation_loss = min(val_loss, best_validation_loss)
            
            iteration += 1
            # end of iteration loop
        # end of epoch loop


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='outdir',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str, default='logdir',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--warm_start_force', action='store_true',
                        help='load model weights only, ignore all missing/non-matching layers')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='detects NaN/Infs in autograd backward pass and gives additional debug info.')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    
    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    
    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    
    if args.detect_anomaly: # checks backprop for NaN/Infs and outputs very useful stack-trace. Runs slowly while enabled.
        torch.autograd.set_detect_anomaly(True)
        print("Autograd Anomaly Detection Enabled!\n(Code will run slower but backward pass will output useful info if crashing or NaN/inf values)")
    
    # these are needed for fp16 training, not inference
    if hparams.fp16_run:
        from apex import amp
        from apex import optimizers as apexopt

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.warm_start_force, args.n_gpus, args.rank, args.group_name, hparams)