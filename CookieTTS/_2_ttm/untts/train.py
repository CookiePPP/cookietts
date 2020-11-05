import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
import time
import argparse
import math
import numpy as np
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import UnTTS, load_model
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from CookieTTS.utils.model.GPU import to_gpu
import time
from math import e

from tqdm import tqdm
import CookieTTS.utils.audio.stft as STFT
from CookieTTS.utils.dataset.utils import load_wav_to_torch
from scipy.io.wavfile import read

import os.path

save_file_check_path = "save"
num_workers_ = 3 # DO NOT USE ABOVE 1 WHEN USING TRUNCATION
start_from_checkpoints_from_zero = 0
gen_new_mels = 0

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
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
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
        train_sampler = DistributedSampler(trainset, shuffle=False)#True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = False#True
    
    train_loader = DataLoader(trainset, num_workers=num_workers_, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler, trainset


def prepare_directories_and_logger(hparams, output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory), hparams)
    else:
        logger = None
    return logger


def warm_start_force_model(checkpoint_path, model):
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
    
    iteration = 0
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    iteration = checkpoint_dict['iteration']
    iteration = 0
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def load_checkpoint(checkpoint_path, model, optimizer, best_val_loss_dict, best_loss_dict, best_validation_loss=1e3):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint_dict['state_dict']) # original
    
    if 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if 'amp' in checkpoint_dict.keys() and amp is not None:
        amp.load_state_dict(checkpoint_dict['amp'])
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    #if 'hparams' in checkpoint_dict.keys():
    #    hparams = checkpoint_dict['hparams']
    if 'best_validation_loss' in checkpoint_dict.keys():
        best_validation_loss = checkpoint_dict['best_validation_loss']
    if 'best_val_loss_dict' in checkpoint_dict.keys():
        best_val_loss_dict = checkpoint_dict['best_val_loss_dict']
    if 'best_loss_dict' in checkpoint_dict.keys():
        best_loss_dict = checkpoint_dict['best_loss_dict']
    if 'average_loss' in checkpoint_dict.keys():
        average_loss = checkpoint_dict['average_loss']
	
	
	iteration = 0 if start_from_checkpoints_from_zero else checkpoint_dict['iteration']
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict


def save_checkpoint(model, optimizer, learning_rate, iteration, hparams, best_validation_loss, average_loss, best_val_loss_dict, best_loss_dict, speaker_id_lookup, filepath):
    from CookieTTS.utils.dataset.utils import load_filepaths_and_text
    tqdm.write("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    
    # get speaker names to ID
    speakerlist = load_filepaths_and_text(hparams.speakerlist)
    speaker_name_lookup = {x[1]: speaker_id_lookup[x[2]] for x in speakerlist if x[2] in speaker_id_lookup.keys()}
    
    save_dict = {'iteration'           : iteration,
                 'state_dict'          : model.state_dict(),
                 'optimizer'           : optimizer.state_dict(),
                 'learning_rate'       : learning_rate,
                 'hparams'             : hparams,
                 'speaker_id_lookup'   : speaker_id_lookup,
                 'speaker_name_lookup' : speaker_name_lookup,
                 'best_validation_loss': best_validation_loss,
                 'best_val_loss_dict'  : best_val_loss_dict,
                 'best_loss_dict'      : best_loss_dict,
                 'average_loss'        : average_loss}
    if hparams.fp16_run:
        save_dict['amp'] = amp.state_dict()
    torch.save(save_dict, filepath)
    tqdm.write("Saving Complete")


def validate(model, criterion, valset, loss_scalars, best_val_loss_dict, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=num_workers_,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, drop_last=True, collate_fn=collate_fn)
        val_loss_t = len_loss_t = loss_z_t = loss_w_t = loss_s_t = loss_att_t = dur_loss_z_t = dur_loss_w_t = dur_loss_s_t = 0.0
        loss_dict_total = None
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            x, y = model.parse_batch(batch)
            with torch.random.fork_rng(devices=[0,]):
                torch.random.manual_seed(0)# use same seed during validation so results are more consistent and comparable.
                y_pred = model(x)
            
            loss_dict = criterion(y_pred, y, loss_scalars)
            if loss_dict_total is None:
                loss_dict_total = {k: 0. for k, v in loss_dict.items()}
            
            if hparams.distributed_run:
                reduced_loss_dict = {k: reduce_tensor(v.data, n_gpus).item() if v is not None else 0. for k, v in loss_dict.items()}
            else:
                reduced_loss_dict = {k: v.item() if v is not None else 0. for k, v in loss_dict.items()}
            
            for k in loss_dict_total.keys():
                loss_dict_total[k] = loss_dict_total[k] + reduced_loss_dict[k]
            loss_terms_arr.append(loss_terms)
            val_loss += reduced_val_loss
            # end forloop
        loss_dict_total = {k: v/(i+1) for k, v in loss_dict_total.items()}
        # end torch.no_grad()
    model.train()
    
    if best_val_loss_dict is None:
        best_val_loss_dict = loss_dict_total
    else:
        best_val_loss_dict = {k: min(best_val_loss_dict[k], loss_dict_total[k]) for k in best_val_loss_dict.keys()}
    
    if rank == 0:
        tqdm.write(f"Validation loss {iteration}: {val_loss:9f}")
        logger.log_validation(loss_dict_total, best_val_loss_dict, model, y, y_pred, iteration)
    
    return best_val_loss_dict['loss'], best_val_loss_dict



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
    print('Initializing UnTTS...')
    model = load_model(hparams)
    print('Done')
    model.eval()
    learning_rate = hparams.learning_rate
    
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
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=hparams.weight_decay)
    optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=hparams.weight_decay)
    
    if True and rank == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("{:,} total parameters in model".format(pytorch_total_params))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("{:,} trainable parameters.".format(pytorch_total_params))
    
    if hparams.fp16_run:
        model, optimizer = amp.initialize(model, optimizer, opt_level=f'O{hparams.fp16_run_optlvl}')
    
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    
    criterion = Tacotron2Loss(hparams)
    
    logger = prepare_directories_and_logger(
        hparams, output_directory, log_directory, rank)
    
    # Load checkpoint if one exists
    best_validation_loss = 1e3 # used to see when "best_model" should be saved

    n_restarts = 0
    checkpoint_iter = 0
    iteration = 0
    epoch_offset = 0
    _learning_rate = 1e-3
    saved_lookup = None
    
    global best_val_loss_dict
    best_val_loss_dict = None
    global best_loss_dict
    best_loss_dict = None
    global expavg_loss_dict
    expavg_loss_dict = None
    expavg_loss_dict_iters = 0# initial iters expavg_loss_dict has been fitted
    loss_dict_smoothness = 0.95 # smoothing factor
    
    if checkpoint_path is not None:
        if warm_start:
            model, iteration, saved_lookup = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        elif warm_start_force:
            model, iteration, saved_lookup = warm_start_force_model(
                checkpoint_path, model)
        else:
            _ = load_checkpoint(checkpoint_path, model, optimizer, best_val_loss_dict, best_loss_dict)
            model, optimizer, _learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict = _
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
        checkpoint_iter = iteration
        iteration += 1  # next iteration is iteration + 1
        print('Model Loaded')
    
    # define datasets/dataloaders
    train_loader, valset, collate_fn, train_sampler, trainset = prepare_dataloaders(hparams, saved_lookup)
    epoch_offset = max(0, int(iteration / len(train_loader)))
    speaker_lookup = trainset.speaker_ids
    
    # load and/or generate global_mean
    if hparams.drop_frame_rate > 0.:
        if rank != 0: # if global_mean not yet calcuated, wait for main thread to do it
            while not os.path.exists(hparams.global_mean_npy): time.sleep(1)
        global_mean = calculate_global_mean(train_loader, hparams.global_mean_npy, hparams)
        hparams.global_mean = global_mean
        model.global_mean = global_mean
    
    # define scheduler
    use_scheduler = 0
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1**(1/5), patience=10)
    
    model.train()
    is_overflow = False
    validate_then_terminate = 0
    if validate_then_terminate:
        val_loss = validate(model, criterion, valset, iteration,
            hparams.batch_size, n_gpus, collate_fn, logger,
            hparams.distributed_run, rank)
        raise Exception("Finished Validation")
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    rolling_loss = StreamingMovingAverage(min(int(len(train_loader)), 200))
    # ================ MAIN TRAINNIG LOOP! ===================
    training = True
    while training:
        try:
            for epoch in tqdm(range(epoch_offset, hparams.epochs), initial=epoch_offset, total=hparams.epochs, desc="Epoch:", position=1, unit="epoch"):
                tqdm.write("Epoch:{}".format(epoch))
                
                if hparams.distributed_run: # shuffles the train_loader when doing multi-gpu training
                    train_sampler.set_epoch(epoch)
                start_time = time.time()
                # start iterating through the epoch
                for i, batch in tqdm(enumerate(train_loader), desc="Iter:  ", smoothing=0, total=len(train_loader), position=0, unit="iter"):
                    # run external code every iter, allows the run to be adjusted without restarts
                    if (i==0 or iteration % param_interval == 0):
                        try:
                            with open("run_every_epoch.py") as f:
                                internal_text = str(f.read())
                                if len(internal_text) > 0:
                                    #code = compile(internal_text, "run_every_epoch.py", 'exec')
                                    ldict = {'iteration': iteration, 'checkpoint_iter': checkpoint_iter, 'n_restarts': n_restarts}
                                    exec(internal_text, globals(), ldict)
                                else:
                                    print("[info] tried to execute 'run_every_epoch.py' but it is empty")
                        except Exception as ex:
                            print(f"[warning] 'run_every_epoch.py' FAILED to execute!\nException:\n{ex}")
                        globals().update(ldict)
                        locals().update(ldict)
                        if show_live_params:
                            print(internal_text)
                    n_restarts = n_restarts_override if (n_restarts_override is not None) else n_restarts or 0
                    # Learning Rate Schedule
                    if custom_lr:
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
                        if decrease_lr_on_restart:
                            learning_rate = learning_rate/(2**(n_restarts/3))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                    # /run external code every epoch, allows the run to be adjusting without restarts/
                    model.zero_grad()
                    x, y = model.parse_batch(batch) # move batch to GPU (async)
                    y_pred = model(x)
                    
                    loss_scalars = {
                        "MelGlow_ls": MelGlow_ls,
                        "DurGlow_ls": DurGlow_ls,
                        "VarGlow_ls": VarGlow_ls,
                        "Sylps_ls"  : Sylps_ls  ,
                    }
                    loss_dict = criterion(y_pred, y, loss_scalars)
                    loss = loss_dict['loss']
                    
                    if hparams.distributed_run:
                        reduced_loss_dict = {k: reduce_tensor(v.data, n_gpus).item() if v is not None else 0. for k, v in loss_dict.items()}
                    else:
                        reduced_loss_dict = {k: v.item() if v is not None else 0. for k, v in loss_dict.items()}
                    reduced_loss = reduced_loss_dict['loss']
                    
                    if hparams.fp16_run:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    
                    if grad_clip_thresh:
                        if hparams.fp16_run:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), grad_clip_thresh)
                            is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_thresh)
                    else:
                        grad_norm = 0.0
                    
                    optimizer.step()
                    
                    # get current Loss Scale of first optimizer
                    loss_scale = amp._amp_state.loss_scalers[0]._loss_scale if hparams.fp16_run else 32768.
                    
                    # restart if training/model has collapsed
                    if (iteration > 1e3 and (reduced_loss > LossExplosionThreshold)) or (math.isnan(reduced_loss)) or (loss_scale < 1/4):
                        raise LossExplosion(f"\nLOSS EXPLOSION EXCEPTION ON RANK {rank}: Loss reached {reduced_loss} during iteration {iteration}.\n\n\n")
                    
                    if expavg_loss_dict is None:
                        expavg_loss_dict = reduced_loss_dict
                    else:
                        expavg_loss_dict = {k: (reduced_loss_dict[k]*(1-loss_dict_smoothness))+(expavg_loss_dict[k]*loss_dict_smoothness) for k in expavg_loss_dict.keys()}
                        expavg_loss_dict_iters += 1
                    
                    if expavg_loss_dict_iters > 100:
                        if best_loss_dict is None:
                            best_loss_dict = expavg_loss_dict
                        else:
                            best_loss_dict = {k: min(best_loss_dict[k], expavg_loss_dict[k]) for k in best_loss_dict.keys()}
                    
                    if rank == 0:
                        duration = time.time() - start_time
                        if not is_overflow:
                            average_loss = rolling_loss.process(reduced_loss)
                            tqdm.write("{} [Train_loss:{:.4f} Avg:{:.4f}] [Grad Norm {:.4f}] "
                                "[{:.2f}s/it] [{:.3f}s/file] [{:.7f} LR] [{} LS]".format(
                                iteration, reduced_loss, average_loss, grad_norm,
                                    duration, (duration/(hparams.batch_size*n_gpus)), learning_rate, round(loss_scale)))
                            logger.log_training(reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration, iteration)
                        else:
                            tqdm.write("Gradient Overflow, Skipping Step")
                        start_time = time.time()
                    
                    if not is_overflow and ((iteration % (hparams.iters_per_checkpoint/1) == 0) or (os.path.exists(save_file_check_path))):
                        # save model checkpoint like normal
                        if rank == 0:
                            checkpoint_path = os.path.join(
                                output_directory, "checkpoint_{}".format(iteration))
                            save_checkpoint(model, optimizer, learning_rate, iteration, hparams, best_validation_loss, average_loss, best_val_loss_dict, best_loss_dict, speaker_lookup, checkpoint_path)
                    
                    if not is_overflow and ((iteration % int(validation_interval) == 0) or (os.path.exists(save_file_check_path)) or (iteration < 1000 and (iteration % 250 == 0))):
                        if rank == 0 and os.path.exists(save_file_check_path):
                            os.remove(save_file_check_path)
                        # perform validation and save "best_model" depending on validation loss
                        val_loss, best_val_loss_dict = validate(model, criterion, valset, loss_scalars, best_val_loss_dict,
                                 iteration, hparams.val_batch_size, n_gpus, collate_fn, logger,
                                 hparams.distributed_run, rank) #validate (0.8 forcing)
                        if use_scheduler:
                            scheduler.step(val_loss)
                        if (val_loss < best_validation_loss):
                            best_validation_loss = val_loss
                            if rank == 0:
                                checkpoint_path = os.path.join(output_directory, "best_model")
                                save_checkpoint(model, optimizer, learning_rate, iteration, hparams, best_validation_loss, average_loss, best_val_loss_dict, best_loss_dict, speaker_lookup, checkpoint_path)
                    
                    iteration += 1
                    # end of iteration loop
                # end of epoch loop
            training = False # exit the While loop
        
        #except Exception as ex: # print Exception and continue from checkpoint. (turns out it takes < 4 seconds to restart like this, fucking awesome)
        except LossExplosion as ex: # print Exception and continue from checkpoint. (turns out it takes < 4 seconds to restart like this, fucking awesome)
            print(ex) # print Loss
            checkpoint_path = os.path.join(output_directory, "best_model")
            assert os.path.exists(checkpoint_path), "best_model checkpoint must exist for automatic restarts"
            
            if hparams.fp16_run:
                amp._amp_state.loss_scalers[0]._loss_scale = 32768
            
            # clearing VRAM for load checkpoint
            model.zero_grad()
            x=y=y_pred=loss=len_loss=loss_z=loss_w=loss_s=loss_att=dur_loss_z=dur_loss_w=dur_loss_s=None
            torch.cuda.empty_cache()
            
            model.eval()
            model, optimizer, _learning_rate, iteration, best_validation_loss, saved_lookup = load_checkpoint(checkpoint_path, model, optimizer)
            learning_rate = optimizer.param_groups[0]['lr']
            epoch_offset = max(0, int(iteration / len(train_loader)))
            model.train()
            checkpoint_iter = iteration
            iteration += 1
            n_restarts += 1


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
    parser.add_argument('--gen_mels', action='store_true',
                        help='Generate mel spectrograms. This will help reduce the memory required.')
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

    if args.gen_mels:
        print("Generating Mels...")
        create_mels(hparams)
        print("Finished Generating Mels")
    
    if args.detect_anomaly: # checks backprop for NaN/Infs and outputs very useful stack-trace. Runs slower while enabled.
        torch.autograd.set_detect_anomaly(True)
        print("Autograd Anomaly Detection Enabled!\n(Code will run slower but backward pass will output useful info if crashing or NaN/inf values)")
    
    # these are needed for fp16 training, not inference
    if hparams.fp16_run:
        from apex import amp
    else:
        global amp
        amp = None
    try:
        from apex import optimizers as apexopt
    except:
        pass
    
    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.warm_start_force, args.n_gpus, args.rank, args.group_name, hparams)
