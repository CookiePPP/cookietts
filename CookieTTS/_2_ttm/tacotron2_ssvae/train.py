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

from model import Tacotron2, GANDiscriminator, load_model
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from CookieTTS.utils.model.GPU import to_gpu
import time
from math import e

from tqdm import tqdm
import CookieTTS.utils.audio.stft as STFT
from CookieTTS.utils.dataset.utils import load_wav_to_torch, load_filepaths_and_text
from scipy.io.wavfile import read

import os.path

from metric import alignment_metric

save_file_check_path = "save"
num_workers_ = 1 # DO NOT CHANGE WHEN USING TRUNCATION
start_from_checkpoints_from_zero = 0


def create_mels(hparams):
    stft = STFT.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)

    def save_mel(file):
        audio, sampling_rate = load_wav_to_torch(file)
        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(file,
                sampling_rate, stft.sampling_rate))
        melspec = stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0).cpu().numpy()
        np.save(file.replace('.wav', '.npy'), melspec)

    # Get the filepath for training and validation files
    wavs = [x[0] for x in load_filepaths_and_text(hparams.training_files) + load_filepaths_and_text(hparams.validation_files)]

    print(str(len(wavs))+" files being converted to mels")
    for audiopath in tqdm(wavs):
        try:
            save_mel(audiopath)
        except Exception as ex:
            tqdm.write(audiopath, " failed to process\n",ex,"\n")

    assert 0


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


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_force_model(checkpoint_path, model, model_d):
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
    
    
    if 'model_d' in checkpoint_dict.keys():
        pretrained_dict = checkpoint_dict['model_d']
        model_dict = model_d.state_dict()
        # Fiter out unneccessary keys
        filtered_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
        model_dict_missing = {k: v for k,v in pretrained_dict.items() if k not in model_dict}
        model_dict_mismatching = {k: v for k,v in pretrained_dict.items() if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape}
        pretrained_missing = {k: v for k,v in model_dict.items() if k not in pretrained_dict}
        if model_dict_missing: print(list(model_dict_missing.keys()),'does not exist in the current model and is being ignored')
        if model_dict_mismatching: print(list(model_dict_mismatching.keys()),"is the wrong shape and has been reset")
        if pretrained_missing: print(list(pretrained_missing.keys()),"doesn't have pretrained weights and has been reset")
        model_dict.update(filtered_dict)
        model_d.load_state_dict(model_dict)
    
    iteration = 0
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, model_d, iteration, saved_lookup


def warm_start_model(checkpoint_path, model, model_d, ignore_layers):
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
    return model, model_d, iteration, saved_lookup


def load_checkpoint(checkpoint_path, model, model_d, optimizer, optimizer_d):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    #state_dict = {k.replace("encoder_speaker_embedding.weight","encoder.encoder_speaker_embedding.weight"): v for k,v in torch.load(checkpoint_path)['state_dict'].items()}
    #model.load_state_dict(state_dict) # tmp for updating old models

    model.load_state_dict(checkpoint_dict['state_dict']) # original
    
    if 'model_d' in checkpoint_dict.keys(): model_d.load_state_dict(checkpoint_dict['model_d'])
    if 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if 'amp' in checkpoint_dict.keys() and amp is not None:
        amp.load_state_dict(checkpoint_dict['amp'])
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    if 'optimizer_d' in checkpoint_dict.keys(): optimizer_d.load_state_dict(checkpoint_dict['optimizer_d'])
    #if 'hparams' in checkpoint_dict.keys(): hparams = checkpoint_dict['hparams']
    if 'best_validation_loss' in checkpoint_dict.keys(): best_validation_loss = checkpoint_dict['best_validation_loss']
    if 'average_loss' in checkpoint_dict.keys(): average_loss = checkpoint_dict['average_loss']
    if (start_from_checkpoints_from_zero):
        iteration = 0
    else:
        iteration = checkpoint_dict['iteration']
    saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, model_d, optimizer, optimizer_d, learning_rate, iteration, best_validation_loss, saved_lookup


def save_checkpoint(model, model_d, optimizer, optimizer_d, learning_rate, iteration, hparams, best_validation_loss, average_loss, speaker_id_lookup, filepath):
    from CookieTTS.utils.dataset.utils import load_filepaths_and_text
    tqdm.write("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))

    # get speaker names to ID
    speakerlist = load_filepaths_and_text(hparams.speakerlist)
    speaker_name_lookup = {x[1]: speaker_id_lookup[x[2]] for x in speakerlist if x[2] in speaker_id_lookup.keys()}
    
    save_dict = {'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,
                #'amp': amp.state_dict(),
                'hparams': hparams,
                'speaker_id_lookup': speaker_id_lookup,
                'speaker_name_lookup': speaker_name_lookup,
                'best_validation_loss': best_validation_loss,
                'average_loss': average_loss }
    if model_d is not None and optimizer_d is not None:
        save_dict['model_d'] = model_d.state_dict()
        save_dict['optimizer_d'] = optimizer_d.state_dict()
    torch.save(save_dict, filepath)
    tqdm.write("Saving Complete")


def average_loss_terms(loss_terms_arr):
    # loss_terms_arr =
    # [
    #    [
    #       [1.12, mse_scalar],
    #       [0.75, mae_scalar],
    #    ],
    #    [
    #       [1.51, mse_scalar],
    #       [0.84, mae_scalar],
    #    ]
    # ]
    total = len(loss_terms_arr)
    loss_terms = loss_terms_arr[0]
    for terms in loss_terms_arr[1:]:
        for i in range(len(loss_terms)):
            loss_terms[i][0] = loss_terms[i][0] + terms[i][0]

    for i in range(len(loss_terms)):
        loss_terms[i][0] = loss_terms[i][0]/total
    return loss_terms


def validate(hparams, model, model_d, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, val_teacher_force_till, val_p_teacher_forcing, teacher_force=1, p_emotionnet_embed=0.0):
    """Handles all the validation scoring and printing"""
    model.eval()
    model_d.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=num_workers_,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, drop_last=True, collate_fn=collate_fn)
        if teacher_force == 1:
            val_teacher_force_till = 0
            val_p_teacher_forcing = 1.0
            p_emotionnet_embed = 1.0
        elif teacher_force == 2:
            val_teacher_force_till = 0
            val_p_teacher_forcing = 0.0
            p_emotionnet_embed = 0.0
        val_loss = 0.0
        diagonality = torch.zeros(1)
        avg_prob = torch.zeros(1)
        loss_terms_arr = []
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            x, y = model.parse_batch(batch)
            with torch.random.fork_rng(devices=[0,]):
                torch.random.manual_seed(0)# use same seed during validation so results are more consistent and comparable.
                y_pred = model(x, teacher_force_till=val_teacher_force_till, p_teacher_forcing=val_p_teacher_forcing, p_emotionnet_embed=p_emotionnet_embed)
            
            rate, prob = alignment_metric(x, y_pred)
            diagonality += rate
            avg_prob += prob
            criterion_dict = { "amp": None,
                            "n_gpus": n_gpus,
                             "model": model,
                           "model_d": model_d,
                           "hparams": hparams,
                         "optimizer": None,
                       "optimizer_d": None,
                  "grad_clip_thresh": 0.0,   }
            loss, gate_loss, loss_terms, reduced_val_loss, reduced_gate_loss, grad_norm, is_overflow = criterion(
                 y_pred, y, criterion_dict, 0)
            
            loss_terms_arr.append(loss_terms)
            val_loss += reduced_val_loss
            # end forloop
        val_loss = val_loss / (i + 1)
        diagonality = (diagonality / (i + 1)).item()
        avg_prob = (avg_prob / (i + 1)).item()
        # end torch.no_grad()
    model.train()
    model_d.train()
    if rank == 0:
        tqdm.write("Validation loss {}: {:9f}  Average Max Attention: {:9f}".format(iteration, val_loss, avg_prob))
        if iteration != 0:
            if teacher_force == 1:
                logger.log_teacher_forced_validation(val_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob)
            elif teacher_force == 2:
                logger.log_infer(val_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob)
            else:
                loss_terms = average_loss_terms(loss_terms_arr)
                logger.log_validation(val_loss, model, y, y_pred, iteration, loss_terms, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob)
    return val_loss


def calculate_global_mean(data_loader, global_mean_npy, hparams):
    if global_mean_npy and os.path.exists(global_mean_npy):
        global_mean = np.load(global_mean_npy)
        return to_gpu(torch.tensor(global_mean).half()) if hparams.fp16_run else to_gpu(torch.tensor(global_mean).float())
    sums = []
    frames = []
    print('calculating global mean...')
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), smoothing=0.001):
        text_padded, input_lengths, mel_padded, gate_padded,\
            output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states, *_ = batch
        # padded values are 0.
        sums.append(mel_padded.double().sum(dim=(0, 2)))
        frames.append(output_lengths.double().sum())
        if i > 100:
            break
    global_mean = sum(sums) / sum(frames)
    global_mean = to_gpu(global_mean.half()) if hparams.fp16_run else to_gpu(global_mean.float())
    if global_mean_npy:
        np.save(global_mean_npy, global_mean.cpu().numpy())
    return global_mean


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
    print('Initializing Tacotron2...')
    model = load_model(hparams)
    print('Done')
    model.eval()
    model_d = GANDiscriminator(hparams).cuda()
    model_d.eval()
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
    
    if hparams.use_postnet_generator_and_discriminator:
        optimizer_d = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model_d.parameters()), lr=learning_rate, weight_decay=hparams.weight_decay)
    else:
        optimizer_d = None
    optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=hparams.weight_decay)
    
    if True and rank == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("{:,} total parameters in model".format(pytorch_total_params))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("{:,} trainable parameters.".format(pytorch_total_params))
    
    if hparams.fp16_run:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        if hparams.use_postnet_generator_and_discriminator:
            model_d, optimizer_d = amp.initialize(model_d, optimizer_d, opt_level='O2')
    
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
        if hparams.use_postnet_generator_and_discriminator:
            model_d = apply_gradient_allreduce(model_d)
    
    criterion = Tacotron2Loss(hparams)
    
    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)
    
    # Load checkpoint if one exists
    best_validation_loss = 1e3 # used to see when "best_model" should be saved
    iteration = 0
    epoch_offset = 0
    _learning_rate = 1e-3
    saved_lookup = None
    if checkpoint_path is not None:
        if warm_start:
            model, model_d, iteration, saved_lookup = warm_start_model(
                checkpoint_path, model, model_d, hparams.ignore_layers)
        elif warm_start_force:
            model, model_d, iteration, saved_lookup = warm_start_force_model(
                checkpoint_path, model, model_d)
        else:
            model, model_d, optimizer, optimizer_d, _learning_rate, iteration, best_validation_loss, saved_lookup = load_checkpoint(
                checkpoint_path, model, model_d, optimizer, optimizer_d)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
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
    model_d.train()
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
    for epoch in tqdm(range(epoch_offset, hparams.epochs), initial=epoch_offset, total=hparams.epochs, desc="Epoch:", position=1, unit="epoch"):
        tqdm.write("Epoch:{}".format(epoch))

        if hparams.distributed_run: # shuffles the train_loader when doing multi-gpu training
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        # start iterating through the epoch
        for i, batch in tqdm(enumerate(train_loader), desc="Iter:  ", smoothing=0, total=len(train_loader), position=0, unit="iter"):
            # run external code every epoch or 1000 iters, allows the run to be adjusted without restarts
            if (iteration % 10 == 0 or i==0):
                try:
                    with open("run_every_epoch.py") as f:
                        internal_text = str(f.read())
                        if len(internal_text) > 0:
                            #code = compile(internal_text, "run_every_epoch.py", 'exec')
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
                learning_rate = max(min_learning_rate, learning_rate) # output the largest number
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                if optimizer_d is not None:
                    for param_group in optimizer_d.param_groups:
                        param_group['lr'] = learning_rate * discriminator_lr_scale
            # /run external code every epoch, allows the run to be adjusting without restarts/
            
            optimizer.zero_grad()
            x, y = model.parse_batch(batch) # move batch to GPU (async)
            y_pred = model(x, teacher_force_till=teacher_force_till, p_teacher_forcing=p_teacher_forcing, drop_frame_rate=drop_frame_rate)
            
            criterion_dict = {
                "amp": amp,
                "n_gpus": n_gpus,
                "model": model,
                "model_d": model_d,
                "hparams": hparams,
                "optimizer": optimizer,
                "optimizer_d": optimizer_d,
                "grad_clip_thresh": grad_clip_thresh,
            }
            loss, gate_loss, loss_terms, reduced_loss, reduced_gate_loss, grad_norm, is_overflow = criterion(
                 y_pred, y, criterion_dict, iteration, em_kl_weight=em_kl_weight, DiagonalGuidedAttention_scalar=DiagonalGuidedAttention_scalar)
            
            
            if not is_overflow and rank == 0:
                duration = time.time() - start_time
                average_loss = rolling_loss.process(reduced_loss)
                tqdm.write("{} [Train_loss {:.4f} Avg {:.4f}] [Gate_loss {:.4f}] [Grad Norm {:.4f}] "
                      "[{:.2f}s/it] [{:.3f}s/file] [{:.7f} LR]".format(
                    iteration, reduced_loss, average_loss, reduced_gate_loss, grad_norm, duration, (duration/(hparams.batch_size*n_gpus)), learning_rate))
                if iteration % 20 == 0:
                    diagonality, avg_prob = alignment_metric(x, y_pred)
                    logger.log_training(
                        reduced_loss, grad_norm, learning_rate, duration, iteration, loss_terms, teacher_force_till, p_teacher_forcing, diagonality=diagonality, avg_prob=avg_prob)
                else:
                    logger.log_training(
                        reduced_loss, grad_norm, learning_rate, duration, iteration, loss_terms, teacher_force_till, p_teacher_forcing)
                start_time = time.time()
            if is_overflow and rank == 0:
                tqdm.write("Gradient Overflow, Skipping Step")
            
            if not is_overflow and ((iteration % (hparams.iters_per_checkpoint/1) == 0) or (os.path.exists(save_file_check_path))):
                # save model checkpoint like normal
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, model_d, optimizer, optimizer_d, learning_rate, iteration, hparams, best_validation_loss, average_loss, speaker_lookup, checkpoint_path)
            
            if not is_overflow and ((iteration % int((hparams.iters_per_validation)/1) == 0) or (os.path.exists(save_file_check_path)) or (iteration < 1000 and (iteration % 250 == 0))):
                if rank == 0 and os.path.exists(save_file_check_path):
                    os.remove(save_file_check_path)
                # perform validation and save "best_model" depending on validation loss
                if iteration > 0:
                    val_loss = validate(hparams, model, model_d, criterion, valset, iteration,
                             hparams.val_batch_size, n_gpus, collate_fn, logger,
                             hparams.distributed_run, rank, val_teacher_force_till, val_p_teacher_forcing, teacher_force=1) #teacher_force
                    val_loss = validate(hparams, model, model_d, criterion, valset, iteration,
                             hparams.val_batch_size, n_gpus, collate_fn, logger,
                             hparams.distributed_run, rank, val_teacher_force_till, val_p_teacher_forcing, teacher_force=2) #infer
                val_loss = validate(hparams, model, model_d, criterion, valset, iteration,
                         hparams.val_batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank, val_teacher_force_till, val_p_teacher_forcing, teacher_force=0) #validate (0.8 forcing)
                if use_scheduler:
                    scheduler.step(val_loss)
                if (val_loss < best_validation_loss):
                    best_validation_loss = val_loss
                    if rank == 0:
                        checkpoint_path = os.path.join(output_directory, "best_model")
                        save_checkpoint(model, model_d, optimizer, optimizer_d, learning_rate, iteration, hparams, best_validation_loss, average_loss, speaker_lookup, checkpoint_path)

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
    
    if args.detect_anomaly: # checks backprop for NaN/Infs and outputs very useful stack-trace. Runs slowly while enabled.
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
