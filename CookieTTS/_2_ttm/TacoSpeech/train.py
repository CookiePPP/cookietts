import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
import time
import argparse
import math
import random
import pickle
import numpy as np
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import TacoSpeech, load_model
from HiFiGAN_wrapper import HiFiGAN
from CookieTTS.utils.dataset.data_utils import TTSDataset, Collate, generate_filelist_from_datasets
from CookieTTS.utils import get_args, force

from loss_function import TacoSpeechLoss
from logger import TacoSpeechLogger
from hparams import create_hparams
from CookieTTS.utils.model.GPU import to_gpu
import time
from math import e
from math import ceil

from tqdm import tqdm
import CookieTTS.utils.audio.stft as STFT
from CookieTTS.utils.dataset.utils import load_wav_to_torch, load_filepaths_and_text
from scipy.io.wavfile import read

import os.path

save_file_check_path = "save"
start_from_checkpoints_from_zero = 0

class LossExplosion(Exception):
    """Custom Exception Class. If Loss Explosion, raise Error and automatically restart training script from previous best_val_model checkpoint."""
    pass

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


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


def get_filelist(hparams, val=False):
    if hparams.data_source == 1:# if filelist is a folder, check all datasets inside the folder for audio files and transcripts.
        # convert dataset folder into a filelist
        filelist = generate_filelist_from_datasets(hparams.dataset_folder,
                              METAPATH_APPEND=hparams.dataset_metapath_append,
                              AUDIO_FILTER   =hparams.dataset_audio_filters,
                              AUDIO_REJECTS  =hparams.dataset_audio_rejects,
                              MIN_DURATION   =hparams.dataset_min_duration,
                              MAX_DURATION   =hparams.dataset_max_duration,
                              MIN_CHAR_LEN   =hparams.dataset_min_chars,
                              MAX_CHAR_LEN   =hparams.dataset_max_chars,)
    elif hparams.data_source == 0:# else filelist is a ".txt" file, load the easy way
        if val:
            filelist = load_filepaths_and_text(hparams.validation_files)
        else:
            filelist = load_filepaths_and_text(hparams.training_files)
    return filelist


def prepare_dataloaders(hparams, dataloader_args, args, speaker_ids, audio_offset=0):
    # Get data, data loaders and collate function ready
    if hparams.data_source == 1:
        if args.rank == 0:
            fl_dict = get_filelist(hparams)
        
        if args.n_gpus > 1:
            if args.rank == 0:
                with open('fl_dict.pkl', 'wb') as pickle_file:
                    pickle.dump(fl_dict, pickle_file, pickle.HIGHEST_PROTOCOL)
            torch.distributed.barrier()# wait till all graphics cards reach this point.
            if args.rank > 0:
                fl_dict = pickle.load(open('fl_dict.pkl', "rb"))
        
        speakerlist = fl_dict['speakerlist']
        
        filelist    = fl_dict['filelist']
        speaker_ids = fl_dict['speaker_ids']
        random.Random(0).shuffle(filelist)
        training_filelist   = filelist[ int(len(filelist)*hparams.dataset_p_val):]
        validation_filelist = filelist[:int(len(filelist)*hparams.dataset_p_val) ]

        
        if args.n_gpus > 1:
            torch.distributed.barrier()# wait till all graphics cards reach this point.
            if args.rank == 0 and os.path.exists('fl_dict.pkl'):
                os.remove('fl_dict.pkl')
    else:
        # get speaker names to ID
        if os.path.exists(hparams.speakerlist):
            # expects speakerlist to look like below
            # |Nancy|0
            # |Bob|1
            # |Phil|2
            #
            # every line must have at least 2 pipe symbols
            speakerlist = load_filepaths_and_text(hparams.speakerlist)
            speaker_name_lookup = {x[1]: speaker_id_lookup[x[2]] for x in speakerlist if x[2] in speaker_id_lookup.keys()}
        else:
            print("'speakerlist' in hparams.py not found! Speaker names will be IDs instead.")
            speakerlist = [['Dataset',i,i,'Source','Source Type'] for i in range(speaker_id_lookup.keys())]
            speaker_name_lookup = speaker_id_lookup# if there is no speaker
        
        training_filelist   = get_filelist(hparams, val=False)
        validation_filelist = get_filelist(hparams, val=True)
        speaker_ids = speaker_ids if hparams.use_saved_speakers else None
    
    trainset = TTSDataset(training_filelist, hparams, dataloader_args, check_files=hparams.check_files, shuffle=False,
                           deterministic_arpabet=False, speaker_ids=speaker_ids,          audio_offset=audio_offset)
    valset = TTSDataset(validation_filelist, hparams, dataloader_args, check_files=hparams.check_files, shuffle=False,
                           deterministic_arpabet=True,  speaker_ids=getattr(trainset, 'speaker_ids', None), audio_offset=audio_offset)
    collate_fn = Collate(hparams)
    
    use_shuffle = False# using custom Shuffle function inside dataloader.dataset which works with TBPTT
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset, shuffle=use_shuffle)
        shuffle = False
    else:
        train_sampler = None
        shuffle = use_shuffle
    
    train_loader = DataLoader(trainset, shuffle=shuffle, sampler=train_sampler,
                              num_workers=hparams.num_workers,# prefetch_factor=hparams.prefetch_factor,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler, trainset, speakerlist


def prepare_directories_and_logger(hparams, args):
    if args.rank == 0:
        if not os.path.isdir(args.output_directory):
            os.makedirs(args.output_directory)
            os.chmod(args.output_directory, 0o775)
        logger = TacoSpeechLogger(os.path.join(args.output_directory, args.log_directory), hparams)
    else:
        logger = None
    return logger


def warm_start_force_model(checkpoint_paths, model):
    if type(checkpoint_paths) is str:
        checkpoint_paths = [checkpoint_paths,]
    
    for checkpoint_path in checkpoint_paths:
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
        iteration = checkpoint_dict.get('iteration', 0)
        
        saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def warm_start_model(checkpoint_paths, model, ignore_layers):
    if type(checkpoint_paths) is str:
        checkpoint_paths = [checkpoint_paths,]
    
    for checkpoint_path in checkpoint_paths:
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
        
        iteration = 0
        #iteration = checkpoint_dict.get('iteration', 0)
        
        saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def load_checkpoint(checkpoint_paths, model, optimizer, best_val_loss_dict, best_loss_dict, best_validation_loss=1e3,):
    if type(checkpoint_paths) is str:
        checkpoint_paths = [checkpoint_paths,]
    
    for checkpoint_path in checkpoint_paths:
        assert os.path.isfile(checkpoint_path)
        print(f"Loading checkpoint '{checkpoint_path}'")
        
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint_dict['state_dict'])# load model weights
        
        if 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])# load optimizer state
        #if 'amp' in checkpoint_dict.keys() and amp is not None:
        #    amp.load_state_dict(checkpoint_dict['amp']) # load AMP (fp16) state.
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
        
        print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict


def save_checkpoint(model, optimizer, hifiGAN, learning_rate, iteration, hparams, best_validation_loss, average_loss,
                    best_val_loss_dict, best_loss_dict, speaker_id_lookup, speakerlist, filepath):
    from CookieTTS.utils.dataset.utils import load_filepaths_and_text
    
    if hifiGAN is not None:
        hifiGAN.save_state_dict()
    
    tqdm.write(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    
    assert all(str(line[2]).isdigit() for line in speakerlist), 'speakerlist got str in speaker_id section!'
    speaker_name_lookup = {x[1].strip(): x[2] for x in speakerlist}
    
    save_dict = {'iteration'           : iteration,
                 'state_dict'          : model.state_dict(),
                 'optimizer'           : optimizer.state_dict(),
                 'learning_rate'       : learning_rate,
                 'hparams'             : hparams,
                 'speaker_id_lookup'   : speaker_id_lookup,
                 'speaker_name_lookup' : speaker_name_lookup,
                 'speakerlist'         : speakerlist,
                 'best_validation_loss': best_validation_loss,
                 'best_val_loss_dict'  : best_val_loss_dict,
                 'best_loss_dict'      : best_loss_dict,
                 'average_loss'        : average_loss}
    if hparams.fp16_run:
        save_dict['amp'] = amp.state_dict()
    torch.save(save_dict, filepath)
    tqdm.write("Saving Complete")


def write_dict_to_file(file_losses, fpath, sfpath, speakerlist, n_gpus, rank, deliminator='","', ends='"'):
    if n_gpus > 1:
        # synchronize data between graphics cards
        import pickle
        
        # dump file_losses for each graphics card into files
        with open(fpath+f'_rank{rank}', 'wb') as pickle_file:
            pickle.dump(file_losses, pickle_file, pickle.HIGHEST_PROTOCOL)
        
        torch.distributed.barrier()# wait till all graphics cards reach this point.
        
        # merge file losses from other graphics cards into this process
        list_of_dicts = []
        for f_rank in [x for x in list(range(n_gpus)) if x != rank]:
            list_of_dicts.append(pickle.load(open(f'{fpath}_rank{f_rank}', "rb")))
        
        for new_file_losses in list_of_dicts:
            for path, loss_dict in new_file_losses.items():
                if path in file_losses:
                    if loss_dict['time'] > file_losses[path]['time']:
                        file_losses[path] = loss_dict
                else:
                    file_losses[path] = loss_dict
        
        torch.distributed.barrier()# wait till all graphics cards reach this point.
        
        os.remove(fpath+f'_rank{rank}')
    
    if len(file_losses.keys()) == 0:
        return file_losses
    
    if rank == 0:# write file_losses data to .CSV file.
        print(f"Writing CSV to {fpath}")
        with open(fpath, 'w') as f:
            f.write(ends+deliminator.join(['path',]+[str(key) for key in next(iter(file_losses.values())).keys()])+ends)
            for path, loss_dict in file_losses.items():
                line = []
                line.append(path)
                for loss_name, loss_value in loss_dict.items():
                    line.append(str(loss_value))
                f.write('\n'+ends+deliminator.join(line)+ends)
    
    # collect losses of each file for each speaker into lists
    speaker_losses = {}
    for loss_dict in file_losses.values():
        speaker_id = int(loss_dict['speaker_id_ext'])
        if speaker_id not in speaker_losses:
            speaker_losses[speaker_id] = {k:[v,] for k,v in list(loss_dict.items())[2:] if v is not None}
        else:
            for loss_name, loss_value in list(loss_dict.items())[2:]:
                if loss_name not in speaker_losses[speaker_id]:
                    speaker_losses[speaker_id][loss_name] = [loss_value,]
                elif loss_value is not None:
                    speaker_losses[speaker_id][loss_name].append(loss_value)
    assert len(speaker_losses.keys())
    
    # then average the loss list for each speaker
    speaker_avg_losses = {k: {} for k in sorted(speaker_losses.keys())}
    for speaker in speaker_avg_losses.keys():
        for loss_name in speaker_losses[speaker].keys():
            speaker_avg_losses[speaker][loss_name] = sum([x for x in speaker_losses[speaker][loss_name] if x is not None])/len(speaker_losses[speaker][loss_name])
    assert len(speaker_avg_losses.keys())
    
    speaker_id_exts = [str(x[2]) for x in speakerlist]
    speaker_names   = [x[1] for x in speakerlist]
    speaker_dataset = [x[0] for x in speakerlist]
    speaker_name_lookup    = {speaker_id_exts[i]:   speaker_names[i] for i in range(len(speaker_id_exts))}
    speaker_dataset_lookup = {speaker_id_exts[i]: speaker_dataset[i] for i in range(len(speaker_id_exts))}
    
    if rank == 0:# write speaker_avg_losses data to .CSV file.
        print(f"Writing speaker CSV to {sfpath}")
        with open(sfpath, 'w') as f:
            f.write(ends+deliminator.join(['speaker_id','speaker_name','dataset']+[str(key) for key in next(iter(speaker_avg_losses.values())).keys()])+ends)
            for speaker, loss_dict in speaker_avg_losses.items():
                line = []
                line.append(str(speaker))
                line.append(   speaker_name_lookup[str(speaker)])
                line.append(speaker_dataset_lookup[str(speaker)])
                for loss_name, loss_value in loss_dict.items():
                    line.append(str(loss_value))
                f.write('\n'+ends+deliminator.join(line)+ends)
    
    return file_losses


def get_mse_sampled_filelist(original_filelist, file_losses, exp_factor, seed=None, max_weighting=8.0):
    # collect losses of each file for each speaker into lists
    speaker_losses = {}
    for loss_dict in file_losses.values():
        speaker_id = int(loss_dict['speaker_id_ext'])
        if speaker_id not in speaker_losses:
            speaker_losses[speaker_id] = {k:[v,] for k,v in list(loss_dict.items())[2:] if v is not None}
        else:
            for loss_name, loss_value in list(loss_dict.items())[2:]:
                if loss_name not in speaker_losses[speaker_id]:
                    speaker_losses[speaker_id][loss_name] = [loss_value,]
                elif loss_value is not None:
                    speaker_losses[speaker_id][loss_name].append(loss_value)
    assert len(speaker_losses.keys())
    
    # then average the loss list for each speaker
    speaker_avg_losses = {k: {} for k in sorted(speaker_losses.keys())}
    for speaker in speaker_avg_losses.keys():
        for loss_name in speaker_losses[speaker].keys():
            speaker_avg_losses[speaker][loss_name] = sum([x for x in speaker_losses[speaker][loss_name] if x is not None])/len(speaker_losses[speaker][loss_name])
    assert len(speaker_avg_losses.keys())
    
    # generate speaker filelists
    spkr_filelist = {int(spkr_id): [] for spkr_id in set([x[2] for x in original_filelist])}
    for path, quote, speaker_id, *_ in original_filelist:
        spkr_filelist[int(speaker_id)].append([path, quote, speaker_id, *_])
    assert len(spkr_filelist.keys()) and any(len(x) for x in spkr_filelist.values())
    
    # shuffle speaker filelists
    for k in spkr_filelist.keys():
        random.Random(seed).shuffle(spkr_filelist[k])
    
    # calculate dataset portion for each speaker and build new filelist
    dataset_len = len(original_filelist)
    new_filelist = []
    spec_MSE_total = sum([loss_dict['spec_MSE']**exp_factor for loss_dict in speaker_avg_losses.values()])
    for speaker_id, loss_dict in speaker_avg_losses.items():
        if int(speaker_id) in spkr_filelist.keys():
            sample_chance = (loss_dict['spec_MSE']**exp_factor)/spec_MSE_total# chance to sample this speaker
            assert sample_chance > 0.0
            n_files = round(sample_chance * dataset_len)
            spkr_files = spkr_filelist[int(speaker_id)]
            if (n_files == 0) or ( len(spkr_files) == 0 ):
                continue
            if len(spkr_files) < n_files:
                spkr_files = spkr_files * min(ceil(n_files/len(spkr_files)), max_weighting)# repeat filelist if needed
            new_filelist.extend(spkr_files[:n_files])
    assert len(new_filelist)
    
    return new_filelist


def update_smoothed_dict(orig_dict, new_dict, smoothing_factor=0.6):
    for key, value in new_dict.items():
        if key in orig_dict:# if audio file already in dict, merge new with old using smoothing_factor
            loss_names, loss_values = orig_dict[key].keys(), orig_dict[key].values()
            for loss_name in loss_names:
                if all(loss_name in dict_ for dict_ in [orig_dict[key], new_dict[key]]) and all(type(loss) in [int, float] for loss in [orig_dict[key][loss_name], new_dict[key][loss_name]]):
                    orig_dict[key][loss_name] = orig_dict[key][loss_name]*(smoothing_factor) + new_dict[key][loss_name]*(1-smoothing_factor)
                elif loss_name in new_dict[key] and type(new_dict[key][loss_name]) in [int, float]:
                    orig_dict[key][loss_name] = new_dict[key][loss_name]
        
        else:# if audio file not in dict, assign new key to dict
            orig_dict[key] = new_dict[key]
    return orig_dict


def validate(hparams, args, file_losses, model, criterion, hifiGAN, valset, loss_scalars, best_val_loss_dict, iteration, collate_fn, logger):
    """Handles all the validation scoring and printing"""
    model.eval()
    if hparams.HiFiGAN_enable:
        hifiGAN.eval()
    with torch.no_grad():
        if False:# if inference, sample from each speaker equally. So speakers with smaller datasets get the same weighting onto the val loss.
            orig_filelist = valset.filelist
            new_filelist = get_mse_sampled_filelist(orig_filelist, file_losses, 0.0, seed=1234)
            if len(new_filelist) >= hparams.n_gpus*hparams.batch_size:
                valset.update_filelist(new_filelist)
        if not len(valset.filelist) >= hparams.n_gpus*hparams.batch_size:
            print(f'too few files in validation set! Found {len(valset.filelist)}, expected {hparams.n_gpus*hparams.batch_size} or more.\nIf you have a small amount of data, increase `dataset_p_val` or decrease `val_batch_size`')
        val_sampler = DistributedSampler(valset) if hparams.distributed_run else None
        val_loader  = DataLoader(valset, sampler=val_sampler,
                            num_workers=hparams.val_num_workers,# prefetch_factor=hparams.prefetch_factor,
                            shuffle=False, batch_size=hparams.batch_size,
                            pin_memory=False, drop_last=True, collate_fn=collate_fn)
        
        tmp_file_losses = {}
        loss_dict_total = None
        for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            y = model.parse_batch(batch)
            with torch.random.fork_rng(devices=[0,]):
                torch.random.manual_seed(i)# use repeatable seeds during validation so results are more consistent and comparable.
                y_pred = force(model, valid_kwargs=model_args, **y)
            
            loss_dict, file_losses_batch = criterion(iteration, model, y_pred, y, loss_scalars, hifiGAN)
            tmp_file_losses = update_smoothed_dict(tmp_file_losses, file_losses_batch, file_losses_smoothness)
            if loss_dict_total is None:
                loss_dict_total = {k: 0. for k, v in loss_dict.items()}
            
            if hparams.distributed_run:
                reduced_loss_dict = {k: reduce_tensor(v.data, args.n_gpus).item() if v is not None else 0. for k, v in loss_dict.items()}
            else:
                reduced_loss_dict = {k: v.item() if v is not None else 0. for k, v in loss_dict.items()}
            reduced_loss = reduced_loss_dict['loss']
            
            for k in loss_dict_total.keys():
                loss_dict_total[k] = loss_dict_total[k] + reduced_loss_dict[k]
            # end forloop
        loss_dict_total = {k: v/(i+1) for k, v in loss_dict_total.items()}
    
        # invert model so glow/flow modules can produce example outputs
        inv_y_pred = None
        
        # end torch.no_grad()
    
    # reverse changes to valset and model
    if False:# if inference, sample from each speaker equally. So speakers with smaller datasets get the same weighting onto the val loss.
        valset.update_filelist(orig_filelist)
    
    model.train()
    if hparams.HiFiGAN_enable:
        hifiGAN.train()
    
    # update best losses
    best_val_loss_dict = loss_dict_total if best_val_loss_dict is None else {k: min(best_val_loss_dict[k], loss_dict_total[k]) for k in best_val_loss_dict.keys() if k in loss_dict_total}
    
    if args.rank == 0:# print, log data and return.
        tqdm.write(f"Validation loss {iteration}: {loss_dict_total['loss']:9f}")
        if iteration > 1:
            log_terms = (loss_dict_total, best_val_loss_dict, model, y, y_pred, inv_y_pred, iteration,)
            logger.log_validation(*log_terms)
    
    file_losses = update_smoothed_dict(file_losses, tmp_file_losses, file_losses_smoothness)
    return loss_dict_total['loss'], best_val_loss_dict, file_losses


def train(args, rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout
    
    Params
    ------
    args.output_directory (string): directory to save checkpoints
    args.log_directory (string) directory to save tensorboard logs
    args.checkpoint_path(string): checkpoint path
    args.n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    # setup distributed
    hparams.n_gpus = args.n_gpus
    hparams.rank = rank
    if hparams.distributed_run:
        init_distributed(hparams, args.n_gpus, rank, group_name)
    
    # reproducablilty stuffs
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    # initialize blank model
    print('Initializing TacoSpeech...')
    model = load_model(hparams)
    print('Done')
    global model_args
    model_args = get_args(model.forward)
    model.eval()
    learning_rate = hparams.learning_rate
    
    # (optional) show the names of each layer in model, mainly makes it easier to copy/paste what you want to adjust
    if hparams.print_layer_names_during_startup:
        print(*[f"Layer{i} = "+str(x[0])+" "+str(x[1].shape) for i,x in enumerate(list(model.named_parameters()))], sep="\n")
    
    if len(hparams.frozen_modules):# (optional) Freeze layers by disabling grads
        for layer, params in list(model.named_parameters()):
            if any(layer.startswith(module) for module in hparams.frozen_modules):
                params.requires_grad = False
                print(f"Layer: {layer} has been frozen")
    
    if len(hparams.unfrozen_modules):# (optional) Unfreeze layers by disabling grads
        for layer, params in list(model.named_parameters()):
            if any(layer.startswith(module) for module in hparams.unfrozen_modules):
                params.requires_grad = True
                print(f"Layer: {layer} has been unfrozen")
    
    if True:# define optimizer (any params without requires_grad are ignored)
        #optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=hparams.weight_decay)
        optimizer =  torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=hparams.weight_decay)
    
    hifiGAN = None
    if hparams.HiFiGAN_enable:
        hifiGAN = HiFiGAN(hparams)
        hifiGAN.load_state_dict_from_file(hparams.HiFiGAN_cp_folder)
        hifiGAN.cuda()
    
    if True and rank == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("{:,} total parameters in model".format(pytorch_total_params))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("{:,} trainable parameters.".format(pytorch_total_params))
    
    if hparams.fp16_run:# convert models to mixed precision
        print("Initializing AMP Model(s) / Optimzier(s)")
        models     = [model,]
        optimizers = [optimizer,]
        
        if hparams.HiFiGAN_enable:
            models.append(hifiGAN.generator)
            models.append(hifiGAN.mp_discriminator)
            models.append(hifiGAN.ms_discriminator)
            optimizers.append(hifiGAN.g_optimizer)
            optimizers.append(hifiGAN.d_optimizer)
        models, optimizers = amp.initialize(models, optimizers, opt_level=f'O{hparams.fp16_run_optlvl}')
        if hparams.HiFiGAN_enable:
            hifiGAN.ms_discriminator = models.pop()
            hifiGAN.mp_discriminator = models.pop()
            hifiGAN.generator        = models.pop()
            hifiGAN.d_optimizer      = optimizers.pop()
            hifiGAN.g_optimizer      = optimizers.pop()
            hifiGAN.amp = amp
        model     = models.pop()
        optimizer = optimizers.pop()
        assert     len(models) == 0
        assert len(optimizers) == 0
        del models, optimizers
    
    print("Initializing Gradient AllReduce model wrapper.")
    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)
    
    if hparams.HiFiGAN_enable and hparams.distributed_run:
        hifiGAN.generator        = apply_gradient_allreduce(hifiGAN.generator)
        hifiGAN.mp_discriminator = apply_gradient_allreduce(hifiGAN.mp_discriminator)
        hifiGAN.ms_discriminator = apply_gradient_allreduce(hifiGAN.ms_discriminator)
    
    print("Initializing TacoSpeech Loss func.")
    criterion = TacoSpeechLoss(hparams)
    
    print("Initializing TacoSpeech Logger.")
    logger = prepare_directories_and_logger(hparams, args)
    
    # Load checkpoint if one exists
    best_validation_loss = 1e3# used to see when "best_val_model" should be saved
    
    is_overflow = False
    average_loss = 9e9
    n_restarts = 0
    checkpoint_iter = 0
    iteration = 0
    epoch_offset = 0
    _learning_rate = 1e-3
    saved_lookup = None
    original_filelist = None
    n_restarts_override = None
    
    global file_losses
    file_losses = {}
    global file_losses_smoothness
    file_losses_smoothness = 0.6
    
    global best_val_loss_dict
    best_val_loss_dict = None
    global best_loss_dict
    best_loss_dict = None
    global expavg_loss_dict
    expavg_loss_dict = None
    expavg_loss_dict_iters = 0# initial iters expavg_loss_dict has been fitted
    loss_dict_smoothness = 0.95 # smoothing factor
    
    if args.checkpoint_path is not None:# load checkpoint
        if args.warm_start:
            model, iteration, saved_lookup = warm_start_model(
                args.checkpoint_path, model, hparams.ignore_layers)
        elif args.warm_start_force:
            model, iteration, saved_lookup = warm_start_force_model(
                args.checkpoint_path, model)
        else:
            _ = load_checkpoint(args.checkpoint_path, model, optimizer, best_val_loss_dict, best_loss_dict)
            model, optimizer, _learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict = _
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
        checkpoint_iter = iteration
        iteration += 1  # next iteration is iteration + 1
        print('Model Loaded')
    
    # define datasets/dataloaders
    dataloader_args = [*get_args(criterion.forward), *model_args]
    if hparams.HiFiGAN_enable:
        dataloader_args.extend(['gt_audio','sampling_rate',])
    if rank == 0:
        dataloader_args.extend(get_args(logger.log_training))
    train_loader, valset, collate_fn, train_sampler, trainset, speakerlist = prepare_dataloaders(hparams, dataloader_args, args, saved_lookup)
    epoch_offset = max(0, int(iteration / len(train_loader)))
    speaker_lookup = trainset.speaker_ids
    
    model.train()
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    just_did_val = True
    rolling_loss = StreamingMovingAverage(min(int(len(train_loader)), 200))
    
    if True:# reproducablilty stuffs
        torch.manual_seed(hparams.seed)
        torch.cuda.manual_seed(hparams.seed)
    
    # ================ MAIN TRAINNIG LOOP! ===================
    training = True
    while training:
        try:
            for epoch in tqdm(range(epoch_offset, hparams.epochs), initial=epoch_offset, total=hparams.epochs, desc="Epoch:", position=1, unit="epoch"):
                tqdm.write("Epoch:{}".format(epoch))
                
                train_loader.dataset.shuffle_dataset()# Shuffle Dataset
                dataset_len = len(train_loader)
                
                start_time = time.time()
                # start iterating through the epoch
                for i, batch in tqdm(enumerate(train_loader), desc="Iter:  ", smoothing=0, total=len(train_loader), position=0, unit="iter"):
                    # run external code every epoch or 1000 iters, allows the run to be adjusted without restarts
                    if (i<2 or iteration % param_interval == 0):
                        try:
                            with open("run_every_epoch.py", encoding='utf-8') as f:
                                internal_text = str(f.read())
                                if len(internal_text) > 0:
                                    #code = compile(internal_text, "run_every_epoch.py", 'exec')
                                    ldict = {'iteration': iteration, 'checkpoint_iter': checkpoint_iter, 'n_restarts': n_restarts, "expavg_loss_dict": expavg_loss_dict}
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
                    if custom_lr:# Learning Rate Schedule
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
                        if just_did_val:
                            learning_rate = 0.0
                            just_did_val=False
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                    
                    HiFiGAN_learning_rate = ldict['HiFiGAN_learning_rate']
                    if hparams.HiFiGAN_enable:# update HiFiGAN Learning Rate
                        if HiFiGAN_learning_rate == 0.:
                            HiFiGAN_learning_rate = learning_rate
                        for group in hifiGAN.g_optimizer.param_groups:
                            group['lr'] = HiFiGAN_learning_rate
                        for group in hifiGAN.d_optimizer.param_groups:
                            group['lr'] = HiFiGAN_learning_rate
                    # /run external code every epoch, allows the run to be adjusting without restarts/
                    
                    optimizer.zero_grad()# clear/zero the gradients from the last iter
                    y = model.parse_batch(batch)# move batch to GPU (async)
                    y['gt_mel'].requires_grad_()
                    y_pred = force(model, valid_kwargs=model_args, **y)
                    
                    loss_scalars = {
                      "decoder_MAE_weight": decoder_MAE_weight,
                      "decoder_MSE_weight": decoder_MSE_weight,
                      "f_fesvd_MAE_weight": f_fesvd_MAE_weight,
                      "f_fesvd_MSE_weight": f_fesvd_MSE_weight,
                        "fesvd_MAE_weight": fesvd_MAE_weight,
                        "fesvd_MSE_weight": fesvd_MSE_weight,
                  "varpred_hdn_NLL_weight": varpred_hdn_NLL_weight,
                         "mdn_loss_weight": mdn_loss_weight,
                         "dur_loss_weight": dur_loss_weight,
                        "sylps_MAE_weight": sylps_MAE_weight,
                        "sylps_MSE_weight": sylps_MSE_weight,
                         "diag_att_weight": diag_att_weight,
              "HiFiGAN_g_all_class_weight": HiFiGAN_g_all_class_weight,
         "HiFiGAN_g_all_featuremap_weight": HiFiGAN_g_all_featuremap_weight,
            "HiFiGAN_g_all_mel_mae_weight": HiFiGAN_g_all_mel_mae_weight,
              "HiFiGAN_d_all_class_weight": HiFiGAN_d_all_class_weight,
                    }
                    loss_dict, file_losses_batch = criterion(iteration, model, y_pred, y, loss_scalars,
                                                hifiGAN if hparams.HiFiGAN_enable else None,)
                    
                    file_losses = update_smoothed_dict(file_losses, file_losses_batch, file_losses_smoothness)
                    
                    if hparams.distributed_run:
                        reduced_loss_dict = {k: reduce_tensor(v.data, args.n_gpus).item() if v is not None else 0. for k, v in loss_dict.items()}
                    else:
                        reduced_loss_dict = {k: v.item() if v is not None else 0. for k, v in loss_dict.items()}
                    
                    reduced_loss = reduced_loss_dict['loss']
                    
                    loss = loss_dict['loss']
                    if hparams.fp16_run:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    
                    if rank==0 and show_gradients:# debug/extreme verbose
                        try:
                            _=avg_grads
                        except:
                            avg_grads = {}
                        for param_name, params in model.named_parameters():
                            if params.requires_grad and params.grad is not None:
                                norm_grad = 1.0
                                grad = params.grad.abs().sum().item()
                                if param_name not in avg_grads:
                                    avg_grads[param_name] = grad
                                elif grad*5. < avg_grads[param_name]:
                                    avg_grads[param_name] = (avg_grads[param_name]*0.9)+(grad*0.1)
                                norm_grad = grad/avg_grads[param_name]
                                if grad > 30.0 or norm_grad > 2.0:
                                    print(f'{norm_grad:03.1f} | {grad:020.6f} | {params.grad.abs().mean().item():06.6f}| {params.grad.abs().max().item():010.6f} | {param_name}')
                    
                    if hparams.HiFiGAN_enable and y_pred['hifigan_enabled']:# HiFiGAN isn't supposed to use gradient clipping so the optimizer.
                        hifiGAN.g_optimizer_step_and_clear()#      should be ran before gradient clipping occurs.
                    
                    if grad_clip_thresh:# apply gradient clipping to params
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
                    
                    # calcuate the effective learning rate after gradient clipping is applied, and use the effective learning rate on the GAN modules.
                    effective_lr = 0.0 if is_overflow else (learning_rate*min((grad_clip_thresh/grad_norm+1e-6), 1.0) if grad_clip_thresh else learning_rate)
                    
                    # (Optional) Discriminator Forward+Backward Pass
                    if hparams.HiFiGAN_enable and y_pred['hifigan_enabled']:
                        hifiGAN.train(model.training)
                        with torch.random.fork_rng(devices=[0,]):
                            hifiGAN(y_pred, y, reduced_loss_dict, loss_dict, loss_scalars)
                    
                    # get current Loss Scale of first optimizer
                    loss_scale = amp._amp_state.loss_scalers[0]._loss_scale if hparams.fp16_run else 32768.
                    
                    # restart if training/model has collapsed
                    if (iteration > 1e3 and (reduced_loss > LossExplosionThreshold)) or (math.isnan(reduced_loss)):
                        raise LossExplosion(f"\nLOSS EXPLOSION EXCEPTION ON RANK {rank}: Loss reached {reduced_loss} during iteration {iteration}.\n\n\n")
                    if (loss_scale < 1/4):
                        raise LossExplosion(f"\nLOSS EXCEPTION ON RANK {rank}: Loss Scaler reached {loss_scale} during iteration {iteration}.\n\n\n")
                    
                    if expavg_loss_dict is None:
                        expavg_loss_dict = reduced_loss_dict
                    else:
                        expavg_loss_dict.update({k:v for k, v in reduced_loss_dict.items() if k not in expavg_loss_dict.keys()})# if new loss term appears in reduced_loss_dict, add it to the expavg_loss_dict.
                        expavg_loss_dict = {k: (reduced_loss_dict[k]*(1-loss_dict_smoothness))+(expavg_loss_dict[k]*loss_dict_smoothness) for k in expavg_loss_dict.keys() if k in reduced_loss_dict}
                        expavg_loss_dict_iters += 1
                    
                    if expavg_loss_dict_iters > 100:# calc smoothed loss dict
                        if best_loss_dict is None:
                            best_loss_dict = expavg_loss_dict
                        else:
                            best_loss_dict = {k: min(best_loss_dict[k], expavg_loss_dict[k]) for k in best_loss_dict.keys() if k in expavg_loss_dict}
                    
                    if rank == 0:# print + log metrics
                        duration = time.time() - start_time
                        if not is_overflow:
                            average_loss = rolling_loss.process(reduced_loss)
                            dbGANAccStr  = expavg_loss_dict.get('dbGAN_accuracy',  None) or reduced_loss_dict.get('dbGAN_accuracy',  0.5)
                            InfGANAccStr = expavg_loss_dict.get('InfGAN_accuracy', None) or reduced_loss_dict.get('InfGAN_accuracy', 0.5)
                            WScoreStr    = expavg_loss_dict.get('weighted_score' , None) or reduced_loss_dict.get('weighted_score' , 0.0)
                            tqdm.write(
                                f"{iteration} [TrainLoss:{reduced_loss:.3f} Avg:{average_loss:.3f}] "
                                f"[{grad_norm:03.1f}GradNorm] [{duration:.2f}s/it] "
                                f"[{(duration/(hparams.batch_size*args.n_gpus)):.3f}s/file] "
                                f"[{learning_rate:.1e}LR] [{loss_scale:.0f}LS] "
                                f"[{WScoreStr:.1%}AttSc] [{dbGANAccStr:.1%}dbGANAcc] [{InfGANAccStr:.1%}InfGANAcc]")
                            logger.log_training(model, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration, iteration)
                        else:
                            tqdm.write("Gradient Overflow, Skipping Step")
                        start_time = time.time()
                    
                    if iteration%checkpoint_interval==0 or os.path.exists(save_file_check_path):# save model checkpoint every X iters
                        if rank == 0:
                            checkpoint_path = os.path.join(args.output_directory, f"checkpoint_{iteration}")
                            save_checkpoint(model, optimizer, hifiGAN, learning_rate, iteration, hparams, best_validation_loss, average_loss, best_val_loss_dict, best_loss_dict, speaker_lookup, speakerlist, checkpoint_path)
                    
                    if iteration%dump_filelosses_interval==0:# syncronise file_losses between graphics cards
                        print("Updating File_losses dict!")
                        file_losses = write_dict_to_file(file_losses, os.path.join(args.output_directory, 'file_losses.csv'),
                                                                      os.path.join(args.output_directory, 'speaker_losses.csv'), speakerlist, args.n_gpus, rank)
                    
                    if (iteration % int(validation_interval) == 0) or (os.path.exists(save_file_check_path)):# validate models and save 'best_val_model' checkpoints
                        if rank == 0 and os.path.exists(save_file_check_path):
                            os.remove(save_file_check_path)
                        # perform validation and save "best_val_model" depending on validation loss
                        val_loss, best_val_loss_dict, file_losses = validate(hparams, args, file_losses, model, criterion, hifiGAN, valset, loss_scalars, best_val_loss_dict, iteration, collate_fn, logger)# validate/teacher_force
                        file_losses = write_dict_to_file(file_losses, os.path.join(args.output_directory, 'file_losses.csv'),
                                                                      os.path.join(args.output_directory, 'speaker_losses.csv'), speakerlist, args.n_gpus, rank)
                        if (val_loss < best_validation_loss):
                            best_validation_loss = val_loss
                            if rank == 0 and hparams.save_best_val_model:
                                checkpoint_path = os.path.join(args.output_directory, "best_val_model")
                                save_checkpoint(
                                    model, optimizer, hifiGAN, learning_rate, iteration, hparams, best_validation_loss,
                                    average_loss, best_val_loss_dict, best_loss_dict, speaker_lookup, speakerlist, checkpoint_path)
                        just_did_val = True
                    
                    del y_pred, y, batch, loss_dict, reduced_loss_dict
                    iteration += 1
                    # end of iteration loop
                
                # update filelist of training dataloader
                print("Updating File_losses dict!")
                file_losses = write_dict_to_file(file_losses, os.path.join(args.output_directory, 'file_losses.csv'),
                                                              os.path.join(args.output_directory, 'speaker_losses.csv'), speakerlist, args.n_gpus, rank)
                print("Done!")
                
                # end of epoch loop
            training = False # exit the While loop
        
        #except Exception as ex: # print Exception and continue from checkpoint. (turns out it takes < 4 seconds to restart like this, fucking awesome)
        except LossExplosion as ex: # print Exception and continue from checkpoint. (turns out it takes < 4 seconds to restart like this, fucking awesome)
            print(ex) # print Loss
            checkpoint_path = os.path.join(args.output_directory, "best_val_model")
            assert os.path.exists(checkpoint_path), "best_val_model checkpoint must exist for automatic restarts"
            
            if hparams.fp16_run:
                amp._amp_state.loss_scalers[0]._loss_scale = 32768
            
            # clearing VRAM for load checkpoint
            model.zero_grad()
            x=batch=y=y_pred=loss=len_loss=loss_z=loss_w=loss_s=loss_att=dur_loss_z=dur_loss_w=dur_loss_s=None
            torch.cuda.empty_cache()
            
            model.eval()
            _ = load_checkpoint(args.checkpoint_path, model, optimizer, best_val_loss_dict, best_loss_dict)
            model, optimizer, _learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict = _
            learning_rate = optimizer.param_groups[0]['lr']
            epoch_offset = max(0, int(iteration / len(train_loader)))
            model.train()
            checkpoint_iter = iteration
            iteration  += 1
            n_restarts += 1
        except KeyboardInterrupt as ex:
            print(ex)
            training = False
            raise KeyboardInterrupt()

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
    
    torch.backends.cudnn.enabled   = hparams.cudnn_enabled
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
    else:
        global amp
        amp = None
    try:
        from apex import optimizers as apexopt
    except:
        pass
    
    train(args, args.rank, args.group_name, hparams)

