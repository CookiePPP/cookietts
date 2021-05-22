import os
os.environ["LRU_CACHE_CAPACITY"] = "3"# reduces RAM usage massively with pytorch 1.4 or older
import time
import argparse
import math
import gc
import random
import pickle
import numpy as np

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import load_model
from CookieTTS.utils.dataset.data_utils import TTSDataset, Collate, generate_filelist_from_datasets
from CookieTTS.utils import get_args, force

from loss_function import Loss
from logger import Logger
from hparams import create_hparams
from CookieTTS.utils.model.GPU import to_gpu
from math import e, ceil

from tqdm import tqdm
import CookieTTS.utils.audio.stft as STFT
from CookieTTS.utils.dataset.utils import load_wav_to_torch, load_filepaths_and_text
from scipy.io.wavfile import read

import os.path

save_file_check_path = "save"
load_checkpoints_with_zero_iters = 0

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


def init_distributed(h, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Initialize distributed communication
    dist.init_process_group(
        backend=h.dist_backend, init_method=h.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def get_filelist(h, val=False):
    if h.data_source == 1:# if filelist is a folder, check all datasets inside the folder for audio files and transcripts.
        # convert dataset folder into a filelist
        filelist = generate_filelist_from_datasets(h.dataset_folder,
                              METAPATH_APPEND=h.dataset_metapath_append,
                              AUDIO_FILTER   =h.dataset_audio_filters,
                              AUDIO_REJECTS  =h.dataset_audio_rejects,
                              MIN_DURATION   =h.dataset_min_duration,
                              MAX_DURATION   =h.dataset_max_duration,
                              MIN_CHAR_LEN   =h.dataset_min_chars,
                              MAX_CHAR_LEN   =h.dataset_max_chars,
                          SKIP_EMPTY_DATASETS=h.skip_empty_datasets,)
    elif h.data_source == 0:# else filelist is a ".txt" file, load the easy way
        if val:
            filelist = load_filepaths_and_text(h.validation_files)
        else:
            filelist = load_filepaths_and_text(h.training_files)
    return filelist


def prepare_dataloaders(h, dataloader_args, args, speaker_ids, audio_offset=0):
    # Get data, data loaders and collate function ready
    if h.data_source == 1:
        if args.rank == 0:
            fl_dict = get_filelist(h)
        
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
        training_filelist   = filelist[ int(len(filelist)*h.dataset_p_val):]
        validation_filelist = filelist[:int(len(filelist)*h.dataset_p_val) ]
        while any(y for y in set([x[2] for x in validation_filelist]) if y not in [x[2] for x in training_filelist]):
            random.Random(0).shuffle(filelist)
            training_filelist   = filelist[ int(len(filelist)*h.dataset_p_val):]
            validation_filelist = filelist[:int(len(filelist)*h.dataset_p_val) ]
        
        if args.n_gpus > 1:
            torch.distributed.barrier()# wait till all graphics cards reach this point.
            if args.rank == 0 and os.path.exists('fl_dict.pkl'):
                os.remove('fl_dict.pkl')
    else:
        # get speaker names to ID
        if os.path.exists(h.speakerlist):
            # expects speakerlist to look like below
            # |Nancy|0
            # |Bob|1
            # |Phil|2
            #
            # every line must have at least 2 pipe symbols
            speakerlist = load_filepaths_and_text(h.speakerlist)
            speaker_name_lookup = {x[1]: speaker_id_lookup[x[2]] for x in speakerlist if x[2] in speaker_id_lookup.keys()}
        else:
            print("'speakerlist' in h.py not found! Speaker names will be IDs instead.")
            speakerlist = [['Dataset',i,i,'Source','Source Type'] for i in range(speaker_id_lookup.keys())]
            speaker_name_lookup = speaker_id_lookup# if there is no speaker
        
        training_filelist   = get_filelist(h, val=False)
        validation_filelist = get_filelist(h, val=True)
        speaker_ids = speaker_ids if h.use_saved_speakers else None
    
    trainset = TTSDataset(training_filelist, h, dataloader_args, check_files=h.check_files, shuffle=False,
                          speaker_ids=speaker_ids, audio_offset=audio_offset,
                          )
    valset = TTSDataset(validation_filelist, h, dataloader_args, check_files=h.check_files, shuffle=False,
                          speaker_ids=getattr(trainset, 'speaker_ids', None), audio_offset=audio_offset,
                          deterministic_arpabet=True,)
    collate_fn = Collate(h)
    
    use_shuffle = False# using custom Shuffle function inside dataloader.dataset which works with TBPTT
    if h.distributed_run:
        train_sampler = DistributedSampler(trainset, shuffle=use_shuffle)
        shuffle = False
    else:
        train_sampler = None
        shuffle = use_shuffle
    
    train_loader = DataLoader(trainset, shuffle=shuffle, sampler=train_sampler,
                              num_workers=h.num_workers,# prefetch_factor=h.prefetch_factor,
                              batch_size=h.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, train_sampler, trainset, speakerlist


def prepare_directories_and_logger(h, args):
    if args.rank == 0:
        if not os.path.isdir(args.output_directory):
            os.makedirs(args.output_directory)
            os.chmod(args.output_directory, 0o775)
        logger = Logger(os.path.join(args.output_directory, args.log_directory), h)
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
        
        if (not any(k.startswith('discriminator') for k in pretrained_dict.keys())) and any(k.startswith('discriminator') for k in model_dict.keys()):
            for k, v in {k: v for k, v in pretrained_dict.items() if 'generator' in k}.items():
                pretrained_dict[k.replace('generator','discriminator')] = v.clone()
        
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
        iteration = checkpoint_dict.get('iteration', 0)
        
        saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
    return model, iteration, saved_lookup


def load_checkpoint(checkpoint_paths, model, optimizer, d_optimizer, best_val_loss_dict, best_loss_dict, best_validation_loss=1e3,):
    if type(checkpoint_paths) is str:
        checkpoint_paths = [checkpoint_paths,]
    
    for checkpoint_path in checkpoint_paths:
        assert os.path.isfile(checkpoint_path)
        print(f"Loading checkpoint '{checkpoint_path}'")
        
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint_dict['state_dict'])# load model weights
        
        if 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])# load optimizer state
        if 'd_optimizer' in checkpoint_dict.keys():
            d_optimizer.load_state_dict(checkpoint_dict['d_optimizer'])# load optimizer state
        #if 'amp' in checkpoint_dict.keys() and amp is not None:
        #    amp.load_state_dict(checkpoint_dict['amp']) # load AMP (fp16) state.
        if 'learning_rate' in checkpoint_dict.keys():
            learning_rate = checkpoint_dict['learning_rate']
        #if 'h' in checkpoint_dict.keys():
        #    h = checkpoint_dict['h']
        if 'best_validation_loss' in checkpoint_dict.keys():
            best_validation_loss = checkpoint_dict['best_validation_loss']
        if 'best_val_loss_dict' in checkpoint_dict.keys():
            best_val_loss_dict = checkpoint_dict['best_val_loss_dict']
        if 'best_loss_dict' in checkpoint_dict.keys():
            best_loss_dict = checkpoint_dict['best_loss_dict']
        if 'average_loss' in checkpoint_dict.keys():
            average_loss = checkpoint_dict['average_loss']
        
        iteration = 0 if load_checkpoints_with_zero_iters else checkpoint_dict['iteration']
        saved_lookup = checkpoint_dict['speaker_id_lookup'] if 'speaker_id_lookup' in checkpoint_dict.keys() else None
        
        print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, d_optimizer, learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict


def save_checkpoint(model, optimizer, d_optimizer, learning_rate, iteration, h, best_validation_loss, average_loss,
                    best_val_loss_dict, best_loss_dict, speaker_id_lookup, speakerlist, dataset, filepath):
    from CookieTTS.utils.dataset.utils import load_filepaths_and_text
    
    tqdm.write(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    
    assert all(str(line[2]).isdigit() for line in speakerlist), 'speakerlist got str in speaker_id section!'
    speaker_name_lookup = {x[1].strip(): x[2] for x in speakerlist}
    
    save_dict = {'iteration'            : iteration,
                 'state_dict'           : model.state_dict(),
                 'optimizer'            : optimizer.state_dict(),
                 'learning_rate'        : learning_rate,
                 'h'                    : h,
                 'speaker_f0_meanstd'   : dataset.speaker_f0_meanstd,
                 'speaker_sylps_meanstd': dataset.speaker_sylps_meanstd,
                 'speaker_id_lookup'    : speaker_id_lookup,
                 'speaker_name_lookup'  : speaker_name_lookup,
                 'speakerlist'          : speakerlist,
                 'best_validation_loss' : best_validation_loss,
                 'best_val_loss_dict'   : best_val_loss_dict,
                 'best_loss_dict'       : best_loss_dict,
                 'average_loss'         : average_loss}
    if h.fp16_run:
        save_dict['amp'] = amp.state_dict()
    if h.GAN_enable:
        save_dict['d_optimizer']  = d_optimizer.state_dict()
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


def validate(h, args, file_losses, model, criterion, valset, loss_scalars, best_val_loss_dict, iteration, collate_fn, logger, inference_mode=False):
    """Handles all the validation scoring and printing"""
    model.eval()
    
    with torch.no_grad():
        if not len(valset.filelist) >= h.n_gpus*h.batch_size:
            print(f'too few files in validation set! Found {len(valset.filelist)}, expected {h.n_gpus*h.batch_size} or more.\nIf you have a small amount of data, increase `dataset_p_val` or decrease `val_batch_size`')
        val_sampler = DistributedSampler(valset) if h.distributed_run else None
        val_loader  = DataLoader(valset, sampler=val_sampler,
                            num_workers=h.val_num_workers,# prefetch_factor=h.prefetch_factor,
                            shuffle=False, batch_size=h.val_batch_size,
                            pin_memory=False, drop_last=True, collate_fn=collate_fn)
        
        if inference_mode:# if force_monotonic_alignment
            orig_window_offset = model.generator.decoder.attention.window_offset
            orig_window_range  = model.generator.decoder.attention.window_range
            model.generator.decoder.attention.window_offset = h.inf_att_window_offset
            model.generator.decoder.attention.window_range  = h.inf_att_window_range
        
        tmp_file_losses = {}
        loss_dict_total = None
        for i, batch in tqdm(enumerate(val_loader), desc=("Inference" if inference_mode else "Validation"), total=len(val_loader), smoothing=0): # i = index, batch = stuff in array[i]
            y = model.parse_batch(batch)
            y['inference_mode'] = inference_mode
            with torch.random.fork_rng(devices=[0,]):
                torch.random.manual_seed(i)# use repeatable seeds during validation so results are more consistent and comparable.
                y_pred = force(model, valid_kwargs=model_args, **y)
            
            # Update Discriminator
            loss_dict = {}; file_losses_batch = {}
            if not h.fast_discriminator_mode and h.GAN_enable:
                loss_dict, file_losses_batch = criterion.d_forward(iteration, model, y_pred, y, loss_scalars, {})
            loss_dict_g, file_losses_batch_g = criterion.g_forward(iteration, model, y_pred, y, loss_scalars, {})
            if h.fast_discriminator_mode and h.GAN_enable:
                loss_dict, file_losses_batch = criterion.d_forward(iteration, model, y_pred, y, loss_scalars, {})
            
            loss_dict = {**loss_dict, **loss_dict_g}
            tmp_file_losses = update_smoothed_dict(tmp_file_losses, {**file_losses_batch, **file_losses_batch_g}, file_losses_smoothness)
            
            if h.distributed_run:
                reduced_loss_dict = {k: reduce_tensor(v.data, args.n_gpus).item() if v is not None else 0. for k, v in loss_dict.items()}
            else:
                reduced_loss_dict = {k: v.item() if v is not None else 0. for k, v in loss_dict.items()}
            reduced_loss = reduced_loss_dict['loss']
            
            if loss_dict_total is None:
                loss_dict_total = {k: 0. for k, v in loss_dict.items()}
            
            for k in loss_dict_total.keys():
                loss_dict_total[k] = loss_dict_total[k] + reduced_loss_dict[k]
            # end forloop
        loss_dict_total = {k: v/(i+1) for k, v in loss_dict_total.items()}
        # end torch.no_grad()
    
    model.train()
    
    if inference_mode:# if force_monotonic_alignment
        model.generator.decoder.attention.window_offset = orig_window_offset
        model.generator.decoder.attention.window_range  = orig_window_range 
    
    # update best losses
    best_val_loss_dict = loss_dict_total if best_val_loss_dict is None else {k: min(best_val_loss_dict[k], loss_dict_total[k]) for k in best_val_loss_dict.keys() if k in loss_dict_total}
    
    if args.rank == 0:# print, log data and return.
        tqdm.write(f"Validation loss {iteration}: {loss_dict_total['loss']:9f}")
        if iteration > 1:
            log_terms = (loss_dict_total, best_val_loss_dict, model, y, y_pred, iteration,)
            if inference_mode:
                logger.log_validation(*log_terms, prepend='inference')
            else:
                logger.log_validation(*log_terms)
    
    file_losses = update_smoothed_dict(file_losses, tmp_file_losses, file_losses_smoothness)
    return loss_dict_total['loss'], best_val_loss_dict, file_losses


def clip_grad_norm_(model, optimizer, grad_clip_thresh, fp16_run=False):
    is_overflow = False
    if grad_clip_thresh:# apply gradient clipping to params
        if fp16_run:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), grad_clip_thresh)
            is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
            if not math.isfinite(grad_norm):
                print(f'Rank NaN Gradnorm(s) from;\n'+'\n'.join([f'- {key} - {torch.norm(p.grad.detach(), 2.0).mean()} - {torch.norm(p.grad.detach(), 2.0).sum()} - {p.grad.detach().mean()} - {p.grad.detach().sum()}' for key, p in model.named_parameters() if p.grad is not None and (not torch.isfinite(torch.norm(p.grad.detach(), 2.0)).all())]))
    else:
        grad_norm = 0.0
    return grad_norm, is_overflow

def backprop_loss_(loss, optimizer, fp16_run=False, inputs=None, retain_graph=None):
    if fp16_run:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(inputs=inputs, retain_graph=retain_graph)
    else:
        loss.backward(inputs=inputs, retain_graph=retain_graph)

def optimizer_step_(optimizer, grad_norm):
    if math.isfinite(grad_norm):
        optimizer.step()

def complete_optimizer_step(loss, model, optimizer, grad_clip_thresh, fp16_run=False, inputs=None, retain_graph=None):
    backprop_loss_(loss, optimizer, fp16_run, inputs, retain_graph)
    if not (isinstance(model, tuple) or isinstance(model, list)):
        model = (model,)
    grad_norm, is_overflow = 0.0, False
    for m in model:
        _ = clip_grad_norm_(m, optimizer, grad_clip_thresh, fp16_run)
        grad_norm += _[0]
        is_overflow = is_overflow or _[1]
    optimizer_step_(optimizer, grad_norm)
    return grad_norm, is_overflow

def update_loss_dict(expavg_loss_dict, best_loss_dict, reduced_loss_dict, expavg_loss_dict_iters, loss_dict_smoothness):
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
    return expavg_loss_dict, best_loss_dict, loss_dict_smoothness

def train(args, rank, group_name, h):
    """Training and validation logging results to tensorboard and stdout
    
    Params
    ------
    args.output_directory (string): directory to save checkpoints
    args.log_directory (string) directory to save tensorboard logs
    args.checkpoint_path(string): checkpoint path
    args.n_gpus (int): number of gpus
    rank (int): rank of current gpu
    h (object): comma separated list of "name=value" pairs.
    """
    # setup distributed
    h.n_gpus = args.n_gpus
    h.rank = rank
    if h.distributed_run:
        init_distributed(h, args.n_gpus, rank, group_name)
    
    # reproducablilty stuffs
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    
    # initialize blank model
    print('Initializing GANTron...')
    model = load_model(h)
    print('Done')
    global model_args
    model_args = get_args(model.forward, kwargs=True)
    model.cuda().eval()
    learning_rate = h.learning_rate
    
    # (optional) show the names of each layer in model, mainly makes it easier to copy/paste what you want to adjust
    if h.print_layer_names_during_startup:
        print(*[f"Layer{i} = "+str(x[0])+" "+str(x[1].shape) for i,x in enumerate(list(model.named_parameters()))], sep="\n")
    
    if len(h.frozen_modules):# (optional) Freeze layers by disabling grads
        for layer, params in list(model.named_parameters()):
            if any(layer.startswith(module) for module in h.frozen_modules):
                params.requires_grad = False
                print(f"Layer: {layer} has been frozen")
    
    if len(h.unfrozen_modules):# (optional) Unfreeze layers by disabling grads
        for layer, params in list(model.named_parameters()):
            if any(layer.startswith(module) for module in h.unfrozen_modules):
                params.requires_grad = True
                print(f"Layer: {layer} has been unfrozen")
    
    if True:# define optimizer (any params without requires_grad are ignored)
        #optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=h.weight_decay)
        optimizer =  torch.optim.Adam(filter(lambda p: p.requires_grad, model.generator.parameters()), lr=learning_rate, weight_decay=h.weight_decay)
    
    d_optimizer = None
    if h.GAN_enable:
        #d_optimizer = apexopt.FusedAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=h.weight_decay)
        d_optimizer =  torch.optim.Adam(filter(lambda p: p.requires_grad, model.discriminator.parameters()), lr=learning_rate, weight_decay=h.weight_decay)
    
    if True and rank == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("{:,} total parameters in model".format(pytorch_total_params))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("{:,} trainable parameters.".format(pytorch_total_params))
    
    if h.fp16_run:# convert models to mixed precision
        print("Initializing AMP Model(s) / Optimzier(s)")
        models     = [model,]
        optimizers = [optimizer,]
        if h.GAN_enable:
            optimizers.append(d_optimizer)
        models, optimizers = amp.initialize(models, optimizers, opt_level=f'O{h.fp16_run_optlvl}', max_loss_scale=8192)
        if h.GAN_enable:
            d_optimizer = optimizers.pop()
        model = models.pop()
        optimizer = optimizers.pop()
        assert     len(models) == 0
        assert len(optimizers) == 0
        del models, optimizers
    
    print("Initializing Gradient AllReduce model wrapper.")
    if h.distributed_run:
        model = apply_gradient_allreduce(model)
    
    print("Initializing GANTron Loss func.")
    criterion = Loss(h)
    criterion.cuda()
    
    print("Initializing GANTron Logger.")
    logger = prepare_directories_and_logger(h, args)
    
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
                args.checkpoint_path, model, h.ignore_layers)
        elif args.warm_start_force:
            model, iteration, saved_lookup = warm_start_force_model(
                args.checkpoint_path, model)
        else:
            _ = load_checkpoint(args.checkpoint_path, model, optimizer, d_optimizer, best_val_loss_dict, best_loss_dict)
            model, optimizer, d_optimizer, _learning_rate, iteration, best_validation_loss, saved_lookup, best_val_loss_dict, best_loss_dict = _
        checkpoint_iter = iteration
        iteration += 1  # next iteration is iteration + 1
        print('Model Loaded')
    
    # define datasets/dataloaders
    print("Initializing Dataloader(s).")
    dataloader_args = [*get_args(criterion.forward), *model_args]
    train_loader, valset, collate_fn, train_sampler, trainset, speakerlist = prepare_dataloaders(h, dataloader_args, args, saved_lookup)
    epoch_offset = max(0, int(iteration / len(train_loader)))
    speaker_id_lookup = trainset.speaker_ids
    
    model.train()
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    just_did_val = True
    rolling_loss = StreamingMovingAverage(min(int(len(train_loader)), 200))
    
    # reproducablilty stuffs
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    
    # ================ MAIN TRAINNIG LOOP! ===================
    training = True
    while training:
        try:
            for epoch in tqdm(range(epoch_offset, h.epochs), initial=epoch_offset, total=h.epochs, desc="Epoch:", position=1, unit="epoch"):
                tqdm.write(f"Epoch:{epoch}")
                
                train_loader.dataset.shuffle_dataset()# Shuffle Dataset
                dataset_len = len(train_loader)
                
                start_time = time.time()
                dataloader_start_time = time.time()
                # start iterating through the epoch
                for i, batch in tqdm(enumerate(train_loader), desc="Iter:  ", smoothing=0, total=len(train_loader), position=0, unit="iter"):
                    if h.distributed_run:
                        torch.distributed.barrier()# wait till all graphics cards reach this point.
                    dataloader_elapsed_time = time.time() - dataloader_start_time
                    # run external code every epoch or 1000 iters, allows the run to be adjusted without restarts
                    if (i<2 or iteration % param_interval == 0):
                        try:
                            with open("run_every_epoch.py", encoding='utf-8') as f:
                                internal_text = str(f.read())
                                if len(internal_text) > 0:
                                    #code = compile(internal_text, "run_every_epoch.py", 'exec')
                                    ldict = {'iteration': iteration, 'checkpoint_iter': checkpoint_iter, 'n_restarts': n_restarts, "expavg_loss_dict": expavg_loss_dict}
                                    gc.disable()
                                    exec(internal_text, globals(), ldict)
                                    gc.enable()
                                else:
                                    print("[info] tried to execute 'run_every_epoch.py' but it is empty")
                        except Exception as ex:
                            print(f"[warning] 'run_every_epoch.py' FAILED to execute!\nException:\n{ex}")
                        globals().update(ldict)
                        locals() .update(ldict)
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
                        if h.GAN_enable:
                            for param_group in d_optimizer.param_groups:
                                param_group['lr'] = learning_rate
                    # /run external code every epoch, allows the run to be adjusting without restarts/
                    
                    loss_scalars = {
                    'g_class_weight'  :      g_class_weight,
                    'g_fm_weight'     :      g_fm_weight,
                    'g_mel_MAE_weight':      g_mel_MAE_weight,
                    'g_mel_MSE_weight':      g_mel_MSE_weight,
                    'g_gate_BCE_weight':     g_gate_BCE_weight,
                    'g_att_loss_weight':     g_att_loss_weight,
                    'g_hard_att_MSE_weight': g_hard_att_MSE_weight,
                    'g_hard_att_MAE_weight': g_hard_att_MAE_weight,
                    'd_class_weight'  :     d_class_weight,
                    'd_att_loss_weight':    d_att_loss_weight,
                    'd_atte_gd_mse_weight': d_atte_gd_mse_weight,
                    'g_att_diagonality_weight':       g_att_diagonality_weight,
                    'g_att_top1_avg_prob_weight':     g_att_top1_avg_prob_weight,
                    'g_att_top2_avg_prob_weight':     g_att_top2_avg_prob_weight,
                    'g_att_top3_avg_prob_weight':     g_att_top3_avg_prob_weight,
                    'g_att_avg_max_dur_weight':       g_att_avg_max_dur_weight,
                    'g_att_avg_min_dur_weight':       g_att_avg_min_dur_weight,
                    'g_att_avg_avg_dur_weight':       g_att_avg_avg_dur_weight,
                    'g_att_avg_missing_enc_p_weight': g_att_avg_missing_enc_p_weight,
                    'g_att_avg_attscore_weight':      g_att_avg_attscore_weight,
                    }
                    loss_params = {
                    'g_class_active_thresh': g_class_active_thresh,
                    'd_class_active_thresh': d_class_active_thresh,
                    }
                    
                    # Generate Fake Data
                    y = model.parse_batch(batch)# move batch to GPU (async)
                    y['teacher_force_prob'] = teacher_force_prob
                    y_pred = force(model, valid_kwargs=model_args, **y)
                    
                    # Update Discriminato
                    if h.fast_discriminator_mode:
                        # Update Generator
                        optimizer.zero_grad(set_to_none=True)# clear/zero the gradients from the last iter
                        loss_dict_g, file_losses_batch_g = criterion.g_forward(iteration, model, y_pred, y, loss_scalars, loss_params, file_losses)
                        
                        grad_norm_g, is_overflow = complete_optimizer_step(loss_dict_g['loss'], model.generator, optimizer, grad_clip_thresh, h.fp16_run, inputs=tuple(model.generator.parameters()), retain_graph=True)
                        optimizer.zero_grad(set_to_none=True)# clear/zero the gradients from the last iter
                        
                        # Update Discriminator
                        if h.GAN_enable:
                            for j in range(n_discriminator_loops):
                                d_optimizer.zero_grad(set_to_none=True)# clear/zero the gradients from the last iter
                                loss_dict, file_losses_batch = criterion.d_forward(iteration, model, y_pred, y, loss_scalars, loss_params)
                                
                                if loss_dict['loss_d'] is not None:
                                    grad_norm_d, is_overflow_d = complete_optimizer_step(loss_dict['loss_d'], model.discriminator, d_optimizer, grad_clip_thresh, h.fp16_run, inputs=tuple(model.discriminator.parameters()))
                                    d_optimizer.zero_grad(set_to_none=True)# clear/zero the gradients from the last iter
                        else:
                            loss_dict = {}; file_losses_batch = {}; grad_norm_d = 0.0; is_overflow_d = False
                    else:
                        # Update Discriminator
                        if h.GAN_enable:
                            for j in range(n_discriminator_loops):
                                d_optimizer.zero_grad()# clear/zero the gradients from the last iter
                                loss_dict, file_losses_batch = criterion.d_forward(iteration, model, y_pred, y, loss_scalars, loss_params)
                                
                                if loss_dict['loss_d'] is not None:
                                    grad_norm_d, is_overflow_d = complete_optimizer_step(loss_dict['loss_d'], model.discriminator, d_optimizer, grad_clip_thresh, h.fp16_run, inputs=tuple(model.discriminator.parameters()))
                        else:
                            loss_dict = {}; file_losses_batch = {}; grad_norm_d = 0.0; is_overflow_d = False
                        
                        # Update Generator
                        optimizer.zero_grad()# clear/zero the gradients from the last iter
                        loss_dict_g, file_losses_batch_g = criterion.g_forward(iteration, model, y_pred, y, loss_scalars, loss_params, file_losses)
                        
                        grad_norm_g, is_overflow = complete_optimizer_step(loss_dict_g['loss'], model.generator, optimizer, grad_clip_thresh, h.fp16_run, inputs=tuple(model.generator.parameters()))
                    
                    # Collate Metrics
                    loss_dict = {**loss_dict, **loss_dict_g}
                    file_losses = update_smoothed_dict(file_losses, {**file_losses_batch, **file_losses_batch_g}, file_losses_smoothness)
                    
                    is_overflow = is_overflow or is_overflow_d
                    grad_norm = grad_norm_g + grad_norm_d
                    
                    # Average+Reduce loss terms across every GPU
                    if h.distributed_run:
                        reduced_loss_dict = {k: reduce_tensor(v.data, args.n_gpus).item() if v is not None else 0. for k, v in loss_dict.items()}
                    else:
                        reduced_loss_dict = {k: v.item() if v is not None else 0. for k, v in loss_dict.items()}
                    
                    reduced_loss   = reduced_loss_dict['loss']
                    reduced_loss_d = reduced_loss_dict.get('loss_d', 0.0)
                    
                    expavg_loss_dict, best_loss_dict, loss_dict_smoothness = update_loss_dict(
                        expavg_loss_dict, best_loss_dict, reduced_loss_dict, expavg_loss_dict_iters, loss_dict_smoothness)
                    
                    # get current Loss Scale of first optimizer
                    loss_scale = amp._amp_state.loss_scalers[0]._loss_scale if h.fp16_run else 32768.
                    # restart if training/model has collapsed
                    if (iteration > 1e3 and (reduced_loss > LossExplosionThreshold)) or (math.isnan(reduced_loss)):
                        raise LossExplosion(f"\nLOSS EXPLOSION EXCEPTION ON RANK {rank}: Loss reached {reduced_loss} during iteration {iteration}.\n\n\n")
                    if (loss_scale < 1/4):
                        raise LossExplosion(f"\nLOSS EXCEPTION ON RANK {rank}: Loss Scaler reached {loss_scale} during iteration {iteration}.\n\n\n")
                    
                    if rank == 0:# print + log metrics
                        duration = time.time() - start_time
                        if not is_overflow:
                            average_loss = rolling_loss.process(reduced_loss)
                            logger.log_training(model, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration, iteration)
                        
                        s = (f"{iteration} [TrainLoss:{reduced_loss:.3f} Avg:{average_loss:.3f}] "
                             f"[{grad_norm_g:04.2f}g {grad_norm_d:04.2f}d GradNorm] [{duration:.2f}s/it] "
                             f"[{(duration/(h.batch_size*args.n_gpus)):.3f}s/file] "
                             f"[{learning_rate:.1e}LR] [{loss_scale:.0f}LS]")
                        if dataloader_elapsed_time > duration*0.1:
                            s += f" [{dataloader_elapsed_time:.2f}IO s/it]"
                        tqdm.write(s)
                        if is_overflow:
                            tqdm.write("Gradient Overflow, Skipping Step\n")
                        start_time = time.time()
                    
                    if iteration%checkpoint_interval==0 or os.path.exists(save_file_check_path):# save model checkpoint every X iters
                        if rank == 0:
                            checkpoint_path = os.path.join(args.output_directory, f"checkpoint_{iteration}")
                            save_checkpoint(
                                model, optimizer, d_optimizer, learning_rate, iteration, h, best_validation_loss,
                                average_loss, best_val_loss_dict, best_loss_dict, speaker_id_lookup, speakerlist, trainset, checkpoint_path)
                    
                    if iteration%dump_filelosses_interval==0:# syncronise file_losses between graphics cards
                        print("Updating File_losses dict!")
                        file_losses = write_dict_to_file(file_losses, os.path.join(args.output_directory, 'file_losses.csv'),
                                                                      os.path.join(args.output_directory, 'speaker_losses.csv'), speakerlist, args.n_gpus, rank)
                    
                    if (iteration % int(validation_interval) == 0) or (os.path.exists(save_file_check_path)):# validate models and save 'best_val_model' checkpoints
                        if rank == 0 and os.path.exists(save_file_check_path):
                            os.remove(save_file_check_path)
                        # perform validation and save "best_val_model" depending on validation loss
                        val_args = (h, args, file_losses, model, criterion, valset, loss_scalars, best_val_loss_dict, iteration, collate_fn, logger)
                        _ = validate(*val_args, inference_mode=True)
                        val_loss, best_val_loss_dict, file_losses = validate(*val_args)
                        file_losses = write_dict_to_file(file_losses, os.path.join(args.output_directory, 'file_losses.csv'),
                                                                      os.path.join(args.output_directory, 'speaker_losses.csv'), speakerlist, args.n_gpus, rank)
                        best_validation_loss = val_loss
                        if rank == 0:
                            checkpoint_path = os.path.join(args.output_directory, "latest_val_model")
                            save_checkpoint(
                                model, optimizer, d_optimizer, learning_rate, iteration, h, best_validation_loss,
                                average_loss, best_val_loss_dict, best_loss_dict, speaker_id_lookup, speakerlist, trainset, checkpoint_path)
                        just_did_val = True
                    
                    del y_pred, y, batch, loss_dict, reduced_loss_dict
                    iteration += 1
                    dataloader_start_time = time.time()
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
            checkpoint_path = os.path.join(args.output_directory, "latest_val_model")
            assert os.path.exists(checkpoint_path), "latest_val_model checkpoint must exist for automatic restarts"
            
            if h.fp16_run:
                amp._amp_state.loss_scalers[0]._loss_scale = 32768
            
            # clearing VRAM for load checkpoint
            model.zero_grad()
            x=batch=y=y_pred=loss=len_loss=loss_z=loss_w=loss_s=loss_att=dur_loss_z=dur_loss_w=dur_loss_s=None
            torch.cuda.empty_cache()
            
            model.eval()
            _ = load_checkpoint(checkpoint_path, model, optimizer, best_val_loss_dict, best_loss_dict)
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
    h = create_hparams(args.hparams)
    
    torch.backends.cudnn.enabled   = h.cudnn_enabled
    torch.backends.cudnn.benchmark = h.cudnn_benchmark
    
    print("FP16 Run:", h.fp16_run)
    print("Dynamic Loss Scaling:", h.dynamic_loss_scaling)
    print("Distributed Run:", h.distributed_run)
    print("cuDNN Enabled:", h.cudnn_enabled)
    print("cuDNN Benchmark:", h.cudnn_benchmark)
    
    if args.detect_anomaly: # checks backprop for NaN/Infs and outputs very useful stack-trace. Runs slowly while enabled.
        torch.autograd.set_detect_anomaly(True)
        print("Autograd Anomaly Detection Enabled!\n(Code will run slower but backward pass will output useful info if crashing or NaN/inf values)")
    
    # these are needed for fp16 training, not inference
    if h.fp16_run:
        from apex import amp
    else:
        global amp
        amp = None
    try:
        from apex import optimizers as apexopt
    except:
        pass
    
    train(args, args.rank, args.group_name, h)

