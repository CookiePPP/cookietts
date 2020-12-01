import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from CookieTTS._4_mtw.hifigan.env import AttrDict, build_env
from CookieTTS._4_mtw.hifigan.meldataset import MelDataset, get_dataset_filelist
from CookieTTS._4_mtw.hifigan.nvSTFT import STFT as STFT_Class
from CookieTTS._4_mtw.hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, del_old_checkpoints, save_checkpoint
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)
    
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))
    
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator(h["discriminator_periods"] if "discriminator_periods" in h.keys() else None).to(device)
    msd = MultiScaleDiscriminator().to(device)
    
    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        gsd = generator.state_dict()
        gsd.update({k: v for k,v in state_dict_g['generator'].items() if k in gsd and state_dict_g['generator'][k].shape == gsd[k].shape})
        missing_keys = {k: v for k,v in state_dict_g['generator'].items() if not (k in gsd and state_dict_g['generator'][k].shape == gsd[k].shape)}.keys()
        generator.load_state_dict(gsd)
        del gsd, state_dict_g
    
    if cp_do is None or len(missing_keys) or a.from_zero:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_do = load_checkpoint(cp_do, device)
        mpd.load_state_dict(state_dict_do['mpd'])
        del state_dict_do['mpd']
        msd.load_state_dict(state_dict_do['msd'])
        del state_dict_do['msd']
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        del state_dict_do
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a, h.segment_size, h.sampling_rate)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, trim_non_voiced=a.trim_non_voiced)
    
    STFT = STFT_Class(h.sampling_rate, h.num_mels, h.n_fft, h.win_size, h.hop_size, h.fmin, h.fmax)
    
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    assert len(train_loader), 'No audio files in dataset!'
    
    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              trim_non_voiced=a.trim_non_voiced)
        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)
        
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'), max_queue=10000)
        sw.logged_gt_plots = False
    
    if h.num_gpus > 1:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = STFT.get_mel(y_g_hat.squeeze(1))
            
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                torch.set_grad_enabled(False)
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, loss_mel.item(), time.time() - start_b))
                
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})
                    del_old_checkpoints(a.checkpoint_path, 'g_' , a.n_models_to_keep)
                    del_old_checkpoints(a.checkpoint_path, 'do_', a.n_models_to_keep)
                
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", loss_mel.item(), steps)
                
                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    print("Validating...")
                    n_audios_to_plot = 6
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    for j, batch in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                        x, y, _, y_mel = batch
                        y_g_hat = generator(x.to(device))
                        y_hat_spec = STFT.get_mel(y_g_hat.squeeze(1))
                        val_err_tot += F.l1_loss(y_mel, y_hat_spec.to(y_mel)).item() 
                        
                        if j < n_audios_to_plot and not sw.logged_gt_plots:
                            sw.add_audio(f'gt/y_{j}', y[0], steps, h.sampling_rate)
                            sw.add_figure(f'spec_{j:02}/gt_spec', plot_spectrogram(y_mel[0]), steps)
                        if j < n_audios_to_plot:
                            sw.add_audio(f'generated/y_hat_{j}', y_g_hat[0], steps, h.sampling_rate)
                            sw.add_figure(f'spec_{j:02}/pred_spec', plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)
                        
                        if j > 64:# I am NOT patient enough to complete an entire validation cycle with over 1536 files.
                            break
                    sw.logged_gt_plots = True
                    val_err = val_err_tot / (j+1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    generator.train()
                    print(f"Done. Val_loss = {val_err}")
                torch.set_grad_enabled(True)
            steps += 1
        
        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--group_name',       default=None)
    parser.add_argument('--input_wavs_dir',   default=None)
    parser.add_argument('--input_training_file',   default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path',  default='cp_hifigan')
    parser.add_argument('--config',           default='')
    parser.add_argument('--training_epochs',  default=3100, type=int)
    parser.add_argument('--stdout_interval',  default=5, type=int)# how often to print to terminal
    parser.add_argument('--validation_interval', default=1000, type=int)# how often to test the model on the val set
    parser.add_argument('--checkpoint_interval', default=5000, type=int)# how often to save the model state to disk
    parser.add_argument('--n_models_to_keep', default=2,  type=int)# how many copies of the model can be saved to the disk at the same time, old checkpoints will be replaced with new checkpoints when the limit is reach.
    parser.add_argument('--summary_interval', default=20, type=int)
    parser.add_argument('--skip_file_checks', action='store_true')
    parser.add_argument('--trim_non_voiced',  action='store_true')# trim start/end of audio where pitch is 0
    parser.add_argument('--fine_tuning',      action='store_true')# load predicted spectrograms as inputs, will increase audio quality of generated samples when using in end-to-end fashion.
    parser.add_argument('--from_zero',      action='store_true')# when loading checkpoints, reset the iteration and epoch counters.
    
    a = parser.parse_args()
    
    assert a.config, '--config not specified!'
    
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
