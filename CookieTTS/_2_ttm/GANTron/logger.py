import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_time_series_to_numpy

class Logger(SummaryWriter):
    def __init__(self, logdir, h):
        super(Logger, self).__init__(logdir)
        self.sampling_rate = h.sampling_rate
        self.n_frames_per_step = h.n_frames_per_step
        self.n_items = h.n_tensorboard_outputs
        self.plotted_targets_val = False# validation/teacher-forcing
        self.plotted_targets_val_lens = None
        self.best_loss_dict = None
    
    def plot_loss_dict(self, loss_dict, iteration, prepend=''):
        # plot datapoints/graphs
        for loss_name, reduced_loss in loss_dict.items():
            self.add_scalar(f"{prepend}/{loss_name}", reduced_loss, iteration)
    
    def plot_model_params(self, model, iteration):
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
    
    def log_training(self, model, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration, iteration):
        # plot distribution of parameters
        if (iteration==1 or
            (iteration%1000 == 0)
            #(iteration%  500 == 0 and iteration >=     0 and iteration <=  5000) or
            #(iteration% 5000 == 0 and iteration >=  5000 and iteration <= 50000) or
            #(iteration%25000 == 0 and iteration >= 50000)
           ):
            print("Plotting Params. This may take open a bit.")
            self.plot_model_params(model, iteration)
        
        prepend = 'training'
        
        if iteration%20 == 0:
            self.plot_loss_dict(reduced_loss_dict, iteration, f'{prepend}')
            
            if expavg_loss_dict is not None:
                self.plot_loss_dict(expavg_loss_dict, iteration, f'{prepend}_smoothed')
            
            if best_loss_dict is not None:
                if self.best_loss_dict is None:
                    self.best_loss_dict = {k: 0. for k in best_loss_dict.keys()}
                
                for loss_name, reduced_loss in best_loss_dict.items():# for each loss value in the dictionary
                    if self.best_loss_dict[loss_name] != reduced_loss or iteration%10000 == 0:# if loss has updated or changed since last time
                        self.best_loss_dict[loss_name] = reduced_loss
                        self.add_scalar(f'{prepend}_smoothed_best/{loss_name}', reduced_loss, iteration)# plot the new value
        
            self.add_scalar("grad.norm", grad_norm, iteration)
        
        if iteration%25 == 0:
            self.add_scalar(f"{prepend}.learning_rate", learning_rate, iteration)
            self.add_scalar(f"{prepend}.duration",      duration,      iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, prepend='validation'):
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        mel_lengths_cpu = y['mel_lengths'].cpu()
        is_len_changed = (self.plotted_targets_val_lens is None) or (mel_lengths_cpu != self.plotted_targets_val_lens).any()
        if is_len_changed:
            self.plotted_targets_val_lens = mel_lengths_cpu
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            # HiFi-GAN plots
            mel_len   = y['mel_lengths'][idx].item()
            txt_len   = y['text_lengths'][idx].item()
            align_len = y['mel_lengths'][idx].item()//self.n_frames_per_step
            
            mag_range = [y['gt_mel'][idx].min().item(), y['gt_mel'][idx].max().item()]
            self.add_image(f"{prepend}_{idx}/mel_gt",
                plot_spectrogram_to_numpy(y['gt_mel'][idx, :, :mel_len].data.float().cpu().numpy(), range=mag_range),
                iteration, dataformats='HWC')
            
            self.add_image(f"{prepend}_{idx}/mel_pred",
                plot_spectrogram_to_numpy(y_pred['pred_mel'][idx, :, :mel_len].data.float().cpu().numpy(), range=mag_range),
                iteration, dataformats='HWC')
            
            self.add_image(f"{prepend}_{idx}/alignment",
                plot_alignment_to_numpy(y_pred['alignments'][idx, :align_len, :txt_len].data.float().cpu().numpy().T),
                iteration, dataformats='HWC')
            
            if 'alignments_d' in y_pred:
                self.add_image(f"{prepend}_{idx}/alignment_d",
                    plot_alignment_to_numpy(y_pred['alignments_d'][idx, :align_len, :txt_len].data.float().cpu().numpy().T),
                    iteration, dataformats='HWC')