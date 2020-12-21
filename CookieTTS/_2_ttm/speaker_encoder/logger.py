import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class AutoVCLogger(SummaryWriter):
    def __init__(self, logdir, hparams):
        super(AutoVCLogger, self).__init__(logdir)
        self.n_items = hparams.n_tensorboard_outputs
        self.plotted_targets_val = False
    
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
        if iteration==1 or (iteration%500 == 0 and iteration > 1 and iteration < 4000) or (iteration%5000 == 0 and iteration > 4999 and iteration < 50000) or (iteration%25000 == 0 and iteration > 49999):
            print("Plotting Params. This may take open a bit.")
            self.plot_model_params(model, iteration)
        
        prepend = 'training'
        
        if iteration%20 == 0:
            self.plot_loss_dict(reduced_loss_dict, iteration, f'{prepend}')
            
            if expavg_loss_dict is not None:
                self.plot_loss_dict(expavg_loss_dict, iteration, f'{prepend}_smoothed')
        
            self.add_scalar("grad.norm", grad_norm, iteration)
        
        if iteration%100 == 0:
            self.add_scalar(f"{prepend}.learning_rate",      learning_rate,      iteration)
            self.add_scalar(f"{prepend}.duration",           duration,           iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration,):
        prepend = 'validation'
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict, iteration, f'{prepend}')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(
                f"{prepend}_{idx}/mel_pred",
                plot_spectrogram_to_numpy(y_pred['pred_mel_postnet'][idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if self.plotted_targets_val < 2:
                self.add_image(
                    f"{prepend}_{idx}/mel_gt",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
        self.plotted_targets_val +=1 # target spect doesn't change so only needs to be plotted once.
