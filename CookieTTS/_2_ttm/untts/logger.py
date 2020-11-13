import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, hparams):
        super(Tacotron2Logger, self).__init__(logdir)
        self.n_items = hparams.n_tensorboard_outputs
        self.plotted_targets = False
        self.best_loss_dict = None
    
    def log_training(self, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration,
                     iteration):
        for loss_name, reduced_loss in reduced_loss_dict.items():
            self.add_scalar(f"training/{loss_name}", reduced_loss, iteration)
        
        if iteration%50 == 0:
            if expavg_loss_dict is not None:
                for loss_name, reduced_loss in expavg_loss_dict.items():
                    self.add_scalar(f"training_smoothed/{loss_name}", reduced_loss, iteration)
            
            if best_loss_dict is not None:
                if self.best_loss_dict is None:
                    self.best_loss_dict = {k: 0. for k in best_loss_dict.keys()}
                
                for loss_name, reduced_loss in best_loss_dict.items():# for each loss value in the dictionary
                    if self.best_loss_dict[loss_name] != reduced_loss or iteration%10000 == 0:# if loss has updated or changed since last time
                        self.best_loss_dict[loss_name] = reduced_loss
                        self.add_scalar(f"training_smoothed_best/{loss_name}", reduced_loss, iteration)# plot the new value
        
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration):
        # plot distribution of parameters
        if iteration%5000 == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        # plot datapoints/graphs
        for loss_name, reduced_loss in reduced_loss_dict.items():
            self.add_scalar(f"validation/{loss_name}", reduced_loss, iteration)
        
        for loss_name, reduced_loss in reduced_bestval_loss_dict.items():
            self.add_scalar(f"validation_best/{loss_name}", reduced_loss, iteration)
        
        # pickup predicted model outputs
        melglow_package, durglow_package, varglow_package, *_ = y_pred
        mel_targets, *_ = y
        
        # plot spects / imgs
        n_items = min(self.n_items, mel_targets.shape[0])
        
        if not self.plotted_targets:
            for idx in range(n_items): # plot target spectrogram of longest audio file(s)
                self.add_image(
                    f"mel_target/{idx}",
                    plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
            self.plotted_targets = True # target spect doesn't change so only needs to be plotted once.