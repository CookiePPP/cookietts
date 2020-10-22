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
        
    def log_training(self, reduced_loss_dict, grad_norm, learning_rate, duration,
                     iteration):
        for loss_name, reduced_loss in reduced_loss_dict.items():
            self.add_scalar(f"training/{loss_name}", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
    
    def log_validation(self, reduced_loss_dict, model, y, y_pred, iteration):
        # plot distribution of parameters
        if iteration%5000 == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        # plot datapoints/graphs
        for loss_name, reduced_loss in reduced_loss_dict.items():
            self.add_scalar(f"validation/{loss_name}", reduced_loss, iteration)
        
        # pickup predicted model outputs
        melglow_package, durglow_package, varglow_package = y_pred
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