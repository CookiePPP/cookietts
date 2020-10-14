import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, sample_rate):
        super(Tacotron2Logger, self).__init__(logdir)
        self.sr = sample_rate

    def log_training(self, iteration, reduced_model_loss, reduced_discriminator_loss, grad_norm, grad_norm_d, learning_rate, duration):
        self.add_scalar("g_training.loss", reduced_model_loss, iteration)
        self.add_scalar("g_grad.norm", grad_norm, iteration)
        self.add_scalar("d_training.loss", reduced_discriminator_loss, iteration)
        self.add_scalar("g_grad.norm", grad_norm_d, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
    
    def log_validation(self, val_loss, model, x, pred_audio, iteration):
        self.add_scalar("validation.loss", val_loss, iteration)
        
        for idx in range(2):
            if iteration%10000 == 0:
                self.add_audio(f"real_audio_{str(idx)}", x[0][idx], iteration, sample_rate=self.sr)
            self.add_audio(f"pred_audio_{str(idx)}", pred_audio[idx], iteration, sample_rate=self.sr)