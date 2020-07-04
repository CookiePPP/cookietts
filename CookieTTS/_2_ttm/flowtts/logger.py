import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
    
    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        z, alignments, pred_output_lengths, log_s_sum, logdet_w_sum = y_pred
        mel_targets, *_ = y
        
        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        # plot alignment, mel target and predicted, gate target and predicted
        for head_i in range(alignments.shape[1]):
            idx = 0 # plot longest audio file
            self.add_image(
                f"alignment1/h{head_i}",
                plot_alignment_to_numpy(alignments[idx][head_i].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            
            if alignments.shape[0] > 1: # if batch_size > 1...
                idx = 1 # pick a second plot
                self.add_image(
                    f"alignment2/h{head_i}",
                    plot_alignment_to_numpy(alignments[idx][head_i].data.cpu().numpy().T),
                    iteration, dataformats='HWC')