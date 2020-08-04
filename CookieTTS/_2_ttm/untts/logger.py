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
        
    def log_training(self, reduced_loss, reduced_len_loss, reduced_loss_z, reduced_loss_w, reduced_loss_s, reduced_loss_att, reduced_dur_loss_z, reduced_dur_loss_w, reduced_dur_loss_s, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training/loss", reduced_loss, iteration)
        self.add_scalar("training/loss_len", reduced_len_loss, iteration)
        self.add_scalar("training/loss_z", reduced_loss_z, iteration)
        self.add_scalar("training/loss_w", reduced_loss_w, iteration)
        self.add_scalar("training/loss_s", reduced_loss_s, iteration)
        self.add_scalar("training/loss_att", reduced_loss_att, iteration)
        self.add_scalar("training/loss_d_z", reduced_dur_loss_z, iteration)
        self.add_scalar("training/loss_d_w", reduced_dur_loss_w, iteration)
        self.add_scalar("training/loss_d_s", reduced_dur_loss_s, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
        
    def log_validation(self, reduced_loss, len_loss, loss_z, loss_w, loss_s, loss_att, dur_loss_z, dur_loss_w, dur_loss_s, model, y, y_pred, iteration):
        self.add_scalar("validation/loss", reduced_loss, iteration)
        self.add_scalar("validation/loss_len", len_loss, iteration)
        self.add_scalar("validation/loss_decoder", (loss_z+loss_w+loss_s), iteration)
        self.add_scalar("validation/loss_z", loss_z, iteration)
        self.add_scalar("validation/loss_w", loss_w, iteration)
        self.add_scalar("validation/loss_s", loss_s, iteration)
        self.add_scalar("validation/loss_att", loss_att, iteration)
        self.add_scalar("validation/loss_d_z", dur_loss_z, iteration)
        self.add_scalar("validation/loss_d_w", dur_loss_w, iteration)
        self.add_scalar("validation/loss_d_s", dur_loss_s, iteration)
        mel_out, alignments, log_s_sum, logdet_w_sum, pred_output_lengths, pred_output_lengths_std, dur_z, dur_log_s_sum, dur_logdet_w_sum, len_pred_attention, *_ = y_pred
        mel_targets, *_ = y
        
        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        n_items = min(self.n_items, mel_targets.shape[0])
        
        # plot alignment, mel target and predicted, gate target and predicted
        if len(alignments.shape) == 4:
            for idx in range(n_items): # plot longest audio file(s)
                for layer in range(alignments.shape[1]):
                    self.add_image(
                        f"alignment/{layer}_{idx}",
                        plot_alignment_to_numpy(alignments[idx, layer].data.cpu().numpy().T),
                        iteration, dataformats='HWC')
        else:
            for idx in range(n_items): # plot longest audio file(s)
                self.add_image(
                    f"alignment/{idx}",
                    plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                    iteration, dataformats='HWC')
        
        if not self.plotted_targets:
            for idx in range(n_items): # plot target spectrogram of longest audio file(s)
                self.add_image(
                    f"mel_target/{idx}",
                    plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
            self.plotted_targets = True # target spect doesn't change so only needs to be plotted once.