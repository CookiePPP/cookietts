import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, teacher_force_till, p_teacher_forcing, diagonality=None, avg_prob=None):
        if diagonality is not None:
            self.add_scalar("training.attention_alignment_diagonality", diagonality, iteration)
        if avg_prob is not None:
            self.add_scalar("training.average_max_attention_weight", avg_prob, iteration)
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("training.p_teacher_forcing", p_teacher_forcing, iteration)
        self.add_scalar("training.teacher_force_till", teacher_force_till, iteration)
        self.add_scalar("duration", duration, iteration)
    
    def log_validation(self, reduced_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        self.add_scalar("validation.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("validation.average_max_attention_weight", avg_prob, iteration)
        self.add_scalar("validation.p_teacher_forcing", val_p_teacher_forcing, iteration)
        self.add_scalar("validation.teacher_force_till", val_teacher_force_till, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets, *_ = y
    
        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        # plot alignment, mel target and predicted, gate target and predicted
        idx = 0 # plot longest audio file
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        
        idx = 1 # and randomly pick a second one to plot
        self.add_image(
            "alignment2",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
    
    def log_infer(self, reduced_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob):
        self.add_scalar("infer.loss", reduced_loss, iteration)
        self.add_scalar("infer.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("infer.average_max_attention_weight", avg_prob, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets, *_ = y
        
        # plot alignment, mel target and predicted, gate target and predicted
        idx = 0 # plot longest audio file
        self.add_image(
            "infer_alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "infer_mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "infer_mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "infer_gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        idx = 1 # and plot 2nd longest audio file
        self.add_image(
            "infer_alignment2",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "infer_mel_target2",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "infer_mel_predicted2",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "infer_gate2",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
    
    def log_teacher_forced_validation(self, reduced_loss, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing, diagonality, avg_prob):
        self.add_scalar("teacher_forced_validation.loss", reduced_loss, iteration)
        self.add_scalar("teacher_forced_validation.attention_alignment_diagonality", diagonality, iteration)
        self.add_scalar("teacher_forced_validation.average_max_attention_weight", avg_prob, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets, *_ = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        
        # plot alignment, mel target and predicted, gate target and predicted
        idx = 0 # plot longest audio file
        self.add_image(
            "teacher_forced_alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        idx = 1 # and plot 2nd longest audio file
        self.add_image(
            "teacher_forced_alignment2",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target2",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted2",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate2",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
