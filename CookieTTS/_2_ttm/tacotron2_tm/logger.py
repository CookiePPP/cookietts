import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, hparams):
        super(Tacotron2Logger, self).__init__(logdir)
        self.n_items = hparams.n_tensorboard_outputs
        self.plotted_targets_val = False# validation
        self.plotted_targets_tf = False # teacher forcing
        self.plotted_targets_inf = False# infer
        self.best_loss_dict = None
    
    def plot_loss_dict(self, loss_dict, iteration, prepend=''):
        # plot datapoints/graphs
        for loss_name, reduced_loss in loss_dict.items():
            self.add_scalar(f"{prepend}/{loss_name}", reduced_loss, iteration)
    
    def plot_model_params(self, model, iteration):
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
    
    def log_training(self, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration,
                     iteration, teacher_force_till, p_teacher_forcing, drop_frame_rate):
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
        
        if iteration%100 == 0:
            self.add_scalar(f"{prepend}.learning_rate", learning_rate, iteration)
            self.add_scalar(f"{prepend}/p_teacher_forcing", p_teacher_forcing, iteration)
            self.add_scalar(f"{prepend}/teacher_force_till", teacher_force_till, iteration)
            self.add_scalar(f"{prepend}/drop_frame_rate", p_teacher_forcing, iteration)
            self.add_scalar(f"{prepend}.duration", duration, iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing):
        prepend = 'validation'
        
        # plot distribution of parameters
        if iteration%20000 == 0:
            self.plot_model_params(model, iteration)
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(
                f"{prepend}_alignment/{idx}",
                plot_alignment_to_numpy(y_pred['alignments'][idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f"{prepend}_mel_pred/{idx}",
                plot_spectrogram_to_numpy(y_pred['pred_mel_postnet'][idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if not self.plotted_targets_val:
                self.add_image(
                    f"{prepend}_mel_gt/{idx}",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
        self.plotted_targets_val = True # target spect doesn't change so only needs to be plotted once.
    
    def log_infer(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing):
        prepend = 'inference'
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(
                f"{prepend}_alignment/{idx}",
                plot_alignment_to_numpy(y_pred['alignments'][idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f"{prepend}_mel_pred/{idx}",
                plot_spectrogram_to_numpy(y_pred['pred_mel_postnet'][idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if not self.plotted_targets_inf:
                self.add_image(
                    f"{prepend}_mel_gt/{idx}",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
        self.plotted_targets_inf = True # target spect doesn't change so only needs to be plotted once.
    
    def log_teacher_forced_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing):
        prepend = 'teacher_forced'
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        mel_L1_map = torch.nn.L1Loss(reduction='none')(y_pred['pred_mel_postnet'], y['gt_mel'])
        mel_L1_map[:, -1, -1] = 10.0 # because otherwise the color map scale is crap
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(
                f"{prepend}_alignment/{idx}",
                plot_alignment_to_numpy(y_pred['alignments'][idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f"{prepend}_mel_pred/{idx}",
                plot_spectrogram_to_numpy(y_pred['pred_mel_postnet'][idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(
                f"{prepend}_mel_SE/{idx}",
                plot_spectrogram_to_numpy(mel_L1_map[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if not self.plotted_targets_tf:
                self.add_image(
                    f"{prepend}_mel_gt/{idx}",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
        self.plotted_targets_tf = True # target spect doesn't change so only needs to be plotted once.