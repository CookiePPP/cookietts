import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir, hparams):
        super(Tacotron2Logger, self).__init__(logdir)
        self.n_items = hparams.n_tensorboard_outputs
        self.plotted_targets_val = False# validation/teacher-forcing
        self.plotted_targets_val_lens = None
        self.plotted_targets_inf = False# infer
        self.plotted_targets_inf_lens = None
        self.best_loss_dict = None
    
    def plot_loss_dict(self, loss_dict, iteration, prepend=''):
        # plot datapoints/graphs
        for loss_name, reduced_loss in loss_dict.items():
            self.add_scalar(f"{prepend}/{loss_name}", reduced_loss, iteration)
    
    def plot_model_params(self, model, iteration):
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
    
    def log_training(self, model, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration,
                     iteration, teacher_force_till, p_teacher_forcing, drop_frame_rate):
        # plot distribution of parameters
        if iteration==1 or (iteration%500 == 0 and iteration > 1 and iteration < 4000) or (iteration%5000 == 0 and iteration > 4999 and iteration < 50000) or (iteration%25000 == 0 and iteration > 49999):
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
        
        if iteration%100 == 0:
            self.add_scalar(f"{prepend}.learning_rate",      learning_rate,      iteration)
            self.add_scalar(f"{prepend}/p_teacher_forcing" , p_teacher_forcing,  iteration)
            self.add_scalar(f"{prepend}/teacher_force_till", teacher_force_till, iteration)
            self.add_scalar(f"{prepend}/drop_frame_rate",    drop_frame_rate,    iteration)
            self.add_scalar(f"{prepend}.duration",           duration,           iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing):
        prepend = 'validation'
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        mel_L1_map = torch.nn.L1Loss(reduction='none')(y_pred['pred_mel_postnet'], y['gt_mel'])
        mel_L1_map[:, -1, -1] = 5.0 # because otherwise the color map scale is crap
        
        mel_lengths_cpu = y['mel_lengths'].cpu()
        is_len_changed = (self.plotted_targets_val_lens is None) or (mel_lengths_cpu != self.plotted_targets_val_lens).any()
        if is_len_changed:
            self.plotted_targets_val_lens = mel_lengths_cpu
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(f"{prepend}_{idx}/alignment",
                plot_alignment_to_numpy(y_pred['alignments'][idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(f"{prepend}_{idx}/mel_pred",
                plot_spectrogram_to_numpy(y_pred['pred_mel'][idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            self.add_image(f"{prepend}_{idx}/mel_SE",
                plot_spectrogram_to_numpy(mel_L1_map[idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if self.plotted_targets_val < 2 or is_len_changed:
                self.add_image(f"{prepend}_{idx}/mel_gt",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
                self.plotted_targets_val +=1 # target spect doesn't change so only needs to be plotted once.
            if 'hifigan_gt_mel' in y and len(y['hifigan_gt_mel']) > idx:
                self.add_image(f"{prepend}_{idx}/hifi_mel_gt",
                    plot_spectrogram_to_numpy(y['hifigan_gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
            if 'hifigan_pred_mel' in y_pred and len(y_pred['hifigan_pred_mel']) > idx:
                self.add_image(f"{prepend}_{idx}/hifi_mel_pred",
                    plot_spectrogram_to_numpy(y_pred['hifigan_pred_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
    
    def log_infer(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, val_teacher_force_till, val_p_teacher_forcing):
        prepend = 'inference'
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        mel_lengths_cpu = y['mel_lengths'].cpu()
        is_len_changed = (self.plotted_targets_inf_lens is None) or (mel_lengths_cpu != self.plotted_targets_inf_lens).any()
        if is_len_changed:
            self.plotted_targets_inf_lens = mel_lengths_cpu
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(f"{prepend}_{idx}/alignment",
                plot_alignment_to_numpy(y_pred['alignments'][idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(f"{prepend}_{idx}/mel_pred",
                plot_spectrogram_to_numpy(y_pred['pred_mel'][idx].data.cpu().numpy()),
                iteration, dataformats='HWC')
            if self.plotted_targets_inf < 2 or is_len_changed:
                self.add_image(f"{prepend}_{idx}/mel_gt",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
                self.plotted_targets_inf +=1 # target spect doesn't change so only needs to be plotted ~~once~~ a couple times.
