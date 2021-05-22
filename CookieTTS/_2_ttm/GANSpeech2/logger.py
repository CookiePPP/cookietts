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
        
        if 'pred_melf0s' in y_pred:
            g_pred_mel, g_pred_f0s = y_pred['pred_melf0s'][:, :y['gt_mel'].shape[1]], y_pred['pred_melf0s'][:, y['gt_mel'].shape[1]:]
            g_gt_mel,     g_gt_f0s = y_pred[  'gt_melf0s'][:, :y['gt_mel'].shape[1]], y_pred[  'gt_melf0s'][:, y['gt_mel'].shape[1]:]
        
        alignment = y.get('alignment', None)
        if alignment is None:
            alignment = y_pred['hard_alignments']
        gt_char_dur = alignment.detach().sum(dim=1)
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        mel_lengths_cpu = y['mel_lengths'].cpu()
        is_len_changed = (self.plotted_targets_val_lens is None) or (mel_lengths_cpu != self.plotted_targets_val_lens).any()
        if is_len_changed:
            self.plotted_targets_val_lens = mel_lengths_cpu
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            # HiFi-GAN plots
            mel_len = y['mel_lengths'][idx].item()
            txt_len = y['text_lengths'][idx].item()
            dec_len = y['mel_lengths'][idx].item()//self.n_frames_per_step
            
            mag_range = [y['gt_mel'][idx].min().item(), y['gt_mel'][idx].max().item()]
            self.add_image(f"{prepend}_{idx}/mel_gt",
                plot_spectrogram_to_numpy(y['gt_mel'][idx, :, :mel_len].data.float().cpu().numpy(), range=mag_range),
                iteration, dataformats='HWC')
            
            if 'pred_mel' in y_pred:
                self.add_image(f"{prepend}_{idx}/mel_pred",
                    plot_spectrogram_to_numpy(y_pred['pred_mel'][idx, :, :mel_len].data.float().cpu().numpy(), range=mag_range),
                    iteration, dataformats='HWC')
            
            if 'pred_mel_a' in y_pred:
                self.add_image(f"{prepend}_{idx}/mel_pred_ga",
                    plot_spectrogram_to_numpy(y_pred['pred_mel_a'][idx, :, :mel_len].data.float().cpu().numpy(), range=mag_range),
                    iteration, dataformats='HWC')
            
            if 'soft_alignments' in y_pred:
                self.add_image(f"{prepend}_{idx}/ga_alignment_soft",
                    plot_alignment_to_numpy(y_pred['soft_alignments'][idx, :dec_len, :txt_len].data.float().cpu().numpy().T),
                    iteration, dataformats='HWC')
            
            if 'hard_alignments' in y_pred:
                self.add_image(f"{prepend}_{idx}/ga_alignment_hard",
                    plot_alignment_to_numpy(y_pred['hard_alignments'][idx, :dec_len, :txt_len].data.float().cpu().numpy().T),
                    iteration, dataformats='HWC')
            
            # Pred Char Durations
            if 'pred_dur' in y_pred:
                pred_dur = y_pred['pred_dur'][idx, 0, :txt_len].data.cpu()# -> [T]
                gt_dur   = gt_char_dur       [idx,    :txt_len].data.cpu()# -> [T]
                pred_dur[pred_dur<0.5] = float('nan')
                gt_dur  [  gt_dur<0.5] = float('nan')
                self.add_image(f"{prepend}_{idx}/g_char_dur",
                    plot_time_series_to_numpy(gt_dur.numpy(), pred_dur.numpy(), ylabel="Frames", xlabel="Chars (Green Target, Red predicted)"),
                    iteration, dataformats='HWC')
            
            # Pred Char Log F0
            if 'pred_char_logf0' in y_pred:
                pred_f0s = y_pred['pred_char_logf0'][idx, 0, :txt_len].data.cpu()# -> [T]
                gt_f0s   =      y[  'gt_char_logf0'][idx,    :txt_len].data.cpu()# -> [T]
                pred_f0s[y_pred['pred_char_voiced'][idx, 0, :txt_len].data.cpu()<=0.01] = float('nan')
                gt_f0s  [     y[  'gt_char_voiced'][idx,    :txt_len].data.cpu()<=0.01] = float('nan')
                pred_f0s[pred_f0s==0.0] = float('nan')
                gt_f0s  [  gt_f0s==0.0] = float('nan')
                self.add_image(f"{prepend}_{idx}/g_char_f0",
                    plot_time_series_to_numpy(gt_f0s.numpy(), pred_f0s.numpy(), ylabel="Log Hz", xlabel="Chars (Green Target, Red predicted)"),
                    iteration, dataformats='HWC')
            
            # Pred Char Voiced Logits
            if 'pred_char_voiced' in y_pred:
                pred_vo = y_pred['pred_char_voiced'][idx, 0, :txt_len].data.cpu()# -> [T]
                gt_vo   =      y[  'gt_char_voiced'][idx,    :txt_len].data.cpu()# -> [T]
                self.add_image(f"{prepend}_{idx}/g_char_voiced",
                    plot_time_series_to_numpy(gt_vo.numpy(), pred_vo.numpy(), ylabel="Prob", xlabel="Chars (Green Target, Red predicted)"),
                    iteration, dataformats='HWC')
            
            # Pred Frame Log F0s
            if 'pred_logf0s' in y_pred:
                pred_f0s = y_pred['pred_logf0s'    ][idx, 1, :mel_len].data.cpu()# -> [T]
                gt_f0s   =      y['gt_frame_logf0s'][idx, 1, :mel_len].data.cpu()# -> [T]
                pred_f0s[y['gt_frame_voiceds'][idx, 1, :mel_len].data.cpu()<=0.5] = float('nan')
                gt_f0s  [y['gt_frame_voiceds'][idx, 1, :mel_len].data.cpu()<=0.5] = float('nan')
                self.add_image(f"{prepend}_{idx}/g_frame_f0s",
                    plot_time_series_to_numpy(gt_f0s.numpy(), pred_f0s.numpy(), ylabel="Log Hz", xlabel="Frames (Green Target, Red predicted)"),
                    iteration, dataformats='HWC')
            
            # Pred Frame Voiced Logits
            if 'pred_voiced' in y_pred:
                pred_vo = y_pred['pred_voiced'     ][idx, 1, :mel_len].data.cpu()# -> [T]
                gt_vo   =      y['gt_frame_voiceds'][idx, 1, :mel_len].data.cpu()# -> [T]
                self.add_image(f"{prepend}_{idx}/g_frame_voiced",
                    plot_time_series_to_numpy(gt_vo.numpy(), pred_vo.numpy(), ylabel="Prob", xlabel="Frames (Green Target, Red predicted)"),
                    iteration, dataformats='HWC')