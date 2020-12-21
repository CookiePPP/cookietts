import random
import torch
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

class AlignTTSLogger(SummaryWriter):
    def __init__(self, logdir, hparams):
        super(AlignTTSLogger, self).__init__(logdir)
        self.n_items = hparams.n_tensorboard_outputs
        self.plotted_targets_val = False# validation/teacher-forcing
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
    
    def log_training(self, model, reduced_loss_dict, expavg_loss_dict, best_loss_dict, grad_norm, learning_rate, duration,
                     iteration):
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
            self.add_scalar(f"{prepend}.duration",           duration,           iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, align, iteration):
        prepend = 'validation'
        
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')
        
        # plot spects / imgs
        n_items = min(self.n_items, y['gt_mel'].shape[0])
        
        # get pred_spect
        with torch.no_grad():
            align = align.transpose(1, 2).float()# [B, txt_T, mel_T] -> [B, mel_T, txt_T]
            char_pred_spect = y_pred['mu_logvar'].chunk(2, dim=-1)[0].cpu().float()# [B, txt_T, n_mel]
            pred_mel = align[:n_items] @ char_pred_spect[:n_items]# [B, mel_T, txt_T] @ [B, txt_T, n_mel] -> [B, mel_T, n_mel]
        
        for idx in range(n_items):# plot target spectrogram of longest audio file(s)
            self.add_image(
                f"{prepend}_{idx}/alignment",
                plot_alignment_to_numpy(align[idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            self.add_image(
                f"{prepend}_{idx}/mel_pred",
                plot_spectrogram_to_numpy(pred_mel[idx].data.cpu().numpy().T),
                iteration, dataformats='HWC')
            if self.plotted_targets_val < 2:
                self.add_image(
                    f"{prepend}_{idx}/mel_gt",
                    plot_spectrogram_to_numpy(y['gt_mel'][idx].data.cpu().numpy()),
                    iteration, dataformats='HWC')
        self.plotted_targets_val +=1 # target spect doesn't change so only needs to be plotted once.