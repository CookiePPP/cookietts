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
            
            self.add_scalar("misc/grad.norm", grad_norm, iteration)
            self.add_scalar(f"misc/{prepend}.learning_rate", learning_rate, iteration)
            self.add_scalar(f"misc/{prepend}.duration",      duration,      iteration)
    
    def log_validation(self, reduced_loss_dict, reduced_bestval_loss_dict, model, y, y_pred, iteration, prepend='validation'):
        # plot datapoints/graphs
        self.plot_loss_dict(reduced_loss_dict,         iteration, f'{prepend}')
        self.plot_loss_dict(reduced_bestval_loss_dict, iteration, f'{prepend}_best')