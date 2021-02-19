current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
# expavg_loss_dict # smoothed loss dict
############################################################################################
##                                                                                        ##
##   ████████╗ █████╗  ██████╗ ██████╗ ███████╗██████╗ ███████╗███████╗ ██████╗██╗  ██╗   ##
##   ╚══██╔══╝██╔══██╗██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║  ██║   ##
##      ██║   ███████║██║     ██║   ██║███████╗██████╔╝█████╗  █████╗  ██║     ███████║   ##
##      ██║   ██╔══██║██║     ██║   ██║╚════██║██╔═══╝ ██╔══╝  ██╔══╝  ██║     ██╔══██║   ##
##      ██║   ██║  ██║╚██████╗╚██████╔╝███████║██║     ███████╗███████╗╚██████╗██║  ██║   ##
##      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚══════╝╚══════╝ ╚═════╝╚═╝  ╚═╝   ##
##                                                                                        ##
##                                                                                        ##
############################################################################################
## VDVAETTS ##
################
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e5 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

# Learning Rate / Optimization
decay_start = 350000
A_ = 1.0e-4
B_ = 20000
C_ = 0e-5
min_learning_rate = 1e-11
grad_clip_thresh  = 10.0 if iteration > 12500 else 40.0
warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*1e6 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval =  125 if iteration < 2000 else (500 if iteration < 8000 else 500)
checkpoint_interval = 1000

# Loss Scalars (set to None to load from hparams.py)
decoder_MAE_weight = 0.00
decoder_MSE_weight = 1.00
decoder_KLD_weight = 0.12

margin = 0#param_interval
about_to_val = (iteration-1)%validation_interval >= validation_interval-margin-1
kld_iteration = iteration
if iteration < 50000 and not about_to_val:
    decoder_KLD_weight *= min(kld_iteration%5000, 2000)/2000# cyclic annealing. (at 0 iters KLD_scale==0.0), (at 2000 iters KLD_scale==1.0), (at 5000 iters KLD_scale=1.0)
                                                      # see https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Annealing-with-the-monotonic-schedule.png

# Pitch Postnet
postnet_f0_MAE_weight     = 0.00# prior outputs
postnet_f0_MSE_weight     = 1.00# prior outputs
postnet_voiced_MAE_weight = 0.00# prior outputs
postnet_voiced_BCE_weight = 1.00# prior outputs

postnet_MAE_weight        = 0.00# decoder outputs
postnet_MSE_weight        = 1.00# decoder outputs

postnet_KLD_weight        = 0.10# [prior <-> encoder] similarity/link
if iteration < 50000 and not about_to_val:
    postnet_KLD_weight *= min(kld_iteration%5000, 2000)/2000# cyclic annealing. (at 0 iters KLD_scale==0.0), (at 2000 iters KLD_scale==1.0), (at 5000 iters KLD_scale=1.0)
                                                            # see https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Annealing-with-the-monotonic-schedule.png

# Varpred (Variance Predictor)
varpred_MAE_weight = 0.00
varpred_MSE_weight = 1.00
varpred_KLD_weight = 1.00
if iteration < 50000 and not about_to_val:
    varpred_KLD_weight *= min(kld_iteration%5000, 2000)/2000# cyclic annealing. at 0 iters, KLD_scale==0.0. at 2500 iters, KLD_scale==1.0, at 2000 iters, KLD_scale=1.0
                                                            # see https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Annealing-with-the-monotonic-schedule.png

# MDN (Mixture Density Network)
mdn_loss_weight     = 1.0000
dur_loss_weight     = 0.5000
gt_align_mse_weight = 0.1000

# To be removed
diag_att_weight     = 0.1000# To be removed
if iteration > 5000:        # To be removed
    diag_att_weight *= 0.25 # To be removed

# (Sylps) Syllables per Second
sylps_MAE_weight = 0.00
sylps_MSE_weight = 0.01

# Residual Encoder. Hasn't been used in a long time and might not work anymore. Not recommended on this model.
res_enc_gMSE_weight = 0.0200# negative classification/regression weight for discriminator.
res_enc_dMSE_weight = 0.0200# positive classification/regression weight for discriminator.
res_enc_kld_weight  = 0.00004# try to hold res_enc_kld between 0.5 and something something.
if expavg_loss_dict is not None and 'res_enc_kld' in expavg_loss_dict:
    res_enc_kld_weight *= expavg_loss_dict['res_enc_kld']

# If using HiFiGAN
if expavg_loss_dict is not None and 'HiFiGAN_g_all_mel_mae' in expavg_loss_dict:
    A_ *= 0.25# decreasing learning rate significantly

# HiFiGAN
HiFiGAN_learning_rate = 0.0# base/initial learning rate for HiFiGAN weights # set to 0. to use learning_rate from generator model.
HiFiGAN_lr_half_life  = 20000# number of iters it takes for learning rate to half

HiFiGAN_weight = 0.05# weight for entire HiFiGAN Losses

HiFiGAN_g_all_class_weight      =  1.0 * HiFiGAN_weight
HiFiGAN_g_all_featuremap_weight =  1.0 * HiFiGAN_weight
HiFiGAN_g_all_mel_mae_weight    = 45.0 * HiFiGAN_weight#45.0 * HiFiGAN_weight

HiFiGAN_d_all_class_weight      =  1.0 * HiFiGAN_weight

# Misc
n_restarts_override = None
