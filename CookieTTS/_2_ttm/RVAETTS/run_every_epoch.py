current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
# expavg_loss_dict # smoothed loss dict
#####################################################################
##                                                                 ##
##   ██████╗ ██╗   ██╗ █████╗ ███████╗████████╗████████╗███████╗   ##
##   ██╔══██╗██║   ██║██╔══██╗██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝   ##
##   ██████╔╝██║   ██║███████║█████╗     ██║      ██║   ███████╗   ##
##   ██╔══██╗╚██╗ ██╔╝██╔══██║██╔══╝     ██║      ██║   ╚════██║   ##
##   ██║  ██║ ╚████╔╝ ██║  ██║███████╗   ██║      ██║   ███████║   ##
##   ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝   ╚═╝      ╚═╝   ╚══════╝   ##
##                                                                 ##
#####################################################################
## RVAETTS ##
#############
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e5 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

# Learning Rate / Optimization
decay_start = 350000
A_ = 0.25e-4# halfing the base LR when using HiFiGAN
B_ = 20000
C_ = 0e-5
min_learning_rate = 1e-11
grad_clip_thresh  = 99.#1.0 if iteration > 20000 else 4.0

warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*1e6 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval = 500# 125 if iteration < 2000 else (500 if iteration < 8000 else 500)
checkpoint_interval = 500

# Loss Scalars (set to None to load from hparams.py)
decoder_MAE_weight = 0.5
decoder_MSE_weight = 0.0

fesvd_MAE_weight   = 0.0
fesvd_MSE_weight   = 1.0

varpred_KLD_weight = 0.0016
varpred_KLD_weight *= min(iteration%5000, 2500)/2500# cyclic annealing. at 0 iters, KLD_scale==0.0. at 2500 iters, KLD_scale==1.0, at 2000 iters, KLD_scale=1.0
                                                  # see https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Annealing-with-the-monotonic-schedule.png

vmelenc_KLD_weight = 0.0016
vmelenc_KLD_weight *= min(iteration%5000, 2500)/2500# cyclic annealing. at 0 iters, KLD_scale==0.0. at 2500 iters, KLD_scale==1.0, at 2000 iters, KLD_scale=1.0
                                                  # see https://www.microsoft.com/en-us/research/uploads/prod/2019/04/Annealing-with-the-monotonic-schedule.png

mdn_loss_weight     = 1.0000
dur_loss_weight     = 0.5000
gt_align_mse_weight = 0.1000

diag_att_weight     = 0.1000# you only want to use this at high strength to warmup the attention, it will mask problems later into training.
if iteration > 5000:
    diag_att_weight *= 0.25

sylps_MAE_weight = 0.00
sylps_MSE_weight = 0.01

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

res_enc_gMSE_weight = 0.0200# negative classification/regression weight for discriminator.
res_enc_dMSE_weight = 0.0200# positive classification/regression weight for discriminator.
res_enc_kld_weight  = 0.00004# try to hold res_enc_kld between 0.5 and something something.
if expavg_loss_dict is not None and 'res_enc_kld' in expavg_loss_dict:
    res_enc_kld_weight *= expavg_loss_dict['res_enc_kld']

# HiFiGAN
HiFiGAN_learning_rate = 0.#0.25e-4# base/initial learning rate for HiFiGAN weights # set to 0. to use learning_rate from generator model.
HiFiGAN_lr_half_life  = 20000# number of iters it takes for learning rate to half

HiFiGAN_weight = 0.05# weight for entire HiFiGAN Losses

HiFiGAN_g_all_class_weight      =  1.0 * HiFiGAN_weight
HiFiGAN_g_all_featuremap_weight =  1.0 * HiFiGAN_weight
HiFiGAN_g_all_mel_mae_weight    = 25.0 * HiFiGAN_weight#45.0 * HiFiGAN_weight

HiFiGAN_d_all_class_weight      =  1.0 * HiFiGAN_weight

# Misc
n_restarts_override = None