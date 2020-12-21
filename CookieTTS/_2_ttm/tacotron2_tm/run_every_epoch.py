current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
# expavg_loss_dict # smoothed loss dict
######################################################################################
##                                                                                  ##
## ████████╗ █████╗  ██████╗ ██████╗ ████████╗██████╗  ██████╗ ███╗   ██╗  ██████╗  ##
## ╚══██╔══╝██╔══██╗██╔════╝██╔═══██╗╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║  ╚════██╗ ##
##    ██║   ███████║██║     ██║   ██║   ██║   ██████╔╝██║   ██║██╔██╗ ██║   █████╔╝ ##
##    ██║   ██╔══██║██║     ██║   ██║   ██║   ██╔══██╗██║   ██║██║╚██╗██║  ██╔═══╝  ##
##    ██║   ██║  ██║╚██████╗╚██████╔╝   ██║   ██║  ██║╚██████╔╝██║ ╚████║  ███████╗ ##
##    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝  ╚══════╝ ##
##                                                                                  ##
######################################################################################
## Tacotron2 ##
###############
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e5 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

offset = 0
# Learning Rate / Optimization
decay_start = 120000
#if iteration <  10000+offset:
#    A_ = 3.600e-4
#elif iteration <  50000+offset:
#    A_ = 2.000e-4
#elif iteration <  80000+offset:
#    A_ = 1.000e-4
#elif iteration < 110000+offset:
#    A_ = 0.500e-4
#elif iteration < 140000+offset:
#    A_ = 0.250e-4
#elif iteration < 170000+offset:
#    A_ = 0.125e-4
#else:
#    A_ = 0.050e-4
A_ = 1.0e-4
A_ *= 0.25**0.5# for HiFiGAN TTFT-ing
B_ = 30000
C_ = 0e-5
min_learning_rate = 1e-11
grad_clip_thresh  = 9999.#1.0 if iteration > 20000 else 4.0

warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter+10
warmup_end   = warmup_start + (A_-warmup_start_lr)*2e6 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval =  125 if iteration < 2000 else (250 if iteration < 8000 else 500)
checkpoint_interval = 1000

# Loss Scalars (set to None to load from hparams.py)
spec_MSE_weight     = 0.0000
spec_MFSE_weight    = 1.0000
postnet_MSE_weight  = 0.0000
postnet_MFSE_weight = 1.0000
gate_loss_weight    = 1.0000
sylps_kld_weight    = 0.0500
sylps_MSE_weight    = 0.0100
sylps_MAE_weight    = 0.0010
diag_att_weight     = 0.1000# you only want to use this at high strenght to warmup the attention, it will mask problems later into training.
if iteration >  5000:
    diag_att_weight *= 0.25

pn_code_MAE_weight = 0.00200# prenet code loss

VE_KLD_weight = 0.01000# increase if VE_KLD goes above 1.0 for more than 500 iters
VE_KLD_weight *= max(min((iteration-250)/12000, 1.0), 0.000005)# linear warmup over 5000 iters
#if expavg_loss_dict is not None and 'VE_KLD' in expavg_loss_dict:
#    VE_KLD_weight *= expavg_loss_dict['VE_KLD']

enable_dbGAN_iter = 60000
dbGAN_gLoss_weight = 0.01 if iteration > enable_dbGAN_iter else 1e-5# De-Blur-GAN Generator Loss
dbGAN_dLoss_weight = 0.1#dbGAN_gLoss_weight*0.5
if expavg_loss_dict is not None and 'dbGAN_accuracy' in expavg_loss_dict:
    if expavg_loss_dict['dbGAN_accuracy'] < 0.75:# disable GAN loss from propagating to Tacotron till the GAN learns useful gradients
        dbGAN_gLoss_weight = 1e-10
    if expavg_loss_dict['dbGAN_accuracy'] > 0.98:# disable GAN loss from propagating to Tacotron till the GAN learns useful gradients
        dbGAN_dLoss_weight = 1e-10

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

enable_InfGAN_iter = 190000
enable_InfGAN = bool( iteration >= enable_InfGAN_iter )# enable this near the end of training as a fine-tuning process.
disable_zoneout_when_InfGAN_enabled = False
InfGAN_gLoss_weight = 5e-3
InfGAN_dLoss_weight = InfGAN_gLoss_weight*0.5
InfGAN_max_accuracy = 0.95# if smoothed InfGAN accuracy goes above this value, do not update InfGAN.
if enable_InfGAN:
    A_ *= (0.5**0.5)# multiply LR by sqrt of 0.5 if InfGAN enables (which halfs the Teacher-forced batch size)
if expavg_loss_dict is not None and 'InfGAN_accuracy' in expavg_loss_dict:
    if expavg_loss_dict['InfGAN_accuracy'] < 0.75:# disable GAN loss from propagating to Tacotron till the GAN learns useful gradients
        InfGAN_gLoss_weight = 1e-8
    if expavg_loss_dict['InfGAN_accuracy'] > 0.98:# disable GAN loss from propagating to Tacotron till the GAN learns useful gradients
        InfGAN_dLoss_weight = 1e-8

res_enc_gMSE_weight = 0.0200# negative classification/regression weight for discriminator.
res_enc_dMSE_weight = 0.0200# positive classification/regression weight for discriminator.
res_enc_kld_weight  = 0.00004# try to hold res_enc_kld between 0.5 and something something.
if expavg_loss_dict is not None and 'res_enc_kld' in expavg_loss_dict:
    res_enc_kld_weight *= expavg_loss_dict['res_enc_kld']

# Drop Frame Rate
dfr_warmup_start =   0
dfr_warmup_iters = 100
dfr_max_value    = 0.20
drop_frame_rate = dfr_max_value if dfr_max_value < 0.01 else min(max(iteration-dfr_warmup_start,0)/(dfr_warmup_iters*dfr_max_value), dfr_max_value) # linearly increase DFR from 0.0 to 0.2 from iteration 1 to 10001.

# Teacher-forcing Config
p_teacher_forcing  = 1.00# 1.00 = teacher force, 0.00 = inference
teacher_force_till = 0
val_p_teacher_forcing  = 1.00
val_teacher_force_till = 0

# HiFiGAN
HiFiGAN_learning_rate = 1e-4# base/initial learning rate for HiFiGAN fine-tuning
HiFiGAN_lr_half_life  = 8000# number of iters it takes for learning rate to half

HiFiGAN_weight = 0.25# weight for entire HiFiGAN Losses

HiFiGAN_g_all_class_weight      =  1.0 * HiFiGAN_weight
HiFiGAN_g_all_featuremap_weight =  1.0 * HiFiGAN_weight
HiFiGAN_g_all_mel_mae_weight    = 45.0 * HiFiGAN_weight

HiFiGAN_d_all_class_weight      =  1.0 * HiFiGAN_weight

HiFiGAN_g_msd_class_weight      =  0.0
HiFiGAN_g_mpd_class_weight      =  0.0
HiFiGAN_g_msd_featuremap_weight =  0.0
HiFiGAN_g_mpd_featuremap_weight =  0.0

# Misc
n_restarts_override = None
teacher_force_till = int(teacher_force_till)
