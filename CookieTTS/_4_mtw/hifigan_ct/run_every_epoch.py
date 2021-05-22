current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
# expavg_loss_dict # smoothed loss dict
##########################################################
##                                                      ##
##   ██╗  ██╗██╗███████╗██╗ ██████╗  █████╗ ███╗   ██╗  ##
##   ██║  ██║██║██╔════╝██║██╔════╝ ██╔══██╗████╗  ██║  ##
##   ███████║██║█████╗  ██║██║  ███╗███████║██╔██╗ ██║  ##
##   ██╔══██║██║██╔══╝  ██║██║   ██║██╔══██║██║╚██╗██║  ##
##   ██║  ██║██║██║     ██║╚██████╔╝██║  ██║██║ ╚████║  ##
##   ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝  ##
##                                                      ##
##########################################################
## HiFiGAN ##
#############
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e5 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

# Learning Rate / Optimization
decay_start = 160000
A_ = 3.0e-3
B_ = 60000
C_ = 0e-5
min_learning_rate = 1e-11
grad_clip_thresh  = 0.0
warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*1e6 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval =   100 if iteration <= 500 else (1000 if iteration < 10000 else 1000)
checkpoint_interval = 10000

g_mel_mae_weight = 1.0

# Misc
n_restarts_override = None
