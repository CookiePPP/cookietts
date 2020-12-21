current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
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
LossExplosionThreshold = 1e3 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

offset = 0
# Learning Rate / Optimization
decay_start = 99999999
A_ = 1.0e-4
B_ = 40000
C_ = 0e-5
min_learning_rate = 1e-6
grad_clip_thresh  = 10.0

warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*1e5 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval = 125 if iteration < 2000 else (250 if iteration < 8000 else 250)
checkpoint_interval = 2000

# Loss Scalars (set to None to load from hparams.py)
code_L1_weight      = 1.0
spec_MSE_weight     = 1.0
spec_MFSE_weight    = 0.0
postnet_MSE_weight  = 1.0
postnet_MFSE_weight = 0.0

# Misc
n_restarts_override = None
