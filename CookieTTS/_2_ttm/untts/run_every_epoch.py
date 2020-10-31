# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- # 
iteration = iteration # reminder that iteration var exists within this scope
checkpoint_iter = checkpoint_iter # iteration of last load checkpoint, can be used to warmup the learning rate after a restart
#n_restarts = n_restarts # number of restarts this run, allows LR to be changed based on instability of the current schedule
###################################################
##                                               ##
## ██╗   ██╗███╗   ██╗████████╗████████╗███████╗ ##
## ██║   ██║████╗  ██║╚══██╔══╝╚══██╔══╝██╔════╝ ##
## ██║   ██║██╔██╗ ██║   ██║      ██║   ███████╗ ##
## ██║   ██║██║╚██╗██║   ██║      ██║   ╚════██║ ##
## ╚██████╔╝██║ ╚████║   ██║      ██║   ███████║ ##
##  ╚═════╝ ╚═╝  ╚═══╝   ╚═╝      ╚═╝   ╚══════╝ ##
##                                               ##
###################################################
## UnTTS ##
###########

param_interval = 5
show_live_params = False

LossExplosionThreshold = 1e3 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = 1 # use Live Custom Learning Rate instead of Scheduler.
decrease_lr_on_restart = 1 # Decrease the Learning Rate on a LossExplosionThreshold exception
n_restarts_override = None

# Custom LR
decay_start = 200000 # wait till decay_start to start decaying learning rate
A_ = 0.10e-4
B_ = 40000
C_ = 0.00000000
min_learning_rate = 0.1e-8

# Loss Scalars
MelGlow_ls = 1.000# dec_T*n_mel samples
DurGlow_ls = 0.050# enc_T*4     samples
VarGlow_ls = 0.050# dec_T*3     samples
Sylps_ls   = 0.200
# set to None to load from hparams.py

warmup_start_lr = 0.1e-4
warmup_start = checkpoint_iter + 0
warmup_end   = warmup_start + (A_-warmup_start_lr)*1e6*2.0 # warmup will linearly increase LR by 1e-6 each iter till LR hits A_

grad_clip_thresh = 50. if iteration > 10000 else 1000.

best_model_margin = 0.01 # training loss margin
validation_interval = 50 if iteration < 101 else (125 if iteration < 2000 else 250)
# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- #