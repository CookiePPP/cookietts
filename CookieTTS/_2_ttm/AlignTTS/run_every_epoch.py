current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
######################################################################################
##                                                                                  ##
## ################################################################################ ##
## ################################################################################ ##
## ################################################################################ ##
## ################################################################################ ##
## ################################################################################ ##
## ################################################################################ ##
##                                                                                  ##
######################################################################################
## AlignTTS ##
##############
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e9 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

offset = 0
# Learning Rate / Optimization
decay_start = 40000
A_ = 1.0e-4
B_ = 20000
C_ = 0e-5
min_learning_rate = 1e-8
grad_clip_thresh  = 100.0

warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*1e4 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval = 1000#125 if iteration < 2000 else (500 if iteration < 8000 else 2000)
checkpoint_interval = 10000

# Tacotron Loss Scalars (set to None to load from hparams.py)

save_alignments = False

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

# Teacher-forcing Config
p_teacher_forcing  = 1.00# 1.00 = teacher force, 0.00 = inference
teacher_force_till = 0# decay this value **very** slowly
val_p_teacher_forcing  = 1.00
val_teacher_force_till = 0

# Misc
n_restarts_override = None
teacher_force_till = int(teacher_force_till)
