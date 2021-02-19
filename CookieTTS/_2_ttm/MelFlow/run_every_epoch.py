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
## MelFlow ##
##############
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e30 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

offset = 0
# Learning Rate / Optimization
decay_start = 36000
A_ = 1.0e-4
B_ = 30000
C_ = 0e-5
min_learning_rate = 1e-6
grad_clip_thresh  = 50.0

warmup_start_lr = 0.0e-4
warmup_start = 8600#checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*2e6 # warmup will linearly increase LR by 1e-5 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

# Validate at; 0, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, ...
validation_interval = 2**int(math.log2(iteration+1))
validation_interval = max(validation_interval,  256)
validation_interval = min(validation_interval, 2048)
checkpoint_interval = 4096

# MelFlow Loss Scalars (set to None to load from hparams.py)
melglow_total_loss_weight = 1.00# mel reconstruction loss
mdn_loss_weight           = 1.00# alignment loss
dur_loss_weight           = 0.05# duration predictor loss
melenc_kld_weight         = 0.00010# regularization loss for mel-encoder outputs (melenc isn't used during inference anyway so this loss may not even be required)

# MelFlow Forward Pass Configs/Options
align_with_z    = bool(0)#True if iteration > 1000 else False # whether to use Z latent or spectrogram for aligning.
mdn_align_grads = bool(0)# whether to keep grads for the alignment loss. This uses a TON of VRAM so you probably want to disable this once the alignments converge/finish learning.

save_alignments = False
show_gradients  = False# print abs().sum() gradients of every param tensor in model.

# Teacher-forcing Config
p_teacher_forcing  = 1.00# 1.00 = teacher force, 0.00 = inference
teacher_force_till = 0# decay this value **very** slowly
val_p_teacher_forcing  = 1.00
val_teacher_force_till = 0

# Misc
n_restarts_override = None
teacher_force_till = int(teacher_force_till)
