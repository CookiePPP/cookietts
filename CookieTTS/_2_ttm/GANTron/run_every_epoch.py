gc.disable()
current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
# expavg_loss_dict # smoothed loss dict
####################################################################################################
## GANTron ## GANTron ## GANTron ## GANTron ## GANTron ## GANTron ## GANTron ## GANTron ## GANTron #
####################################################################################################
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e5 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

# Learning Rate / Optimization
decay_start = 160000
A_ = 2.0e-4
B_ = 60000
C_ = 0e-5
min_learning_rate = 1e-11
grad_clip_thresh  = 1000.0 if iteration < 2000 else 1.0
warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*5e5 # warmup will linearly increase LR by 1e-4 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval = 100 if iteration <= 500 else (1000 if iteration < 10000 else 1000)
checkpoint_interval = validation_interval

# Training Parameters
n_discriminator_loops = 1
teacher_force_prob = 0.5

# Loss Term Weights
g_mel_MAE_weight  = 0.40# use 0.37 for MAE if enabled
g_mel_MSE_weight  = 0.00# use 1.00 for MSE if enabled
g_gate_BCE_weight = 0.05

g_hard_att_MSE_weight = 0.00# if iteration > 50000 else 0.00
g_hard_att_MAE_weight = 0.00

g_att_loss_weight = 0.05# * expavg_loss_dict.get('g_att_loss', 0.0) * 5.0
d_att_loss_weight = 0.05

d_atte_gd_mse_weight = 0.00

g_fm_weight    = 0.0#1e-11# has been removed "temporarily"
g_class_weight = 0.08
d_class_weight = 0.08 * 2.

d_class_active_thresh = 0.2
g_class_active_thresh = 0.2

# Metric Weights (recommend always leaving at 0.0)
g_att_diagonality_weight       = 0.0
g_att_top1_avg_prob_weight     = 0.0
g_att_top2_avg_prob_weight     = 0.0
g_att_top3_avg_prob_weight     = 0.0
g_att_avg_max_dur_weight       = 0.0
g_att_avg_min_dur_weight       = 0.0
g_att_avg_avg_dur_weight       = 0.0
g_att_avg_missing_enc_p_weight = 0.0
g_att_avg_attscore_weight      = 0.0

# Misc
n_restarts_override = None
