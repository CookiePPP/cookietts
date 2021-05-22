gc.disable()
current_iteration = iteration
# checkpoint_iter # this is the iteration of the loaded checkpoint. If starting from sratch this value will be zero.
# expavg_loss_dict # smoothed loss dict
#############################################################################################################################################
## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 ## GANSpeech2 #
#############################################################################################################################################
param_interval = 1# how often this file is ran
dump_filelosses_interval = 1000# how often to update file_losses.cvs
show_live_params = False
LossExplosionThreshold = 1e5 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = True
decrease_lr_on_restart = True # Decrease the Learning Rate on a LossExplosionThreshold exception

show_gradients = False# print abs().sum() gradients of every param tensor in tacotron model.

# Learning Rate / Optimization
decay_start = 1e9
A_ = 0.5e-4
B_ = 120000
C_ = 0e-5
min_learning_rate = 1e-6
grad_clip_thresh  = 1000.0 if iteration < 2000 else 1000.0
warmup_start_lr = 0.0e-4
warmup_start = checkpoint_iter
warmup_end   = warmup_start + (A_-warmup_start_lr)*5e5 # warmup will linearly increase LR by 1e-4 each iter till LR hits A_

best_model_margin = 0.01 # training loss margin

validation_interval = 50 if iteration < 100 else (100 if iteration <= 500 else (1000 if iteration <= 10000 else 1000))
checkpoint_interval = 200000#validation_interval

# Training Parameters
teacher_force_prob = 1.00

# Loss Term Weights
aligner_scalar = 1e-2
ga_mel_MAE_weight  = aligner_scalar * 0.00# use 0.37 for MAE if enabled
ga_mel_MSE_weight  = aligner_scalar * 0.50# use 1.00 for MSE if enabled
ga_gate_BCE_weight = aligner_scalar * 0.01

ga_hard_att_MSE_weight = aligner_scalar * 0.00# if iteration > 50000 else 0.00
ga_hard_att_MAE_weight = aligner_scalar * 0.00

ga_att_loss_weight = aligner_scalar * (0.3 if iteration < 2000 else 0.1)

n_discriminator_loops = 1# <-- will significantly increase time-per-iter
dur_scalar = 1.0
rec_scalar = 1.0
# Generator Reconstruction
g_dur_MAE_weight    = rec_scalar* dur_scalar* 0.33
g_dur_MSE_weight    = rec_scalar* dur_scalar* 0.
g_logf0_MAE_weight  = rec_scalar* dur_scalar* 0.33
g_logf0_MSE_weight  = rec_scalar* dur_scalar* 0.
g_voiced_MAE_weight = rec_scalar* dur_scalar* 0.33
g_voiced_MSE_weight = rec_scalar* dur_scalar* 0.
g_voiced_BCE_weight = rec_scalar* dur_scalar* 0.

g_voiceds_MAE_weight = 0. # <- doesn't enter disciminator. Used as gating for pred f0s during inference only.
g_voiceds_MSE_weight = 0. # <- doesn't enter disciminator. Used as gating for pred f0s during inference only.
g_voiceds_BCE_weight = 0.1# <- doesn't enter disciminator. Used as gating for pred f0s during inference only.
g_logf0s_MAE_weight  = rec_scalar* 1.00
g_logf0s_MSE_weight  = rec_scalar* 0.
g_mel_MAE_weight     = rec_scalar* 1.00
g_mel_MSE_weight     = rec_scalar* 0.

g_class_dur_weight = 0.09 *dur_scalar
g_class_f0s_weight = 0.09
g_class_mel_weight = 0.09
d_class_dur_weight = 0.09 *2.0 *dur_scalar
d_class_f0s_weight = 0.09 *2.0
d_class_mel_weight = 0.09 *2.0

g_class_dur_active_thresh = 0.10# value required for this term to be used for gradients. Skipping updates from an unbalanced GAN *might* restabilize it. [0.0 to disable.   0.50 to skip almost every update.   0.1 recommended]
g_class_f0s_active_thresh = 0.10# value required for this term to be used for gradients. Skipping updates from an unbalanced GAN *might* restabilize it. [0.0 to disable.   0.50 to skip almost every update.   0.1 recommended]
g_class_mel_active_thresh = 0.30# value required for this term to be used for gradients. Skipping updates from an unbalanced GAN *might* restabilize it. [0.0 to disable.   0.50 to skip almost every update.   0.1 recommended]
d_class_dur_active_thresh = 0.10# value required for this term to be used for gradients. Skipping updates from an unbalanced GAN *might* restabilize it. [0.0 to disable.   0.50 to skip almost every update.   0.1 recommended]
d_class_f0s_active_thresh = 0.10# value required for this term to be used for gradients. Skipping updates from an unbalanced GAN *might* restabilize it. [0.0 to disable.   0.50 to skip almost every update.   0.1 recommended]
d_class_mel_active_thresh = 0.30# value required for this term to be used for gradients. Skipping updates from an unbalanced GAN *might* restabilize it. [0.0 to disable.   0.50 to skip almost every update.   0.1 recommended]

# Metric Weights (recommend always leaving at 0.0)
ga_att_diagonality_weight       = 0.0
ga_att_top1_avg_prob_weight     = 0.0
ga_att_top2_avg_prob_weight     = 0.0
ga_att_top3_avg_prob_weight     = 0.0
ga_att_avg_max_dur_weight       = 0.0
ga_att_avg_min_dur_weight       = 0.0
ga_att_avg_avg_dur_weight       = 0.0
ga_att_avg_missing_enc_p_weight = 0.0
ga_att_avg_attscore_weight      = 0.0

# Misc
n_restarts_override = None
