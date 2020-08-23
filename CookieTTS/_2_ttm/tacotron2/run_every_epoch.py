current_iteration = iteration

# Learning Rate / Optimization
decay_start = 99999999
A_ = 4e-5
B_ = 40000
C_ = 0e-5
min_learning_rate = 1e-6

grad_clip_thresh = 1.5

# Loss Scalars
em_kl_weight = 0.0001 # set to None to load from hparams
DiagonalGuidedAttention_scalar = 0.001#0.05 # set to None to load from hparams

# Prenet/Teacher Forcing Stuffs
drop_frame_rate = 0.0
#drop_frame_rate = min(0.000010 * max(current_iteration-5000,0), 0.2) # linearly increase DFR from 0.0 to 0.25 from iteration 5000 to 55000.
p_teacher_forcing = 1.00
teacher_force_till = 0
val_p_teacher_forcing=0.80
val_teacher_force_till=30