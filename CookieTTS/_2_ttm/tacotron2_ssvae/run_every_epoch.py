current_iteration = iteration
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

# Learning Rate / Optimization
decay_start = 99999999
A_ = 1.0e-4 #+ max(0., min(5e-4, 0.001e-4*(iteration-10000)))
B_ = 40000
C_ = 0e-5
min_learning_rate = 1e-6

discriminator_lr_scale = 1.0

grad_clip_thresh = 1.0

# Loss Scalars
em_kl_weight = 0.0005 # set to None to load from hparams.py
DiagonalGuidedAttention_scalar = 0.05 # set to None to load from hparams.py

# Prenet/Teacher Forcing Stuffs
dfr_warmup_start = 0
dfr_warmup_iters = 2000
dfr_max_value = 0.05
drop_frame_rate = dfr_max_value if dfr_max_value < 0.01 else min(max(current_iteration-dfr_warmup_start,0)/(dfr_warmup_iters*dfr_max_value), dfr_max_value) # linearly increase DFR from 0.0 to 0.2 from iteration 1 to 10001.

p_teacher_forcing = 0.95
teacher_force_till = 0
val_p_teacher_forcing=0.80
val_teacher_force_till=30