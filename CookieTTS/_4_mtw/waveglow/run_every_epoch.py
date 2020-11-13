# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- # 
iteration = iteration # reminder that iteration var exists within this scope
seconds_elapsed = seconds_elapsed # reminder that seconds_elapsed var exists within this scope
day = 86400
hr = 3600
#############################################################################
##                                                                         ##
## ██╗    ██╗ █████╗ ██╗   ██╗███████╗ ██████╗ ██╗      ██████╗ ██╗    ██╗ ##
## ██║    ██║██╔══██╗██║   ██║██╔════╝██╔════╝ ██║     ██╔═══██╗██║    ██║ ##
## ██║ █╗ ██║███████║██║   ██║█████╗  ██║  ███╗██║     ██║   ██║██║ █╗ ██║ ##
## ██║███╗██║██╔══██║╚██╗ ██╔╝██╔══╝  ██║   ██║██║     ██║   ██║██║███╗██║ ##
## ╚███╔███╔╝██║  ██║ ╚████╔╝ ███████╗╚██████╔╝███████╗╚██████╔╝╚███╔███╔╝ ##
##  ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝  ╚══╝╚══╝  ##
##                                                                         ##
#############################################################################
## WaveGlow / WaveFlow ##
#########################

param_interval = 5
show_live_params = False

LossExplosionThreshold = 9e9#999 if iteration > 100 else 1000
# maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = 1 # use Live Custom Learning Rate instead of Scheduler.

# Custom LR
decay_start = 9990000 # wait till decay_start to start decaying learning rate
hrs_elapsed = max(  0, (seconds_elapsed-(hr*30))//hr  ) # start dropping Learning Rate 24 hours after training starts.

#A_ = 0.00010000 if seconds_elapsed > (42*60) else 0.00040000 # drop LR after 42 minutes (Gated Unit Testing reproducability)
#*min((seconds_elapsed/(2*hr)), 1.0) # warmup over 4 hours

A_ = 0.0005000
A_ = A_/(2**(hrs_elapsed/12))# LR updates every hour, half-life of 12 hours.
B_ = 80000
C_ = 0.0000000

warmup_start = 12760
warmup_end   = warmup_start + 300
warmup_start_lr = 0.0002000

best_model_margin = 0.001 if iteration > 7000 else 0.05 # training loss margin

validation_interval = 250
if iteration > 9900:
    validation_interval = 1000
if iteration > 99000:
    validation_interval = 2500

use_grad_clip = True
grad_clip_thresh = 150 if iteration > 7000 else 250

# Scheduled LR
patience_iterations = 12000 # number of iterations without improvement to decrease LR
scheduler_patience = patience_iterations//validation_interval
scheduler_cooldown = 3
if 0:
    override_scheduler_best = 30 # WARNING: Override LR scheduler internal best value.
    override_scheduler_last_lr = A_ #[A_+C_]
else:
    override_scheduler_best = None
    override_scheduler_last_lr = None

# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- #