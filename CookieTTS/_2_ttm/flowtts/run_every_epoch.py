# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- # 
iteration = iteration # reminder that iteration var exists within this scope

param_interval = 5
show_live_params = False

LossExplosionThreshold = 1e9 # maximum loss value (which will trigger a restart from latest checkpoint)

custom_lr = 1 # use Live Custom Learning Rate instead of Scheduler.

# Custom LR
decay_start = 40000 # wait till decay_start to start decaying learning rate
A_ = 5e-4
B_ = 40000
C_ = 0.00000000
min_learning_rate = 1e-8

warmup_start = 0
warmup_end   = 1000
warmup_start_lr = 0.5e-4

grad_clip_thresh = 100

best_model_margin = 1.50 # training loss margin
validation_interval = 250
# ----------------------------------- LIVE PARAMS UPDATE ----------------------------------- #
