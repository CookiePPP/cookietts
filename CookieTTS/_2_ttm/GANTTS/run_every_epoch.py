current_iteration = iteration
##########################################################################
### GAN-TTS : HIGH FIDELITY SPEECH SYNTHESIS WITH ADVERSARIAL NETWORKS ###
##########################################################################

# Learning Rate / Optimization
decay_start = 99999999
A_ = 0.2e-5
B_ = 40000
C_ = 0e-5
min_learning_rate = 1e-6

grad_clip_thresh = 75

descriminator_loss_scale = 0.1