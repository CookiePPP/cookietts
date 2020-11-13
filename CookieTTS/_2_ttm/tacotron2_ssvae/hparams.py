import tensorflow as tf
from CookieTTS.utils.text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    from CookieTTS.utils.utils_hparam import HParams
    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=250,
        iters_per_validation=250,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers= ["layers_here"],
        frozen_modules=["layers_here"], # only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        print_layer_names_during_startup=True,
        
        ################################
        # Data Parameters              #
        ################################
        check_files=True, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
                          # This can take a little as it has to simulate an entire EPOCH of dataloading.
        load_mel_from_disk=True, # Saves significant RAM and CPU.
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt', # lets the checkpoints include speaker names.
        dict_path='../../dict/merged.dict.txt',
        p_arpabet=0.5, # probability to use ARPAbet / pronounciation dictionary.
        use_saved_speakers=True,# use the speaker lookups saved inside the model instead of generating again
        numeric_speaker_ids=False, # sort speaker_ids in filelist numerically, rather than alphabetically.
                                   # e.g:
                                   #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                   # instead of,
                                   #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
                                   # Mellotron repo has this off by default, but ON makes the most logical sense to me.
        raw_speaker_ids=False,  # use the speaker IDs found in filelists for the internal IDs. Values greater than n_speakers will crash (as intended).
                                # This will disable sorting the ids
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_train_taca2.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_validation_taca2.txt',
        text_cleaners=['basic_cleaners'],
        
        silence_value=-11.52,
        silence_pad_start=0,# frames to pad the start of each clip
        silence_pad_end=0,  # frames to pad the end of each clip
                            # These frames will be added to the loss functions and Tacotron must predict and generate the padded silence.
        
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=48000,
        filter_length=2400,
        hop_length=600,
        win_length=2400,
        n_mel_channels=256,
        mel_fmin=0.0,
        mel_fmax=16000.0,
        
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # Gate
        gate_positive_weight=10, # how much more valuable 1 positive frame is to 1 zero frame. 80 Frames per seconds, therefore values around 10 are fine.
        
        # Synthesis/Inference Related
        gate_threshold=0.5,
        gate_delay=10,
        max_decoder_steps=3000,
        low_vram_inference=False, # doesn't save alignment and gate information, frees up some vram, especially for large input sequences.
        
        # Teacher-forcing Config
        p_teacher_forcing=1.00,    # 1.00 baseline
        teacher_force_till=20,     # int, number of starting frames with teacher_forcing at 100%, helps with clips that have challenging starting conditions i.e breathing before the text begins.
        val_p_teacher_forcing=0.80,
        val_teacher_force_till=20,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim=64, # speaker_embedding before encoder
        encoder_concat_speaker_embed='before_conv', # concat before encoder convs, or just before the LSTM inside decode. Options 'before_conv','before_lstm'
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_conv_hidden_dim=512,
        encoder_LSTM_dim=768,
        
        # (SylpsNet) Predicts speaking speed
        sylpsnet_layer_dims = [32, 32],# width of each layer, LeakyReLU() is used between hiddens
        
        # (EmotionNet) Semi-supervised VAE/Classifier
        emotion_classes = ['neutral','anxious','happy','annoyed','sad','confused','smug','angry','whispering','shouting','sarcastic','amused','surprised','singing','fear','serious'],
        emotionnet_latent_dim=32,# unsupervised Latent Dim
        emotionnet_encoder_outputs_dropout=0.75,# Encoder Outputs Dropout
        emotionnet_RNN_dim=128, # GRU dim to summarise Encoder Outputs
        emotionnet_classifier_layer_dropout=0.25, # Dropout ref, speaker and summarised Encoder outputs.
                                                  # Which are used to predict zs and zu
        
        # (EmotionNet) Reference encoder
        emotionnet_ref_enc_convs=[32, 32, 64, 64, 128, 128],
        emotionnet_ref_enc_rnn_dim=64, # GRU dim to summarise RefSpec Conv Outputs
        emotionnet_ref_enc_use_bias=False,
        emotionnet_ref_enc_droprate=0.3, # Dropout for Reference Spectrogram Encoder Conv Layers
        
        # (AuxEmotionNet)
        auxemotionnet_layer_dims=[256,],# width of each layer, LeakyReLU() is used between hiddens
                                        # input is TorchMoji hidden, outputs to classifier layer and zu param predictor
        auxemotionnet_encoder_outputs_dropout=0.75,# Encoder Outputs Dropout
        auxemotionnet_RNN_dim=128, # GRU dim to summarise Encoder outputs
        auxemotionnet_classifier_layer_dropout=0.25, # Dropout ref, speaker and summarised Encoder outputs.
                                                     # Which are used to predict zs and zu params
        
        # (AuxEmotionNet) TorchMoji
        torchMoji_attDim=2304,# published model uses 2304
        
        # (Speaker) Speaker embedding
        n_speakers=512, # maximum number of speakers the model can support.
        speaker_embedding_dim=256, # speaker embedding size # 128 baseline
        
        # (Decoder/Encoder) Bottleneck parameters
        # The outputs from the encoder, speaker, emotionnet and sylpsnet need to be mixed.
        # By default the information is mixed by the DecoderRNN, but this is repeated every spectrogram frame so likely wastes a massive amount of compute performing the same operations repeatedly.
        # Thus, this memory bottleneck can be used to mix the above mentioned outputs into a more compressed representation before decoding, allowing the DecoderRNN to be made smaller and more effective.
        use_memory_bottleneck=True,# False baseline
        memory_bottleneck_dim=512,# new memory size. 512 would be equivalent to the original Tacotron2.
        memory_bottleneck_bias=False,
        
        # (Decoder) Decoder parameters
        start_token = "",#"☺"
        stop_token = "",#"␤"
        hide_startstop_tokens=False, # trim first/last encoder output before feeding to attention.
        n_frames_per_step=1,# currently only 1 is supported
        context_frames=1,   # TODO TODO TODO TODO TODO
        
        # (Decoder) Prenet
        prenet_dim=512,         # 256 baseline
        prenet_layers=2,        # 2 baseline
        prenet_batchnorm=False,  # False baseline
        p_prenet_dropout=0.5,   # 0.5 baseline
        prenet_speaker_embed_dim=0, # speaker_embedding before encoder
        prenet_noise=0.0, # Apply Gaussian Noise (std defined here) to the Teacher Forced Prenet inputs.
        prenet_blur_min=0.0,# Apply random vertical blur between prenet_blur_min
        prenet_blur_max=0.0,#                                and prenet_blur_max
                            # Set max to False or Zero to disable
        
        # (Decoder) AttentionRNN
        attention_rnn_dim=1280, # 1024 baseline
        AttRNN_extra_decoder_input=True,# False baseline # Feed DecoderRNN Hidden State into AttentionRNN
        AttRNN_hidden_dropout_type='dropout',# options ('dropout','zoneout')
        p_AttRNN_hidden_dropout=0.1,# 0.1 baseline
        
        # (Decoder) DecoderRNN
        decoder_rnn_dim=512, # 1024 baseline
        DecRNN_hidden_dropout_type='dropout',# options ('dropout','zoneout')
        p_DecRNN_hidden_dropout=0.0,# 0.1 baseline
        decoder_residual_connection=False,# residual connections with the AttentionRNN hidden state and Attention/Memory Context
        # Optional Second Decoder
        second_decoder_rnn_dim=0,# 0 baseline # Extra DecoderRNN to learn more complex patterns # set to 0 to disable layer.
        second_decoder_residual_connection=True,# residual connections between the DecoderRNNs
        
        # (Decoder) Attention parameters
        attention_type=0,
        # 0 -> Hybrid Location-Based Attention (Vanilla Tacotron2)
        # 1 -> GMMAttention (Long-form Synthesis)
        # 1 -> Dynamic Convolution Attention (Long-form Synthesis)
        attention_dim=128, # 128 Layer baseline # Used for Key-Query Dim
        
        # (Decoder) Attention Type 0 Parameters
        windowed_attention_range = 64,# set to 0 to disable
                                     # will set the forward and back distance the model can attend to.
                                     # 2 will give the model 5 characters it can attend to at any one time.
                                     # This will also allow more stable generation with longer text inputs and save VRAM during inference.
        windowed_att_pos_offset=1.25,# Offset the current_pos by this amount.
        windowed_att_pos_learned=True,
        
        # (Decoder) Attention Type 0 (and 2) Parameters
        attention_location_n_filters=32,   # 32 baseline
        attention_location_kernel_size=31, # 31 baseline
        
        # (Decoder) Attention Type 1 Parameters
        num_att_mixtures=1,# 5 baseline
        attention_layers=1,# 1 baseline
        delta_offset=0.005,    # 0 baseline, values around 0.005 will push the model forwards. Since we're using the sigmoid function caution is suggested.
        delta_min_limit=0.0, # 0 baseline, values around 0.010 will force the model to move forward, in this example, the model cannot spend more than 100 steps on the same encoder output.
        lin_bias=False, # I need to figure out what that layer is called.
        initial_gain='relu', # initial weight distribution 'tanh','relu','sigmoid','linear'
        normalize_attention_input=True, # False baseline
        normalize_AttRNN_output=False,  # True baseline
        
        # (Decoder) Attention Type 2 Parameters
        dynamic_filter_num=128, # 8 baseline
        dynamic_filter_len=21, # 21 baseline # currently only 21 is supported
        
        # (Postnet) Mel-post processing network parameters
        use_postnet=False,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=6,
        postnet_residual_connections=2,# False baseline, int > 0 == n_layers in each residual block
        
        # (Adversarial Postnet Generator) - modifies the tacotron output to look convincingly fake instead of just accurate.
        use_postnet_generator_and_discriminator=False,
        adv_postnet_noise_dim=128,
        adv_postnet_embedding_dim=384,
        adv_postnet_kernel_size=3,
        adv_postnet_n_convolutions=6,
        adv_postnet_residual_connections=3,
        
        # (Adversarial Postnet Discriminator) - Learns the difference between real and fake spectrograms, teaches the postnet generator how to make convincing looking outputs.
        dis_postnet_embedding_dim=512,
        dis_postnet_kernel_size=5,
        dis_postnet_n_convolutions=8,
        dis_postnet_residual_connections=4,
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=0.1e-5,# overriden by 'run_every_epoch.py'
        weight_decay=1e-6,
        grad_clip_thresh=1.0,# overriden by 'run_every_epoch.py'
        
        batch_size=40,     # controls num of files processed in parallel per GPU
        val_batch_size=40, # for more precise comparisons between models, constant batch_size is useful
        
        use_TBPTT=False,# continue truncated files into the next training iteration
        truncated_length=1000, # max mel length till truncation.
        mask_padding=True,#mask values by setting them to the same values in target and predicted
        masked_select=True,#mask values by removing them from the calculation
        
        # (DFR) Drop Frame Rate
        global_mean_npy='global_mean.npy',
        drop_frame_rate=0.25,# overriden by 'run_every_epoch.py'
        
        ################################
        # Loss Weights/Scalars         #
        ################################
        LL_SpectLoss=False,# Use Log-likelihood loss on Decoder and Postnet outputs.
                           # This will cause Tacotron to produce a normal Spectrogram and a logvar Spectrogram.
                           # The logvar spectrogram will represent the confidence of the model on it's prediction,
                           # and can be used to calcuate the std (error) tacotron expects that element to have.
        
        # if LL_SpectLoss is True:
        melout_LL_scalar  = 1.0,  # Log-likelihood Loss
        postnet_LL_scalar = 1.0,  # Log-likelihood Loss
        # else:
        melout_MSE_scalar = 1.0,  #      MSE Spectrogram Loss Before Postnet
        melout_MAE_scalar = 0.0,  #       L1 Spectrogram Loss Before Postnet
        melout_SMAE_scalar = 0.0, # SmoothL1 Spectrogram Loss Before Postnet
        postnet_MSE_scalar = 1.0, #      MSE Spectrogram Loss After Postnet
        postnet_MAE_scalar = 0.0, #       L1 Spectrogram Loss After Postnet
        postnet_SMAE_scalar = 0.0,# SmoothL1 Spectrogram Loss After Postnet
        
        # if use_postnet_generator_and_discriminator is True:
        adv_postnet_scalar = 0.1,# Global Loss Scalar for the entire Adversarial System
                                 #                  (Postnet, Reconstruction, Discriminator, Etc)
        adv_postnet_reconstruction_weight = 10.0,# Reconstruction Loss to force the GAN to follow the input somewhat
        adv_postnet_grad_propagation = 0.05, # Multiply GAN gradients before they connect to the main network.
                                            # 0.0 is the same as detaching the GAN loss from the main network, so the GAN loss will only update the Adversarial Postnet.
                                            # 1.0 is apply GAN Loss gradients to the entire tacotron2 network like *normal*
        dis_postnet_scalar = 0.1,# Loss Scalar for discriminator on normal not-GAN postnet
        dis_spect_scalar   = 0.1,# Loss Scalar for discriminator on normal recurrent spect outputs
        
        zsClassificationNCELoss = 0.00, # EmotionNet Classification Loss (Negative Cross Entropy)
        zsClassificationMAELoss = 0.00, # EmotionNet Classification Loss (Mean Absolute Error)
        zsClassificationMSELoss = 0.00, # EmotionNet Classification Loss (Mean Squared Error)
        
        auxClassificationNCELoss = 0.00, # AuxEmotionNet NCE Classification Loss
        auxClassificationMAELoss = 0.00, # AuxEmotionNet MAE Classification Loss
        auxClassificationMSELoss = 0.00, # AuxEmotionNet MSE Classification Loss
        
        em_kl_weight   = 0.0010, # EmotionNet KDL weight # Can be overriden by 'run_every_epoch.py'
        syl_KDL_weight = 0.0020, # SylNet KDL Weight
        
        pred_sylpsMSE_weight = 0.01,# Encoder Pred Sylps MSE weight
        pred_sylpsMAE_weight = 0.00,# Encoder Pred Sylps MAE weight
        
        predzu_MSE_weight = 0.02, # AuxEmotionNet Pred Zu MSE weight
        predzu_MAE_weight = 0.00, # AuxEmotionNet Pred Zu MAE weight
        
        DiagonalGuidedAttention_scalar=0.05, # Can be overriden by 'run_every_epoch.py
                                             # 'dumb' guided attention. Simply punishes the model for attention that is non-diagonal. Decreases training time and increases training stability with English speech. 
                                             # As an example of how this works, if you imagine there is a 10 letter input that lasts 1 second. The first 0.1s is pushed towards using the 1st letter, the next 0.1s will be pushed towards using the 2nd letter, and so on for each chunk of audio. Since each letter has a different natural duration (especially punctuation), this attention guiding is not particularly accurate, so it's not recommended to use a high loss scalar later into training.
        DiagonalGuidedAttention_sigma=0.5, # how to *curve?* the attention loss? Just leave this one alone.
        
        rescale_for_volume=0.0, # Not implemented # Rescale spectrogram losses to prioritise louder sounds, and put less (or zero) priority on quieter sounds
                                # Valid values between 0.0 and 1.0
                                # will rescale spectrogram magnitudes (which range from -11.52 for silence, and 4.5 for deafeningly loud)
                                # https://www.desmos.com/calculator/bmbakslmtc
    )

    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s', hparams.values())

    return hparams