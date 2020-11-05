from CookieTTS.utils.text.symbols import symbols
import tensorflow as tf

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    from CookieTTS.utils.utils_hparam import HParams
    hparams = HParams(
        random_segments=False,# DONT MODIFY
        
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=1000,
        iters_per_validation=250,
        seed=1234,
        
        dynamic_loss_scaling=True,
        fp16_run=True,# requires 20 Series or Better (e.g: RTX 2080 Ti, RTX 2060, Tesla V100, Tesla A100)
        fp16_run_optlvl='2',
        
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        
        cudnn_enabled  = True,
        cudnn_benchmark=False,
        
        ignore_layers= ["layers_here"],# for `warm_start`-ing
        frozen_modules=["layers_here"],# only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        
        #########################
        ## Logging / Verbosity ##
        #########################
        print_layer_names_during_startup=True,
        n_tensorboard_outputs=5,# number of items from validation so show in Tensorboard
        n_tensorboard_outputs_highloss=5,# top X tacotron outputs with worst validation loss.
        n_tensorboard_outputs_badavgatt=5,# top X tacotron outputs with weakest average attention.
        
        #################################
        ## Batch Size / Segment Length ##
        #################################
        batch_size=32,    # controls num of files processed in parallel per GPU
        val_batch_size=32,# for more precise comparisons between models, constant batch_size is useful
        use_TBPTT=False,  # continue processing longer files into the next training iteration
        max_segment_length=800,# max mel length till a segment is sliced.
        
        ###################################
        ## Dataset / Filelist Parameters ##
        ###################################
        data_source=0,# 0 to use nvidia/tacotron2 filelists, 1 to use automatic dataset processor
        
        # if data_source is 0:
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/train_taca2.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/validation_taca2.txt',
        
        # if data_source is 1:
        dataset_folder='/media/cookie/Samsung 860 QVO/ClipperDatasetV2',
        dataset_audio_filters=['*.wav',],
        dataset_audio_rejects=['*_Noisy_*','*_Very Noisy_*'],
        dataset_p_val = 0.03,# portion of dataset for Validation
        dataset_min_duration=2.0,# minimum duration of audio files to be added
        
        ##################################
        ## Text / Speaker Parameters    ##
        ##################################
        text_cleaners=['basic_cleaners'],
        dict_path='../../dict/merged.dict.txt',
        p_arpabet=0.5, # probability to use ARPAbet / pronounciation dictionary on the text
        
        use_saved_speakers =False, # use the speaker lookups saved inside the model instead of generating again
        numeric_speaker_ids=True,  # sort speaker_ids in filelist numerically, rather than alphabetically.
                                   # e.g:
                                   #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                   # instead of,
                                   #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
                                   # Mellotron repo has this off by default, but ON makes the most logical sense to me.
        
        check_files=True, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
                          # This can take a while as it has to load the entire dataset once.
        ##################################
        ## Audio Parameters             ##
        ##################################
        sampling_rate=44100,
        target_lufs=-27.0, # Loudness each file is rescaled to, use None for original file loudness.
        
        trim_enable = False,# set to False to disable trimming completely
        trim_cache_audio = False,# save trimmed audio to disk to load later. Saves CPU usage, uses more disk space.
                                 # modifications to params below do not apply to already cached files while True.
        trim_margin_left  = [0.0125]*5,
        trim_margin_right = [0.0125]*5,
        trim_top_db       = [50  ,46  ,46  ,46  ,46  ],
        trim_window_length= [8192,4096,2048,1024,512 ],
        trim_hop_length   = [1024,512 ,256 ,128 ,128 ],
        trim_ref          = ['amax']*5,
        trim_emphasis_str = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ],
        
        ##################################
        ## Spectrogram Parameters       ##
        ##################################
        filter_length=2048,
        hop_length=512,
        win_length=2048,
        n_mel_channels=80,
        mel_fmin=20.0,
        mel_fmax=11025.0,
        stft_clamp_val=1e-5,# 1e-5 = original
        
        silence_value=-11.52,
        silence_pad_start=0,# frames to pad the start of each spectrogram
        silence_pad_end=0,  # frames to pad the end   of each spectrogram
                            # These frames will be added to the loss functions and Tacotron must predict and generate the padded silence.
        
        ######################################
        ## Synthesis / Inference Parameters ##
        ######################################
        gate_threshold=0.5,# confidence required to end the audio file.
        gate_delay    =10, # allows the model to continue generative audio slightly after the audio file ends.
        max_decoder_steps=3000,# max duration of an audio file during a single generation.
        
        ##################################
        ## Model Parameters             ##
        ##################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # Gate
        gate_positive_weight=10, # how much more valuable 1 positive frame is to 1 zero frame. 80 Frames per seconds, therefore values around 10 are fine.
        
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
        
        # (AuxEmotionNet) TorchMoji
        torchMoji_attDim=2304,# published model uses 2304
        torchMoji_crushedDim=32,
        torchMoji_BatchNorm=True,
        
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
        prenet_dim=256,        # 256 baseline
        prenet_layers=2,       # 2 baseline
        prenet_batchnorm=False,# False baseline
        prenet_bn_momentum=0.5,# Inverse smoothing factor, 0.1 = high smoothing, 0.9 = Almost no smoothing
        p_prenet_dropout  =0.5,# 0.5 baseline
        
        prenet_speaker_embed_dim=0,# speaker_embedding before encoder
        prenet_noise   =0.05,# Add Gaussian Noise to Prenet inputs. std defined here.
        prenet_blur_min=0.00,# Apply random vertical blur between prenet_blur_min
        prenet_blur_max=0.00,#                                and prenet_blur_max
                             # Set max to False or Zero to disable
        
        # (Decoder) AttentionRNN
        attention_rnn_dim=1280, # 1024 baseline
        AttRNN_extra_decoder_input=True,# False baseline # Feed DecoderRNN Hidden State into AttentionRNN
        AttRNN_hidden_dropout_type='dropout',# options ('dropout','zoneout')
        p_AttRNN_hidden_dropout=0.05,# 0.1 baseline
        
        # (Decoder) DecoderRNN
        decoder_rnn_dim=384, # 1024 baseline
        DecRNN_hidden_dropout_type='dropout',# options ('dropout','zoneout')
        p_DecRNN_hidden_dropout=0.1,# 0.1 baseline
        decoder_residual_connection=False,# residual connections with the AttentionRNN hidden state and Attention/Memory Context
        # Optional Second Decoder
        second_decoder_rnn_dim=0,# 0 baseline # Extra DecoderRNN to learn more complex patterns # set to 0 to disable layer.
        second_decoder_residual_connection=True,# residual connections between the DecoderRNNs
        
        # (Decoder) Attention parameters
        attention_type=0,
        # 0 -> Hybrid Location-Based Attention (Vanilla Tacotron2)
        # 1 -> GMMAttention (Long-form Synthesis)
        # 1 -> Dynamic Convolution Attention (Long-form Synthesis)
        attention_dim=192, # 128 Layer baseline # Used for Key-Query Dim
        
        # (Decoder) Attention Type 0 Parameters
        windowed_attention_range = 32,# set to 0 to disable
                                     # will set the forward and back distance the model can attend to.
                                     # 2 will give the model 5 characters it can attend to at any one time.
                                     # This will also allow more stable generation with extremely long text inputs and save VRAM during inference.
        windowed_att_pos_offset=1.25,# Offset the current_pos by this amount.
        windowed_att_pos_learned=True,
        
        # (Decoder) Attention Type 0 (and 2) Parameters
        attention_location_n_filters=32,  # 32 baseline
        attention_location_kernel_size=31,# 31 baseline
        
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
        use_postnet=True,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=6,
        postnet_residual_connections=3,# False baseline, int > 0 == n_layers in each residual block
        
        ##################################
        ## Optimization Hyperparameters ##
        ##################################
        use_saved_learning_rate=False,
        learning_rate=0.1e-5,# overriden by 'run_every_epoch.py'
        weight_decay=1e-6,
        grad_clip_thresh=1.0,# overriden by 'run_every_epoch.py'
        
        mask_padding=True,#mask values by setting them to the same values in target and predicted
        masked_select=True,#mask values by removing them from the calculation
        
        # (DFR) Drop Frame Rate
        global_mean_npy='global_mean.npy',
        drop_frame_rate=0.25,# overriden by 'run_every_epoch.py'
        
        ##################################
        ## Loss Weights/Scalars         ##
        ##################################
        # All of these can be overriden from 'run_every_epoch.py'
        spec_MSE_weight    = 1.0,# MSE Spectrogram Loss Before Postnet
        postnet_MSE_weight = 1.0,# MSE Spectrogram Loss After Postnet
        gate_loss_weight   = 1.0,# Gate Loss
        
        sylps_kld_weight = 0.0020, # SylNet KDL Weight
        sylps_MSE_weight = 0.01,# Encoder Pred Sylps MSE weight
        sylps_MAE_weight = 0.00,# Encoder Pred Sylps MAE weight
        
        diag_att_weight=0.05,# 'dumb' guided attention. Simply punishes the model for attention that is non-diagonal. Decreases training time and increases training stability with English speech. 
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
