from CookieTTS.utils.text.symbols import symbols

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    from CookieTTS.utils.utils_hparam import HParams
    hparams = HParams(
        random_segments=False,# DONT MODIFY
        
        #################################
        ## Experiment Parameters       ##
        #################################
        epochs = 1000,
        
        n_models_to_keep=4,# NOT IMPLEMENTED # if over this number, will delete oldest checkpoint(s). Will never delete "best" checkpoints (as shown below).
        save_best_val_model = True,# save best teacher forced postnet MFSE as a seperate checkpoint.
                                   # This is basically best audio quality model.
        save_best_inf_attsc = True,# save best inference weighted attention score as a seperate checkpoint.
                              # This is basically best stability model.
        
        dynamic_loss_scaling=True,
        fp16_run        = False,# requires 20 Series or Better (e.g: RTX 2080 Ti, RTX 2060, Tesla V100, Tesla A100)
        fp16_run_optlvl = '2',
        
        distributed_run = True,
        dist_backend = "nccl",
        dist_url     = "tcp://127.0.0.1:54321",
        
        cudnn_enabled   =  True,
        cudnn_benchmark = False,
        seed = 1234,
        
        #################################
        ## Freezing/Reseting Modules   ##
        #################################
        print_layer_names_during_startup = False,# will print every modules key to be used below.
        ignore_layers    = ["layers_here"],# for `--warm_start`-ing
        frozen_modules   = ["layers_here"],# only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        unfrozen_modules = ["layers_here"],# modules that are unfrozen
        
        #################################
        ## Logging / Verbosity         ##
        #################################
        n_tensorboard_outputs=8,# number of items from validation so show in Tensorboard
        
        #################################
        ## Batch Size / Segment Length ##
        #################################
        batch_size    =12,# controls num of files processed in parallel per GPU
        val_batch_size=12,# for more precise comparisons between models, constant batch_size is useful
        
        max_segment_length=9999,# max mel length till a segment is sliced.
        max_chars_length  =9999,# max text input till text is sliced. I use segment_length/4.
                               # text slicing is ignored when using TBPTT
        
        sort_text_len_decending = False,# Leave disabled
        
        num_workers    =8,# (train) Number of threads for dataloading per GPU
        val_num_workers=2,# (eval)  Number of threads for dataloading per GPU
        prefetch_factor=8,# NOT IMPLEMENTED - Requires Pytorch 1.7 (so not right now)# Number of samples loaded in advance by each worker.
        
        ###################################
        ## Dataset / Filelist Parameters ##
        ###################################
        data_source = 1,# 0 to use nvidia/arflow_hifigan filelists, 1 to use automatic dataset processor
        
        # if data_source is 0:
        speakerlist     ='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        training_files  ='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/train_taca2.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/validation_taca2.txt',
        
        # if data_source is 1:
        dataset_folder = '/media/cookie/WD6TB/TTS/HiFiDatasets',
        dataset_metapath_append = '_arflowhifigan',
        dataset_audio_filters= ['*.wav','*.flac',],
        dataset_audio_rejects= ['*_Noisy_*','*_Very Noisy_*',],
        dataset_p_val = 0.005,# portion of dataset for Validation # default of 0.5% may be too small depending on the size of your dataset.
        dataset_min_duration =  0.9,# minimum duration in seconds for audio files to be added.
        dataset_max_duration = 12.0,# maximum duration in seconds for audio files being added.
                                    # use max_segment_length to control how much of each audio file can be used to fill VRAM during training.
        dataset_min_chars    =   12,# min number of letters/text that a transcript should have to be added to the audiofiles list.
        dataset_max_chars    =  256,# min number of letters/text that a transcript should have to be added to the audiofiles list.
                                    # use max_chars_length to control how much of text from each audio file can be used to fill VRAM during training.
        
        force_load  = True,# if a file fails to load, replace it with a random other file.
        
        ##################################
        ## Text / Speaker Parameters    ##
        ##################################
        text_cleaners=['basic_cleaners'],
        dict_path='../../dict/merged.dict.txt',
        p_arpabet=0.5, # probability to use ARPAbet / pronounciation dictionary on the text
        
        use_saved_speakers  = False,# use the speaker lookups saved inside the model instead of generating again
        numeric_speaker_ids =  True,# sort speaker_ids in filelist numerically, rather than alphabetically.
                                    # e.g:
                                    #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                    # instead of,
                                    #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
                                    # Mellotron repo has this off by default, but ON makes the most logical sense to me.
        
        check_files= True,# check all files exist, aren't corrupted, have text, good length, and other stuff before training.
                          # This can take a while as it has to load the entire dataset once.
        ##################################
        ## Audio Parameters             ##
        ##################################
        sampling_rate= 44100,
        target_lufs  = -25.0,# Loudness each file is rescaled to, use None for original file loudness.
        
        trim_enable      = True,# set to False to disable trimming completely
        trim_cache_audio = True,# save trimmed audio to disk to load later. Saves CPU usage, uses more disk space.
        filt_min_freq =    60.,# low freq
        filt_max_freq = 18000.,# top freq
        filt_order    =     6 ,# filter strength/agressiveness/whatever - don't set too high or things will break
        trim_margin_left  = [0.125, 0.05, 0.0375],
        trim_margin_right = [0.125, 0.05, 0.0375],
        trim_ref          = ['amax']*3,
        trim_top_db       = [   36,   41,   46],# volume/dB under reference that should be trimmed
        trim_window_length= [16384, 4096, 2048],
        trim_hop_length   = [ 2048, 1024,  512],
        trim_emphasis_str = [  0.0,  0.0,  0.0],
        
        ##################################
        ## Spectrogram Parameters       ##
        ##################################
        filter_length  =  2048,
        hop_length     =   512,
        win_length     =  2048,
        n_mel_channels =   160,
        mel_fmin       =    20.0,
        mel_fmax       = 11025.0,
        stft_clamp_val = 1e-5,# 1e-5 = original
        
        cache_mel=True,# save spectrograms to disk to load later. Saves CPU usage, uses more disk space.
                        # modifications to params below do not apply to already cached files.
        
        silence_value = -11.5129,# = ln(1e-5)
        silence_pad_start = 0,# frames to pad the start of each spectrogram
        silence_pad_end   = 0,# frames to pad the  end  of each spectrogram
                            # These frames will be added to the loss functions and Tacotron must predict and generate the padded silence.
        
        ######################################
        ## Synthesis / Inference Parameters ##
        ######################################
        max_decoder_steps = 3000,# max duration of an audio file during a single generation.
        
        ##################################
        ## Model Parameters             ##
        ##################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim    = 64, # speaker_embedding before encoder
        encoder_concat_speaker_embed = 'before_conv',# concat before encoder convs, or just before the LSTM inside decode. Options 'before_conv','before_lstm'
        encoder_kernel_size     =    5,# baseline 5
        encoder_n_convolutions  =    3,# baseline 3
        encoder_conv_hidden_dim =  512,# baseline 512
        encoder_LSTM_dim        = 1024,# baseline 512
        encoder_LSTM_n_layers   =    1,# baseline 1
        
        # (SylpsNet) Predicts speaking speed
        sylpsnet_layer_dims = [32, 32],# width of each layer, LeakyReLU() is used between hiddens
        
        # (Residual Encoder) Learns information related to anything that isn't related to Text or Speakers. (e.g: background noise)
        # https://openreview.net/pdf?id=Bkg9ZeBB37
        # THIS IS NOT WORKING CORRECTLY RIGHT NOW AND MAY BE REMOVED AT ANY TIME.
        use_res_enc = False,
        res_enc_filters   = [32, 32, 64, 64, 128, 128],
        res_enc_gru_dim   = 128,
        res_enc_n_tokens  =   4,
        res_enc_embed_dim =  64,
        
        use_res_enc_dis = False,# Enable/Disable Discriminator
        res_enc_dis_dim  = 128,
        res_enc_n_layers =   3,
        
        # (AuxEmotionNet) TorchMoji
        torchMoji_attDim     = 2304,# published model uses 2304
        torchMoji_crushedDim =   32,
        torchMoji_BatchNorm  = True,
        
        # (Speaker) Speaker embedding
        n_speakers            = 1024,# maximum number of speakers the model can support.
        speaker_embedding_dim =  256,# speaker embedding size # 128 baseline
        
        # (Decoder/Encoder) Bottleneck parameters
        # The outputs from the encoder, speaker, emotionnet and sylpsnet need to be mixed.
        # By default the information is mixed by the DecoderRNN, but this is repeated every spectrogram frame so likely wastes a massive amount of compute performing the same operations repeatedly.
        # Thus, this memory bottleneck can be used to mix the above mentioned outputs into a more compressed representation before decoding, allowing the DecoderRNN to be made smaller and more effective.
        use_memory_bottleneck  =  True,# False baseline
        memory_bottleneck_dim  =   768,# new memory size. 512 would be equivalent to the original tacotron2.
        memory_bottleneck_bias = False,
        
        # Memory Encoder
        mem_fft_n_heads  =    2,
        mem_fft_ff_dim   = 1536,
        mem_fft_n_layers =    4,
        
        # (Mixture Density Network) Guided Alignment using AlignTTS algorithm as target.
        enable_MDN = True,
        MDN_mel_downscale = 8,# Resize n_mel_channels to a smaller height to reduce VRAM usage of the alignment algorithm. 160/8 = 20, n_mel_channels/mel_downscale = 20
        mdn_n_heads  =    2,
        mdn_ff_dim   = 1536,
        mdn_n_layers =    4,# If you need more capacity, increase n_layers
        mdn_mel_enc_variational_tokens = False,
        mdn_mel_enc_conv_dim = 256,
        mdn_mel_enc_lstm_dim =  32,
        mdn_mel_enc_n_tokens =   4,
        durpred_n_heads  =    2,# 
        durpred_ff_dim   = 1024,# Duration Predictor, used to generate an Alignment when target audio doesn't exist. (aka Inference)
        durpred_n_layers =    4,# If you need more capacity, increase n_layers
        
        memory_efficient = True,# Saves forward pass states to recompute the gradients in chunks
                                 # Will reduce VRAM usage significantly at the cost of running parts of the model twice.
                                 # (The reduction in VRAM allows 4x batch size on an RTX 2080 Ti at the cost of 60% higher time-per-iter)
        
        # FESVD Normalizing Flow
        # Predicts Pitch + Energy + Spectral Tilt on a per-letter level
        arflow_n_flows              =  6,
        arflow_n_cond_layers        =  0,
        arflow_cond_hidden_channels =  0,
        arflow_cond_output_channels =  0,
        arflow_cond_kernel_size     =  0,
        arflow_cond_residual        = True,
        arflow_cond_padding_mode    = 'zeros',
        arflow_WN_config            = {
            'n_cond_layers'       :       1,
            'cond_hidden_channels':     512,
            'cond_kernel_size'    :       1,
            'cond_padding_mode'   : 'zeros',
            'seperable_conv'      :   False,
            'merge_res_skip'      :   False,
            'n_layers'            :       6,
            'n_channels'          :     384,
            'kernel_size_w'       :       3,
            'use_weight_norm'     :    True,
        },
        
        # (Decoder) Decoder parameters
        start_token = "~",#"☺"
        stop_token  = "~",#"␤"
        
        dec_n_heads  =    2,
        dec_ff_dim   = 1536,
        dec_n_layers =    4,
        dec_n_blocks =    2,
        
        # (optional) use a glow based decoder
        dec_glow = False,
        dec_glow_n_flows              =  8,
        dec_glow_n_cond_layers        =  0,
        dec_glow_cond_hidden_channels =  0,
        dec_glow_cond_output_channels =  0,
        dec_glow_cond_kernel_size     =  0,
        dec_glow_cond_residual        = True,
        dec_glow_cond_padding_mode    = 'zeros',
        dec_glow_WN_config            = {
            'n_cond_layers'       :       1,
            'cond_hidden_channels':     512,
            'cond_kernel_size'    :       1,
            'cond_padding_mode'   : 'zeros',
            'seperable_conv'      :   False,
            'merge_res_skip'      :   False,
            'n_layers'            :       5,
            'n_channels'          :     192,
            'kernel_size_w'       :       3,
            'use_weight_norm'     :    True,
        },
        dec_glow_shift_spect = 0.0,
        dec_glow_scale_spect = 1.0,
        
        # (Very Experimental) HiFi-GAN Latent Inputs
        HiFiGAN_enable    = False,
        HiFiGAN_cp_folder = '../../_4_mtw/hifigan/cp_hifigan_935universal44Khz_0_latent_ft',
        
        # used for reconstruction loss, not input
        HiFiGAN_filter_length = 2048,
        HiFiGAN_hop_length    =  512,
        HiFiGAN_win_length    = 2048,
        HiFiGAN_n_mel_channels=  160,
        HiFiGAN_clamp_val     = 1e-5,
        
        HiFiGAN_batch_size  =     4,# affects VRAM usage
        HiFiGAN_segment_size= 32768,# affects VRAM usage
        HiFiGAN_learning_rate= 1e-4,# overriden by 'run_every_epoch.py'
        HiFiGAN_lr_half_life =30000,# overriden by 'run_every_epoch.py'
        
        ##################################
        ## Optimization Hyperparameters ##
        ##################################
        use_saved_learning_rate=False,
        learning_rate   = 0.1e-5,# overriden by 'run_every_epoch.py'
        grad_clip_thresh= 1.0,   # overriden by 'run_every_epoch.py'
        weight_decay    = 1.0e-6,
        
        mask_padding  = True,#mask values by setting them to the same values in target and predicted
        masked_select = True,#mask values by removing them from the calculation
        
        # (DFR) Drop Frame Rate
        global_mean_npy = 'global_mean.npy',
        drop_frame_rate=  0.25,# overriden by 'run_every_epoch.py'
        
        ##################################
        ## Loss Weights/Scalars         ##
        ##################################
        # All of these can be overriden from 'run_every_epoch.py'
        spec_MSE_weight     = 0.0,# MSE  Spectrogram Loss Before Postnet
        spec_MFSE_weight    = 1.0,# MFSE Spectrogram Loss Before Postnet
        postnet_MSE_weight  = 0.0,# MSE  Spectrogram Loss After Postnet
        postnet_MFSE_weight = 1.0,# MFSE Spectrogram Loss After Postnet
        gate_loss_weight    = 1.0,# Gate Loss
        
        res_enc_kld_weight  = 0.002,
        res_enc_gMSE_weight = 0.1,
        
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
