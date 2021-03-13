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
        fp16_run        = True,# requires 20 Series or Better (e.g: RTX 2080 Ti, RTX 2060, Tesla V100, Tesla A100)
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
        n_tensorboard_outputs_highloss=5, # NOT IMPLEMENTED # top X tacotron outputs with worst validation loss.
        n_tensorboard_outputs_badavgatt=5,# NOT IMPLEMENTED # top X tacotron outputs with weakest average attention.
        
        #################################
        ## Batch Size / Segment Length ##
        #################################
        batch_size    =8,# controls num of files processed in parallel per GPU
        val_batch_size=8,# for more precise comparisons between models, constant batch_size is useful
        
        use_TBPTT = False,# continue processing longer files into the next training iteration.
                         # allows very large inputs to be learned from
                         # and a large speedup in training speed by decreasing max_segment_length and increasing batch_size.
        
        max_segment_length=1024,# max mel length till a segment is sliced.
        max_chars_length  =192,# max text input till text is sliced. I use segment_length/4.
                               # text slicing is ignored when using TBPTT
        
        gradient_checkpoint      = False,# Saves forward pass states to recompute the gradients in chunks
                                         # Will reduce VRAM usage significantly at the cost of running parts of the model twice.
                                         # (The reduction in VRAM allows 4x batch size on an RTX 2080 Ti at the cost of 80% higher time-per-iter)
        checkpoint_decode_chunksize=2048,# recommend the square root of max_segment_length or tune manually
                                         # Changing this should have a large impact on VRAM usage when using gradient_checkpoint 
                                         # and make no difference to anything when gradient_checkpoint is disabled.
        
        sort_text_len_decending = False,# This is legacy, leave disabled
        
        min_avg_max_att       =  0.45,# files under this alignment strength are filtered out of the dataset during training.
        max_diagonality       =  1.25,# files  over this     diagonality    are filtered out of the dataset during training.
        max_spec_mse          =  1.00,# files  over this mean squared error are filtered out of the dataset during training.
        p_missing_enc         =  0.08,# 
        
        min_avg_max_att_start =9e9,# when to start filtering out weak alignments.
                                      # (normally mis-labelled files or files that are too challenging to learn)
                                      # Only applies to training dataset.
                                      # Only updates at the end of each epoch.
        
        num_workers    =16,# (train) Number of threads for dataloading per GPU
        val_num_workers=16,# (eval)  Number of threads for dataloading per GPU
        prefetch_factor=8,# NOT IMPLEMENTED - Requires Pytorch 1.7 (so not right now)# Number of samples loaded in advance by each worker.
        
        ###################################
        ## Dataset / Filelist Parameters ##
        ###################################
        data_source = 1,# 0 to use nvidia/tacotron2 filelists, 1 to use automatic dataset processor
        
        # if data_source is 0:
        speakerlist     ='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        training_files  ='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/train_taca2.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/validation_taca2.txt',
        
        # if data_source is 1:
        dataset_folder = '/media/cookie/WD6TB/TTS/HiFiDatasets',
        dataset_audio_filters= ['*.wav','*.flac',],
        dataset_audio_rejects= ['*_Noisy_*','*_Very Noisy_*',],
        dataset_p_val = 0.005,# portion of dataset for Validation # default of 0.5% may be too small depending on the size of your dataset.
        dataset_min_duration =  0.9,# minimum duration in seconds for audio files to be added.
        dataset_max_duration = 30.0,# maximum duration in seconds for audio files being added.
                                    # use max_segment_length to control how much of each audio file can be used to fill VRAM during training.
        dataset_min_chars    =   12,# min number of letters/text that a transcript should have to be added to the audiofiles list.
        dataset_max_chars    =  256,# min number of letters/text that a transcript should have to be added to the audiofiles list.
                                    # use max_chars_length to control how much of text from each audio file can be used to fill VRAM during training.
        
        force_load  = True,# if a file fails to load, replace it with a random other file.
        
        inference_equally_sample_speakers=True,# Will change the 'inference' results to use the same number of files from each speaker.
                                               # This makes sense if the speakers you want to clone aren't the same as the speakers with the most audio data.
        
        speaker_mse_sampling_start = 999999,# when True, instead of loading each audio file in order, load audio files
                                         #        randomly with higher probability given to more challenging speakers.
                                         # This must start after "min_avg_max_att_start" has started.
                                         #  - Without filtering out outlier files, this would just enchance the damage that mislablled files and noisy speakers do to the model.
                                         # EDIT: After testing, this option does not seem to be beneficial for most cases
        speaker_mse_exponent       = 1.0,# Power for weighting speakers sampling chances.
                                         # 0.0 = All speakers are sampled equally often. error values also have no effect.
                                         # 1.0 = Speakers are sampled proportional to their spectrogram error,
                                         #     e.g: double MSE = double chance to appear in training set.
                                         # 2.0 = Speakers are sampled with weighting by the square of their errors,
                                         #     e.g: double MSE = quadrulple chance to appear in training set.
        
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
        target_lufs  = -27.0,# Loudness each file is rescaled to, use None for original file loudness.
        
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
        
        cache_mel=False,# save spectrograms to disk to load later. Saves CPU usage, uses more disk space.
                        # modifications to params below do not apply to already cached files.
        
        silence_value = -11.5129,# = ln(1e-5)
        silence_pad_start = 0,# frames to pad the start of each spectrogram
        silence_pad_end   = 0,# frames to pad the  end  of each spectrogram
                            # These frames will be added to the loss functions and Tacotron must predict and generate the padded silence.
        
        ######################################
        ## Synthesis / Inference Parameters ##
        ######################################
        gate_threshold    = 0.5, # confidence required to end the audio file.
        gate_delay        = 10,  # allows the model to continue generative audio slightly after the audio file ends.
        max_decoder_steps = 3000,# max duration of an audio file during a single generation.
        
        ##################################
        ## Outer Model Parameters       ##
        ##################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        start_token = "",#"☺"
        stop_token  = "",#"␤"
        
        ##################################
        ## Tacotron Model Parameters    ##
        ##################################
        # (AuxEmotionNet) TorchMoji
        tt_torchMoji_attDim     = 2304,# published model uses 2304
        tt_torchMoji_crushedDim =   32,
        tt_torchMoji_BatchNorm  = True,
        
        # (Speaker) Speaker embedding
        n_speakers            = 2048,# maximum number of speakers the model can support.
        speaker_embedding_dim =  256,# speaker embedding size # 128 baseline
        
        # (Encoder) Encoder parameters
        encoder_dim        =512,
        encoder_n_layers   =  3,
        encoder_kernel_size=  3,
        encoder_n_blocks   = 12,
        use_stilt_40  = True,
        use_stilt_80  = True,
        use_stilt_120 = True,
        
        # (Decoder) Decoder parameters
        decoder_dim        =512,
        decoder_n_layers   =  3,
        decoder_kernel_size=  3,
        decoder_n_blocks   = 12,
        
        ##################################
        ## Optimization Hyperparameters ##
        ##################################
        use_saved_learning_rate=False,
        learning_rate = 0.1e-5,# overriden by 'run_every_epoch.py'
        grad_clip_thresh=1.0,  # overriden by 'run_every_epoch.py'
        weight_decay  = 1e-6,
        
        mask_padding  = True,#mask values by setting them to the same values in target and predicted
        masked_select = True,#mask values by removing them from the calculation
        
        # (DFR) Drop Frame Rate
        global_mean_npy = 'global_mean.npy',
        drop_frame_rate=  0.25,# overriden by 'run_every_epoch.py'
    )

    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s', hparams.values())

    return hparams