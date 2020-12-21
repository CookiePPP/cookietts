from CookieTTS.utils.text.symbols import symbols

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    from CookieTTS.utils.utils_hparam import HParams
    hparams = HParams(
        random_segments=True,# DONT MODIFY
        
        #################################
        ## Experiment Parameters       ##
        #################################
        epochs = 1000,
        
        n_models_to_keep=4,# NOT IMPLEMENTED # if over this number, will delete oldest checkpoint(s). Will never delete "best" checkpoints (as shown below).
        save_best_val_model = True,# save best MFSE as a seperate checkpoint.
                                   # This is basically best audio quality model, it does not represent most accurate speaker
        
        dynamic_loss_scaling=True,
        fp16_run        = False,# requires 20 Series or Better (e.g: RTX 2080 Ti, RTX 2060, Tesla V100, Tesla A100)
        fp16_run_optlvl = '2',
        
        distributed_run = False,
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
        batch_size    =16,# controls num of files processed in parallel per GPU
        val_batch_size=16,# for more precise comparisons between models, constant val_batch_size is useful
        
        use_TBPTT  =False,# continue processing longer files into the next training iteration
        max_segment_length=1024,# max mel length till a segment is sliced.
        
        num_workers    =8,# (train) Number of threads for dataloading per GPU
        val_num_workers=8,# (eval)  Number of threads for dataloading per GPU
        prefetch_factor=8,# NOT IMPLEMENTED # Number of samples loaded in advance by each worker.
        
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
        dataset_min_duration =  1.5,# minimum duration in seconds for audio files to be added.
        dataset_max_duration = 30.0,# maximum duration in seconds for audio files being added.
                                    # use max_segment_length to control how much of each audio file can be used to fill VRAM during training.
        dataset_min_chars    =   16,# min number of letters/text that a transcript should have to be added to the audiofiles list.
        dataset_max_chars    =  256,# min number of letters/text that a transcript should have to be added to the audiofiles list.
                                    # use max_chars_length to control how much of text from each audio file can be used to fill VRAM during training.
        
        n_speakers = 2048,
        
        force_load  = True,# if a file fails to load, replace it with a random other file.
        ##################################
        ## Text / Speaker Parameters    ##
        ##################################
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
        
        trim_enable = True,# set to False to disable trimming completely
        trim_cache_audio = False,# save trimmed audio to disk to load later. Saves CPU usage, uses more disk space.
                                 # modifications to params below do not apply to already cached files.
        trim_margin_left  = [0.0125]*3,
        trim_margin_right = [0.0125]*3,
        trim_ref          = ['amax']*3,
        trim_top_db       = [   48,   46,   46],
        trim_window_length= [16384, 4096, 2048],
        trim_hop_length   = [ 2048, 1024,  512],
        trim_emphasis_str = [  0.0,  0.0,  0.0],
        
        ##################################
        ## Spectrogram Parameters       ##
        ##################################
        filter_length  =  2048,
        hop_length     =   512,
        win_length     =  2048,
        n_mel_channels =   80,
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
        gate_threshold    = 0.5,      # to be removed
        gate_delay        = 10,       # to be removed
        max_decoder_steps = 3000,     # to be removed
        n_symbols=len(symbols),       # to be removed
        symbols_embedding_dim=512,    # to be removed
        gate_positive_weight =10,     # to be removed
        p_teacher_forcing      = 1.00,# to be removed
        teacher_force_till     = 20,  # to be removed
        val_p_teacher_forcing  = 0.80,# to be removed
        val_teacher_force_till = 20,  # to be removed
        
        ##################################
        ## Model Parameters             ##
        ##################################
        
        # (Misc)
        use_causal_convs=False,# this will remove the ability for any conv layers to use information from the future
                              # which will roughly half the inference latency of this network, but may negatively affect audio quality
        
        # (Encoder) Encoder parameters
        lstm_dim      = 768,
        n_lstm_layers =   3,
    
        speaker_encoder_dim = 256,
        
        ##################################
        ## Optimization Hyperparameters ##
        ##################################
        use_saved_learning_rate=False,
        learning_rate = 0.1e-5,# overriden by 'run_every_epoch.py'
        grad_clip_thresh=1.0,  # overriden by 'run_every_epoch.py'
        weight_decay  = 1e-6,
    )

    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s', hparams.values())

    return hparams
