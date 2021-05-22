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
        
        dynamic_loss_scaling=True,
        fp16_run        = False,# requires 20 Series or Better (e.g: RTX 2080 Ti, RTX 2060, Tesla V100, Tesla A100)
        fp16_run_optlvl = '2',
        
        distributed_run = True,
        dist_backend = "nccl",
        dist_url     = "tcp://127.0.0.1:54321",
        
        cudnn_enabled   = True,
        cudnn_benchmark = True,
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
        n_tensorboard_outputs=8,# number of items from validation to show in Tensorboard
        
        #################################
        ## Batch Size / Segment Length ##
        #################################
        batch_size    =8,# controls num of files processed in parallel per GPU
        val_batch_size=8,# for more precise comparisons between models, constant batch_size is useful
        
        max_segment_length=32,# mel length
        val_max_segment_length=192,# mel length
        
        sort_text_len_decending = False,# Leave disabled
        
        num_workers    =2,# (train) Number of threads for dataloading per GPU
        val_num_workers=2,# (eval)  Number of threads for dataloading per GPU
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
        dataset_metapath_append = '_Oggless',
        dataset_audio_filters= ['*.wav','*.flac'],
        dataset_audio_rejects= ['*_Noisy_*','*_Very Noisy_*',],
        dataset_p_val = 0.005,# portion of dataset for Validation # default of 0.5% may be too small depending on the size of your dataset.
        dataset_min_duration =  0.9,# minimum duration in seconds for audio files to be added.
        dataset_max_duration = 10.0,# maximum duration in seconds for audio files being added.
                                    # use max_segment_length to control how much of each audio file can be used to fill VRAM during training.
        dataset_min_chars    =   12,# min number of letters/text that a transcript should have to be added to the audiofiles list.
        dataset_max_chars    =  256,# min number of letters/text that a transcript should have to be added to the audiofiles list.
                                    # use max_chars_length to control how much of text from each audio file can be used to fill VRAM during training.
        skip_empty_datasets = True,
        
        minimum_sampling_rate = 44000.0,
        
        force_load  = True,# if a file fails to load, replace it with a random other file.
        
        check_files= True,# check all files exist, aren't corrupted, have text, good length, and other stuff before training.
                          # This can take a while as it has to load the entire dataset once.
        ##################################
        ## Audio Parameters             ##
        ##################################
        sampling_rate= 44100,
        target_lufs  = -24.0,# Loudness each file is rescaled to, use None for original file loudness.
        
        trim_enable      = True,# set to False to disable trimming completely
        trim_cache_audio = True,# save trimmed audio to disk to load later. Saves CPU usage, uses more disk space.
        filt_min_freq =    60.,# low freq
        filt_max_freq = 18000.,# top freq
        filt_order    =     3 ,# filter strength/agressiveness/whatever - don't set too high or things will break
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
        n_mel_channels =   128,
        mel_fmin       =    20.0,
        mel_fmax       = 16000.0,
        stft_clamp_val = 1e-5,# 1e-5 = original
        
        cache_mel=False,# save spectrograms to disk to load later. Saves CPU usage, uses more disk space and IOPS.
                        # modifications to params below do not apply to already cached files.
        
        silence_value = -11.5129,# = ln(1e-5)
        silence_pad_start = 0,# frames to pad the start of each spectrogram
        silence_pad_end   = 0,# frames to pad the  end  of each spectrogram
                            # These frames will be added to the loss functions and Tacotron must predict and generate the padded silence.
        
        
        HiFiGAN_cp_folder = '../../_4_mtw/hifigan/cp_hifigan_44Khz_VDVAE_latent_01',
        
        # mel params used for reconstruction loss, not input
        HiFiGAN_filter_length = 2048,
        HiFiGAN_hop_length    =  512,
        HiFiGAN_win_length    = 2048,
        HiFiGAN_n_mel_channels=  128,
        HiFiGAN_clamp_val     = 1e-5,
        
        ##################################
        ## Optimization Hyperparameters ##
        ##################################
        learning_rate   = 0.1e-5,# overriden by 'run_every_epoch.py'
        grad_clip_thresh= 1.0,   # overriden by 'run_every_epoch.py'
        weight_decay    = 0.0,
    )

    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s', hparams.values())

    return hparams
