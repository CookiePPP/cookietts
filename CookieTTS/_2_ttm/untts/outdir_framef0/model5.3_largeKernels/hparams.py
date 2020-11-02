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
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        fp16_run_optlvl=2, # options: [1, 2] | 1 is recommended for typical mixed precision usage
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers =["insert_module_here"],
        frozen_modules=["insert_module_here"], # only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        print_layer_names_during_startup=True,
        n_tensorboard_outputs=4, # Number of items to show in tensorboard images section (up to val_batch_size). Ordered by text length e.g: clip with most text will be first.
        
        ################################
        # Data Parameters              #
        ################################
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        use_saved_speakers=True, # use the speaker lookups saved inside the model instead of generating again
        raw_speaker_ids=True, # use the speaker IDs found in filelists for the internal IDs. Values over max_speakers will crash (as intended).
        
        check_files=True, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
        load_mel_from_disk=True,
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_train_taca2.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_validation_taca2.txt',
        
        dict_path='../../dict/merged.dict.txt',
        text_cleaners=['basic_cleaners'],
        p_arpabet = 0.5,
        start_token = "",#"☺"
        stop_token = "",#"␤"
        
        decoder_padding_value = 0.0,#-11.51,# padding value for first WN module
        spect_padding_value   = 0.0,#-11.51,# padding value used to fill batches of spects during training
        silence_pad_start = 0,
        silence_pad_end   = 0,
        
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
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=0.1e-5,
        weight_decay=1e-6,
        batch_size=16,
        val_batch_size=16,# for more precise comparisons between models, constant batch_size is useful
        use_TBPTT=False,
        truncated_length=1000, # max mel length till truncation.
        mask_padding=True,#mask values by setting them to the same values in target and predicted
        masked_select=True,#mask values by removing them from the calculation
        
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        n_speakers=512,
        
        # Synthesis/Inference Related
        max_decoder_steps=3000,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim=64, # speaker_embedding before encoder
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_conv_hidden_dim=512,
        encoder_LSTM_dim=768,
        
        Sylps_loss_scalar = 0.5,
        
        # (GST) TorchMoji
        torchMoji_attDim=2304,# published model uses 2304
        torchMoji_crushed_dim=32,# linear to crush dim
        
        # (MelEncoder) MelEncoder parameters
        melenc_enable=False,
        melenc_ignore_at_inference=True,
        melenc_drop_frame_rate=0.0,# replace mel channels at this fraction of timesteps to zeros.
        melenc_drop_rate=0.0, # chance to drop the ENTIRE melenc output
        melenc_speaker_embed_dim=64,
        melenc_kernel_size=3,
        melenc_stride     =3,
        melenc_n_layers=2,
        melenc_conv_dim=512,
        melenc_n_tokens=2,
        
        # (Memory)
        speaker_embedding_dim=64,
        
        # (Decoder) Decoder parameters
        MelGlow_loss_scalar = 1.0,# Can be overriden by 'run_every_epoch.py'
        sigma=1.0,
        grad_checkpoint=0,
        n_flows=6,
        n_group=256,
        n_early_every=4,
        n_early_size=40,
        mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (Decoder) Decoder Cond parameters
        cond_residual=False,
        cond_res_rezero=False,
        cond_weightnorm=True, # Causes NaN Gradients late into training.
        cond_layers=0,
        cond_act_func='lrelu',
        cond_padding_mode='zeros',
        cond_kernel_size=1,
        cond_hidden_channels=256,
        cond_output_channels=256,
        
        # (Decoder) WN parameters
        wn_n_channels=512,
        wn_kernel_size=17,
        wn_dilations_w=None, # use list() to specify multiple dilations
        wn_n_layers=1,# 1,  2,  3,  4,  5,  6,  7
                      # 1,  2,  4,  8, 16, 32, 64
                      # 3,  5,  9, 17, 33, 65,129
        wn_res_skip=False,      # enable when using more than 1 layer
        wn_merge_res_skip=True, # disable when using more than 1 layer
        wn_seperable_conv=True,
        
        # (Decoder) WN Cond parameters
        wn_cond_layers=1,
        wn_cond_act_func='none',
        wn_cond_padding_mode='zeros',
        wn_cond_kernel_size=1,
        wn_cond_hidden_channels=256,
        
        # (DurationGlow) Generative Duration Predictor
        DurGlow_enable=True, # use Duration Glow?
        DurGlow_loss_scalar = 0.1,# Can be overriden by 'run_every_epoch.py'
        dg_sigma=1.0,
        dg_grad_checkpoint=0,
        dg_n_flows=6,
        dg_n_group=2,
        dg_n_early_every=10,
        dg_n_early_size=2,
        dg_mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (DurationGlow) Decoder Cond parameters
        dg_cond_residual=False,
        dg_cond_res_rezero=False,
        dg_cond_weightnorm=True, # Can cause NaN Gradients late into training.
        dg_cond_layers=0,
        dg_cond_act_func='lrelu',
        dg_cond_padding_mode='zeros',
        dg_cond_kernel_size=1,
        dg_cond_hidden_channels=256,
        dg_cond_output_channels=256,
        
        # (DurationGlow) WN parameters
        dg_wn_n_channels=512,
        dg_wn_kernel_size=7,
        dg_wn_dilations_w=1, # use list() to specify multiple dilations
        dg_wn_n_layers=1,
        dg_wn_res_skip=False,      # ignore unless using more than 1 layer
        dg_wn_merge_res_skip=True, # ignore unless using more than 1 layer
        dg_wn_seperable_conv=False,
        
        # (DurationGlow) WN Cond parameters
        dg_wn_cond_layers=1,
        dg_wn_cond_act_func='none',
        dg_wn_cond_padding_mode='zeros',
        dg_wn_cond_kernel_size=1,
        dg_wn_cond_hidden_channels=256,
        
        # (VarGlow) Variational (Perceived Loudness, F0, Energy) Predictor
        VarGlow_loss_scalar = 1.0,# Can be overriden by 'run_every_epoch.py'
        var_sigma=1.0,        # Ignore
        var_grad_checkpoint=0,# Ignore
        var_n_flows=12,
        var_n_group=6,        # Ignore
        var_n_early_every=10, # Ignore
        var_n_early_size =2,  # Ignore
        var_mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (VarGlow) Decoder Cond parameters
        var_cond_residual=False,
        var_cond_res_rezero=False,
        var_cond_weightnorm=True, # Can cause NaN Gradients late into training.
        var_cond_layers=1,
        var_cond_act_func='lrelu',
        var_cond_padding_mode='zeros',
        var_cond_kernel_size=1,
        var_cond_hidden_channels=512,
        var_cond_output_channels=512,
        
        # (VarGlow) WN parameters
        var_wn_n_channels=512,
        var_wn_kernel_size=17,
        var_wn_dilations_w=1, # use list() to specify multiple dilations
        var_wn_n_layers=1,
        var_wn_res_skip=False,      # ignore unless using more than 1 layer
        var_wn_merge_res_skip=True, # ignore unless using more than 1 layer
        var_wn_seperable_conv=False,
        
        # (VarGlow) WN Cond parameters
        var_wn_cond_layers=1,
        var_wn_cond_act_func='none',
        var_wn_cond_padding_mode='zeros',
        var_wn_cond_kernel_size=1,
        var_wn_cond_hidden_channels=512,
    )
    
    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)
    
    if verbose:
        print('Final parsed hparams: %s', hparams.values())
    
    return hparams
