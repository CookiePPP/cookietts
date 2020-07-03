import tensorflow as tf
from CookieTTS.utils.text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=2000,
        iters_per_validation=2000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=["none-N/A"],
        frozen_modules=["none-N/A"], # only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        print_layer_names_during_startup=True,
        
        ################################
        # Data Parameters              #
        ################################
        check_files=False, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
        load_mel_from_disk=True,
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        use_saved_speakers=True, # use the speaker lookups saved inside the model instead of generating again
        raw_speaker_ids=True, # use the speaker IDs found in filelists for the internal IDs. Values over max_speakers will crash (as intended).
        #training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/train_taca2_arpa.txt',
        #validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/validation_taca2_arpa.txt',
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_train_taca2_merged.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_validation_taca2_merged.txt',
        text_cleaners=['basic_cleaners'],
        start_token="",
        stop_token ="",
        
        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=48000,
        filter_length=2400,
        hop_length=600,
        win_length=2400,
        n_mel_channels=160,
        mel_fmin=0.0,
        mel_fmax=16000.0,
        
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # Synthesis/Inference Related
        max_decoder_steps=3000,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim=64, # speaker_embedding before encoder
        encoder_concat_speaker_embed='before_conv', # concat before encoder convs, or just before the LSTM inside decode. Options 'before_conv','before_lstm'
        n_speakers=512,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_conv_hidden_dim=512,
        encoder_LSTM_dim=512,
        
        # (Length Predictor) Length Predictor parameters
        len_pred_filter_size=256,
        len_pred_kernel_size=3,
        len_pred_dropout=0.3,
        len_pred_n_layers=2,
        
        # (Attention) Positional Attention parameters
        pos_att_head_num=4,
        
        # (Attention) Speaker Embed
        speaker_embedding_dim=128,
        
        # (Decoder) Decoder parameters
        sigma=1.0,
        grad_checkpoint=0,
        n_flows=10,
        n_group=160,
        n_early_every=4,
        n_early_size=2,
        mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (Decoder) Cond parameters
        cond_residual=False,
        cond_res_rezero=False,
        cond_layers=0,
        cond_act_func='lrelu',
        cond_padding_mode='zeros',
        cond_kernel_size=1,
        cond_hidden_channels=256,
        cond_output_channels=256,
        
        # (Decoder) WN parameters
        wn_n_channels=256,
        wn_kernel_size=3,
        wn_dilations_w=1, # use list() to specify multiple dilations
        wn_n_layers=1,
        wn_res_skip=False,      # ignore unless using more than 1 layer
        wn_merge_res_skip=True, # ignore unless using more than 1 layer
        wn_seperable_conv=False,
        
        # (Decoder) WN Cond parameters
        wn_cond_layers=1,
        wn_cond_act_func='none',
        wn_cond_padding_mode='zeros',
        wn_cond_kernel_size=1,
        wn_cond_hidden_channels=256,
        
        # (GST) TorchMoji
        torchMoji_attDim=2304,# published model uses 2304
        torchMoji_linear=True,# load/save text infer linear layer.
        torchMoji_training=True,# switch GST to torchMoji mode
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=0.1e-5,
        weight_decay=1e-7,
        grad_clip_thresh=100.0,
        batch_size=32,
        val_batch_size=32, # for more precise comparisons between models, constant batch_size is useful
        use_TBPTT=False,
        truncated_length=1000, # max mel length till truncation.
        mask_padding=True,#mask values by setting them to the same values in target and predicted
        masked_select=True,#mask values by removing them from the calculation
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
