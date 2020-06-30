import tensorflow as tf
from text.symbols import symbols


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
        #ignore_layers=["decoder.attention_layer.F.2.weight", "decoder.attention_layer.F.2.bias","decoder.attention_layer.F.0.linear_layer.weight","decoder.attention_layer.F.0.linear_layer.bias"],
        ignore_layers=["encoder.lstm.weight_ih_l0","encoder.lstm.weight_hh_l0","encoder.lstm.bias_ih_l0","encoder.lstm.bias_hh_l0","encoder.lstm.weight_ih_l0_reverse","encoder.lstm.weight_hh_l0_reverse","encoder.lstm.bias_ih_l0_reverse","encoder.lstm.bias_hh_l0_reverse","decoder.attention_rnn.weight_ih","decoder.attention_rnn.weight_hh","decoder.attention_rnn.bias_ih","decoder.attention_rnn.bias_hh","decoder.attention_layer.query_layer.linear_layer.weight","decoder.attention_layer.memory_layer.linear_layer.weight","decoder.decoder_rnn.weight_ih","decoder.linear_projection.linear_layer.weight","decoder.gate_layer.linear_layer.weight"],
        frozen_modules=["none-N/A"], # only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        print_layer_names_during_startup=True,
        
        ################################
        # Data Parameters              #
        ################################
        check_files=False, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
        load_mel_from_disk=True,
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt',
        use_saved_speakers=True, # use the speaker lookups saved inside the model instead of generating again
        raw_speaker_ids=False, # use the speaker IDs found in filelists for the internal IDs. Values over max_speakers will crash (as intended).
        training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/train_taca2_arpa.txt',
        validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/validation_taca2_arpa.txt',
        #training_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_train_taca2_merged.txt',
        #validation_files='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/mel_validation_taca2_merged.txt',
        text_cleaners=['basic_cleaners'],
        
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
        
        # Gate
        gate_positive_weight=10, # how much more valuable 1 positive frame is to 1 zero frame. 80 Frames per seconds, therefore values around 20 are fine.
        
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
        
        # (Decoder) Decoder parameters
        start_token = "",#"☺"
        stop_token = "",#"␤"
        hide_startstop_tokens=False, # trim first/last encoder output before feeding to attention.
        n_frames_per_step=1,    # currently only 1 is supported
        context_frames=1,   # TODO TODO TODO TODO TODO
        
        # (Decoder) Prenet
        prenet_dim=512,         # 256 baseline
        prenet_layers=2,        # 2 baseline
        prenet_batchnorm=False,  # False baseline
        p_prenet_dropout=0.5,   # 0.5 baseline
        prenet_speaker_embed_dim=0, # speaker_embedding before encoder
        
        # (Decoder) AttentionRNN
        attention_rnn_dim=1280, # 1024 baseline
        AttRNN_extra_decoder_input=True,# False baseline
        AttRNN_hidden_dropout_type='zoneout',# options ('dropout','zoneout')
        p_AttRNN_hidden_dropout=0.10,     # 0.1 baseline
        p_AttRNN_cell_dropout=0.00,       # 0.0 baseline
        
        # (Decoder) AttentionRNN Speaker embedding
        n_speakers=512,
        speaker_embedding_dim=256, # speaker embedding size # 128 baseline
        
        # (Decoder) DecoderRNN
        decoder_rnn_dim=1792,   # 1024 baseline
        extra_projection=False, # another linear between decoder_rnn and the linear projection layer (hopefully helps with high sampling rates and hopefully doesn't help decoder_rnn overfit)
        DecRNN_hidden_dropout_type='zoneout',# options ('dropout','zoneout')
        p_DecRNN_hidden_dropout=0.2,     # 0.1 baseline
        p_DecRNN_cell_dropout=0.00,       # 0.0 baseline
        
        # (Decoder) Attention parameters
        attention_type=0,
        # 0 -> Hybrid Location-Based Attention (Vanilla Tacotron2)
        # 1 -> GMMAttention (Long-form Synthesis)
        # 1 -> Dynamic Convolution Attention (Long-form Synthesis)
        attention_dim=128,      # 128 Layer baseline
        
        # (Decoder) Attention Type 0 (and 2) Parameters
        attention_location_n_filters=32,   # 32 baseline
        attention_location_kernel_size=31, # 31 baseline
        
        # (Decoder) Attention Type 1 Parameters
        num_att_mixtures=1,# 5 baseline
        attention_layers=1,# 1 baseline
        delta_offset=0,    # 0 baseline, values around 0.005 will push the model forwards. Since we're using the sigmoid function caution is suggested.
        delta_min_limit=0, # 0 baseline, values around 0.010 will force the model to move forward, in this example, the model cannot spend more than 100 steps on the same encoder output.
        lin_bias=False, # I need to figure out what that layer is called.
        initial_gain='relu', # initial weight distribution 'tanh','relu','sigmoid','linear'
        normalize_attention_input=True, # False baseline
        normalize_AttRNN_output=False,  # True baseline
        
        # (Decoder) Attention Type 2 Parameters
        dynamic_filter_num=128, # 8 baseline
        dynamic_filter_len=21, # 21 baseline # currently only 21 is supported
        
        # (Postnet) Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        
        # (GST) Reference encoder
        with_gst=True,
        ref_enc_pack_padded_seq=True, # TODO
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,
        
        # (GST) Multi-headed Attention Layer
        gstAtt_dim=128,
        num_heads=8,
        
        # (GST) Style Token Layer
        token_num=5, # acts as the information bottleneck.
        token_activation_func='tanh', # default 'softmax', options 'softmax','sigmoid','tanh','absolute'
        token_embedding_size=256, # token embedding size
        
        # (GST) TorchMoji
        torchMoji_attDim=2304,# published model uses 2304
        torchMoji_linear=True,# load/save text infer linear layer.
        torchMoji_training=True,# switch GST to torchMoji mode
        
        # (GST) Drop Style Tokens
        p_drop_tokens=0.0, # Nudge the decoder to infer style without GST's input
        drop_tokens_mode='zeros',#Options: ('zeros','halfs','embedding','speaker_embedding','emotion_embedding') # Replaces style_tokens with either a scaler or an embedding, or speaker embeddings
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        loss_func = 'MSELoss', # options 'MSELoss','SmoothL1Loss'
        learning_rate=0.1e-5,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        val_batch_size=64, # for more precise comparisons between models, constant batch_size is useful
        use_TBPTT=True,
        truncated_length=1000, # max mel length till truncation.
        mask_padding=True,#mask values by setting them to the same values in target and predicted
        masked_select=True,#mask values by removing them from the calculation
        
        # (DFR) Drop Frame Rate
        global_mean_npy='global_mean.npy',
        drop_frame_rate=0.25,
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
