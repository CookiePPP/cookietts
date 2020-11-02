import tensorflow as tf
from text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=[0],

        ################################
        # Data Parameters              #
        ################################
        load_mel_from_disk=True,
        training_files='mel_train_taca2_merged.txt',
        validation_files='mel_validation_taca2_merged.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=48000,
        filter_length=2400,
        hop_length=600,
        win_length=2400,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=18000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512, # text embedding size

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1280, # main decoder param # 1024 original
        prenet_dim=256, # original 256
        max_decoder_steps=3500,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        p_teacher_forcing=0.8,

        # Attention parameters
        attention_rnn_dim=1024, # 1024 original
        attention_dim=128, # 128 original

        # Location Layer parameters
        attention_location_n_filters=128, # 32 original
        attention_location_kernel_size=31, # 31 original

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Speaker embedding
        n_speakers=240,
        speaker_embedding_dim=128, # speaker embedding size

        # Reference encoder
        with_gst=True,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,

        # Style Token Layer
        token_embedding_size=256, # token embedding size
        token_num=10,
        num_heads=8,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=100e-5,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=27, # 32*3 = 0.377 val loss, # 2 = 0.71 val loss
        mask_padding=True,  # set model's padded outputs to padded values

        ##################################
        # MMI options                    #
        ##################################
        drop_frame_rate=0.00,
        use_mmi=True,
        use_gaf=True,
        max_gaf=0.02,
        global_mean_npy='global_mean.npy'
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
