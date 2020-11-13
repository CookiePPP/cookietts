import tensorflow as tf
from CookieTTS.utils.text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=1000,
        iters_per_checkpoint=1000,
        iters_per_validation=1000,
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
        check_files=1, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
                          # This can take a little as it has to simulate an entire EPOCH of dataloading.
        speakerlist='/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/speaker_ids.txt', # lets the checkpoints include speaker names.
        dict_path='../../dict/merged.dict.txt',
        use_saved_speakers=True,# use the speaker lookups saved inside the model instead of generating again
        numeric_speaker_ids=False, # sort speaker_ids in filelist numerically, rather than alphabetically.
                                   # e.g:
                                   #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                   # instead of,
                                   #    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] -> [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
                                   # Mellotron repo has this off by default, but ON makes the most logical sense to me.
        raw_speaker_ids=False,  # use the speaker IDs found in filelists for the internal IDs. Values greater than n_speakers will crash (as intended).
                                # This will disable sorting the ids
        training_files="/media/cookie/Samsung PM961/TwiBot/CookiePPPTTS/CookieTTS/_2_ttm/tacotron2/EncDurFilelist/map_train.txt",
        validation_files="/media/cookie/Samsung PM961/TwiBot/CookiePPPTTS/CookieTTS/_2_ttm/tacotron2/EncDurFilelist/map_val.txt",
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
        emotionnet_encoder_outputs_dropout=0.0,# Encoder Outputs Dropout
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
        auxemotionnet_encoder_outputs_dropout=0.0,# Encoder Outputs Dropout
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
        
        # (Duration Predictor) parameters
        len_pred_filter_size=512,
        len_pred_kernel_size=3,
        len_pred_dropout=0.2,
        len_pred_n_layers=3,
        
        # (Decoder) parameters
        z_dim = 128,
        gblock_kernel_size = 3,
        in_channels = 512,
        decoder_dims   = [768, 768, 768, 384, 384, 384, 256, 192, 192],
        decoder_scales = [1  , 1  , 1  , 2  , 2  , 2  , 3  , 5  , 5  ],# upsample from 12.5ms hop_length features
                        #  80,  80,  80, 160, 320, 640,1920,9600,48000 Hz
        dilations = [1,2,4,8], # dilations of each layer in each block.
        
        # (Destriminator(s)) parameters
        d_dilations = [1, 2],
        descriminator_base_window = 600, # scaled by in_channels for each descriminator.
        descriminator_configs = [
            # Using Conditional Features
            [
                1,# in_channels
                [128, 128, 128, 256, 256, 384, 512, 512, 512], # dims
                [  0,   0,   0,   0,   0,   0,   1,   0,   0], # use_cond
                [  5,   5,   3,   2,   2,   2,   1,   1,   1], # scales
            ], [
                2,# in_channels
                [128, 128, 128, 256, 256, 512, 512, 512], # dims
                [  0,   0,   0,   0,   0,   1,   0,   0], # use_cond
                [  5,   5,   3,   2,   2,   1,   1,   1], # scales
            ], [
                4,# in_channels
                [128, 128, 128, 256, 512, 512, 512], # dims
                [  0,   0,   0,   0,   1,   0,   0], # use_cond
                [  5,   5,   3,   2,   1,   1,   1], # scales
            ], [
                8,# in_channels
                [128, 128, 256, 512, 512, 512], # dims
                [  0,   0,   0,   1,   0,   0], # use_cond
                [  5,   5,   3,   1,   1,   1], # scales
            ], [
                15,# in_channels
                [128, 256, 256, 384, 512, 512, 512], # dims
                [  0,   0,   0,   0,   1,   0,   0], # use_cond
                [  5,   2,   2,   2,   1,   1,   1], # scales
            ], [
                30,# in_channels
                [128, 256, 384, 512, 512, 512], # dims
                [  0,   0,   0,   1,   0,   0], # use_cond
                [  5,   2,   2,   1,   1,   1], # scales
            ],
            # Without Features
            [
                8,# in_channels
                [128, 128, 256, 512, 512, 512], # dims
                [  0,   0,   0,   0,   0,   0], # use_cond
                [  5,   5,   3,   1,   1,   1], # scales
            ], [
                8,# in_channels
                [128, 128, 256, 512, 512, 512], # dims
                [  0,   0,   0,   0,   0,   0], # use_cond
                [  5,   5,   3,   1,   1,   1], # scales
            ], [
                8,# in_channels
                [128, 128, 256, 512, 512, 512], # dims
                [  0,   0,   0,   0,   0,   0], # use_cond
                [  5,   5,   3,   1,   1,   1], # scales
            ], [
                150,# in_channels
                [256, 384, 512, 512, 512], # dims
                [  0,   0,   0,   0,   0], # use_cond
                [  2,   2,   1,   1,   1], # scales
            ],
        ],
        ################################
        # Optimization Hyperparameters #
        ################################
        weight_decay=1e-6,
        batch_size=4,     # controls num of files processed in parallel per GPU
        val_batch_size=4, # for more precise comparisons between models, constant batch_size is useful
        segment_length=96000,
        ################################
        # Loss Weights/Scalars         #
        ################################
        duration_predictor_weight = 1.0,
    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams