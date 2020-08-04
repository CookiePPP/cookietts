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
        check_files=True, # check all files exist, aren't corrupted, have text, good length, and other stuff before training.
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
        encoder_LSTM_dim=640,
        
        # (Length Predictor) Length Predictor parameters
        len_pred_filter_size=256,
        len_pred_kernel_size=3,
        len_pred_dropout=0.3,
        len_pred_n_layers=2,
        
        # (Attention) Positional Attention parameters
        pos_att_head_num=2,  # Number of Attention heads
        
        #    Sin/Cos Position Encoding
        pos_att_inv_freq=10000,# default 10000, defines maximum frequency of decoder positional encoding
        pos_att_enc_inv_freq=10000,# default 10000, defines maximum frequency of encoder outputs positional encoding
        pos_att_positional_encoding_for_key=True,   # add position information to encoder outputs key   (used for key-query match to generate alignments)
        pos_att_positional_encoding_for_value=True,# add position information to encoder outputs value (multiplied by alignment and sent to the decoder)
        
        #    Learned Embedding Position Encoding
        pos_enc_positional_embedding_kv=False, # Learned position embedding
        pos_enc_positional_embedding_q=False,  # Learned position embedding
        
        #    Attention Guiding
        pos_att_guided_attention=False, # 'dumb' guided attention, simply punishes the model for attention that is non-diagonal. Decreases training time and increases training stability with English speech. # As an example of how this works, if you imagine there is a 10 letter input that lasts 1 second. The first 0.1s is pushed towards using the 1st letter, the next 0.1s will be pushed towards using the 2nd letter, and so on for each chunk of audio. Since each letter has a different natural duration (especially punctuation), this attention guiding is not particularly accurate, so it's not recommended to use a high loss scalar later into training.
        pos_att_guided_attention_sigma=0.5,  # how relaxed the diagonal constraint is, default should be good for any speakers.
        pos_att_guided_attention_alpha=10.0, # loss scalar (the strength of the attention loss), high values can used during the start of training to keep all the attention heads in line, lower values should be used once the alignment has become diagonal.
        
        # (Attention) Speaker Embed
        speaker_embedding_dim=128,
        
        # (Decoder) Decoder parameters
        sigma=1.0,
        grad_checkpoint=0,
        n_flows=10,
        n_group=160,
        n_early_every=4,
        n_early_size=20,
        mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (Decoder) Decoder Cond parameters
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
        weight_decay=1e-6,
        batch_size=24,
        val_batch_size=24, # for more precise comparisons between models, constant batch_size is useful
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
