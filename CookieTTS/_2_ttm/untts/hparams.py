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
        fp16_run_optlvl=1, # options: [1, 2] | 1 is recommended for typical mixed precision usage
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://127.0.0.1:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=["decoder.positional_attention.positional_embedding.inv_freq", "decoder.positional_attention.enc_positional_embedding.inv_freq", "decoder.positional_attention.positional_embedding.ext_scalar"],
        frozen_modules=["none-N/A"], # only the module names are required e.g: "encoder." will freeze all parameters INSIDE the encoder recursively
        print_layer_names_during_startup=True,
        n_tensorboard_outputs=4, # Number of items to show in tensorboard images section (up to val_batch_size). Ordered by text length e.g: clip with most text will be first.
        
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
        start_token = "☺",#"☺"
        stop_token = "",#"␤"
        decoder_padding_value = -11.51,# padding value for first WN module
        spect_padding_value = -11.51, # padding value used to fill batches of spects during training
        
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
        n_speakers=512,
        
        # Synthesis/Inference Related
        max_decoder_steps=3000,
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim=64, # speaker_embedding before encoder
        encoder_concat_speaker_embed='before_conv', # concat before encoder convs, or just before the LSTM inside decode. Options 'before_conv','before_lstm'
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_conv_hidden_dim=512,
        encoder_LSTM_dim=768,
        
        # (MelEncoder) MelEncoder parameters
        melenc_enable=True,
        melenc_ignore_at_inference=True,
        melenc_drop_frame_rate=0.0,# replace mel channels at this fraction of timesteps to zeros.
        melenc_drop_rate=0.0, # chance to drop the ENTIRE melenc output
        melenc_speaker_embed_dim=64,
        melenc_kernel_size=3,
        melenc_n_layers=2,
        melenc_conv_dim=512,
        
        # (Length Predictor) Length Predictor parameters
        len_pred_input='embedding', # 'encoder' or 'embedding'
        len_pred_filter_size=512,
        len_pred_kernel_size=3,
        len_pred_dropout=0.3,
        len_pred_n_layers=3,
        len_pred_loss_weight=0.1,
        
        # (Attention Guiding) Diagonal Attention Guiding
        pos_att_guided_attention=True, # 'dumb' guided attention, simply punishes the model for attention that is non-diagonal. Decreases training time and increases training stability with English speech. # As an example of how this works, if you imagine there is a 10 letter input that lasts 1 second. The first 0.1s is pushed towards using the 1st letter, the next 0.1s will be pushed towards using the 2nd letter, and so on for each chunk of audio. Since each letter has a different natural duration (especially punctuation), this attention guiding is not particularly accurate, so it's not recommended to use a high loss scalar later into training.
        pos_att_guided_attention_sigma=0.5, # how relaxed the diagonal constraint is, default should be good for any speakers.
        pos_att_guided_attention_alpha=2.5, # loss scalar (the strength of the attention loss)
        
        # (Attention Guiding) Length Predictor Attention Guiding
        pos_att_len_pred_guided_att=False, # use length predictor to guide attention.
        pos_att_len_pred_att_weight=0.5, # use length predictor to guide attention.
        
        # (Attention Guiding) Path Loss
        att_path_loss=True, # Loss based on difference between adjacent alignment frames, force the model to attend to text in order.
        att_path_loss_weight=1.0,
        
        # (Attention Option 1) Attention parameters
        pos_att_dim=512,     # Width of Attention Layers (and inputs to Decoder) #  !! USED BY ALL ATTENTION TYPES !!
        n_mha_layers=1,      # Number of Attention Layers
        pos_att_head_num=1,  # Number of Attention Heads in each Layer
        pos_att_conv_block_groups=1,# See 'groups' param in pytorch Conv1d
        speaker_embedding_dim=256,
        pos_att_t_min = 1/2, # set to None to disable random attention temperature.
        pos_att_t_max = 2,
        use_duration_predictor_for_inference=False,# this will override option 1 during inference, however may produce better results.
        use_durationGlow_for_inference=False,      # this will override option 1 during inference, however may produce better results.
        
        # (Attention Option 1) Decoder Position Encoding
        pos_att_inv_freq=10000, # default 10000, defines maximum frequency of decoder positional encoding
        pos_att_step_size=1,
        pos_att_positional_encoding_for_query=True,# add position information to decoder query
        pos_enc_positional_embedding_q=False,       # use Learned Position Embeddings for Query
        rezero_pos_enc_q=False,
        
        # (Attention Option 1) Encoder Position Encoding
        pos_att_enc_inv_freq=10000,# default 10000, defines maximum frequency of encoder outputs positional encoding
        pos_att_enc_step_size=5,
        pos_att_use_duration_predictor_for_scalar=True,# use duration predictor for step size instead of having it static
        pos_att_duration_predictor_learn_scalar=True,# learned mix of ext and static
        pos_att_use_duration_predictor_teacher_forcing=1.0,# Chance to use Ground Truth 'Frames per Letter' instead of Predicted 'Frames per Letter'.
        pos_att_positional_encoding_for_key=True,   # add position information to encoder outputs key   (used for key-query match to generate alignments)
        pos_att_positional_encoding_for_value=False, # add position information to encoder outputs value (multiplied by alignment and sent to the decoder)
        pos_enc_positional_embedding_kv=False,      # use Learned Position Embeddings for Key/Value
        rezero_pos_enc_kv=False,
        
        # (Attention Option 2) Use Duration Predictor to produce Attention during training
        use_duration_predictor_for_attention=False,# this will disable option 1 and use option 2
        
        # (Attention Option 3) Use MoBoAlignerAttention
        use_MoboAttention=False,# this will disable option 1 and option 2
        
        # (Attention Option 4) Use Recurrent Location/Content Dot-Product Attention
        use_BasicAttention=False,# this will disable option 1, 2 and 3
        bas_att_location_kw=7,
        
        # (Attention Option 5) Use GMM Based Relative Attention
        use_GMMAttention=False,# this will disable option 1, 2, 3 and 4
         # use `pos_att_dim=128,` to control the dim of this Attention.
        gmm_att_n_layers=2,     # LeakyReLU operations on the MelEncoder outputs, unknown performance impact
        gmm_att_num_mixtures=1, # Equiv to Head Number, lets the network explore more around the local text area and/or increases attention capacity
        gmm_att_delta_min_limit=0.0,
        gmm_att_delta_offset=0.0, # min change in text position every frame, 0.0 allows the network to stand still and say the same letter forever, -1.0 would let the network speak backwards. 0.005 will force the model forward though the text no matter what. (there is a clip of Pinkie saying the 'o' from 'No!' for over 3.1 seconds, this network cannot align to that while using 0.005, though I believe 'Noooooooooooooo!' would be the correct transcript for that file anyway so this should be a non-issue for correctly labelled files)
        gmm_att_lin_bias=True,
        gmm_att_attrnn_zoneout=0.0,
        gmm_att_attrnn_dim=1280,
        
        # (Decoder) Decoder parameters
        sigma=1.0,
        grad_checkpoint=0,
        n_flows=10,
        n_group=160,
        n_early_every=4,
        n_early_size=40,
        mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (Decoder) Decoder Cond parameters
        cond_residual=False,
        cond_res_rezero=False,
        cond_weightnorm=False, # Causes NaN Gradients late into training.
        cond_layers=0,
        cond_act_func='lrelu',
        cond_padding_mode='zeros',
        cond_kernel_size=1,
        cond_hidden_channels=256,
        cond_output_channels=256,
        
        # (Decoder) WN parameters
        wn_n_channels=384,
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
        DurGlow_enable=False, # use Duration Glow?
        DurGlow_loss_scalar = 0.0001, # keep low for first 50k~ iters
        dg_sigma=1.0,
        dg_grad_checkpoint=0,
        dg_n_flows=1,
        dg_n_group=2,
        dg_n_early_every=10,
        dg_n_early_size=2,
        dg_mix_first=True,#True = WaveGlow style, False = WaveFlow style
        
        # (DurationGlow) Decoder Cond parameters
        dg_cond_residual=False,
        dg_cond_res_rezero=False,
        dg_cond_weightnorm=False, # Causes NaN Gradients late into training.
        dg_cond_layers=0,
        dg_cond_act_func='lrelu',
        dg_cond_padding_mode='zeros',
        dg_cond_kernel_size=1,
        dg_cond_hidden_channels=256,
        dg_cond_output_channels=256,
        
        # (DurationGlow) WN parameters
        dg_wn_n_channels=256,
        dg_wn_kernel_size=3,
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
        val_batch_size=24,# for more precise comparisons between models, constant batch_size is useful
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
