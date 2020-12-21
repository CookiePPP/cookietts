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
        batch_size    =24,# controls num of files processed in parallel per GPU
        val_batch_size=24,# for more precise comparisons between models, constant batch_size is useful
        
        use_TBPTT = True,# continue processing longer files into the next training iteration.
                         # allows very large inputs to be learned from
                         # and a large speedup in training speed by decreasing max_segment_length and increasing batch_size.
        
        max_segment_length=384,# max mel length till a segment is sliced.
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
        
        min_avg_max_att_start =110000,# when to start filtering out weak alignments.
                                      # (normally mis-labelled files or files that are too challenging to learn)
                                      # Only applies to training dataset.
                                      # Only updates at the end of each epoch.
        
        num_workers    =8,# (train) Number of threads for dataloading per GPU
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
        ## Model Parameters             ##
        ##################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        
        # Gate
        gate_positive_weight=10, # how much more valuable 1 positive frame is to 1 zero frame. 80 Frames per seconds, therefore values around 10 are fine.
        
        # Teacher-forcing Config
        p_teacher_forcing      = 1.00,# overriden by 'run_every_epoch.py'
        teacher_force_till     = 20,  # overriden by 'run_every_epoch.py'
        val_p_teacher_forcing  = 0.80,# overriden by 'run_every_epoch.py'
        val_teacher_force_till = 20,  # overriden by 'run_every_epoch.py'
        
        # (Encoder) Encoder parameters
        encoder_speaker_embed_dim    = 64, # speaker_embedding before encoder
        encoder_concat_speaker_embed = 'before_conv',# concat before encoder convs, or just before the LSTM inside decode. Options 'before_conv','before_lstm'
        encoder_kernel_size     =    5,
        encoder_n_convolutions  =    3,
        encoder_conv_hidden_dim =  512,
        encoder_LSTM_dim        = 1024,
        
        # (SylpsNet) Predicts speaking speed
        sylpsnet_layer_dims = [32, 32],# width of each layer, LeakyReLU() is used between hiddens
        
        # NOT IN USE (EmotionNet) Semi-supervised VAE/Classifier
        emotion_classes = ['neutral','anxious','happy','annoyed','sad','confused','smug','angry','whispering','shouting','sarcastic','amused','surprised','singing','fear','serious'],
        
        # (Residual Encoder) Learns information related to anything that isn't related to Text or Speakers. (e.g: background noise)
        # https://openreview.net/pdf?id=Bkg9ZeBB37
        # THIS IS NOT WORKING CORRECTLY RIGHT NOW AND MAY BE REMOVED AT ANY TIME.
        use_res_enc = True,
        res_enc_filters   = [32, 32, 64, 64, 128, 128],
        res_enc_gru_dim   = 128,
        res_enc_n_tokens  =   4,
        res_enc_embed_dim =  64,
        
        use_res_enc_dis = False,# Enable/Disable Discriminator
        res_enc_dis_dim  = 128,
        res_enc_n_layers =   3,
        
        # (DebluraGAN) GAN Spectrogram Loss
        # *Should* reduce the blur for generated spectrograms.
        use_dbGAN      = True,
        
        # Conv2d Blocks
        dbGAN_prenet_dim      = 128,
        dbGAN_prenet_kernel_h = 3,
        dbGAN_prenet_kernel_w = 3,
        dbGAN_prenet_stride   = [
                                  [3, 3],
                                  [3, 3],
                                ],
        dbGAN_prenet_n_layers = 2,
        dbGAN_prenet_n_blocks = 2,
        
        # Conv1d Blocks
        dbGAN_dim  = 128,
        dbGAN_kernel_w =   3,
        dbGAN_stride   =   1,# mel_T stride
        dbGAN_n_layers =   2,
        dbGAN_n_blocks =   2,
        
        # Random Number Generator Input to Decoder, to allow the decoder to produce convincing random noise when required.
        use_DecoderRNG =  True,# highly recommended when using dbGAN!
        DecoderRNG_dim =   160,# highly recommended when using dbGAN! # predict random numbers for the speech part of the decoder to use
        noise_projector = True,# highly recommended when using dbGAN! # predict a scalar for white-noise to add to each element.
        
        # (InferenceGAN) TF/Inf GAN Loss
        # Reduces the difference between Infernce and Training Results
        # requires training 50% batch without teacher forcing
        # recommened only late into training.
        use_InfGAN = True,
        InfGAN_use_speaker =  True,# speaker embeds
        InfGAN_use_spect   = False,# decoder spectrogram
        InfGAN_use_postnet = False,# postnet spectrogram
        InfGAN_use_DecRNN  =  True,#   decoderRNN hidden state(s)
        InfGAN_use_AttRNN  =  True,# AttentionRNN hidden state(s)
        InfGAN_use_context =  True,# Attention Contexts <- the current encoder outputs the attention is focused on
        
        TemporalInfGAN = True,# True  = use Causal Temporal Convs
                              # False = use Uni-Directional LSTMs
        
        # (InferenceGAN) If using TemporalConvInferenceGAN
        InfGAN_n_channels     =   384,
        InfGAN_n_layers       =     7,
        InfGAN_kernel_size    =     3,
        InfGAN_seperable_conv = False,
        InfGAN_merge_res_skip = False,
        InfGAN_res_skip       =  True,
        InfGAN_n_layers_dilations_w=None,
        # (InferenceGAN) If using LSTMInferenceGAN
        InfGAN_LSTM_dim      = 256,
        InfGAN_LSTM_n_layers =   2,
        
        
        # (AuxEmotionNet) TorchMoji
        torchMoji_attDim     = 2304,# published model uses 2304
        torchMoji_crushedDim =   32,
        torchMoji_BatchNorm  = True,
        
        # (Speaker) Speaker embedding
        n_speakers            = 2048,# maximum number of speakers the model can support.
        speaker_embedding_dim =  256,# speaker embedding size # 128 baseline
        
        # (Variational Encoder)
        # Gives slightly random inputs into the decoder that contain information about the target frame.
        # This acts kinda like a cheat-sheet for tacotron, where it gets to pick up a small amount of information about it's target
        # As such, using the Variational Encoder will have a large impact on spec and postnet losses without actually making tacotron more accurate.
        # The reason this is useful is because tacotron produces less blurry outputs when it has a cheat sheet,
        # however, since we can't use a cheat-sheet when generating new audio, when generating new audio we feed tacotron completely random cheat sheets.
        # The theory is that tacotron will produce clear outputs using the cheat sheet, and since the cheat sheet is soo small, tacotron will only use the cheat-sheet for really important things it can't already predict.
        # and thus, using random cheat-sheets should only add randomness to things like the background noise and the higher harmonics, where randomness is acceptable anyway.
        # - What i'm saying is, keep the VE_KLD loss quite low, or tacotron will start to use it's cheat sheet instead of thinking for itself.
        #   The cheat sheet should only be informative enough to contain small bits of information that tacotron can't predict.
        #   Any more informative and tacotron would stop trying to learn and just use the cheat sheet for everything.
        
        use_ve = False,# True/False - use Variational Encoder?
        ve_lstm_dim = 768,# - width/dim of each lstm.
        ve_n_lstm   =   1,# - number of lstm layers.
        ve_n_tokens =  20,# n_tokens is the size of the cheat sheet. Too small and tacotron will act like normal (i.e: Blurry outputs)
                          #                                          Too large and tacotron will start becoming very random and unstable as it relies on the cheat sheet for things it shouldn't.
                          # Also keep in mind the VE_KLD loss, VE_KLD is basically how "not random" or "precise" every token is.
                                                     # VE_KLD =  0.0 means the tokens are all just random number generators and contain no information.
                                                     # VE_KLD = 99.0 means the tokens are extremely precise and contain far too much information.
                                                     # I don't know the best value, but between 0.5 and 2.0 is quite common.
                                                     # Use VE_KLD_weight in 'run_every_epoch.py' to influence the VE_KLD.
                                                     # higher VE_KLD_weight = tokens are forced to be more random
                                                     # when you start training, VE_KLD_weight should be kept low so tacotron can build it's cheat sheet and figure out what it wants to put in the cheat sheet.
                                                     # Once VE_KLD starts shooting up that means tacotron has figured out what it wants in the cheat sheet, at that point you need to increase VE_KLD_weight to bring it back down to about 1.0-ish.
        ve_embed_dim = 160,# - dimension to expand the n_tokens to before adding back to the model
        
        # (Decoder/Encoder) Bottleneck parameters
        # The outputs from the encoder, speaker, emotionnet and sylpsnet need to be mixed.
        # By default the information is mixed by the DecoderRNN, but this is repeated every spectrogram frame so likely wastes a massive amount of compute performing the same operations repeatedly.
        # Thus, this memory bottleneck can be used to mix the above mentioned outputs into a more compressed representation before decoding, allowing the DecoderRNN to be made smaller and more effective.
        use_memory_bottleneck  =  True,# False baseline
        memory_bottleneck_dim  =   512,# new memory size. 512 would be equivalent to the original Tacotron2.
        memory_bottleneck_bias = False,
        
        # (Decoder) Decoder parameters
        start_token = "",#"☺"
        stop_token  = "",#"␤"
        hide_startstop_tokens=False, # trim first/last encoder output before feeding to attention.
        n_frames_per_step=1,# currently only 1 is supported
        context_frames   =1,# NOT IMPLEMENTED
        
        # (Decoder) Prenet
        prenet_dim   = 512,    # 256 baseline
        prenet_layers=   2,    # 2 baseline
        prenet_batchnorm=False,# False baseline
        prenet_bn_momentum=0.5,# Inverse smoothing factor, 0.1 = high smoothing, 0.9 = Almost no smoothing
        p_prenet_dropout  =0.5,# 0.5 baseline
        
        prenet_speaker_embed=True,# True/False - use speaker_embedding before encoder
        prenet_noise   =0.00,# Add Gaussian Noise to Prenet inputs. std defined here.
        prenet_blur_min=0.00,# Apply random vertical blur between prenet_blur_min
        prenet_blur_max=0.00,#                                and prenet_blur_max
                             # Set max to False or Zero to disable
        prenet_use_code_loss=True,# L1 Loss between prenet outputs with GT and Pred inputs
                                  # Forces Tacotron to produce reasonable latents from it's own outputs
        
        # (Decoder) AttentionRNN
        AttRNN_extra_decoder_input =  True,# False baseline # Feed DecoderRNN Hidden State into AttentionRNN
        # Optional Second AttentionRNN
        AttRNN_use_global_cond     =  True,# Add speaker_embed, emotion_embed, speaking rate,
        
        attention_rnn_dim          =  1280,  # 1024 baseline
        AttRNN_hidden_dropout_type = 'zoneout',# options ('dropout','zoneout')
        p_AttRNN_hidden_dropout    =  0.10,  # 0.1 baseline
        second_attention_rnn_dim=0,# 0 baseline # Extra AttentionRNN to learn more complex patterns # set to 0 to disable layer.
        second_attention_rnn_residual_connection=True,# residual connections between the AttentionRNNs
                                                      # requires the attention_rnn dims to match to activate/work.
        
        # (Decoder) DecoderRNN
        decoder_rnn_dim            = 1024,  # 1024 baseline
        DecRNN_hidden_dropout_type = 'zoneout',# options ('dropout','zoneout')
        p_DecRNN_hidden_dropout    =  0.10, # 0.1 baseline
        decoder_residual_connection= False, # residual connections with the AttentionRNN hidden state and Attention/Memory Context
        # Optional Second DecoderRNN
        second_decoder_rnn_dim=1024,# 0 baseline # Extra DecoderRNN to learn more complex patterns # set to 0 to disable layer.
        second_decoder_residual_connection=True,# residual connections between the DecoderRNNs
        # Optional Third! DecoderRNN
        third_decoder_rnn_dim =1024,# 0 baseline # Extra DecoderRNN to learn more complex patterns # set to 0 to disable layer.
        third_decoder_residual_connection=True,# residual connections between the DecoderRNNs
        
        # (Decoder) Misc
        decoder_input_residual=True,# add the decoder_input to the outputs so to the rest of the decoder is only learning the modifier
        
        # (Decoder) Attention parameters
        attention_type=0,
        # 0 -> Hybrid Location-Based Attention (Vanilla Tacotron2)
        # 1 -> GMMAttention (Long-form Synthesis)
        # 1 -> Dynamic Convolution Attention (Long-form Synthesis)
        attention_dim=128, # 128 Layer baseline # Used for Key-Query Dim
        
        # (Decoder) Attention Type 0 Parameters
        windowed_attention_range = 16,# set to 0 to disable
                                     # will set the forward and back distance the model can attend to.
                                     # 2 will give the model 5 characters it can attend to at any one time.
                                     # This will also allow more stable generation with extremely long text inputs and save VRAM during inference.
        windowed_att_pos_offset  =  0.00,# Offset the current_pos by this amount.
        windowed_att_pos_learned = False,
        attention_learned_temperature=False,# add temperature param to alignments for softmax.
        
        # (Decoder) Attention Type 0 (and 2) Parameters
        attention_location_n_filters  =32,# 32 baseline
        attention_location_kernel_size=31,# 31 baseline
        
        # (Decoder) Attention Type 1 Parameters
        num_att_mixtures=1,# 5 baseline
        attention_layers=1,# 1 baseline
        delta_offset   = 0.005,    # 0 baseline, values around 0.005 will push the model forwards. Since we're using the sigmoid function caution is suggested.
        delta_min_limit= 0.0, # 0 baseline, values around 0.010 will force the model to move forward, in this example, the model cannot spend more than 100 steps on the same encoder output.
        lin_bias=False, # I need to figure out what that layer is called.
        initial_gain='relu', # initial weight distribution 'tanh','relu','sigmoid','linear'
        normalize_attention_input=  True,# False baseline
        normalize_AttRNN_output  = False,# True baseline
        
        # (Decoder) Attention Type 2 Parameters
        dynamic_filter_num=128,# 8 baseline
        dynamic_filter_len= 21, # 21 baseline # currently only 21 is supported
        
        # (Postnet) Mel-post processing network parameters
        use_postnet=True,
        postnet_embedding_dim = 512,
        postnet_kernel_size   =   5,
        postnet_n_convolutions=   6,
        postnet_residual_connections=3,# False baseline, int > 0 == n_layers in each residual block
        
        # (Very Experimental) HiFi-GAN Latent Inputs
        # - NOT IMPLEMENTED
        HiFiGAN_enable    = True,
        HiFiGAN_cp_folder = '../../_4_mtw/hifigan/cp_hifigan_935universal44Khz_0_latent_ft',
        
        # used for reconstruction loss, not input
        HiFiGAN_filter_length = 2048,
        HiFiGAN_hop_length    =  512,
        HiFiGAN_win_length    = 2048,
        HiFiGAN_n_mel_channels=  160,
        HiFiGAN_clamp_val     = 1e-5,
        
        HiFiGAN_batch_size  =     4,# affects VRAM usage
        HiFiGAN_segment_size= 32768,# affects VRAM usage
        HiFiGAN_learning_rate= 1e-4,# overriden by 'run_every_epoch.py'
        HiFiGAN_lr_half_life =30000,# overriden by 'run_every_epoch.py'
        
        ##################################
        ## Optimization Hyperparameters ##
        ##################################
        use_saved_learning_rate=False,
        learning_rate   = 0.1e-5,# overriden by 'run_every_epoch.py'
        grad_clip_thresh= 1.0,   # overriden by 'run_every_epoch.py'
        weight_decay    = 1.0e-6,
        
        mask_padding  = True,#mask values by setting them to the same values in target and predicted
        masked_select = True,#mask values by removing them from the calculation
        
        # (DFR) Drop Frame Rate
        global_mean_npy = 'global_mean.npy',
        drop_frame_rate=  0.25,# overriden by 'run_every_epoch.py'
        
        ##################################
        ## Loss Weights/Scalars         ##
        ##################################
        # All of these can be overriden from 'run_every_epoch.py'
        spec_MSE_weight     = 0.0,# MSE  Spectrogram Loss Before Postnet
        spec_MFSE_weight    = 1.0,# MFSE Spectrogram Loss Before Postnet
        postnet_MSE_weight  = 0.0,# MSE  Spectrogram Loss After Postnet
        postnet_MFSE_weight = 1.0,# MFSE Spectrogram Loss After Postnet
        gate_loss_weight    = 1.0,# Gate Loss
        
        res_enc_kld_weight  = 0.002,
        res_enc_gMSE_weight = 0.1,
        
        sylps_kld_weight = 0.0020, # SylNet KDL Weight
        sylps_MSE_weight = 0.01,# Encoder Pred Sylps MSE weight
        sylps_MAE_weight = 0.00,# Encoder Pred Sylps MAE weight
        
        diag_att_weight=0.05,# 'dumb' guided attention. Simply punishes the model for attention that is non-diagonal. Decreases training time and increases training stability with English speech. 
                             # As an example of how this works, if you imagine there is a 10 letter input that lasts 1 second. The first 0.1s is pushed towards using the 1st letter, the next 0.1s will be pushed towards using the 2nd letter, and so on for each chunk of audio. Since each letter has a different natural duration (especially punctuation), this attention guiding is not particularly accurate, so it's not recommended to use a high loss scalar later into training.
        DiagonalGuidedAttention_sigma=0.5, # how to *curve?* the attention loss? Just leave this one alone.
        
        rescale_for_volume=0.0, # Not implemented # Rescale spectrogram losses to prioritise louder sounds, and put less (or zero) priority on quieter sounds
                                # Valid values between 0.0 and 1.0
                                # will rescale spectrogram magnitudes (which range from -11.52 for silence, and 4.5 for deafeningly loud)
                                # https://www.desmos.com/calculator/bmbakslmtc
    )

    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s', hparams.values())

    return hparams
