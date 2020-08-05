from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm, LSTMCellWithZoneout, GMMAttention, DynamicConvolutionAttention
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame

from modules.SylpsNet import SylpsNet
from modules.EmotionNet import EmotionNet
from modules.AuxEmotionNet import AuxEmotionNet

drop_rate = 0.5

def load_model(hparams):
    model = Tacotron2(hparams)
    if torch.cuda.is_available(): model = model.cuda()
    if hparams.fp16_run:
        if hparams.attention_type in [0,2]:
            model.decoder.attention_layer.score_mask_value = finfo('float16').min
        elif hparams.attention_type == 1:
            model.decoder.attention_layer.score_mask_value = 0
        else:
            print(f'mask value not found for attention_type {hparams.attention_type}')
            raise
    return model


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim, out_bias=False):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=out_bias, w_init_gain='tanh')
    
    def forward(self, attention_weights_cat): # [B, 2, enc]
        processed_attention = self.location_conv(attention_weights_cat) # [B, 2, enc] -> [B, n_filters, enc]
        processed_attention = processed_attention.transpose(1, 2) # [B, n_filters, enc] -> [B, enc, n_filters]
        processed_attention = self.location_dense(processed_attention) # [B, enc, n_filters] -> [B, enc, attention_dim]
        return processed_attention # [B, enc, attention_dim]


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, # Crushes the Encoder outputs to Attention Dimension used by this module
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
    
    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, enc_time)
        
        RETURNS
        -------
        alignment (batch, enc_time)
        """
        processed = self.location_layer(attention_weights_cat) # [B, 2, enc] # conv1d, matmul
        processed.add_( self.query_layer(query.unsqueeze(1)).expand_as(processed_memory) ) # unsqueeze, matmul, expand_as, add_
        processed.add_( processed_memory ) # add_
        energies = self.v( torch.tanh( processed ) ) # tanh, matmul
        
        return energies.squeeze(-1) # squeeze
    
    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, attention_weights=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        if attention_weights is None:
            alignment = self.get_alignment_energies(
                attention_hidden_state, processed_memory, attention_weights_cat)
            
            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)
            
            attention_weights = F.softmax(alignment, dim=1) # softmax
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # unsqueeze, bmm
        attention_context = attention_context.squeeze(1) # squeeze
        
        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, p_prenet_dropout, prenet_batchnorm):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.p_prenet_dropout = p_prenet_dropout
        self.prenet_batchnorm = prenet_batchnorm
        self.p_prenet_input_dropout = 0
        
        if self.prenet_batchnorm:
            self.batchnorms = nn.ModuleList([ nn.BatchNorm1d(size) for size in sizes ])
    
    def forward(self, x):
        if self.p_prenet_input_dropout: # dropout from the input, definitely a dangerous idea, but I think it would be very interesting to try values like 0.05 and see the effect
            x = F.dropout(x, self.p_prenet_input_dropout, self.training)
        
        for i, linear in enumerate(self.layers):
            x = F.relu(linear(x))
            if self.p_prenet_dropout > 0:
                x = F.dropout(x, p=self.p_prenet_dropout, training=True)
            if self.prenet_batchnorm:
                x = self.batchnorms[i](x)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )
        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )
    
    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), drop_rate, self.training)
        x = F.dropout(self.convolutions[-1](x), drop_rate, self.training)
        
        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__() 
        self.encoder_speaker_embed_dim = hparams.encoder_speaker_embed_dim
        if self.encoder_speaker_embed_dim:
            self.encoder_speaker_embedding = nn.Embedding(
            hparams.n_speakers, self.encoder_speaker_embed_dim)
        
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.encoder_conv_hidden_dim = hparams.encoder_conv_hidden_dim
        
        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            if _ == 0:
                if self.encoder_concat_speaker_embed == 'before_conv':
                    input_dim = hparams.symbols_embedding_dim+self.encoder_speaker_embed_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    input_dim = hparams.symbols_embedding_dim
                else:
                    raise NotImplementedError(f'encoder_concat_speaker_embed is has invalid value {hparams.encoder_concat_speaker_embed}, valid values are "before","inside".')
            else:
                input_dim = self.encoder_conv_hidden_dim
            
            if _ == (hparams.encoder_n_convolutions)-1: # last conv
                if self.encoder_concat_speaker_embed == 'before_conv':
                    output_dim = hparams.encoder_LSTM_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    output_dim = hparams.encoder_LSTM_dim-self.encoder_speaker_embed_dim
            else:
                output_dim = self.encoder_conv_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(input_dim,
                         output_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(output_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(hparams.encoder_LSTM_dim,
                            int(hparams.encoder_LSTM_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        self.LReLU = nn.LeakyReLU(negative_slope=0.01) # LeakyReLU
    
    def forward(self, text, text_lengths, speaker_ids=None):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.repeat(1, 1, text.size(2)) # extend across all encoder steps
            if self.encoder_concat_speaker_embed == 'before_conv':
                text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            text = F.dropout(self.LReLU(conv(text)), drop_rate, self.training)
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        text = text.transpose(1, 2)
        
        # pytorch tensor are not reversible, hence the conversion
        text_lengths = text_lengths.cpu().numpy()
        text = nn.utils.rnn.pack_padded_sequence(
            text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(text)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        
        return outputs
    
    def inference(self, text, speaker_ids=None, text_lengths=None):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.repeat(1, 1, text.size(2))
            if self.encoder_concat_speaker_embed == 'before_conv':
                text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            text = F.dropout(self.LReLU(conv(text)), drop_rate, self.training)
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        text = text.transpose(1, 2) # [B, embed, sequence] -> [B, sequence, embed]
        
        if text_lengths is not None:
            # pytorch tensor are not reversible, hence the conversion
            text_lengths = text_lengths.cpu().numpy()
            text = nn.utils.rnn.pack_padded_sequence(
                text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(text) # -> [B, sequence, embed]
        
        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)
        
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.memory_dim = hparams.encoder_LSTM_dim + hparams.speaker_embedding_dim + len(hparams.emotion_classes) + hparams.zu_latent_dim + 1# size 1 == "sylzu"
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.prenet_layers = hparams.prenet_layers
        self.prenet_batchnorm = hparams.prenet_batchnorm if hasattr(hparams, 'prenet_batchnorm') else False
        self.p_prenet_dropout = hparams.p_prenet_dropout
        self.prenet_speaker_embed_dim = hparams.prenet_speaker_embed_dim if hasattr(hparams, 'prenet_speaker_embed_dim') else 0
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.AttRNN_extra_decoder_input = hparams.AttRNN_extra_decoder_input
        self.AttRNN_hidden_dropout_type = hparams.AttRNN_hidden_dropout_type
        self.p_AttRNN_hidden_dropout = hparams.p_AttRNN_hidden_dropout
        self.p_AttRNN_cell_dropout = hparams.p_AttRNN_cell_dropout
        self.DecRNN_hidden_dropout_type = hparams.DecRNN_hidden_dropout_type
        self.p_DecRNN_hidden_dropout = hparams.p_DecRNN_hidden_dropout
        self.p_DecRNN_cell_dropout = hparams.p_DecRNN_cell_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.teacher_force_till = hparams.teacher_force_till
        self.num_att_mixtures = hparams.num_att_mixtures
        self.normalize_attention_input = hparams.normalize_attention_input
        self.normalize_AttRNN_output = hparams.normalize_AttRNN_output
        self.attention_type = hparams.attention_type
        self.attention_layers = hparams.attention_layers
        self.low_vram_inference = hparams.low_vram_inference if hasattr(hparams, 'low_vram_inference') else False
        self.context_frames = hparams.context_frames
        self.hide_startstop_tokens = hparams.hide_startstop_tokens
        
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step * self.context_frames,
            [hparams.prenet_dim]*hparams.prenet_layers, self.p_prenet_dropout, self.prenet_batchnorm)
        
        if self.AttRNN_extra_decoder_input:
            AttRNN_Dimensions = hparams.prenet_dim + self.memory_dim + hparams.decoder_rnn_dim
        else:
            AttRNN_Dimensions = hparams.prenet_dim + self.memory_dim
        
        if self.AttRNN_hidden_dropout_type == 'dropout':
            self.attention_rnn = nn.LSTMCell(
                AttRNN_Dimensions, # input_size
                hparams.attention_rnn_dim) # hidden_size)
        elif self.AttRNN_hidden_dropout_type == 'zoneout':
            self.attention_rnn = LSTMCellWithZoneout(
                AttRNN_Dimensions, # input_size
                hparams.attention_rnn_dim, zoneout_prob=self.p_DecRNN_hidden_dropout) # hidden_size, bias)
            self.p_AttRNN_hidden_dropout = 0.0 # zoneout assigned inside LSTMCellWithZoneout so don't need normal dropout
        
        if self.attention_type == 0:
            self.attention_layer = Attention(
                hparams.attention_rnn_dim, self.memory_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)
        elif self.attention_type == 1:
            self.attention_layer = GMMAttention(
                hparams.num_att_mixtures, hparams.attention_layers,
                hparams.attention_rnn_dim, self.memory_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size, hparams)
        elif self.attention_type == 2:
            self.attention_layer = DynamicConvolutionAttention(
                hparams.attention_rnn_dim, self.memory_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size,
                hparams.dynamic_filter_num, hparams.dynamic_filter_len)
        else:
            print("attention_type invalid, valid values are... 0 and 1")
            raise
        
        if self.DecRNN_hidden_dropout_type == 'dropout':
            self.decoder_rnn = nn.LSTMCell(
                hparams.attention_rnn_dim + self.memory_dim, # input_size
                hparams.decoder_rnn_dim, 1) # hidden_size, bias)
        elif self.DecRNN_hidden_dropout_type == 'zoneout':
            self.decoder_rnn = LSTMCellWithZoneout(
                hparams.attention_rnn_dim + self.memory_dim, # input_size
                hparams.decoder_rnn_dim, 1, zoneout_prob=self.p_DecRNN_hidden_dropout) # hidden_size, bias)
            self.p_DecRNN_hidden_dropout = 0.0 # zoneout assigned inside LSTMCellWithZoneout so don't need normal dropout
        
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.memory_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)
        
        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.memory_dim, 1,
            bias=True, w_init_gain='sigmoid')
    
    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        
        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input
    
    def initialize_decoder_states(self, memory, mask, preserve=None, override=None):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        preserve: Batch shape bool tensor of decoder states to preserve
        """
        B = memory.size(0)
        MAX_ENCODE = memory.size(1)
        
        if preserve is not None:
            if len(preserve.shape) < 2:
                preserve = preserve[:, None]
            assert preserve.shape[0] == B
        
        if hasattr(self, 'attention_hidden') and preserve is not None:
            self.attention_hidden *= preserve
            self.attention_hidden.detach_()
            self.attention_cell *= preserve
            self.attention_cell.detach_()
        else:
            self.attention_hidden = Variable(memory.data.new( # attention hidden state
                B, self.attention_rnn_dim).zero_())
            self.attention_cell = Variable(memory.data.new( # attention cell state
                B, self.attention_rnn_dim).zero_())
        
        if hasattr(self, 'decoder_hidden') and preserve is not None:
            self.decoder_hidden *= preserve
            self.decoder_hidden.detach_()
            self.decoder_cell *= preserve
            self.decoder_cell.detach_()
        else:
            self.decoder_hidden = Variable(memory.data.new( # LSTM decoder hidden state
                B, self.decoder_rnn_dim).zero_())
            self.decoder_cell = Variable(memory.data.new( # LSTM decoder cell state
                B, self.decoder_rnn_dim).zero_())
        
        if hasattr(self, 'attention_weights') and preserve is not None: # save all the encoder possible
            self.saved_attention_weights = self.attention_weights
            self.saved_attention_weights_cum = self.attention_weights_cum
        
        self.attention_weights = Variable(memory.data.new( # attention weights of that frame
            B, MAX_ENCODE).zero_())
        self.attention_weights_cum = Variable(memory.data.new( # cumulative weights of all frames during that inferrence
            B, MAX_ENCODE).zero_())
        
        if hasattr(self, 'saved_attention_weights') and preserve is not None:
            COMMON_ENCODE = min(MAX_ENCODE, self.saved_attention_weights.shape[1]) # smallest MAX_ENCODE of the saved and current encodes
            self.attention_weights[:, :COMMON_ENCODE] = self.saved_attention_weights[:, :COMMON_ENCODE] # preserve any encoding weights possible (some will be part of the previous iterations padding and are gone)
            self.attention_weights_cum[:, :COMMON_ENCODE] = self.saved_attention_weights_cum[:, :COMMON_ENCODE]
            self.attention_weights *= preserve
            self.attention_weights.detach_()
            self.attention_weights_cum *= preserve
            self.attention_weights_cum.detach_()
            if self.attention_type == 2: # Dynamic Convolution Attention
                    self.attention_weights[:, 0] = ~preserve.bool()[:,0] # [B, 1] -> [B] # initialize the weights at encoder step 0
                    self.attention_weights_cum[:, 0] = ~preserve.bool()[:,0] # [B, 1] -> [B] # initialize the weights at encoder step 0
        elif self.attention_type == 2:
            self.attention_weights[:, 0] = 1 # initialize the weights at encoder step 0
            self.attention_weights_cum[:, 0] = 1 # initialize the weights at encoder step 0
        
        if hasattr(self, 'attention_context') and preserve is not None:
            self.attention_context *= preserve
            self.attention_context = self.attention_context.detach()
        else:
            self.attention_context = Variable(memory.data.new( # attention output
                B, self.encoder_LSTM_dim).zero_())
        
        self.memory = memory
        if self.attention_type == 0:
            self.processed_memory = self.attention_layer.memory_layer(memory) # Linear Layer, [B, enc_T, enc_dim] -> [B, enc_T, attention_dim]
        elif self.attention_type == 1:
            self.previous_location = Variable(memory.data.new(
                B, 1, self.num_att_mixtures).zero_())
        self.mask = mask
    
    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) > 1:
            gate_outputs = gate_outputs.transpose(0, 1)
        else:
            gate_outputs = gate_outputs[None]
        
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        
        return mel_outputs, gate_outputs, alignments
    
    def decode(self, decoder_input, attention_weights=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        if self.AttRNN_extra_decoder_input:
            cell_input = torch.cat((decoder_input, self.attention_context, self.decoder_hidden), -1)# [Processed Previous Spect Frame, Last input Taken from Text/Att, Previous Decoder state used to produce frame]
        else:
            cell_input = torch.cat((decoder_input, self.attention_context), -1)# [Processed Previous Spect Frame, Last input Taken from Text/Att]
        
        if self.normalize_AttRNN_output and self.attention_type == 1:
            cell_input = cell_input.tanh()
        
        self.attention_hidden, self.attention_cell = self.attention_rnn( # predict next step attention based on cell_input
            cell_input, (self.attention_hidden, self.attention_cell))
        
        if self.p_AttRNN_hidden_dropout:
            self.attention_hidden = F.dropout(
                self.attention_hidden, self.p_AttRNN_hidden_dropout, self.training)
        if self.p_AttRNN_cell_dropout:
            self.attention_cell = F.dropout(
                self.attention_cell, self.p_AttRNN_cell_dropout, self.training)
        
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),# attention weights from the last step and 
             self.attention_weights_cum.unsqueeze(1)), dim=1)# the total attention weights from every step previously summed
        
        if self.attention_type == 0:
            self.attention_context, self.attention_weights = self.attention_layer( # attention_context is the encoder output that is to be used at the current frame(?)
                self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask, attention_weights)
        elif self.attention_type == 1:
            self.attention_context, self.attention_weights, self.previous_location = self.attention_layer(
                self.attention_hidden, self.memory, self.previous_location, self.mask)
        elif self.attention_type == 2:
            self.attention_context, self.attention_weights = self.attention_layer( # attention_context is the encoder output that is to be used at the current frame(?)
                self.attention_hidden, attention_weights_cat, self.memory, self.mask, attention_weights)
        else:
            raise NotImplementedError(f"Attention Type {self.attention_type} Invalid")
        
        self.attention_weights_cum += self.attention_weights# [B, enc] # cumulative weights determine how much time has been spent on each encoder_input, should let the model know what has already been said and what still needs to be spoken
        
        decoder_input = torch.cat( (self.attention_hidden, self.attention_context), -1) # cat 6.475ms
        
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn( # lstmcell 12.789ms
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        
        if self.p_DecRNN_hidden_dropout:
            self.decoder_hidden = F.dropout(
                self.decoder_hidden, self.p_DecRNN_hidden_dropout, self.training)
        if self.p_DecRNN_cell_dropout:
            self.decoder_cell = F.dropout(
                self.decoder_cell, self.p_DecRNN_cell_dropout, self.training)
        
        decoder_hidden_attention_context = torch.cat( (self.decoder_hidden, self.attention_context), dim=1) # cat 6.555ms
        
        gate_prediction = self.gate_layer(decoder_hidden_attention_context) # addmm 5.762ms
        
        decoder_output = self.linear_projection(decoder_hidden_attention_context) # addmm 5.621ms
        
        return decoder_output, gate_prediction, self.attention_weights
    
    def forward(self, memory, decoder_inputs, memory_lengths, preserve_decoder=None, teacher_force_till=None, p_teacher_forcing=None):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        if self.hide_startstop_tokens: # remove start/stop token from Decoder
            memory = memory[:,1:-1,:]
            memory_lengths = memory_lengths-2
        
        decoder_input = self.get_go_frame(memory).unsqueeze(0) # create blank starting frame
        if self.context_frames > 1: decoder_input = decoder_input.repeat(self.context_frames, 1, 1)
        # memory -> (1, B, mel_channels) <- which is all 0's
        
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        # (B, mel_channels, T_out) -> (T_out, B, mel_channels)
        
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # concat T_out
        
        if self.prenet_speaker_embed_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            decoder_inputs = torch.cat((decoder_inputs, embedded_speakers), dim=2)
        
        decoder_inputs = self.prenet(decoder_inputs)
        
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths), preserve=preserve_decoder)
        
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if len(mel_outputs) <= teacher_force_till or np.random.uniform(0.0, 1.0) <= p_teacher_forcing:
                decoder_input = decoder_inputs[len(mel_outputs)] # use all-in-one processed output for next step
            else:
                decoder_input = self.prenet(mel_outputs[-1]) # use last output for next step (like inference)
            
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, memory_lengths=None):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        if self.hide_startstop_tokens: # remove start/stop token from Decoder
            memory = memory[:,1:-1,:]
            memory_lengths = memory_lengths-2
        decoder_input = self.get_go_frame(memory)
        
        self.initialize_decoder_states(memory, mask=None if memory_lengths is None else ~get_mask_from_lengths(memory_lengths))
        
        sig_max_gates = torch.zeros(decoder_input.size(0))
        mel_outputs, gate_outputs, alignments, break_point = [], [], [], self.max_decoder_steps
        for i in range(self.max_decoder_steps):
            decoder_input = self.prenet(decoder_input)
            
            mel_output, gate_output_gpu, alignment = self.decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_output_cpu = gate_output_gpu.cpu().float() # small operations e.g min(), max() and sigmoid() are faster on CPU # also .float() because Tensor.min() doesn't work on half precision CPU
            if not self.low_vram_inference:
                gate_outputs += [gate_output_gpu.squeeze(1)]
                alignments += [alignment]
            
            if self.attention_type == 1 and self.num_att_mixtures == 1:# stop when the attention location is out of the encoder_outputs
                if self.previous_location.squeeze().item() + 1. > memory.shape[1]:
                    break
            else:
                # once ALL batch predictions have gone over gate_threshold at least once, set break_point
                if i > 4: # model has very *interesting* starting predictions
                    sig_max_gates = torch.max(torch.sigmoid(gate_output_cpu), sig_max_gates)# sigmoid -> max
                if sig_max_gates.min() > self.gate_threshold: # min()  ( implicit item() as well )
                    break_point = min(break_point, i+self.gate_delay)
            
            if i >= break_point:
                break
            
            decoder_input = mel_output
        else:
            print("Warning! Reached max decoder steps")
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        # apply sigmoid to the GPU as well.
        gate_outputs = torch.sigmoid(gate_outputs)
        
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.teacher_force_till = hparams.teacher_force_till
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        self.drop_frame_rate = hparams.drop_frame_rate
        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(
                hparams.n_speakers, self.speaker_embedding_dim)
        
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.sylps_net = SylpsNet(hparams)
        self.emotion_net = EmotionNet(hparams)
        self.aux_emotion_net = AuxEmotionNet(hparams)
        
        if hparams.use_postnet_discriminator:
            self.postnet_discriminator = PostnetDiscriminator(hparams)
    
    def parse_batch(self, batch):
        text_padded, text_lengths, mel_padded, gate_padded, output_lengths, speaker_ids, \
          torchmoji_hidden, preserve_decoder_states, sylps, emotion_id, emotion_onehot = batch
        text_padded = to_gpu(text_padded).long()
        text_lengths = to_gpu(text_lengths).long()
        output_lengths = to_gpu(output_lengths).long()
        speaker_ids = to_gpu(speaker_ids.data).long()
        mel_padded = to_gpu(mel_padded).float()
        max_len = torch.max(text_lengths.data).item() # used by loss func
        gate_padded = to_gpu(gate_padded).float() # used by loss func
        if torchmoji_hidden is not None:
            torchmoji_hidden = to_gpu(torchmoji_hidden).float()
        if preserve_decoder_states is not None:
            preserve_decoder_states = to_gpu(preserve_decoder_states).float()
        if sylps is not None:
            sylps = to_gpu(sylps).float()
        if emotion_id is not None:
            emotion_id = to_gpu(emotion_id).long()
        if emotion_onehot is not None:
            emotion_onehot = to_gpu(emotion_onehot).float()
        return (
            (text_padded, text_lengths, mel_padded, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states, sylps, emotion_id, emotion_onehot),
            (mel_padded, gate_padded, output_lengths, emotion_id, emotion_onehot))
            # returns ((x),(y)) as (x) for training input, (y) for ground truth/loss calc
    
    def mask_outputs(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            # [B, n_mel, steps]
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        
        return outputs
    
    def forward(self, inputs, teacher_force_till=None, p_teacher_forcing=None, drop_frame_rate=None):
        text, text_lengths, gt_mels, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states, gt_sylps, emotion_id, emotion_onehot = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        
        if teacher_force_till == None: p_teacher_forcing  = self.p_teacher_forcing
        if p_teacher_forcing == None:  teacher_force_till = self.teacher_force_till
        if drop_frame_rate == None:    drop_frame_rate    = self.drop_frame_rate
        
        if drop_frame_rate > 0. and self.training:
            # gt_mels shape (B, n_mel_channels, T_out),
            gt_mels = dropout_frame(gt_mels, self.global_mean, output_lengths, drop_frame_rate)
        
        memory = []
        
        # Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, enc_T]
        encoder_outputs, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_ids) # [B, enc_T, enc_dim]
        memory.append(encoder_outputs)
        
        # speaker_id -> speaker_embed
        speaker_embed = self.speaker_embedding(speaker_ids)
        memory.append( speaker_embed[:, None].repeat(1, encoder_outputs.size(1), 1) )
        
        # Sylps -> sylzu, mu, logvar
        sylzu, syl_mu, syl_logvar = self.sylps_net(gt_sylps)
        memory.append( sylzu[:, None].repeat(1, encoder_outputs.size(1), 1) )
        
        # Gt_mels, speaker, encoder_outputs -> zs, em_zu, em_mu, em_logvar
        zs, em_zu, em_mu, em_logvar, em_params = self.emotion_net(gt_mels, speaker_embed, encoder_outputs,
                                                                   text_lengths=text_lengths, emotion_id=emotion_id, emotion_onehot=emotion_onehot)
        memory.extend(( em_zu[:, None].repeat(1, encoder_outputs.size(1), 1),
                           zs[:, None].repeat(1, encoder_outputs.size(1), 1), ))
        
        # torchMoji, encoder_outputs -> aux(zs, em_mu, em_logvar)
        aux_zs, aux_em_mu, aux_em_logvar, aux_em_params = self.aux_emotion_net(torchmoji_hidden, speaker_embed, encoder_outputs, text_lengths=text_lengths)
        
        # memory -> mel_outputs
        memory = torch.cat(memory, dim=2)# concat along Embed dim
        mel_outputs, gate_outputs, alignments = self.decoder(memory, gt_mels, memory_lengths=text_lengths, preserve_decoder=preserve_decoder_states,
                                                                         teacher_force_till=teacher_force_till, p_teacher_forcing=p_teacher_forcing)
        
        # mel_outputs -> mel_outputs_postnet (learn a modifier for the output)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet.add_(mel_outputs)
        
        return self.mask_outputs(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, pred_sylps
                [sylzu, syl_mu, syl_logvar],
                [zs, em_zu, em_mu, em_logvar, em_params],
                [aux_zs, aux_em_mu, aux_em_logvar, aux_em_params],
            ],
            output_lengths)
    
    def inference(self, text, speaker_ids, style_input=None, style_mode=None, text_lengths=None):
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        encoder_outputs = self.encoder.inference(embedded_text, speaker_ids=speaker_ids, text_lengths=text_lengths) # [B, time, encoder_out]
        
        if self.with_gst:
            assert (style_input is not None) or (style_mode.lower() == 'zeros'), "style mode specified but no style_input"
            if style_mode.lower() == 'mel': # enter any 160 channel mel-spectrogram
                embedded_gst, *_ = self.gst(style_input)
            elif style_mode.lower() == 'zeros': # enter any value, will set style_tokens to 0
                weights = torch.ones(1, self.token_num).cuda()
                embedded_gst, *_ = self.gst(weights*0.0, ref_mode=0).half()
            elif style_mode.lower() == 'style_token' or  style_mode.lower() == 'token': # should input style_token length list/array
                assert len(style_input) == self.token_num
                weights = torch.FloatTensor(style_input).unsqueeze(0).cuda()
                embedded_gst, *_ = self.gst(weights, ref_mode=0).half()
            elif style_mode.lower() == 'torchmoji_hidden':
                assert type(style_input) == torch.Tensor
                embedded_gst, *_ = self.gst(style_input, ref_mode=3).half() # should input hidden_state of torchMoji as tensor
            elif style_mode.lower() == 'torchmoji_string':
                assert type(style_input) == type(list()) or type(style_input) == type('')
                if type(style_input) == type(''):
                    style_input = [style_input,]
                embedded_gst, *_ = self.gst(style_input, ref_mode=2).half() # should input text as string
            else:
                raise NotImplementedError("No style option specified however styles are used in this model.")
            embedded_gst = embedded_gst.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_gst), dim=2) # [batch, time, encoder_out]
        
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_speakers), dim=2) # [batch, time, encoder_out]
        
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, memory_lengths=text_lengths)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet.add_(mel_outputs)
        
        return self.mask_outputs(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
