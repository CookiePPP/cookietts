from math import sqrt
import random
import numpy as np
from numpy import finfo

import torch
from torch import nn
from torch.nn import functional as F

from torch import Tensor
from typing import List, Tuple, Optional

from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm, LSTMCellWithZoneout, GMMAttention, DynamicConvolutionAttention
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame

from CookieTTS._2_ttm.tacotron2_ssvae.nets.SylpsNet import SylpsNet
from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d, LnBatchNorm1d

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


#@torch.jit.script
def scale_grads(input, scale: float):
    """
    Change gradient magnitudes
    Note: Do not use @torch.jit.script on pytorch <= 1.6 with this function!
          no_grad() and detach() do not work correctly till version 1.7 with JIT script.
    """
    out = input.clone()
    out *= scale               # multiply tensor
    out.detach().mul_(1/scale) # reverse multiply without telling autograd
    return out


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
                 attention_location_n_filters, attention_location_kernel_size,
                 windowed_attention_range: int=0,
                 windowed_att_pos_learned: bool=True,
                 windowed_att_pos_offset: float=0.,
                 attention_learned_temperature: bool=False):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, # Crushes the Encoder outputs to Attention Dimension used by this module
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.windowed_attention_range = windowed_attention_range
        if windowed_att_pos_learned is True:
            self.windowed_att_pos_offset = nn.Parameter( torch.zeros(1) )
        else:
            self.windowed_att_pos_offset = windowed_att_pos_offset
        
        self.softmax_temp = nn.Parameter(torch.tensor(1.0)) if attention_learned_temperature else None
        self.score_mask_value = -float("inf")
    
    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat,
                mask: Optional[Tensor] = None,
                memory_lengths: Optional[Tensor] = None,
                attention_weights: Optional[Tensor] = None,
                current_pos: Optional[Tensor] = None,
                score_mask_value: float = -float('inf')) -> Tuple[Tensor, Tensor]:
        """
        PARAMS
        ------
        attention_hidden_state:
            [B, AttRNN_dim] FloatTensor
                attention rnn last output
        memory:
            [B, txt_T, enc_dim] FloatTensor
                encoder outputs
        processed_memory:
            [B, txt_T, proc_enc_dim] FloatTensor
                processed encoder outputs
        attention_weights_cat:
            [B, 2 (or 3), txt_T] FloatTensor
                previous, cummulative (and sometimes exp_avg) attention weights
        mask:
            [B, txt_T] BoolTensor
                mask for padded data
        attention_weights: (Optional)
            [B, txt_T] FloatTensor
                optional override attention_weights
                useful for duration predictor attention or perfectly copying a clip with an alternative speaker.
        """
        B, txt_T, enc_dim = memory.shape
        
        if attention_weights is None:
            processed = self.location_layer(attention_weights_cat) # [B, 2, txt_T] # conv1d, matmul
            processed.add_( self.query_layer(attention_hidden_state.unsqueeze(1)).expand_as(processed_memory) ) # unsqueeze, matmul, expand_as, add_
            processed.add_( processed_memory ) # add_
            alignment = self.v( torch.tanh( processed ) ).squeeze(-1) # tanh, matmul, squeeze
            
            if mask is not None:
                if self.windowed_attention_range > 0 and current_pos is not None:
                    if self.windowed_att_pos_offset:
                        current_pos = current_pos + self.windowed_att_pos_offset
                    max_end = memory_lengths - 1 - self.windowed_attention_range
                    min_start = self.windowed_attention_range
                    current_pos = torch.min(current_pos.clamp(min=min_start), max_end.to(current_pos))
                    
                    mask_start = (current_pos-self.windowed_attention_range).clamp(min=0).round() # [B]
                    mask_end = mask_start+(self.windowed_attention_range*2)                       # [B]
                    pos_mask = torch.arange(txt_T, device=current_pos.device).unsqueeze(0).repeat(B, 1)  # [B, txt_T]
                    pos_mask = (pos_mask >= mask_start.unsqueeze(1).repeat(1, txt_T)) & (pos_mask <= mask_end.unsqueeze(1).repeat(1, txt_T))# [B, txt_T]
                    
                    # attention_weights_cat[pos_mask].view(B, self.windowed_attention_range*2+1) # for inference masked_select later
                    
                    mask = mask | ~pos_mask# [B, txt_T] & [B, txt_T] -> [B, txt_T]
                alignment.data.masked_fill_(mask, score_mask_value)#    [B, txt_T]
            
            softmax_temp = self.softmax_temp
            if softmax_temp is not None:
                alignment *= softmax_temp
            
            attention_weights = F.softmax(alignment, dim=1)# [B, txt_T] # softmax along encoder tokens dim
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)# unsqueeze, bmm
                                      # [B, 1, txt_T] @ [B, txt_T, enc_dim] -> [B, 1, enc_dim]
        
        attention_context = attention_context.squeeze(1)# [B, 1, enc_dim] -> [B, enc_dim] # squeeze
        
        new_pos = (attention_weights*torch.arange(txt_T, device=attention_weights.device).expand(B, -1)).sum(1)
                       # ([B, txt_T] * [B, txt_T]).sum(1) -> [B]
        
        return attention_context, attention_weights, new_pos# [B, enc_dim], [B, txt_T]


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, p_prenet_dropout, prenet_batchnorm, bn_momentum):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.p_prenet_dropout = p_prenet_dropout
        self.prenet_batchnorm = prenet_batchnorm
        self.p_prenet_input_dropout = 0.
        
        self.batchnorms = None
        if self.prenet_batchnorm:
            self.batchnorms = nn.ModuleList([ MaskedBatchNorm1d(size, eval_only_momentum=False, momentum=bn_momentum) for size in sizes ])
            self.batchnorms.append(MaskedBatchNorm1d(in_dim, eval_only_momentum=False, momentum=bn_momentum))
    
    def forward(self, x):
        if self.p_prenet_input_dropout: # dropout from the input, definitely a dangerous idea, but I think it would be very interesting to try values like 0.05 and see the effect
            x = F.dropout(x, self.p_prenet_input_dropout, self.training)
        
        if self.batchnorms is not None:
            x = self.batchnorms[-1](x)
        
        for i, linear in enumerate(self.layers):
            x = F.relu(linear(x))
            if self.p_prenet_dropout > 0:
                x = F.dropout(x, p=self.p_prenet_dropout, training=True)
            if self.batchnorms is not None:
                x = self.batchnorms[i](x)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.b_res = hparams.postnet_residual_connections if hasattr(hparams, 'postnet_residual_connections') else False
        self.convolutions = nn.ModuleList()
        
        prev_output_layer = True
        for i in range(hparams.postnet_n_convolutions):
            is_output_layer = (bool(self.b_res) and bool( i % self.b_res == 0 )) or (i+1 == hparams.postnet_n_convolutions)
            layers = [ ConvNorm(hparams.n_mel_channels if prev_output_layer else hparams.postnet_embedding_dim,
                             hparams.n_mel_channels if is_output_layer else hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='linear' if is_output_layer else 'tanh'), ]
            if not is_output_layer:
                layers.append(nn.BatchNorm1d(hparams.postnet_embedding_dim))
            prev_output_layer = is_output_layer
            self.convolutions.append(nn.Sequential(*layers))
    
    def forward(self, x):
        x_orig = x.clone()
        len_convs = len(self.convolutions)
        for i, conv in enumerate(self.convolutions):
            if (bool(self.b_res) and bool( i % self.b_res == 0 )) or (i+1 == len_convs):
                x_orig = x_orig + conv(x)
                x = x_orig
            else:
                x = F.dropout(torch.tanh(conv(x)), drop_rate, self.training)
        
        return x_orig


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
        
        self.sylps_layer = LinearNorm(hparams.encoder_LSTM_dim, 1)
    
    def forward(self, text, text_lengths=None, speaker_ids=None):
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
        
        if text_lengths is not None:
            # pytorch tensor are not reversible, hence the conversion
            text_lengths = text_lengths.cpu().numpy()
            text = nn.utils.rnn.pack_padded_sequence(
                text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, (hidden_state, _) = self.lstm(text)
        
        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)
        
        hidden_state = hidden_state.transpose(0, 1)# [2, B, h_dim] -> [B, 2, h_dim]
        B, _, h_dim = hidden_state.shape
        hidden_state = hidden_state.contiguous().view(B, -1)# [B, 2, h_dim] -> [B, 2*h_dim]
        pred_sylps = self.sylps_layer(hidden_state)# [B, 2*h_dim] -> [B, 1]
        
        return outputs, hidden_state, pred_sylps


class MemoryBottleneck(nn.Module):
    """
    Crushes the memory/encoder outputs dimension to save excess computation during Decoding.
    (If it works for the Attention then I don't see why it shouldn't also work for the Decoder)
    """
    def __init__(self, hparams):
        super(MemoryBottleneck, self).__init__()
        self.mem_output_dim = hparams.memory_bottleneck_dim
        self.mem_input_dim = hparams.encoder_LSTM_dim + hparams.speaker_embedding_dim + hparams.torchMoji_crushedDim + 1
        self.bottleneck = LinearNorm(self.mem_input_dim, self.mem_output_dim, bias=hparams.memory_bottleneck_bias, w_init_gain='tanh')
    
    def forward(self, memory):
        memory = self.bottleneck(memory)# [B, txt_T, input_dim] -> [B, txt_T, output_dim]
        return memory


@torch.jit.ignore
def HeightGaussianBlur(inp, blur_strength=1.0):
    """
    inp: [B, H, W] FloatTensor
    blur_strength: Float - min 0.0, max 5.0"""
    inp = inp.unsqueeze(1) # [B, height, width] -> [B, 1, height, width]
    var_ = blur_strength
    norm_dist = torch.distributions.normal.Normal(0, var_)
    conv_kernel = torch.stack([norm_dist.cdf(torch.tensor(i+0.5)) - norm_dist.cdf(torch.tensor(i-0.5)) for i in range(int(-var_*3-1),int(var_*3+2))], dim=0)[None, None, :, None]
    input_padding = (conv_kernel.shape[2]-1)//2
    out = F.conv2d(F.pad(inp, (0,0,input_padding,input_padding), mode='reflect'), conv_kernel).squeeze(1) # [B, 1, height, width] -> [B, height, width]
    return out


class Decoder(nn.Module):
    attention_hidden: Optional[torch.Tensor]# init self vars
    attention_cell: Optional[torch.Tensor]# init self vars
    decoder_hidden: Optional[torch.Tensor]# init self vars
    decoder_cell: Optional[torch.Tensor]# init self vars
    second_decoder_hidden: Optional[torch.Tensor]# init self vars
    second_decoder_cell: Optional[torch.Tensor]# init self vars
    attention_weights: Optional[torch.Tensor]# init self vars
    attention_weights_cum: Optional[torch.Tensor]# init self vars
    saved_attention_weights: Optional[torch.Tensor]# init self vars
    saved_attention_weights_cum: Optional[torch.Tensor]# init self vars
    attention_context: Optional[torch.Tensor]# init self vars
    previous_location: Optional[torch.Tensor]# init self vars
    memory: Optional[torch.Tensor]# init self vars
    processed_memory: Optional[torch.Tensor]# init self vars
    mask: Optional[torch.Tensor]# init self vars
    gate_delay: int
    
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        if hparams.use_memory_bottleneck:
            self.memory_dim = hparams.memory_bottleneck_dim
        else:
            self.memory_dim = hparams.encoder_LSTM_dim + hparams.speaker_embedding_dim + len(hparams.emotion_classes) + hparams.emotionnet_latent_dim + 1# size 1 == "sylzu"
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.prenet_layers = hparams.prenet_layers
        self.prenet_batchnorm = hparams.prenet_batchnorm if hasattr(hparams, 'prenet_batchnorm') else False
        self.prenet_bn_momentum = hparams.prenet_bn_momentum
        self.p_prenet_dropout = hparams.p_prenet_dropout
        self.prenet_speaker_embed_dim = hparams.prenet_speaker_embed_dim if hasattr(hparams, 'prenet_speaker_embed_dim') else 0
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold# Float
        self.AttRNN_extra_decoder_input = hparams.AttRNN_extra_decoder_input
        self.AttRNN_hidden_dropout_type = hparams.AttRNN_hidden_dropout_type
        self.p_AttRNN_hidden_dropout = hparams.p_AttRNN_hidden_dropout
        self.DecRNN_hidden_dropout_type = hparams.DecRNN_hidden_dropout_type
        self.p_DecRNN_hidden_dropout = hparams.p_DecRNN_hidden_dropout
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.teacher_force_till = hparams.teacher_force_till
        self.num_att_mixtures = hparams.num_att_mixtures
        self.normalize_attention_input = hparams.normalize_attention_input
        self.normalize_AttRNN_output = hparams.normalize_AttRNN_output
        self.attention_type = hparams.attention_type
        self.windowed_attention_range = hparams.windowed_attention_range if hasattr(hparams, 'windowed_attention_range') else 0
        self.windowed_att_pos_offset =  hparams.windowed_att_pos_offset if hasattr(hparams, 'windowed_att_pos_offset') else 0
        if self.windowed_attention_range:
            self.exp_smoothing_factor = nn.Parameter( torch.ones(1) * 0.0 )
        self.attention_layers = hparams.attention_layers
        
        self.dump_attention_weights = False
        self.prenet_noise = hparams.prenet_noise
        self.prenet_blur_min = hparams.prenet_blur_min
        self.prenet_blur_max = hparams.prenet_blur_max
        self.context_frames = hparams.context_frames
        self.hide_startstop_tokens = hparams.hide_startstop_tokens
        
        # Default States
        self.second_decoder_rnn = None
        self.memory_bottleneck = None
        self.attention_hidden = None
        self.attention_cell = None
        self.decoder_hidden = None
        self.decoder_cell = None
        self.second_decoder_hidden = None
        self.second_decoder_cell = None
        self.attention_weights = None
        self.attention_weights_cum = None
        self.attention_position = None
        self.saved_attention_weights = None
        self.saved_attention_weights_cum = None
        self.attention_context = None
        self.previous_location = None
        self.attention_weights_scaler = None
        self.memory = None
        self.processed_memory = None
        self.mask = None
        self.gate_delay = 0
        
        if hparams.use_memory_bottleneck:
            self.memory_bottleneck = MemoryBottleneck(hparams)
        
        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step * self.context_frames,
            [hparams.prenet_dim]*hparams.prenet_layers, self.p_prenet_dropout,
            self.prenet_batchnorm, self.prenet_bn_momentum)
        
        AttRNN_Dimensions = hparams.prenet_dim + self.memory_dim
        if self.AttRNN_extra_decoder_input:
            AttRNN_Dimensions += hparams.decoder_rnn_dim
        
        self.attention_rnn = LSTMCellWithZoneout(
            AttRNN_Dimensions, hparams.attention_rnn_dim, bias=True,
            zoneout=self.p_AttRNN_hidden_dropout if self.AttRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_AttRNN_hidden_dropout if self.AttRNN_hidden_dropout_type == 'dropout' else 0.0)
        
        if self.attention_type == 0:
            self.attention_layer = Attention(
                hparams.attention_rnn_dim, self.memory_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size,
                self.windowed_attention_range,
                hparams.windowed_att_pos_learned,
                self.windowed_att_pos_offset,
                hparams.attention_learned_temperature)
        else:
            print("attention_type invalid, valid values are... 0 and 1")
            raise
        
        if hasattr(hparams, 'use_cum_attention_scaler') and hparams.use_cum_attention_scaler:
            self.attention_weights_scaler = nn.Parameter(torch.ones(1)*-2.0)
        
        self.decoder_residual_connection = hparams.decoder_residual_connection
        if self.decoder_residual_connection:
            assert (hparams.attention_rnn_dim + self.memory_dim) == hparams.decoder_rnn_dim, f"if using 'decoder_residual_connection', decoder_rnn_dim must equal attention_rnn_dim + memory_dim ({hparams.attention_rnn_dim + self.memory_dim})."
        self.decoder_rnn = LSTMCellWithZoneout(
            hparams.attention_rnn_dim + self.memory_dim, hparams.decoder_rnn_dim, bias=True,
            zoneout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'dropout' else 0.0)
        decoder_rnn_output_dim = hparams.decoder_rnn_dim
        
        if hparams.second_decoder_rnn_dim > 0:
            self.second_decoder_rnn_dim = hparams.second_decoder_rnn_dim
            self.second_decoder_residual_connection = hparams.second_decoder_residual_connection
            if self.second_decoder_residual_connection:
                assert self.second_decoder_rnn_dim == hparams.decoder_rnn_dim, "if using 'second_decoder_residual_connection', both DecoderRNN dimensions must match."
            self.second_decoder_rnn = LSTMCellWithZoneout(
            hparams.decoder_rnn_dim, hparams.second_decoder_rnn_dim, bias=True,
            zoneout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'zoneout' else 0.0,
            dropout=self.p_DecRNN_hidden_dropout if self.DecRNN_hidden_dropout_type == 'dropout' else 0.0)
            decoder_rnn_output_dim = hparams.second_decoder_rnn_dim
        
        self.linear_projection = LinearNorm(
            decoder_rnn_output_dim + self.memory_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)
        
        self.gate_layer = LinearNorm(
            decoder_rnn_output_dim + self.memory_dim, 1,
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
        decoder_input = memory.new_zeros(B, self.n_mel_channels * self.n_frames_per_step)
        return decoder_input
    
    
    def initialize_decoder_states(self, memory, mask: Optional[Tensor] = None, preserve: Optional[Tensor] = None):
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
        
        attention_hidden = self.attention_hidden # https://github.com/pytorch/pytorch/issues/22155#issue-460050914
        attention_cell = self.attention_cell
        if attention_hidden is not None and attention_cell is not None and preserve is not None:
            attention_hidden *= preserve
            attention_hidden.detach_()
            attention_cell *= preserve
            attention_cell.detach_()
        else:
            self.attention_hidden = memory.new_zeros( B, self.attention_rnn_dim)# attention hidden state
            self.attention_cell = memory.new_zeros( B, self.attention_rnn_dim)# attention cell state
        
        decoder_hidden = self.decoder_hidden
        decoder_cell = self.decoder_cell
        if decoder_hidden is not None and decoder_cell is not None and preserve is not None:
            decoder_hidden *= preserve
            decoder_hidden.detach_()
            decoder_cell *= preserve
            decoder_cell.detach_()
        else:
            self.decoder_hidden = memory.new_zeros( B, self.decoder_rnn_dim)# LSTM decoder hidden state
            self.decoder_cell = memory.new_zeros( B, self.decoder_rnn_dim)# LSTM decoder cell state
        
        second_decoder_rnn = self.second_decoder_rnn
        if second_decoder_rnn is not None:
            second_decoder_hidden = self.second_decoder_hidden
            second_decoder_cell = self.second_decoder_cell
            if second_decoder_hidden is not None and second_decoder_cell is not None and preserve is not None:
                second_decoder_hidden *= preserve
                second_decoder_hidden.detach_()
                second_decoder_cell *= preserve
                second_decoder_cell.detach_()
            else:
                self.second_decoder_hidden = memory.new_zeros( B, self.second_decoder_rnn_dim)# LSTM decoder hidden state
                self.second_decoder_cell = memory.new_zeros( B, self.second_decoder_rnn_dim)# LSTM decoder cell state
        
        attention_weights = self.attention_weights
        if attention_weights is not None and preserve is not None: # save all the encoder possible
            self.saved_attention_weights = attention_weights
            self.saved_attention_weights_cum = self.attention_weights_cum
        
        self.attention_weights = memory.new_zeros( B, MAX_ENCODE)# attention weights of that frame
        self.attention_weights_cum = memory.new_zeros( B, MAX_ENCODE)# cumulative weights of all frames during that inferrence
        
        saved_attention_weights = self.saved_attention_weights
        attention_weights = self.attention_weights
        attention_weights_cum = self.attention_weights_cum
        if saved_attention_weights is not None and attention_weights is not None and attention_weights_cum is not None and preserve is not None:
            COMMON_ENCODE = min(MAX_ENCODE, saved_attention_weights.shape[1]) # smallest MAX_ENCODE of the saved and current encodes
            attention_weights[:, :COMMON_ENCODE] = saved_attention_weights[:, :COMMON_ENCODE] # preserve any encoding weights possible (some will be part of the previous iterations padding and are gone)
            attention_weights_cum[:, :COMMON_ENCODE] = saved_attention_weights[:, :COMMON_ENCODE]
            attention_weights *= preserve
            attention_weights.detach_()
            attention_weights_cum *= preserve
            attention_weights_cum.detach_()
            if self.attention_type == 2: # Dynamic Convolution Attention
                    attention_weights[:, 0] = ~(preserve==1)[:,0] # [B, 1] -> [B] # initialize the weights at encoder step 0
                    attention_weights_cum[:, 0] = ~(preserve==1)[:,0] # [B, 1] -> [B] # initialize the weights at encoder step 0
        elif attention_weights is not None and attention_weights_cum is not None and self.attention_type == 2:
            first_attention_weights = attention_weights[:, 0]
            first_attention_weights.fill_(1.) # initialize the weights at encoder step 0
            first_attention_weights_cum = attention_weights_cum[:, 0]
            first_attention_weights_cum.fill_(1.) # initialize the weights at encoder step 0
        
        attention_context = self.attention_context
        if attention_context is not None and preserve is not None:
            attention_context *= preserve
            self.attention_context = attention_context.detach()
        else:
            self.attention_context = memory.new_zeros(B, self.memory_dim)# attention output
        
        self.memory = memory
        if self.attention_type == 0:
            self.processed_memory = self.attention_layer.memory_layer(memory) # Linear Layer, [B, txt_T, enc_dim] -> [B, txt_T, attention_dim]
            if self.windowed_attention_range:
                attention_position = self.attention_position
                if attention_position is not None and preserve is not None:
                    attention_position *= preserve.squeeze(1)
                    attention_position.detach_()
                else:
                    self.attention_position = memory.new_zeros(B)# [B]
        
        elif self.attention_type == 1:
            self.previous_location = memory.new_zeros( B, 1, self.num_att_mixtures)
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
    
    def parse_decoder_outputs(self,
            mel_outputs:  List[Tensor],
            gate_outputs: List[Tensor],
            alignments:   List[Tensor],
            hidden_att_contexts_arr: Optional[List[Tensor]] = None):
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
        gate_outputs = gate_outputs.transpose(0, 1) if len(gate_outputs.size()) > 1 else gate_outputs[None]
        gate_outputs = gate_outputs.contiguous()
        
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view( mel_outputs.size(0), -1, self.n_mel_channels )
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        
        if hidden_att_contexts_arr is not None and len(hidden_att_contexts_arr):
            hidden_att_contexts = torch.stack(hidden_att_contexts_arr, dim=1).transpose(1, 2)# list([B, dim], [B, dim], ...) -> [B, dim, mel_T]
        else:
            hidden_att_contexts = None
        return mel_outputs, gate_outputs, alignments, hidden_att_contexts
    
    def decode(self, decoder_input: Tensor, memory_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        attention_hidden      = self.attention_hidden
        assert attention_hidden is not None
        attention_cell        = self.attention_cell
        assert attention_cell is not None
        attention_weights     = self.attention_weights
        assert attention_weights is not None
        attention_context     = self.attention_context
        assert attention_context is not None
        attention_weights_cum = self.attention_weights_cum
        assert attention_weights_cum is not None
        decoder_hidden        = self.decoder_hidden
        assert decoder_hidden is not None
        decoder_cell          = self.decoder_cell
        second_decoder_rnn = self.second_decoder_rnn
        if second_decoder_rnn is not None:
            assert decoder_cell is not None
            second_decoder_hidden = self.second_decoder_hidden
            assert second_decoder_hidden is not None
            second_decoder_cell   = self.second_decoder_cell
            assert second_decoder_cell is not None
        memory = self.memory
        assert memory is not None
        processed_memory = self.processed_memory
        assert processed_memory is not None
        mask = self.mask
        assert mask is not None
        
        if self.AttRNN_extra_decoder_input:
            cell_input = torch.cat((decoder_input, attention_context, decoder_hidden), -1)# [Processed Previous Spect Frame, Last input Taken from Text/Att, Previous Decoder state used to produce frame]
        else:
            cell_input = torch.cat((decoder_input, attention_context), -1)# [Processed Previous Spect Frame, Last input Taken from Text/Att]
        
        if self.normalize_AttRNN_output and self.attention_type == 1:
            cell_input = cell_input.tanh()
        
        _ = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        self.attention_hidden = attention_hidden = _[0]
        self.attention_cell   = attention_cell   = _[1]
        
        scaled_attention_weights_cum = attention_weights_cum.unsqueeze(1)
        if self.attention_weights_scaler is not None:
            scaled_attention_weights_cum *= self.attention_weights_scaler
        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), scaled_attention_weights_cum), dim=1)
        # [B, 1, txt_T] cat [B, 1, txt_T] -> [B, 2, txt_T]
        
        if self.attention_type == 0:
            _ = self.attention_layer(attention_hidden, memory, processed_memory, attention_weights_cat, mask, memory_lengths, None, self.attention_position)
            self.attention_context = attention_context = _[0]
            self.attention_weights = attention_weights = _[1]
            new_pos = _[2]
            
            attention_position = self.attention_position
            assert attention_position is not None
            exp_smoothing_factor = self.exp_smoothing_factor
            assert exp_smoothing_factor is not None
            
            smooth_factor = torch.sigmoid(exp_smoothing_factor)
            self.attention_position = (attention_position*smooth_factor) + (new_pos*(1-smooth_factor))
            
            attention_weights_cum += attention_weights
            self.attention_weights_cum = attention_weights_cum
        else:
            raise NotImplementedError(f"Attention Type {self.attention_type} Invalid / Not Implemented / Deprecated")
        
        decoder_input = torch.cat( ( attention_hidden, attention_context), -1) # cat 6.475ms
        
        decoderrnn_state = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))# lstmcell 12.789ms
        self.decoder_hidden = decoder_rnn_output = decoderrnn_state[0]
        self.decoder_cell   = decoder_cell       = decoderrnn_state[1]
        if self.decoder_residual_connection:
            decoder_rnn_output = decoder_rnn_output + decoder_input
        
        if second_decoder_rnn is not None:
            second_decoder_state = self.second_decoder_rnn(decoder_rnn_output, (second_decoder_hidden, second_decoder_cell))
            self.second_decoder_hidden = second_decoder_hidden = second_decoder_state[0]
            self.second_decoder_cell   = second_decoder_cell   = second_decoder_state[1]
            if self.second_decoder_residual_connection:
                decoder_rnn_output = decoder_rnn_output + second_decoder_hidden
            else:
                decoder_rnn_output = second_decoder_hidden
        
        decoder_hidden_attention_context = torch.cat( (decoder_rnn_output, attention_context), dim=1) # -> [B, dim] cat 6.555ms
        
        gate_prediction = self.gate_layer(decoder_hidden_attention_context) # -> [B, 1] addmm 5.762ms
        
        decoder_output = self.linear_projection(decoder_hidden_attention_context) # -> [B, context_frames*n_mel] addmm 5.621ms
        
        return decoder_output, gate_prediction, attention_weights, decoder_hidden_attention_context
    
    def forward(self, memory, decoder_inputs, memory_lengths,
                preserve_decoder:    Optional[Tensor] = None,
                decoder_input:       Optional[Tensor] = None,
                teacher_force_till:               int = 0,
                p_teacher_forcing:              float = 1.0,
                return_hidden_state:             bool = False):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory             : Encoder outputs
        decoder_inputs     : Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths     : Encoder output lengths for attention masking.
        preserve_decoder   : [B] Tensor - Preserve model state for True items in batch/Tensor
        decoder_input      : [B, n_mel, context] FloatTensor
        teacher_force_till : INT - Beginning X frames where Teacher Forcing is forced ON.
        p_teacher_forcing  : Float - 0.0 to 1.0 - Change to use Teacher Forcing during training/validation.
        
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        if self.hide_startstop_tokens: # remove first/last tokens from Memory # I no longer believe this is useful.
            memory = memory[:,1:-1,:]
            memory_lengths = memory_lengths-2
        
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory)
        
        #if self.prenet_blur_max > 0.0:
        #    rand_blur_strength = torch.rand(1).uniform_(self.prenet_blur_min, self.prenet_blur_max)
        #    decoder_inputs = HeightGaussianBlur(decoder_inputs, blur_strength=rand_blur_strength)# [B, n_mel, mel_T]
        
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)# [B, n_mel, mel_T] -> [mel_T, B, n_mel]
        
        if decoder_input is None:
            decoder_input = self.get_go_frame(memory).unsqueeze(0) # create blank starting frame
            if self.context_frames > 1:
                decoder_input = decoder_input.repeat(self.context_frames, 1, 1)
        else:
            decoder_input = decoder_input.permute(2, 0, 1) # [B, n_mel, context_frames] -> [context_frames, B, n_mel]
        # memory -> (1, B, n_mel) <- which is all 0's
        
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # [context_frames+mel_T, B, n_mel] concat T_out
        
        if self.prenet_noise and self.training:
            decoder_inputs = decoder_inputs + self.prenet_noise * torch.randn(decoder_inputs.shape, device=decoder_inputs.device, dtype=decoder_inputs.dtype)
        
        #if self.prenet_speaker_embed_dim: # __future__ feature, not added yet!
        #    embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
        #    embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
        #    decoder_inputs = torch.cat((decoder_inputs, embedded_speakers), dim=2)
        
        decoder_inputs = self.prenet(decoder_inputs)
        
        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths), preserve=preserve_decoder)
        
        mel_outputs, gate_outputs, alignments, hidden_att_contexts = [], [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            if teacher_force_till >= len(mel_outputs) or p_teacher_forcing >= torch.rand(1):
                decoder_input = decoder_inputs[len(mel_outputs)] # use teacher forced input
            else:
                decoder_input = self.prenet(mel_outputs[-1][:, :self.n_mel_channels]) # [B, n_mel] use last output for next input (like inference)
            
            mel_output, gate_output, attention_weights, decoder_hidden_attention_context = self.decode(decoder_input, memory_lengths)
            
            if self.dump_attention_weights:
                attention_weights = attention_weights.cpu()
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            if not self.dump_attention_weights or len(mel_outputs) < 2:
                alignments += [attention_weights]
            if return_hidden_state:
                hidden_att_contexts += [decoder_hidden_attention_context]
        
        mel_outputs, gate_outputs, alignments, hidden_att_contexts = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, hidden_att_contexts)
        
        return mel_outputs, gate_outputs, alignments, hidden_att_contexts, memory
    
    @torch.jit.export
    def inference(self, memory,
            memory_lengths: Tensor,
            return_hidden_state: bool = False):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        
        RETURNS
        -------
        mel_outputs  : mel outputs from the decoder
        gate_outputs : gate outputs from the decoder
        alignments   : sequence of attention weights from the decoder
        """
        if self.hide_startstop_tokens: # remove start/stop token from Decoder
            memory = memory[:,1:-1,:]
            memory_lengths = memory_lengths-2
        
        if hasattr(self, 'memory_bottleneck'):
            memory = self.memory_bottleneck(memory)
        
        decoder_input = self.get_go_frame(memory)
        
        self.initialize_decoder_states(memory, mask=None if memory_lengths is None else ~get_mask_from_lengths(memory_lengths))
        
        sig_max_gates = torch.zeros(decoder_input.size(0))
        mel_outputs, gate_outputs, alignments, hidden_att_contexts, break_point = [], [], [], [], self.max_decoder_steps
        for i in range(self.max_decoder_steps):
            decoder_input = self.prenet(decoder_input[:, :self.n_mel_channels])
            
            mel_output, gate_output_gpu, alignment, decoder_hidden_attention_context = self.decode(decoder_input, memory_lengths)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_output_cpu = gate_output_gpu.cpu().float() # small operations e.g min(), max() and sigmoid() are faster on CPU # also .float() because Tensor.min() doesn't work on half precision CPU
            gate_outputs += [gate_output_gpu.squeeze(1)]
            alignments += [alignment]
            if return_hidden_state:
                hidden_att_contexts += [decoder_hidden_attention_context]
            
            if self.attention_type == 1 and self.num_att_mixtures == 1:# stop when the attention location is out of the encoder_outputs
                previous_location = self.previous_location
                assert previous_location is not None
                if previous_location.squeeze().item() + 1. > memory.shape[1]:
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
        
        mel_outputs, gate_outputs, alignments, hidden_att_contexts = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments, hidden_att_contexts)
        
        # apply sigmoid to the GPU as well.
        gate_outputs = torch.sigmoid(gate_outputs)
        
        return mel_outputs, gate_outputs, alignments, hidden_att_contexts


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
        if not hasattr(hparams, 'use_postnet') or hparams.use_postnet:
            self.postnet = Postnet(hparams)
        self.sylps_net = SylpsNet(hparams)
        self.tm_linear = nn.Linear(hparams.torchMoji_attDim, hparams.torchMoji_crushedDim)
        if hparams.torchMoji_BatchNorm:
            self.tm_bn = MaskedBatchNorm1d(hparams.torchMoji_attDim, eval_only_momentum=False, momentum=0.05)
        
        if hparams.use_postnet_generator_and_discriminator if hasattr(hparams, "use_postnet_generator_and_discriminator") else False:
            self.postnet_grad_propagation_scalar = hparams.adv_postnet_grad_propagation
            self.adversarial_postnet = GANPostnet(hparams)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def mask_outputs(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            # [B, n_mel, steps]
            outputs[0].data.masked_fill_(mask, 0.0)
            if outputs[1] is not None:
                outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        
        return outputs
    
    def forward(self, gt_mel, mel_lengths, text, text_lengths, speaker_id, gt_sylps,
                torchmoji_hdn, pres_prev_state, cont_next_iter, init_mel,
                teacher_force_till=None, p_teacher_forcing=None, drop_frame_rate=None, return_hidden_state=False):
        with torch.no_grad():
            p_teacher_forcing  = self.p_teacher_forcing  if teacher_force_till is None else p_teacher_forcing
            teacher_force_till = self.teacher_force_till if p_teacher_forcing  is None else teacher_force_till
            drop_frame_rate    = self.drop_frame_rate    if drop_frame_rate    is None else drop_frame_rate
            
            if drop_frame_rate > 0. and self.training:
                gt_mel = dropout_frame(gt_mel, self.global_mean, mel_lengths, drop_frame_rate)# [B, n_mel, mel_T]
        
        memory = []
        
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, txt_T]
        encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        speaker_embed = self.speaker_embedding(speaker_id)
        memory.append( speaker_embed[:, None].repeat(1, encoder_outputs.size(1), 1) )
        
        # (SylpsNet) Sylps -> sylzu, mu, logvar
        sylzu, syl_mu, syl_logvar = self.sylps_net(gt_sylps)
        memory.append( sylzu[:, None].repeat(1, encoder_outputs.size(1), 1) )
        
        # (TorchMoji)
        torchmoji_hdn = self.tm_bn(torchmoji_hdn).to(sylzu)# [B, hdn_dim]
        torchmoji_hdn = self.tm_linear(torchmoji_hdn)#       [B, hdn_dim] -> [B, crushed_dim]
        torchmoji_hdn = torchmoji_hdn[:, None].repeat(1, encoder_outputs.size(1), 1)# [B, C] -> [B, txt_T, C]
        memory.append(torchmoji_hdn)
        
        # (Decoder/Attention) memory -> pred_mel
        memory = torch.cat(memory, dim=2)# concat along Embed dim # [B, txt_T, dim]
        pred_mel, pred_gate_logits, alignments, hidden_att_contexts, memory = self.decoder(memory, gt_mel, memory_lengths=text_lengths, preserve_decoder=pres_prev_state,
                                                   decoder_input=init_mel, teacher_force_till=teacher_force_till, p_teacher_forcing=p_teacher_forcing,
                                                   return_hidden_state=return_hidden_state)
        
        # (Postnet) pred_mel -> pred_mel_postnet (learn a modifier for the output)
        pred_mel_postnet = self.postnet(pred_mel) if hasattr(self, 'postnet') else None
        
        outputs = {
                   "pred_mel": pred_mel,           # [B, n_mel, mel_T]
           "pred_mel_postnet": pred_mel_postnet,   # [B, n_mel, mel_T]
           "pred_gate_logits": pred_gate_logits,   # [B, mel_T]
                 "pred_sylps": pred_sylps,         # [B]
              "pred_sylps_mu": syl_mu,             # [B]
          "pred_sylps_logvar": syl_logvar,         # [B]
                 "alignments": alignments,         # [B, mel_T, txt_T]
        "hidden_att_contexts": hidden_att_contexts,# [B, hdn_dim, mel_T]
            "encoder_outputs": encoder_outputs,    # [B, txt_T, enc_dim]
        }
        return outputs
    
    def update_device(self, **inputs):
        target_device = next(self.parameters()).device
        target_float_dtype = next(self.parameters()).dtype
        outputs = {}
        for key, input in inputs.items():
            if type(input) == Tensor:# move all Tensor types to GPU
                if input.dtype == torch.float32:
                    outputs[key] = input.to(target_device, target_float_dtype)# convert float to half if required.
                else:
                    outputs[key] = input.to(target_device                    )# leave Long / Bool unchanged in datatype
            else:
                outputs[key] = input
        return outputs
    
    def inference(self, text_seq, text_lengths, speaker_id, torchmoji_hdn, gt_sylps=None, return_hidden_state=False):# [B, enc_T], [B], [B], [B], [B, tm_dim]
        
        memory = []
        
        # (Encoder) Text -> Encoder Outputs, pred_sylps
        embedded_text = self.embedding(text_seq).transpose(1, 2) # [B, embed, txt_T]
        encoder_outputs, hidden_state, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_id) # [B, txt_T, enc_dim]
        memory.append(encoder_outputs)
        
        # (Speaker) speaker_id -> speaker_embed
        speaker_embed = self.speaker_embedding(speaker_id)
        memory.append( speaker_embed[:, None].repeat(1, encoder_outputs.size(1), 1) )
        
        # (SylpsNet) Sylps -> sylzu, mu, logvar
        sylzu = self.sylps_net.infer_auto(gt_sylps or pred_sylps, rand_sampling=False)
        memory.append( sylzu[:, None].repeat(1, encoder_outputs.size(1), 1) )
        
        # (TorchMoji)
        torchmoji_hdn = self.tm_bn(torchmoji_hdn).to(sylzu)# [B, hdn_dim]
        torchmoji_hdn = self.tm_linear(torchmoji_hdn)#       [B, hdn_dim] -> [B, crushed_dim]
        torchmoji_hdn = torchmoji_hdn[:, None].repeat(1, encoder_outputs.size(1), 1)# [B, C] -> [B, txt_T, C]
        memory.append(torchmoji_hdn)
        
        # (Decoder/Attention) memory -> pred_mel
        memory = torch.cat(memory, dim=2)# concat along Embed dim # [B, txt_T, dim]
        pred_mel, pred_gate, alignments, hidden_att_contexts = self.decoder.inference(memory, memory_lengths=text_lengths, return_hidden_state=return_hidden_state)
        
        # (Postnet) pred_mel -> pred_mel_postnet (learn a modifier for the output)
        pred_mel_postnet = self.postnet(pred_mel) if hasattr(self, 'postnet') else mel_outputs
        
        outputs = {
           "pred_mel_postnet": pred_mel_postnet,# [B, n_mel, mel_T]
                  "pred_gate": pred_gate,       # [B, mel_T]
                 "alignments": alignments,      # [B, mel_T, txt_T]
                 "pred_sylps": pred_sylps,      # [B]
        }
        return outputs
