from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths, dropout_frame
#from modules import GST # mellotron GST implementation
from TPGST import GST # Other GST implementation

#from kornia.filters import GaussianBlur2d

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


class LSTMCellWithZoneout(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True, zoneout_prob=0.1):
        super().__init__(input_size, hidden_size, bias)
        self._zoneout_prob = zoneout_prob

    def forward(self, input, hx):
        old_h, old_c = hx
        new_h, new_c = super(LSTMCellWithZoneout, self).forward(input, hx)
        if self.training:
            c_mask = torch.empty_like(new_c).bernoulli_(p=self._zoneout_prob).bool().data
            h_mask = torch.empty_like(new_h).bernoulli_(p=self._zoneout_prob).bool().data
            h = torch.where(h_mask, old_h, new_h)
            c = torch.where(c_mask, old_c, new_c)
            return h, c
        else:
            return new_h, new_c


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
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
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
        
        # easy to read
        #processed_query = self.query_layer(query.unsqueeze(1))
        #processed_attention_weights = self.location_layer(attention_weights_cat)
        #energies = self.v(torch.tanh(
        #    processed_query + processed_attention_weights + processed_memory))
        
        # using inplace addition
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


class DynamicConvolutionAttention(nn.Module):
    """A first attempt at making this Attention."""
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 dynamic_filter_num, dynamic_filter_len): # default 8, 21)
        super(DynamicConvolutionAttention, self).__init__()
        self.dynamic_filter_len = dynamic_filter_len
        self.dynamic_filter_num = dynamic_filter_num
        
        # static
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim, out_bias=True)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        
        # dynamic
        self.dynamic_filter = torch.nn.Sequential(
            LinearNorm(attention_rnn_dim, attention_dim, bias=True, w_init_gain='tanh'), # attention_rnn_dim -> attention dim
            nn.Tanh(),#nn.LeakyReLU(negative_slope=0.1),#nn.ReLU(),#nn.Tanh(),
            LinearNorm(attention_dim, dynamic_filter_num*dynamic_filter_len, bias=False, w_init_gain='tanh'), # filter_num * filter_length
            )
        self.vg = LinearNorm(dynamic_filter_num, attention_dim, bias=True)
        
        # prior
        self.prior_filter = self.get_prior_filter(dynamic_filter_num, dynamic_filter_len).to("cuda")
        
        # misc
        self.score_mask_value = -float("inf")
    
    def get_prior_filter(self, dynamic_filter_num, dynamic_filter_len):
        assert dynamic_filter_len == 21, "Only filter_len of 21 is currently supported" # I don't know how to calcuate this one atm so here's a set of premade values I found on their Reddit post
        prior_filters = torch.tensor( [0.7400209, 0.07474979, 0.04157422, 0.02947039, 0.023170564, 0.019321883, 0.016758798, 0.014978543, 0.013751862, 0.013028075, 0.013172861] ) # [filter_len-10]
        prior_filters = prior_filters.flip(dims=(0,)) # [filter_len-10] -> [filter_len-10]
        prior_filters = prior_filters[None, None, :] # [filter_len-10] -> [1, 1, filter_len-10]
        return prior_filters
    
    def get_alignment_energies(self, attention_RNN_state,
                               attention_weights_cat):
        """
        PARAMS
        ------
        attention_RNN_state: attention rnn last output [B, dim]       ## decoder output (batch, n_mel_channels * n_frames_per_step)
        attention_weights_cat: prev and cumulative att weights (B, 2, enc_time)
        
        RETURNS
        -------
        alignment (batch, enc_time)
        """
        verbose = 0 # debug
        # get Static filter intermediate value
        processed = self.location_layer(attention_weights_cat) # [B, 2, enc_T] -> [B, attention_n_filters, enc_T] -> [B, enc_T, attention_dim] # take prev+cumulative att weights, send through conv -> linear
        if verbose: print("1 processed.shape =", processed.shape) # [16, 90, 1]
        
        # get Dynamic filter intermediate value(s)
        prev_att = attention_weights_cat[:, 0, :][:, :, None] # [B, 2, enc_T] -> [B, enc_T] -> [B, enc_T, 1]
        dynamic_filt = self.dynamic_filter(attention_RNN_state) # [B, AttRNN_dim] -> [B, attention_dim] -> [B, 1, attention_dim] -> [B, 1, dynamic_filter_num*dynamic_filter_len]
        dynamic_filt = dynamic_filt.view([-1, self.dynamic_filter_len, self.dynamic_filter_num]) # [B, 1, dynamic_filter_num*dynamic_filter_len] -> [B, dynamic_filter_len, dynamic_filter_num]
        if verbose: print("1 prev_att.shape =", prev_att.shape) # [16, 90, 1]
        if verbose: print("1 dynamic_filt.shape =", dynamic_filt.shape) # [16, 21, 8]
        
        if True: # calc dynamic energies from matmul with dynamic filter
            # "stack previous alignments into matrices" # https://www.reddit.com/r/MachineLearning/comments/dmo0z1/r_attenchilada_locationrelative_attention/f6vtkmk/
            prev_att_stacked = prev_att.repeat(1,1,self.dynamic_filter_len)
            dynamic = prev_att_stacked @ dynamic_filt # [B, enc_T, dynamic_filter_len] @ [B, dynamic_filter_len, dynamic_filter_num] -> [B, enc_T, dynamic_filter_num]
            if True: # extra linear?
                dynamic = self.vg(dynamic) # [B, enc_T, dynamic_filter_num] -> [B, enc_T, attention_dim]
                pass
        else:  # calc dynamic engeries from F.conv1d with dynamic filter
            dynamic_filt = dynamic_filt.permute(2,0,1)[:,:,None,:] # [B, dynamic_filter_len, dynamic_filter_num] -> [dynamic_filter_num,B                 ,1, dynamic_filter_len]
                                                                                                                   #(out_channels      ,in_channels/groups,kH,kW                )
            if verbose: print("1.9 dynamic_filt.shape =", dynamic_filt.shape) # [8, 24, 1, 21]
            shape = dynamic_filt.shape
            dynamic_filt = dynamic_filt.reshape(shape[0]*shape[1], 1, 1, shape[-1]) # [dynamic_filter_num,B,1, dynamic_filter_len] -> [dynamic_filter_num*B,1,1, dynamic_filter_len]
            prev_att_shaped = prev_att[None, ...].permute(0, 1, 3, 2) # [B, enc_T, 1] -> [1, B, enc_T, 1] -> [1        ,B          ,1 ,enc_T]
                                                                                                            #(minibatch,in_channels,iH,iW   )
            if verbose: print("2 prev_att_shaped.shape =", prev_att_shaped.shape) # [1, 24, 1, 65] -> [24, 8, 1, 65]
            if verbose: print("2 dynamic_filt.shape =", dynamic_filt.shape) # [8, 24, 1, 21]
            if False:
                padd = (self.dynamic_filter_len-1)//2
                dynamic = torch.nn.functional.conv2d(prev_att_shaped, dynamic_filt, bias=None, stride=1, padding=(0,padd), dilation=1, groups=prev_att_shaped.size(1))# [1, B, 1, enc_T] -> [1, B*dyna_f_num, 1, enc_T]
            else:
                padd = self.dynamic_filter_len - 1
                prev_att_shaped = F.pad(prev_att_shaped, (padd, 0)) # [1, 1, B, enc_T] -> [1, 1, B, padd+enc_T]
                dynamic = torch.nn.functional.conv2d(prev_att_shaped, dynamic_filt, bias=None, stride=1, padding=0, dilation=1, groups=prev_att_shaped.size(1))# [1, B, 1, enc_T] -> [1, B*dyna_f_num, 1, enc_T]
            if verbose: print("2 dynamic.shape =", dynamic.shape) # [1, 8, 1, 65]
            dynamic = dynamic.view(shape[1], shape[0], -1).transpose(1,2) # [1, B*dyna_f_num, 1, enc_T] -> [B, dyna_f_num, enc_T] -> [B, enc_T, dyna_f_num]
        
        if verbose: print("2.1 dynamic.shape =", dynamic.shape) # [1, 8, 1, 65]
        
        # I don't currently know how the Dynamic and Static energies are meant to interact (I can't tell from the paper).
        if True: # first try addition
            energies = self.v( torch.tanh( processed + dynamic ) ) # [B, enc_T, attention_dim] -> [B, enc_T, 1] # mix them
        elif True: # then try concatentation
            proc = torch.cat( (processed, dynamic), dim=2) # [B, enc_T, dynamic_filter_num] + [B, enc_T, attention_dim] -> [B, enc_T, dynamic_filter_num+attention_dim]
            energies = self.v( torch.tanh( proc ) ) # [B, enc_T, attention_dim] -> [B, enc_T, 1] # mix them
        elif True: # then try adding seperated energies
            static_energies = self.v( torch.tanh( processed ) ) # [B, enc_T, attention_dim] -> [B, enc_T, 1] # mix them
            if verbose: print("static_energies.shape =", static_energies.shape)
            dynamic_energies = self.vg( torch.tanh(dynamic) ) # [B, enc_T, dynamic_filter_num] -> [B, enc_T, 1] # mix them
            if verbose: print("dynamic_energies.shape =", dynamic_energies.shape)
            energies = static_energies + dynamic_energies # [B, enc_T, 1] + [B, enc_T, 1] -> [B, enc_T, 1]
        else:
            pass
        
        if False: # add the Prior filter
            padd = self.dynamic_filter_len - 11
            prev_att = F.pad(prev_att.transpose(1,2), (padd, 0)) # [B, enc_T, 1] -> [B, 1, enc_T] -> [B, 1, enc_T+padd]
            prior_energy = F.conv1d(prev_att, self.prior_filter.to(prev_att.dtype)) # [B, 1, enc_T+padd] -> [B, 1, enc_T]
            prior_energy = (prior_energy.clamp(min=1e-6)).log() # [B, enc_T, 1] clamp min value so log doesn't underflow
            #prior_energy = prior_energy.clamp(min=1e-6)
            #prior_energy = prior_energy.squeeze(-1) # [B, enc_T, 1] -> [B, enc_T]
            
            if verbose: print("3 energies.shape =", energies.shape) #
            if verbose: print("3 prior_energy.shape =", prior_energy.shape) #
            
            energies += prior_energy.transpose(1,2) # [B, enc_T, 1]
        
        return energies.squeeze(-1) # [B, enc_T, 1] -> [B, enc_T] # squeeze blank dim

    def forward(self, attention_RNN_state, attention_weights_cat, memory, mask, attention_weights=None):
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
                attention_RNN_state, attention_weights_cat) # outputs [B, enc_T]
            
            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)
            
            attention_weights = F.softmax(alignment, dim=1) # softmax
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # unsqueeze, bmm
        attention_context = attention_context.squeeze(1) # squeeze
        
        return attention_context, attention_weights


class GMMAttention(nn.Module): # Experimental from NTT123
    def __init__(self, num_mixtures, attention_layers, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size, hparams):
        super(GMMAttention, self).__init__()
        self.num_mixtures = num_mixtures
        self.normalize_attention_input = hparams.normalize_attention_input
        self.delta_min_limit = hparams.delta_min_limit
        self.delta_offset = hparams.delta_offset
        self.lin_bias = hparams.lin_bias
        self.initial_gain = hparams.initial_gain
        lin = nn.Linear(attention_dim, 3*num_mixtures, bias=self.lin_bias)
        lin.weight.data.mul_(0.01)
        if self.lin_bias:
            lin.bias.data.mul_(0.008)
            lin.bias.data.sub_(2.0)
        
        if attention_layers == 1:
            self.F = nn.Sequential(
                    LinearNorm(attention_rnn_dim, attention_dim, bias=True, w_init_gain=self.initial_gain),
                    nn.Tanh(),
                    lin)
        elif attention_layers == 2:
            self.F = nn.Sequential(
                    LinearNorm(attention_rnn_dim, attention_dim, bias=True, w_init_gain=self.initial_gain),
                    LinearNorm(attention_dim, attention_dim, bias=False, w_init_gain='tanh'),
                    nn.Tanh(),
                    lin)
        else:
            print(f"attention_layers invalid, valid values are... 1, 2\nCurrent Value {attention_layers}")
            raise
        
        self.score_mask_value = 0 # -float("inf")
        self.register_buffer('pos', torch.arange(
            0, 2000, dtype=torch.float).view(1, -1, 1).data)
    
    
    def get_alignment_energies(self, attention_hidden_state, memory, previous_location):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        memory: encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        
        RETURNS
        -------
        alignment (batch, max_time)
        """
        if self.normalize_attention_input:
            attention_hidden_state = attention_hidden_state.tanh()
        w, delta, scale = self.F(attention_hidden_state.unsqueeze(1)).chunk(3, dim=-1)
        delta = delta.sigmoid()#*1.0 # normalize 0.0 - 1.0,
        if self.delta_min_limit: delta = delta.clamp(min=self.delta_min_limit) # supposed to be fine with autograd but not 100% confident.
        if self.delta_offset: delta = delta + self.delta_offset
        loc = previous_location + delta
        scale = scale.sigmoid() * 2 + 1
        
        if False: # I don't know anything about this but both versions exist
            pos = self.pos[:, :memory.shape[1], :]
            z1 = torch.erf((loc-pos+0.5)*scale)
            z2 = torch.erf((loc-pos-0.5)*scale)
            z = (z1 - z2)*0.5
            w = torch.sigmoid(w) #w = torch.softmax(w, dim=-1) # not sure which to use
        else:
            std = torch.nn.functional.softplus(scale + 5) + 1e-5
            pos = self.pos[:, :memory.shape[1], :]
            z1 = torch.tanh((loc-pos+0.5) / std)
            z2 = torch.tanh((loc-pos-0.5) / std)
            z = (z1 - z2)*0.5
            w = torch.softmax(w, dim=-1) + 1e-5
        
        z = torch.bmm(z, w.squeeze(1).unsqueeze(2)).squeeze(-1)
        # z = z.sum(dim=-1)
        return z, loc
    
    def forward(self, attention_hidden_state, memory, previous_location, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment, loc = self.get_alignment_energies(attention_hidden_state, memory, previous_location)
        
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        
        #attention_weights = alignment # without softmax
        attention_weights = F.softmax(alignment, dim=1) # with softmax
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights, loc


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
            #text = F.dropout(F.relu(conv(text)), drop_rate, self.training) # Normal ReLU
            text = F.dropout(self.LReLU(conv(text)), drop_rate, self.training) # LeakyReLU
        
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
            #text = F.dropout(F.relu(conv(text)), drop_rate, self.training) # Normal ReLU
            text = F.dropout(self.LReLU(conv(text)), drop_rate, self.training) # LeakyReLU
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        text = text.transpose(1, 2) # [B, embed, sequence] -> [B, sequence, embed]
        
        if text_lengths is not None:
            #text *= get_mask_from_lengths(text_lengths)[:, :, None]
            # pytorch tensor are not reversible, hence the conversion
            text_lengths = text_lengths.cpu().numpy()
            text = nn.utils.rnn.pack_padded_sequence(
                text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(text) # -> [B, sequence, embed]
        
        if text_lengths is not None:
            #outputs *= get_mask_from_lengths(text_lengths)[:, :, None]
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)
        
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_LSTM_dim = hparams.encoder_LSTM_dim + hparams.token_embedding_size + hparams.speaker_embedding_dim
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
        self.extra_projection = hparams.extra_projection
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
            AttRNN_Dimensions = hparams.prenet_dim + self.encoder_LSTM_dim + hparams.decoder_rnn_dim
        else:
            AttRNN_Dimensions = hparams.prenet_dim + self.encoder_LSTM_dim
        
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
                hparams.attention_rnn_dim, self.encoder_LSTM_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size)
        elif self.attention_type == 1:
            self.attention_layer = GMMAttention(
                hparams.num_att_mixtures, hparams.attention_layers,
                hparams.attention_rnn_dim, self.encoder_LSTM_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size, hparams)
        elif self.attention_type == 2:
            self.attention_layer = DynamicConvolutionAttention(
                hparams.attention_rnn_dim, self.encoder_LSTM_dim,
                hparams.attention_dim, hparams.attention_location_n_filters,
                hparams.attention_location_kernel_size,
                hparams.dynamic_filter_num, hparams.dynamic_filter_len)
        else:
            print("attention_type invalid, valid values are... 0 and 1")
            raise
        
        if self.DecRNN_hidden_dropout_type == 'dropout':
            self.decoder_rnn = nn.LSTMCell(
                hparams.attention_rnn_dim + self.encoder_LSTM_dim, # input_size
                hparams.decoder_rnn_dim, 1) # hidden_size, bias)
        elif self.DecRNN_hidden_dropout_type == 'zoneout':
            self.decoder_rnn = LSTMCellWithZoneout(
                hparams.attention_rnn_dim + self.encoder_LSTM_dim, # input_size
                hparams.decoder_rnn_dim, 1, zoneout_prob=self.p_DecRNN_hidden_dropout) # hidden_size, bias)
            self.p_DecRNN_hidden_dropout = 0.0 # zoneout assigned inside LSTMCellWithZoneout so don't need normal dropout
        
        if self.extra_projection:
            self.linear_projection_pre = LinearNorm(
                hparams.decoder_rnn_dim + self.encoder_LSTM_dim,
                hparams.decoder_rnn_dim + self.encoder_LSTM_dim)
        
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_LSTM_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)
        
        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_LSTM_dim, 1,
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
            cell_input = torch.cat((decoder_input, self.attention_context, self.decoder_hidden), -1)
        else:
            cell_input = torch.cat((decoder_input, self.attention_context), -1)
        
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
        
        self.attention_weights_cum += self.attention_weights# [B, enc]??? # cumulative weights determine how much time has been spent on each encoder_input, should let the model know what has already been said and what still needs to be spoken
        
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
        
        if self.extra_projection:
            decoder_hidden_attention_context = self.linear_projection_pre(
                decoder_hidden_attention_context)
        
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
        
        decoder_inputs = self.prenet(decoder_inputs) # some linear layers
        
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
            
            if self.attention_type == 1 and self.num_att_mixtures == 1:## stop when the attention location is out of the encoder_outputs
                if self.previous_location.squeeze().item() + 1. > memory.shape[1]:
                    break
            else:
                # once ALL batch predictions have gone over gate_threshold at least once, set break_point
                if i > 4: # model has very *interesting* starting predictions
                    sig_max_gates = torch.max(torch.sigmoid(gate_output_cpu), sig_max_gates)# sigmoid -> max
                if sig_max_gates.min() > self.gate_threshold: # min()  ( implicit item() as well )
                    break_point = min(break_point, i+self.gate_delay)
            
            if i >= break_point: # gt
                break
            
            decoder_input = mel_output
        else:
            print("Warning! Reached max decoder steps")
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        # apply modification to the GPU as well.
        gate_outputs = torch.sigmoid(gate_outputs)
        
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.token_num = hparams.token_num
        self.p_teacher_forcing = hparams.p_teacher_forcing
        self.teacher_force_till = hparams.teacher_force_till
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        self.with_gst = hparams.with_gst
        ref_mode_lookup = {
            False: 1, #torchMoji_training false, use normal training
            True: 3,} #torchMoji_training true, train the linear instead
        self.ref_mode = ref_mode_lookup[hparams.torchMoji_training and hparams.torchMoji_linear]
        
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.drop_frame_rate = hparams.drop_frame_rate
        if self.drop_frame_rate > 0.:
            # global mean is not used at inference.
            self.global_mean = getattr(hparams, 'global_mean', None)
        if self.with_gst:
            self.gst = GST(hparams)
            self.drop_tokens_mode = hparams.drop_tokens_mode
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(
                hparams.n_speakers, self.speaker_embedding_dim)
        
    def parse_batch(self, batch):
        text_padded, text_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states = batch
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
        return (
            (text_padded, text_lengths, mel_padded, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states),
            (mel_padded, gate_padded, output_lengths))
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
        text, text_lengths, gt_mels, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        
        if teacher_force_till == None: p_teacher_forcing = self.p_teacher_forcing
        if p_teacher_forcing == None: teacher_force_till = self.teacher_force_till
        if drop_frame_rate == None: drop_frame_rate = self.drop_frame_rate
        
        if drop_frame_rate > 0. and self.training:
            # gt_mels shape (B, n_mel_channels, T_out),
            gt_mels = dropout_frame(gt_mels, self.global_mean, output_lengths, drop_frame_rate)
        
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        encoder_outputs = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_ids) # [B, time, encoder_out]
        
        if self.with_gst:
            embedded_gst = self.gst(gt_mels if (torchmoji_hidden is None) else torchmoji_hidden, ref_mode=self.ref_mode) # create embedding from tokens from reference mel
            embedded_gst = embedded_gst.repeat(1, encoder_outputs.size(1), 1) # repeat token along-side the other embeddings for input to decoder
            encoder_outputs = torch.cat((encoder_outputs, embedded_gst), dim=2) # [batch, time, encoder_out]
        
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_speakers), dim=2) # [batch, time, encoder_out]
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, gt_mels, memory_lengths=text_lengths, preserve_decoder=preserve_decoder_states, teacher_force_till=teacher_force_till, p_teacher_forcing=p_teacher_forcing)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet.add_(mel_outputs)
        
        return self.mask_outputs(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)
    
    def inference(self, text, speaker_ids, style_input=None, style_mode=None, text_lengths=None):
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        encoder_outputs = self.encoder.inference(embedded_text, speaker_ids=speaker_ids, text_lengths=text_lengths) # [B, time, encoder_out]
        
        if self.with_gst:
            assert (style_input is not None) or (style_mode.lower() == 'zeros'), "style mode specified but no style_input"
            if style_mode.lower() == 'mel': # enter any 160 channel mel-spectrogram
                embedded_gst = self.gst(style_input)
            elif style_mode.lower() == 'zeros': # enter any value, will set style_tokens to 0
                weights = torch.ones(1, self.token_num).cuda()
                embedded_gst = self.gst(weights*0.0, ref_mode=0).half()
            elif style_mode.lower() == 'style_token' or  style_mode.lower() == 'token': # should input style_token length list/array
                assert len(style_input) == self.token_num
                weights = torch.FloatTensor(style_input).unsqueeze(0).cuda()
                embedded_gst = self.gst(weights, ref_mode=0).half()
            elif style_mode.lower() == 'torchmoji_hidden':
                assert type(style_input) == torch.Tensor
                embedded_gst = self.gst(style_input, ref_mode=3).half() # should input hidden_state of torchMoji as tensor
            elif style_mode.lower() == 'torchmoji_string':
                assert type(style_input) == type(list()) or type(style_input) == type('')
                if type(style_input) == type(''):
                    style_input = [style_input,]
                embedded_gst = self.gst(style_input, ref_mode=2).half() # should input text as string
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
