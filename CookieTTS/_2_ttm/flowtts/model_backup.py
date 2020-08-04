from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d
from CookieTTS._2_ttm.flowtts.fastpitch.length_predictor import TemporalPredictor
from CookieTTS._2_ttm.flowtts.fastpitch.transformer import PositionalEmbedding
from CookieTTS._2_ttm.flowtts.waveglow.glow import FlowDecoder

drop_rate = 0.5

def load_model(hparams):
    model = FlowTTS(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / sqrt(dk)# [B*n_head, dec_T, enc_dim//n_head] @ [B*n_head, enc_T, enc_dim//n_head].t() -> [B*n_head, dec_T, enc_T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -65500.0)# [B*n_head, dec_T, enc_T]
        attention = F.softmax(scores, dim=-1) # softmax along enc dim
        return attention.matmul(value), attention# [B*n_head, dec_T, enc_T] @ [B*n_head, enc_T, enc_dim//n_head] -> [B*n_head, dec_T, enc_dim//n_head]

# https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
    
    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)# [B, dec_T, enc_dim], [B, enc_T, enc_dim], [B, enc_T, enc_dim]
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        
        q = self._reshape_to_batches(q)# [B, dec_T, enc_dim] -> [B*n_head, dec_T, enc_dim//n_head]
        k = self._reshape_to_batches(k)# [B, enc_T, enc_dim] -> [B*n_head, enc_T, enc_dim//n_head]
        v = self._reshape_to_batches(v)# [B, enc_T, enc_dim] -> [B*n_head, enc_T, enc_dim//n_head]
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)# [B, dec_T, enc_T] -> [B*n_head, dec_T, enc_T]
        y, attention_scores = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)# [B*n_head, dec_T, enc_dim//n_head] -> [B, dec_T, enc_dim]
        
        att_shape = attention_scores.shape
        attention_scores = attention_scores.view(att_shape[0]//self.head_num, self.head_num, *att_shape[1:])
        # [B*n_head, dec_T, enc_T] -> [B, n_head, dec_T, enc_T]
        
        y = self.linear_o(y)# [B, dec_T, enc_dim]
        if self.activation is not None:
            y = self.activation(y)
        return y, attention_scores# [B, dec_T, enc_dim], [B, n_head, dec_T, enc_T]
    
    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)
    
    def _reshape_to_batches(self, x):# [B, enc_T, enc_dim] -> [B*n_head, enc_T, enc_dim//n_head]
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)
    
    def _reshape_from_batches(self, x):# [B*n_head, enc_T, enc_dim//n_head] -> [B, enc_T, enc_dim]
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)
    
    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class PositionalAttention(nn.Module):
    def __init__(self, hparams):
        super(PositionalAttention, self).__init__()
        self.head_num = hparams.pos_att_head_num
        self.merged_pos_enc = True
        self.pos_enc_dim = hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim
        self.pos_enc_dim = self.pos_enc_dim if self.merged_pos_enc else self.pos_enc_dim/hparams.pos_att_head_num
        if False:
            self.positional_embedding = PositionalEmbedding(self.pos_enc_dim, inv_freq=hparams.pos_att_inv_freq)
            self.multi_head_attention = MultiHeadAttention(hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim, hparams.pos_att_head_num)
            self.pytorch_native_mha = False
        else:
            self.positional_embedding = PositionalEmbedding(self.pos_enc_dim, inv_freq=hparams.pos_att_inv_freq)
            self.multi_head_attention = torch.nn.MultiheadAttention(hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim,
                                                                    hparams.pos_att_head_num, dropout=0.1, bias=True,
                                                                    add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
            self.pytorch_native_mha = True
        
        self.pos_enc_k = hparams.pos_att_positional_encoding_for_key
        self.pos_enc_v = hparams.pos_att_positional_encoding_for_value
        if self.pos_enc_k or self.pos_enc_v:
            self.enc_positional_embedding = PositionalEmbedding(self.pos_enc_dim, inv_freq=hparams.pos_att_enc_inv_freq)
        
        if hparams.pos_enc_positional_embedding_kv: # learned positional encoding
            self.pos_embedding_kv_max = 400
            self.pos_embedding_kv = nn.Embedding(self.pos_embedding_kv_max, self.pos_enc_dim)
        if hparams.pos_enc_positional_embedding_q: # learned positional encoding
            self.pos_embedding_q_max = 18000
            self.pos_embedding_q = nn.Embedding(self.pos_embedding_q_max, self.pos_enc_dim)
        
        self.o_residual_weights = nn.Parameter(torch.ones(1)*0.02)
        
        n_self_att_layers = 3
        self.self_att_o_rws = nn.Parameter(torch.ones(n_self_att_layers)*0.02)
        self.self_attention_layers = nn.ModuleList()
        for i in range(n_self_att_layers):
            self_att_layer = torch.nn.MultiheadAttention(hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim,
                                                        hparams.pos_att_head_num, dropout=0.1, bias=True,
                                                        add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
            self.self_attention_layers.append(self_att_layer)
        
    def forward(self, cond_inp, output_lengths, cond_lens=None):# [B, seq_len, dim], int, [B]
        batch_size, enc_T, enc_dim = cond_inp.shape
        
        # get Random Position Offset (this *might* allow better distance generalisation)
        #trandint = torch.randint(10000, (1,), device=cond_inp.device, dtype=cond_inp.dtype)
        
        # get Query from Positional Encoding
        dec_T_max = output_lengths.max().item()
        dec_pos_emb = torch.arange(0, dec_T_max, device=cond_inp.device, dtype=cond_inp.dtype)# + trandint        
        if hasattr(self, 'pos_embedding_q'):
            dec_pos_emb = self.pos_embedding_q(dec_pos_emb.clamp(0, self.pos_embedding_q_max-1).long())[None, ...].repeat(cond_inp.size(0), 1, 1)# [B, enc_T, enc_dim]
        elif hasattr(self, 'positional_embedding'):
            dec_pos_emb = self.positional_embedding(dec_pos_emb, bsz=cond_inp.size(0))# [B, dec_T, enc_dim]
        if not self.merged_pos_enc:
            dec_pos_emb = dec_pos_emb.repeat(1, 1, self.head_num)
        if output_lengths is not None:# masking for batches
            dec_mask = get_mask_from_lengths(output_lengths).unsqueeze(2)# [B, dec_T, 1]
            dec_pos_emb = dec_pos_emb * dec_mask# [B, dec_T, enc_dim] * [B, dec_T, 1] -> [B, dec_T, enc_dim]
        q = dec_pos_emb# [B, dec_T, enc_dim]
        
        # get Key/Value from Encoder Outputs
        k = v = cond_inp# [B, enc_T, enc_dim]
        # (optional) add position encoding to Encoder outputs
        if hasattr(self, 'enc_positional_embedding'):
            enc_pos_emb = torch.arange(0, enc_T, device=cond_inp.device, dtype=cond_inp.dtype)# + trandint
            if hasattr(self, 'pos_embedding_kv'):
                enc_pos_emb = self.pos_embedding_kv(enc_pos_emb.clamp(0, self.pos_embedding_kv_max-1).long())[None, ...].repeat(cond_inp.size(0), 1, 1)# [B, enc_T, enc_dim]
            elif hasattr(self, 'enc_positional_embedding'):
                enc_pos_emb = self.enc_positional_embedding(enc_pos_emb, bsz=cond_inp.size(0))# [B, enc_T, enc_dim]
            if self.pos_enc_k:
                k = k + enc_pos_emb
            if self.pos_enc_v:
                v = v + enc_pos_emb
        enc_mask = get_mask_from_lengths(cond_lens).unsqueeze(1).repeat(1, q.size(1), 1) if (cond_lens is not None) else None# [B, dec_T, enc_T]
        
        if not self.pytorch_native_mha:
            output, attention_scores = self.multi_head_attention(q, k, v, mask=enc_mask)# [B, dec_T, enc_dim], [B, n_head, dec_T, enc_T]
        else:
            q = q.transpose(0, 1)# [B, dec_T, enc_dim] -> [dec_T, B, enc_dim]
            k = k.transpose(0, 1)# [B, enc_T, enc_dim] -> [enc_T, B, enc_dim]
            v = v.transpose(0, 1)# [B, enc_T, enc_dim] -> [enc_T, B, enc_dim]
            
            enc_mask = ~enc_mask[:, 0, :] if (cond_lens is not None) else None# [B, dec_T, enc_T] -> # [B, enc_T]
            attn_mask = ~get_mask_3d(output_lengths, cond_lens).repeat_interleave(self.head_num, 0) if (cond_lens is not None) else None#[B*n_head, dec_T, enc_T]
            attn_mask = attn_mask.float() * -35500.0 if (cond_lens is not None) else None
            
            output, attention_scores = self.multi_head_attention(q, k, v, key_padding_mask=enc_mask, attn_mask=attn_mask)# [dec_T, B, enc_dim], [B, dec_T, enc_T]
            
            output = output.transpose(0, 1)# [dec_T, B, enc_dim] -> [B, dec_T, enc_dim]
            output = output + self.o_residual_weights * dec_pos_emb
            attention_scores = attention_scores*get_mask_3d(output_lengths, cond_lens) if (cond_lens is not None) else attention_scores
            #attention_scores # [B, dec_T, enc_T]
            
            for self_att_layer, residual_weight in zip(self.self_attention_layers, self.self_att_o_rws):
                q = output.transpose(0, 1)# [B, dec_T, enc_dim] -> [dec_T, B, enc_dim]
                
                output, att_sc = self_att_layer(q, k, v, key_padding_mask=enc_mask, attn_mask=attn_mask)# ..., [dec_T, B, enc_dim], [B, dec_T, enc_T])
                
                output = output.transpose(0, 1)# [dec_T, B, enc_dim] -> [B, dec_T, enc_dim]
                output = output * residual_weight + q.transpose(0, 1)# ([B, dec_T, enc_dim] * rw) + [B, dec_T, enc_dim]
                att_sc = att_sc*get_mask_3d(output_lengths, cond_lens) if (cond_lens is not None) else att_sc
                attention_scores = attention_scores + att_sc
            
            attention_scores = attention_scores / (1+len(self.self_att_o_rws))
        
        if output_lengths is not None:
            output = output * dec_mask# [B, dec_T, enc_dim] * [B, dec_T, 1]
        return output, attention_scores


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
    
    def forward(self, text, text_lengths, speaker_ids=None, enc_drop_rate=0.2):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.repeat(1, 1, text.size(2)) # extend across all encoder steps
            if self.encoder_concat_speaker_embed == 'before_conv':
                text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            #text = F.dropout(F.relu(conv(text)), enc_drop_rate, self.training) # Normal ReLU
            text = F.dropout(self.LReLU(conv(text)), enc_drop_rate, self.training) # LeakyReLU
        
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
        self.glow = FlowDecoder(hparams)
    
    def forward(self, gt_mels, cond):
        """ Decoder forward pass for training
        PARAMS
        ------
        gt_mels: Decoder inputs i.e. mel-specs
        cond: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        log_det: log deterimates of Affine Coupling + InvertibleConv
        """
        mel_outputs, log_s_sum, logdet_w_sum = self.glow(gt_mels, cond)
        return mel_outputs, log_s_sum, logdet_w_sum

    def infer(self, cond, speaker_ids=None, sigma=None):
        """ Decoder inference
        PARAMS
        ------
        cond: Encoder outputs
        
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        """
        mel_outputs = self.glow.infer(cond, sigma=sigma)
        return mel_outputs


class FlowTTS(nn.Module):
    def __init__(self, hparams):
        super(FlowTTS, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.length_predictor = TemporalPredictor(hparams)
        self.positional_attention = PositionalAttention(hparams)
        self.decoder = Decoder(hparams)
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
            (mel_padded, gate_padded, output_lengths, text_lengths))
            # returns ((x),(y)) as (x) for training input, (y) for ground truth/loss calc
    
    def mask_outputs(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            # [B, n_mel, steps]
            outputs[0].data.masked_fill_(mask, 0.0) # [B, n_mel, T]
        return outputs
    
    def forward(self, inputs):
        text, text_lengths, gt_mels, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        
        assert not torch.isnan(text).any(), 'text has NaN values.'
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        assert not torch.isnan(embedded_text).any(), 'embedded_text has NaN values.'
        encoder_outputs = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_ids) # [B, enc_T, enc_dim]
        assert not torch.isnan(encoder_outputs).any(), 'encoder_outputs has NaN values.'
        
        # predict length of each input
        enc_out_mask = get_mask_from_lengths(text_lengths).unsqueeze(-1)# [B, enc_T, 1]
        encoder_lengths = self.length_predictor(encoder_outputs, enc_out_mask)# [B, enc_T, enc_dim]
        assert not torch.isnan(encoder_lengths).any(), 'encoder_lengths has NaN values.'
        
        # sum lengths (used to predict mel-spec length)
        encoder_lengths = encoder_lengths.clamp(1e-6, 4096)
        pred_output_lengths = encoder_lengths.sum((1,))
        assert not torch.isnan(encoder_lengths).any(), 'encoder_lengths has NaN values.'
        assert not torch.isnan(pred_output_lengths).any(), 'pred_output_lengths has NaN values.'
        
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_speakers), dim=2) # [batch, enc_T, enc_dim]
        
        # Positional Attention
        cond, attention_scores = self.positional_attention(encoder_outputs, output_lengths, cond_lens=text_lengths)
        cond = cond.transpose(1, 2)
        assert not torch.isnan(cond).any(), 'cond has NaN values.'
        # [B, enc_T, enc_dim] -> [B, enc_dim, dec_T] # Masked Multi-head Attention
        
        # Decoder
        mel_outputs, log_s_sum, logdet_w_sum = self.decoder(gt_mels, cond) # [B, n_mel, dec_T], [B, dec_T, enc_dim] -> [B, n_mel, dec_T], [B] # Series of Flows
        assert not torch.isnan(mel_outputs).any(), 'mel_outputs has NaN values.'
        assert not torch.isnan(log_s_sum).any(), 'mel_outputs has NaN values.'
        assert not torch.isnan(logdet_w_sum).any(), 'mel_outputs has NaN values.'
        
        return self.mask_outputs(
            [mel_outputs, attention_scores, pred_output_lengths, log_s_sum, logdet_w_sum],
            output_lengths)
    
    def inference(self, text, speaker_ids, text_lengths=None, sigma=1.0):
        assert not torch.isnan(text).any(), 'text has NaN values.'
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        assert not torch.isnan(embedded_text).any(), 'encoder_outputs has NaN values.'
        encoder_outputs = self.encoder.inference(embedded_text, speaker_ids=speaker_ids) # [B, enc_T, enc_dim]
        assert not torch.isnan(encoder_outputs).any(), 'encoder_outputs has NaN values.'
        
        # predict length of each input
        enc_out_mask = get_mask_from_lengths(text_lengths) if (text_lengths is not None) else None
        encoder_lengths = self.length_predictor(encoder_outputs, enc_out_mask)
        assert not torch.isnan(encoder_lengths).any(), 'encoder_lengths has NaN values.'
        
        # sum lengths (used to predict mel-spec length)
        encoder_lengths = encoder_lengths.clamp(1, 128)
        pred_output_lengths = encoder_lengths.sum((1,)).long()
        assert not torch.isnan(encoder_lengths).any(), 'encoder_lengths has NaN values.'
        assert not torch.isnan(pred_output_lengths).any(), 'pred_output_lengths has NaN values.'
        
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_speakers), dim=2) # [batch, enc_T, enc_dim]
        
        # Positional Attention
        cond, attention_scores = self.positional_attention(encoder_outputs, pred_output_lengths, cond_lens=text_lengths)
        cond = cond.transpose(1, 2)
        assert not torch.isnan(cond).any(), 'cond has NaN values.'
        # [B, enc_T, enc_dim] -> [B, enc_dim, dec_T] # Masked Multi-head Attention
        
        # Decoder
        mel_outputs = self.decoder.infer(cond, sigma=sigma) # [B, dec_T, emb] -> [B, n_mel, dec_T] # Series of Flows
        assert not torch.isnan(mel_outputs).any(), 'mel_outputs has NaN values.'
        
        return self.mask_outputs(
            [mel_outputs, attention_scores, None, None, None])
