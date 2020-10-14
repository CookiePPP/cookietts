import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d, dropout_frame
from CookieTTS._2_ttm.GANTTS.nets.SylpsNet import SylpsNet
from CookieTTS._2_ttm.GANTTS.nets.EmotionNet import EmotionNet
from CookieTTS._2_ttm.GANTTS.nets.AuxEmotionNet import AuxEmotionNet
from CookieTTS._2_ttm.untts.fastpitch.length_predictor import TemporalPredictor
from math import sqrt


def load_model(hp):
    model = GANTTS(hp)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_model_d(hp):
    model = GANTTS_D(hp)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class ConditionalBatchNorm1d(nn.Module):
    """
    Conditional Batch Normalization
    https://github.com/yanggeng1995/GAN-TTS/blob/master/models/generator.py#L121-L144
    """
    def __init__(self, num_features, z_channels=128):
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.z_channels = z_channels
        self.batch_norm = nn.BatchNorm1d(num_features, affine=False)
        
        self.layer = nn.utils.spectral_norm(nn.Linear(z_channels, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.layer.bias.data.zero_()             # Initialise bias at 0
    
    def forward(self, inputs, noise):
        outputs = self.batch_norm(inputs)
        gamma, beta = self.layer(noise).chunk(2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)
        
        outputs = gamma * outputs + beta
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=None, dilation=1, kernel_size=3, act_func=nn.LeakyReLU(negative_slope=0.1, inplace=True), bias=True, scale:int=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.padding = int((kernel_size*dilation - dilation)/2)
        self.dilation = dilation
        self.bias = bias
        if scale != 1:
            self.scale = scale
        
        if self.z_dim is not None:
            self.bn = ConditionalBatchNorm1d(self.in_dim, self.z_dim)
        self.act_func = act_func
        
        self.conv = nn.Conv1d(self.in_dim, self.out_dim, self.kernel_size, padding=self.padding, dilation=self.dilation, bias=bias)
    
    def forward(self, x, z=None, output_lengths=None):    # [B, in_dim, T]
        if hasattr(self, 'bn'):
            x = self.bn(x, z)
        x = self.act_func(x)
        if hasattr(self, 'scale'):
            if self.downsample:
                F.avg_pool1d(x, kernel_size=self.scale)
            else:
                x = F.interpolate(x, scale_factor=self.scale, mode='linear') # [B, in_dim, T]   -> [B, in_dim, x*T]
        if output_lengths is not None:
            scale_factor = x.shape[2]/output_lengths.sum().max()
            if scale_factor != 1.0:
                output_lengths = (output_lengths.float()*(scale_factor)).long()
            mask = get_mask_from_lengths(output_lengths).unsqueeze(1)
            x.masked_fill_(~mask, 0.0)
        x = self.conv(x)                                  # [B, in_dim, x*T] -> [B, out_dim, x*T]
        return x                                          # [B, out_dim, x*T]


class GBlock(nn.Module):
    def __init__(self, input_dim, output_dim, z_dim, kernel_size=3, dilations=[1,2,4,8], scale:int=1, upsample_block_id=0, res_block_id=1):
        super(GBlock, self).__init__()
        self.resblocks = nn.ModuleList()
        self.upsample_block_id = upsample_block_id
        self.res_block_id = res_block_id
        self.scale = scale
        
        for i, dilation in enumerate(dilations):
            in_dim = input_dim if i == 0 else output_dim
            dilation = dilations[i]
            scale_f = scale if i == self.upsample_block_id else int(1)
            resblock = ResidualBlock(in_dim, output_dim, z_dim=z_dim, dilation=dilation, kernel_size=kernel_size, scale=scale_f)
            self.resblocks.append(resblock)
        
        self.skip_conv = nn.Conv1d(input_dim, output_dim, 1)
    
    def forward(self, h, z, output_lengths=None):
        scaled_h = F.interpolate(h, scale_factor=self.scale, mode='linear') if self.scale != 1 else h# [B, input_dim, T] -> [B, input_dim, x*T]
        if output_lengths is not None:
            scale_factor = scaled_h.shape[2]/output_lengths.sum().max()
            if scale_factor != 1.0:
                output_lengths = (output_lengths.float()*(scale_factor)).long()
            mask = get_mask_from_lengths(output_lengths).unsqueeze(1)
            scaled_h.masked_fill_(~mask, 0.0)
        residual = self.skip_conv(scaled_h)# [B, input_dim, x*T] -> [B, output_dim, x*T]
        
        for i, resblock in enumerate(self.resblocks): # [B, input_dim, T] -> [B, output_dim, x*T]
            h = resblock(h, z, output_lengths)
            if i == self.res_block_id:
                h += residual
                residiual = h
        
        return h + residual # [B, output_dim, x*T]


class DBlock(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim=0, kernel_size=3, dilations=[1,2,4,8], scale:int=1, cond_block_id=0):
        super(DBlock, self).__init__()
        self.resblocks = nn.ModuleList()
        self.cond_block_id = cond_block_id
        self.scale = scale
        
        for i, dilation in enumerate(dilations):
            in_dim = input_dim if i == 0 else (output_dim*2 if cond_dim and i == self.cond_block_id+1 else output_dim)
            dilation = dilations[i]
            out_dim = output_dim*2 if cond_dim and i == self.cond_block_id else output_dim
            resblock = ResidualBlock(in_dim, out_dim, dilation=dilation, kernel_size=kernel_size, downsample=True)
            self.resblocks.append(resblock)
        
        self.skip_conv = nn.Conv1d(input_dim, output_dim, 1)
        if cond_dim:
            self.cond_conv = nn.Conv1d(cond_dim , 2*output_dim, 1)
    
    def forward(self, x, cond):
        residual = self.skip_conv(x)# [B, input_dim, T] -> [B, output_dim, T]
        residual = F.avg_pool1d(residual, kernel_size=self.scale) if self.scale != 1 else residual# [B, output_dim, T] -> [B, output_dim, T/x]
        
        x = F.avg_pool1d(x, kernel_size=self.scale) if self.scale != 1 else x# [B, input_dim, T] -> [B, input_dim, x*T]
        for i, resblock in enumerate(self.resblocks): # [B, input_dim, T] -> [B, output_dim, x*T]
            x = resblock(x.clone())
            if i == self.cond_block_id and hasattr(self, 'cond_conv'):
                cond = self.cond_conv(cond)# [B, cond_dim, T] -> [B, output_dim, T]
                if cond.shape[2] != x.shape[2]:
                    cond = F.interpolate(cond, size=x.shape[2], mode='linear')
                x = x + cond
        
        return x+residual


class GANTTS_Descriminator(nn.Module):
    def __init__(self, in_channels, encoder_dims, cond_dim=0, use_cond=[0, 0, 0, 0, 0, 1, 1, 1], scales=[5, 3, 2, 2, 2, 1, 1, 1], kernel_size=3, dilations=[1,2]):
        super(GANTTS_Descriminator, self).__init__()
        self.in_channels = in_channels
        
        self.Dblocks = nn.ModuleList([
            DBlock(in_channels, encoder_dims[0], cond_dim=cond_dim if use_cond[0] else 0, kernel_size=kernel_size, dilations=dilations, scale=scales[0]),
        ])
        for i, dim in enumerate(encoder_dims[:-1]):
            in_dim = encoder_dims[i]
            out_dim = encoder_dims[i+1]
            scale = scales[i+1]
            dblock = DBlock(in_dim, out_dim, cond_dim=cond_dim if use_cond[i+1] else 0, kernel_size=kernel_size, dilations=dilations, scale=scale)
            self.Dblocks.append(dblock)
    
    def forward(self, x, cond):
        B, _, T = x.shape
        x = x.view(B, self.in_channels, -1)# [B, 1, T] -> [B, in_dim, T//in_dim]
        
        for dblock in self.Dblocks:
            x = dblock(x, cond)
        
        return x.mean(dim=2, keepdim=True) # [B, output_dim, 1]


class GANTTS_D(nn.Module):
    def __init__(self, hp):
        super(GANTTS_D, self).__init__()
        self.base_window_size = hp.descriminator_base_window
        self.descriminators = nn.ModuleList()
        cat_dims = []
        
        for i, (in_channels, encoder_dims, use_cond, scales) in enumerate(hp.descriminator_configs):
            descriminator = GANTTS_Descriminator(in_channels, encoder_dims, hp.memory_bottleneck_dim, use_cond, scales, dilations=hp.d_dilations)
            self.descriminators.append(descriminator)
            cat_dims.append(encoder_dims[-1])
        
        cat_dims = sum(cat_dims)
        self.end = nn.Conv1d(cat_dims, 1, 1)
    
    def forward(self, x, cond):
        cond = cond.transpose(1, 2) # [B, enc_T, dim] -> [B, dim, enc_T]
        pred_fakeness = []
        for descriminator in self.descriminators:
            k = descriminator.in_channels
            w_size = k * self.base_window_size
            max_start = x.shape[1]-w_size
            start = random.randint(0, max_start)
            x_window = x[:, start:start+w_size]# [B, T] -> [B, k*base_window]
            _ = descriminator(x_window.unsqueeze(1), cond)# [B, k*base_window] -> [B, dim]
            pred_fakeness.append(_)
        pred_fakeness = torch.cat(pred_fakeness, dim=1)# [[B, dim, 1], ...] -> [B, 10*dim]
        pred_fakeness = self.end(pred_fakeness).sigmoid() # [B, 10*dim, 1] -> [B, 1, 1]
        return pred_fakeness[:, 0, 0] # [B, 1, 1] -> [B]


class Decoder(nn.Module):
    def __init__(self, hp):
        super(Decoder, self).__init__()
        self.start = nn.Conv1d(hp.in_channels, hp.decoder_dims[0], kernel_size=3)
        
        self.Gblocks = nn.ModuleList([
            GBlock(hp.decoder_dims[0], hp.decoder_dims[0], hp.z_dim, kernel_size=hp.gblock_kernel_size, dilations=hp.dilations, scale=hp.decoder_scales[0]),
        ])
        for i, dim in enumerate(hp.decoder_dims[:-1]):
            in_dim = hp.decoder_dims[i]
            out_dim = hp.decoder_dims[i+1]
            scale = hp.decoder_scales[i+1]
            gblock = GBlock(in_dim, out_dim, hp.z_dim, kernel_size=hp.gblock_kernel_size, dilations=hp.dilations, scale=scale)
            self.Gblocks.append(gblock)
        
        self.end = nn.Conv1d(hp.decoder_dims[-1], 1, kernel_size=3)
    
    def forward(self, x, z, output_lengths=None):
        x = self.start(x)# [B, in_dim, T//scale_factors] -> [B, self.decoder_dims[0], T//scale_factors]
        if output_lengths is not None:
            mask = get_mask_from_lengths(output_lengths).unsqueeze(1)
            x.masked_fill_(~mask, 0.0)
        
        for gblock in self.Gblocks:
            x = gblock(x, z, output_lengths=output_lengths)
        
        if output_lengths is not None:
            scale_factor = x.shape[2]/output_lengths.sum().max()
            if scale_factor != 1.0:
                output_lengths = (output_lengths.float()*(scale_factor)).long()
            mask = ~get_mask_from_lengths(output_lengths).unsqueeze(1)
            x.masked_fill_(mask, 0.0)
        x = self.end(x)# [B, 1, T]
        
        x = x.tanh()
        if output_lengths is not None:
            x.masked_fill_(mask, 0.0)
        
        return x # [B, 1, T]


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hp):
        super(Encoder, self).__init__() 
        self.encoder_speaker_embed_dim = hp.encoder_speaker_embed_dim
        if self.encoder_speaker_embed_dim:
            self.encoder_speaker_embedding = nn.Embedding(
            hp.n_speakers, self.encoder_speaker_embed_dim)
        
        self.encoder_concat_speaker_embed = hp.encoder_concat_speaker_embed
        self.encoder_conv_hidden_dim = hp.encoder_conv_hidden_dim
        
        convolutions = []
        for _ in range(hp.encoder_n_convolutions):
            if _ == 0:
                if self.encoder_concat_speaker_embed == 'before_conv':
                    input_dim = hp.symbols_embedding_dim+self.encoder_speaker_embed_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    input_dim = hp.symbols_embedding_dim
                else:
                    raise NotImplementedError(f'encoder_concat_speaker_embed is has invalid value {hp.encoder_concat_speaker_embed}, valid values are "before","inside".')
            else:
                input_dim = self.encoder_conv_hidden_dim
            
            if _ == (hp.encoder_n_convolutions)-1: # last conv
                if self.encoder_concat_speaker_embed == 'before_conv':
                    output_dim = hp.encoder_LSTM_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    output_dim = hp.encoder_LSTM_dim-self.encoder_speaker_embed_dim
            else:
                output_dim = self.encoder_conv_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(input_dim,
                         output_dim,
                         kernel_size=hp.encoder_kernel_size, stride=1,
                         padding=int((hp.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(output_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(hp.encoder_LSTM_dim,
                            int(hp.encoder_LSTM_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        self.LReLU = nn.LeakyReLU(negative_slope=0.01) # LeakyReLU
        
        self.sylps_layer = LinearNorm(hp.encoder_LSTM_dim, 1)
    
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
    def __init__(self, hp):
        super(MemoryBottleneck, self).__init__()
        self.mem_output_dim = hp.memory_bottleneck_dim
        self.mem_input_dim = hp.encoder_LSTM_dim + hp.speaker_embedding_dim + len(hp.emotion_classes) + hp.emotionnet_latent_dim + 1
        self.bottleneck = LinearNorm(self.mem_input_dim, self.mem_output_dim, bias=hp.memory_bottleneck_bias, w_init_gain='tanh')
    
    def forward(self, memory):
        memory = self.bottleneck(memory)# [B, enc_T, input_dim] -> [B, enc_T, output_dim]
        return memory


class GANTTS(nn.Module):
    def __init__(self, hp):
        super(GANTTS, self).__init__()
        self.embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        std = sqrt(2.0 / (hp.n_symbols + hp.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        self.speaker_embedding_dim = hp.speaker_embedding_dim
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(hp.n_speakers, self.speaker_embedding_dim)
        
        self.encoder = Encoder(hp) # Text -> Encoder Features
        self.durpred = TemporalPredictor(hp.memory_bottleneck_dim, hp) # Encoder Features -> Durations
        self.sylps_net = SylpsNet(hp) # Text -> Sylps
        self.emotion_net = EmotionNet(hp) # Spect -> Unsupervised Latent
        self.aux_emotion_net = AuxEmotionNet(hp) # Text -> Unsupervised Latent
        self.decoder = Decoder(hp) # Encoder Features + Durations + Sylps + Unsupervised Latent -> Audio
        self.z_dim = hp.z_dim
    
    def parse_batch(self, batch):
        audio, attention_contexts, encoder_outputs, text_lengths, durations = batch
        audio = to_gpu(audio).float()
        attention_contexts = to_gpu(attention_contexts).float()
        encoder_outputs = to_gpu(encoder_outputs).float()
        text_lengths = to_gpu(text_lengths).long()
        durations = to_gpu(durations).long()
        return (audio, attention_contexts, encoder_outputs, text_lengths, durations)
    
    #@torch.jit.script
    def parse_encoder_outputs(self, encoder_outputs, durations, output_lengths, text_lengths):
        """
        Acts as Monotonic Attention for Encoder Outputs.
        
        [B, enc_T, enc_dim] x [B, enc_T, durations] -> [B, dec_T, enc_dim]
        """
        B, enc_T, enc_dim = encoder_outputs.shape# [Batch Size, Text Length, Encoder Dimension]
        dec_T = output_lengths.max().item()# Length of Features
        
        start_pos = torch.zeros(B, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B]
        attention_pos = torch.arange(dec_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype).expand(B, dec_T)# [B, dec_T, enc_T]
        attention = torch.zeros(B, dec_T, enc_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B, dec_T, enc_T]
        for enc_inx in range(durations.shape[1]):
            dur = durations[:, enc_inx]# [B]
            end_pos = start_pos + dur# [B]
            if text_lengths is not None: # if last char, extend till end of decoder sequence
                mask = (text_lengths == (enc_inx+1))# [B]
                if mask.any():
                    end_pos.masked_fill_(mask, dec_T)
            
            att = (attention_pos>=start_pos.unsqueeze(-1).repeat(1, dec_T)) & (attention_pos<end_pos.unsqueeze(-1).repeat(1, dec_T))
            attention[:, :, enc_inx][att] = 1.# set predicted duration values to positive
            
            start_pos = start_pos + dur # [B]
        if text_lengths is not None:
            attention = attention * get_mask_3d(output_lengths, text_lengths)
        return attention.matmul(encoder_outputs)
    
    @torch.jit.script
    def generate_noise(x, z_dim:int):
        noise = torch.randn(x.shape[0], z_dim, device=x.device, dtype=x.dtype)
        return noise
    
    def forward(self, inputs):
        audio, attention_contexts, encoder_outputs, text_lengths, _ = inputs
        attention_contexts = attention_contexts.transpose(1, 2) # [B, enc_T, dim] -> [B, dim, enc_T]
        
        noise = self.generate_noise(encoder_outputs, self.z_dim)
        
        pred_audio = self.decoder(attention_contexts, noise).squeeze(1)# [B, 1, T] -> [B, T]
        
        pred_durations = self.durpred(encoder_outputs)
        
        return pred_audio, pred_durations
    
    def inference(self, text, speaker_ids, style_input=None, style_mode=None, text_lengths=None):
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        encoder_outputs = self.encoder(embedded_text, speaker_ids=speaker_ids, text_lengths=text_lengths) # [B, time, encoder_out]
        
        encoder_outputs = self.bottleneck(encoder_outputs, memory_lengths=text_lengths)
        
        pred_durations = self.durpred(encoder_outputs, memory_lengths=text_lengths)
        output_lengths = pred_durations.sum(1)# [B, enc_T] -> [B]
        
        attention_contexts = self.parse_encoder_outputs(encoder_outputs, pred_durations, output_lengths, text_lengths)
        
        pred_audio = self.decoder(attention_contexts, noise, output_lengths=output_lengths)
        
        return pred_audio