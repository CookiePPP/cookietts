from math import sqrt
import random
import numpy as np
from numpy import finfo

from torch.autograd import Variable
from torch import Tensor
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d
from CookieTTS._2_ttm.untts.fastpitch.length_predictor import TemporalPredictor
from CookieTTS._2_ttm.untts.fastpitch.transformer import PositionalEmbedding
from CookieTTS._2_ttm.untts.waveglow.glow import FlowDecoder
from CookieTTS._2_ttm.untts.waveglow.durglow import DurationGlow
from CookieTTS._2_ttm.untts.waveglow.varglow import VarGlow

drop_rate = 0.5

def load_model(hparams):
    model = UnTTS(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class LenPredictorAttention(nn.Module):
    def __init__(self):
        super(LenPredictorAttention, self).__init__()
    
    def forward(self, encoder_outputs, encoder_lengths, output_lengths, cond_lens=None, attention_override=None):
        if attention_override is None:
            B, enc_T, enc_dim = encoder_outputs.shape# [Batch Size, Text Length, Encoder Dimension]
            dec_T = output_lengths.max().item()# Length of Spectrogram
            
            #encoder_lengths = encoder_lengths# [B, enc_T]
            #encoder_outputs = encoder_outputs# [B, enc_T, enc_dim]
            
            start_pos = torch.zeros(B, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B]
            attention_pos = torch.arange(dec_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype).expand(B, dec_T)# [B, dec_T, enc_T]
            attention = torch.zeros(B, dec_T, enc_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B, dec_T, enc_T]
            for enc_inx in range(encoder_lengths.shape[1]):
                dur = encoder_lengths[:, enc_inx]# [B]
                end_pos = start_pos + dur# [B]
                if cond_lens is not None: # if last char, extend till end of decoder sequence
                    mask = (cond_lens == (enc_inx+1))# [B]
                    if mask.any():
                        end_pos.masked_fill_(mask, dec_T)
                
                att = (attention_pos>=start_pos.unsqueeze(-1).repeat(1, dec_T)) & (attention_pos<end_pos.unsqueeze(-1).repeat(1, dec_T))
                attention[:, :, enc_inx][att] = 1.# set predicted duration values to positive
                
                start_pos = start_pos + dur # [B]
            if cond_lens is not None:
                attention = attention * get_mask_3d(output_lengths, cond_lens)
        else:
            attention = attention_override
        return attention.matmul(encoder_outputs)# [B, dec_T, enc_T] @ [B, enc_T, enc_dim] -> [B, dec_T, enc_dim]

def get_attention_from_lengths(
        memory:        Tensor,# FloatTensor[B, enc_T, enc_dim]
        enc_durations: Tensor,# FloatTensor[B, enc_T]
        text_lengths:  Tensor #  LongTensor[B]
        ):
    B, enc_T, mem_dim = memory.shape
    
    mask = get_mask_from_lengths(text_lengths)
    enc_durations.masked_fill_(~mask, 0.0)
    
    enc_durations = enc_durations.round()#  [B, enc_T]
    dec_T = int(enc_durations.sum(dim=1).max().item())# [B, enc_T] -> int
    
    attention_contexts = torch.zeros(B, dec_T, mem_dim, device=memory.device, dtype=memory.dtype)# [B, dec_T, enc_dim]
    for i in range(B):
        mem_temp = []
        for j in range(int(text_lengths[i].item())):
            duration = int(enc_durations[i, j].item())
            
            # [B, enc_T, enc_dim] -> [1, enc_dim] -> [duration, enc_dim]
            mem_temp.append( memory[i, j:j+1].repeat(duration, 1) )
        mem_temp = torch.cat(mem_temp, dim=0)# [[duration, enc_dim], ...] -> [dec_T, enc_dim]
        min_len = min(attention_contexts.shape[1], mem_temp.shape[0])
        attention_contexts[i, :min_len] = mem_temp[:min_len]
    
    return attention_contexts# [B, dec_T, enc_dim]


class MelEncoder(nn.Module):
    """MelEncoder module:
        - Three 1-d convolution banks
    """
    def __init__(self, hparams, input_dim, output_dim):
        super(MelEncoder, self).__init__() 
        
        self.melenc_conv_hidden_dim = hparams.melenc_conv_dim
        self.output_dim = output_dim
        self.drop_chance = hparams.melenc_drop_frame_rate
        
        convolutions = []
        for _ in range(hparams.melenc_n_layers):
            input_dim = input_dim if (_ == 0) else self.melenc_conv_hidden_dim
            output_dim = self.output_dim if (_ == hparams.melenc_n_layers-1) else self.melenc_conv_hidden_dim
            conv_layer = nn.Sequential(
                ConvNorm(input_dim,
                         output_dim,
                         kernel_size=hparams.melenc_kernel_size, stride=hparams.melenc_stride,
                         padding=int((hparams.melenc_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(output_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(hparams.melenc_n_tokens,
                        hparams.melenc_n_tokens//2, 1,
                        batch_first=True, bidirectional=True)
        
        self.LReLU = nn.LeakyReLU(negative_slope=0.01)
    
    def drop_frames(self, spect, drop_chance=0.0):
        if drop_chance > 0.0:# randomly set some frames to zeros
            B, n_mel, dec_T = spect.shape
            frames_to_keep = torch.rand(B, 1, dec_T, device=spect.device, dtype=spect.dtype) > drop_chance
            spect = spect * frames_to_keep
        return spect
    
    def forward(self, spect, output_lengths, speaker_ids=None, enc_drop_rate=0.2):
        spect = self.drop_frames(spect, self.drop_chance)
        
        for conv in self.convolutions:
            spect = F.dropout(self.LReLU(conv(spect)), enc_drop_rate, self.training) # LeakyReLU
        
        #spect *= get_mask_from_lengths(output_lengths).unsqueeze(1)
        #       # [B, dec_dim, dec_T]*[B, 1, dec_T] -> [B, dec_dim, enc_T]
        
        spect = spect.transpose(1, 2)# [B, dec_dim, dec_T] -> [B, dec_T, dec_dim]
        
        self.lstm.flatten_parameters()
        _, (h_n, c_n) = self.lstm(spect)
        
        _, B, C = h_n
        h_n = h_n.permute(1, 0, 2).reshape(B, -1)
        return h_n


class Conv1dResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, n_blocks=1, kernel_size=3,
                act_func=nn.LeakyReLU(negative_slope=0.01, inplace=True),
                hidden_dim=None, dropout=0.2, use_batchnorm=True, residual_act_func=False):
        super(Conv1dResBlock, self).__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_dim  = hidden_dim or output_dim
        self.n_blocks    = n_blocks
        self.n_layers    = n_layers
        self.kernel_size = kernel_size
        self.dropout     = dropout
        self.act_func    = act_func
        self.residual_act_func = residual_act_func
        self.blocks = nn.ModuleList()
        
        self.start_conv = ConvNorm(input_dim, hidden_dim, 1)
        
        for i in range(self.n_blocks):
            convs = nn.ModuleList()
            for j in range(self.n_layers):
                conv = ConvNorm(hidden_dim, hidden_dim, kernel_size,
                                padding=(kernel_size - 1)//2       ,)
                if use_batchnorm:
                    conv = nn.Sequential(conv, nn.BatchNorm1d(hidden_dim))
                convs.append(conv)
            self.blocks.append(convs)
        
        self.end_conv = ConvNorm(hidden_dim, output_dim, 1)
    
    def forward(self, x, dropout: float=0.2): # [B, in_channels, T]
        dropout = self.dropout if dropout is None else dropout
        
        x = self.start_conv(x)
        
        for i, block in enumerate(self.blocks):
            x_identity = x # https://kharshit.github.io/img/resnet_block.png
            for j, layer in enumerate(block):
                is_last_layer = bool (j+1 == self.n_layers)
                x = layer(x)
                if not is_last_layer:
                    x = self.act_func(x)
                    x = F.dropout(x, dropout, self.training, inplace=True)
            x = x + x_identity
            if self.residual_act_func:
                x = self.act_func(x)
        
        return self.end_conv(x) # [B, out_channels, T]

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
            std = sqrt(2.0 / (hparams.n_speakers + self.encoder_speaker_embed_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.encoder_speaker_embedding.weight.data.uniform_(-val, val)
        
        self.encoder_conv_hidden_dim = hparams.encoder_conv_hidden_dim
        
        output_dim = hparams.symbols_embedding_dim+self.encoder_speaker_embed_dim# first layer input_dim
        convolutions = []
        for i in range(hparams.encoder_n_convolutions):
            is_last_layer = bool( i+1==hparams.encoder_n_convolutions )
            is_first_layer= bool( i == 0 )
            
            input_dim = output_dim
            output_dim = hparams.encoder_LSTM_dim if is_last_layer else self.encoder_conv_hidden_dim
            
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
        
        cond_dim = 2
        self.cond_conv = nn.Linear(hparams.encoder_LSTM_dim, cond_dim) # predicts Preceived Loudness Mu/Logvar from LSTM Hidden State
    
    def forward(self, text, text_lengths=None, speaker_ids=None, enc_dropout=0.2):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.repeat(1, 1, text.size(2)) # extend across all encoder steps
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            text = F.dropout(self.LReLU(conv(text)), enc_dropout, self.training) # LeakyReLU
        
        text = text.transpose(1, 2)
        
        if text_lengths is not None:
            text_lengths = text_lengths.cpu().numpy()# pytorch tensor are not reversible, hence the conversion
            text = nn.utils.rnn.pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, (h_n, c_n) = self.lstm(text)# -> [T, B, 2], ([2, B, C], [2, B, C])
        
        if text_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        _, B, C = h_n.shape
        cond = self.cond_conv(h_n.permute(1, 0, 2).reshape(B, -1))# [2, B, C] -> [B, C]
        
        return outputs, cond


def item_dropout(x, p_dropout=0.1, std=0.0):
    """Replace random slices of the 0th dim with zeros."""
    if p_dropout > 0.:
        B, *_ = x.shape
        mask = torch.rand(B, device=x.device) < p_dropout
        for i in range(len(_)):
            mask = mask.unsqueeze(-1)
        x = x * ~mask
        if std > 0.0:
            x += x.clone().normal_(mean=0, std=std)
    else:
        x = x.clone()
    return x

class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.attention = LenPredictorAttention()
        self.melglow = FlowDecoder(hparams)
    
    def forward(self, gt_mels, cond):
        """
        Decoder forward pass for training
        """
        # (Training) Encode Spect into Z Latent
        z, log_s_sum, logdet_w_sum = self.melglow(gt_mels, cond)
        return z, log_s_sum, logdet_w_sum
    
    def infer(self, cond, sigma=None):
        """
        Decoder inference
        """
        # (Inference) Decode Z into Spect
        mel_outputs = self.melglow.infer(cond, sigma=sigma)
        return mel_outputs

class MaskedBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super(MaskedBatchNorm1d, self).__init__(*args, **kwargs)
    
    def forward(self,
            x:               Tensor      ,# [B, C, T]
            x_mask: Optional[Tensor]=None,# [B, T]
            ):
        x_dims = len(x.shape)
        assert x_dims in [2, 3],                        'input must have 2/3 dims of shape [B, C] or [B, C, T]'
        assert x_mask is None or len(x_mask.shape) == 2, 'input must have 3 dims of shape [B, T]'
        
        if x_mask is not None and x_dims == 3:# must be [B, C, T] and have mask
            x.masked_fill_(~x_mask.unsqueeze(1), 0.0)
            masked_y = super(MaskedBatchNorm1d, self).forward(x.permute(0, 2, 1)[x_mask])# [B, C, T] -> [B*T, C] -> [B, C, T]
            y = x.permute(0, 2, 1)# [B, C, T] -> [B, T, C]
            y[x_mask] = masked_y#   [B*T, C]  =  [B*T, C]
            y = y.transpose(1, 2)#  [B, T, C] -> [B, C, T]
        else:
            y = super(MaskedBatchNorm1d, self).forward(x)# [B, C, T] -> [B*T, C] -> [B, C, T]
        return y# [B, C, T] or [B, C]

class LnBatchNorm1d(nn.Module):
    """
    Invertible Log() and BatchNorm1d()
    PARAMS:
        - same args/kwargs as BatchNorm1d
        
        - clamp_min:
            minimum value before log() operation.
    """
    def __init__(self, *args, clamp_min: float=0.01, clamp_max: float=1000., **kwargs):
        super(LnBatchNorm1d, self).__init__()
        self.norm = MaskedBatchNorm1d(*args, **kwargs)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    def forward(self, x, x_mask=None):
        x_log = x.clamp(min=self.clamp_min, max=self.clamp_max).log()
        y = self.norm(x_log, x_mask)
        return y
    
    def inverse(self, y, x_mask=None):
        mean = self.norm.running_mean
        var = self.norm.running_var
        x_log = (y*var.sqrt()[None, :, None])+mean[None, :, None]
        
        x = x_log.exp()
        x.masked_fill_(~x_mask, 0.0)
        x.clamp_(min=self.clamp_min, max=self.clamp_max)
        return x

class UnTTS(nn.Module):
    def __init__(self, hparams):
        super(UnTTS, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.melenc_enable  = hparams.melenc_enable
        
        self.bn_pl     = MaskedBatchNorm1d(1, momentum=0.01, affine=False)
        if (hparams.f0_log_scale if hasattr(hparams, 'f0_log_scale') else False):
            self.bn_f0 =     LnBatchNorm1d(1, momentum=0.01, affine=False, clamp_min=0.01, clamp_max=800.)
        else:
            self.bn_f0 = MaskedBatchNorm1d(1, momentum=0.01, affine=False)
        self.bn_energy = MaskedBatchNorm1d(1, momentum=0.01, affine=False)
        self.lbn_duration =  LnBatchNorm1d(1, momentum=0.01, affine=False, clamp_min=0.75, clamp_max=60.)
        
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        self.torchmoji_linear = LinearNorm(hparams.torchMoji_attDim, hparams.torchMoji_crushed_dim)
        
        self.encoder = Encoder(hparams)
        
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(hparams.n_speakers, self.speaker_embedding_dim)
        
        cond_input_dim = self.speaker_embedding_dim + 1     + hparams.torchMoji_crushed_dim + hparams.encoder_LSTM_dim
                               #           speaker          + sylps +         torchmoji_dim         +     encoder_outputs
        self.duration_glow = DurationGlow(hparams, cond_input_dim) if hparams.DurGlow_enable else None
        
        self.drop_cond_chance = float(hparams.drop_cond_chance) if hasattr(hparams, 'drop_cond_chance') else 0.5
        cond_input_dim += 3# perc_loudness + f0 + energy
        self.variance_inpainter = VarGlow(hparams, cond_input_dim)
        
        melenc_input_dim = None
        self.mel_encoder = MelEncoder(hparams, melenc_input_dim, hparams.melenc_output_dim) if hparams.melenc_enable else None
        
        hparams.cond_input_dim = cond_input_dim
        self.decoder = Decoder(hparams)
    
    @torch.no_grad()
    def parse_batch(self, batch):
        text_padded, mel_padded, speaker_ids, text_lengths, output_lengths, \
                 alignments, torchmoji_hidden, perc_loudness, f0, energy, sylps = batch
        text_padded    = to_gpu(text_padded).long()
        mel_padded     = to_gpu(mel_padded).float()
        speaker_ids    = to_gpu(speaker_ids.data).long()
        text_lengths   = to_gpu(text_lengths).long()
        output_lengths = to_gpu(output_lengths).long()
        alignments     = to_gpu(alignments).float()
        if torchmoji_hidden is not None:
            torchmoji_hidden = to_gpu(torchmoji_hidden).float()
        perc_loudness  = to_gpu(perc_loudness).float()
        f0             = to_gpu(f0).float()
        energy         = to_gpu(energy).float()
        sylps          = to_gpu(sylps).float()
        voiced_mask    = to_gpu(voiced_mask).bool()
        
        return (
            (text_padded, mel_padded, speaker_ids, text_lengths, output_lengths, alignments, torchmoji_hidden, perc_loudness, f0, energy, sylps, voiced_mask),
            (mel_padded, text_lengths, output_lengths, perc_loudness, f0, energy, sylps, voiced_mask))
            # returns ((x),(y)) as (x) for training input, (y) for ground truth/loss calc
    
    def forward(self, inputs):
        text, gt_mels, speaker_ids, text_lengths, output_lengths, \
               alignments, torchmoji_hidden, perc_loudness, f0, energy, sylps, voiced_mask = inputs
        
        # zero mean unit variance normalization of features
        mask = get_mask_from_lengths(output_lengths)# [B, T]
        perc_loudness = self.bn_pl    (perc_loudness.unsqueeze(1)                  )# [B]        -> [B, 1]
        f0            = self.bn_f0    (f0.unsqueeze(1)         , (voiced_mask&mask))# [B, dec_T] -> [B, 1, dec_T]
        energy        = self.bn_energy(energy.unsqueeze(1)     , mask              )# [B, dec_T] -> [B, 1, dec_T]
        
        embedded_text = self.embedding(text).transpose(1, 2)#    [B, embed, sequence]
        encoder_outputs, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_ids)# [B, enc_T, enc_dim]
        memory = [encoder_outputs,]
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            memory.append(embedded_speakers)# [B, enc_T, enc_dim]
        if sylps is not None:
            sylps = sylps[:, None, None]# [B] -> [B, 1, 1]
            sylps = sylps.repeat(1, encoder_outputs.size(1), 1)
            memory.append(sylps)# [B, enc_T, enc_dim]
        if torchmoji_hidden is not None:
            emotion_embed = torchmoji_hidden.unsqueeze(1)# [B, C] -> [B, 1, C]
            emotion_embed = self.torchmoji_linear(emotion_embed)# [B, 1, in_C] -> [B, 1, out_C]
            emotion_embed = emotion_embed.repeat(1, encoder_outputs.size(1), 1)
            memory.append(emotion_embed)#   [B, enc_T, enc_dim]
        memory = torch.cat(memory, dim=2)# [[B, enc_T, enc_dim], [B, enc_T, speaker_dim]] -> [B, enc_T, enc_dim+speaker_dim]
        
        # DurationGlow
        mask = get_mask_from_lengths(text_lengths)# [B, T]
        enc_durations = alignments.sum(dim=1) # [B, dec_T, enc_T] -> [B, enc_T]
        enc_durations = enc_durations.unsqueeze(1)# [B, enc_T] -> [B, 1, enc_T]
        ln_enc_durations = self.lbn_duration(enc_durations, mask).repeat(1, 2, 1)# Norm and [B, 1, enc_T] -> [B, 2, enc_T]
        dur_z, dur_log_s_sum, dur_logdet_w_sum = self.duration_glow(ln_enc_durations, memory.transpose(1, 2))
                                                                #  ([B, enc_T]   , [B, enc_dim, enc_T]   )
        
        attention_contexts = alignments @ memory
        #             [B, dec_T, enc_T] @ [B, enc_T, enc_dim] -> [B, dec_T, enc_dim]
        
        # Variances Inpainter
        # cond -> attention_contexts(+perc_loudness)(+f0)(+energy)
        # x/z  -> perc_loudness + f0 + energy
        perc_loudness = perc_loudness.unsqueeze(-1).repeat(1, 1, f0.size(2))# [B, 1] -> [B, 1, dec_T]
        
        drop_cond_chance = self.drop_cond_chance # Add noise and randomly drop item from batch
        perc_loudness_maybe = item_dropout(perc_loudness, drop_cond_chance, 0.05)# [B, 1, dec_T]
        f0_maybe            = item_dropout(f0           , drop_cond_chance, 0.05)# [B, 1, dec_T]
        energy_maybe        = item_dropout(energy       , drop_cond_chance, 0.05)# [B, 1, dec_T]
        
        incomplete_variances = torch.cat( (attention_contexts.transpose(1, 2),
                                perc_loudness_maybe, f0_maybe, energy_maybe), dim=1)# -> [B, C, dec_T]
        
        var_gt = torch.cat((perc_loudness, f0, energy), dim=1)
        var_gt = var_gt.repeat(1, 2, 1)
        variance_z, variance_log_s_sum, variance_logdet_w_sum = self.variance_inpainter(var_gt, incomplete_variances)
        
        global_cond = None
        if self.melenc_enable: # take all current info, and produce global cond tokens which can be randomly sampled from later
            melenc_input = torch.cat((gt_mels, attention_contexts, perc_loudness, f0, energy), dim=1)
            global_cond = self.mel_encoder(melenc_input, output_lengths)# [B, n_tokens]
        
        # Decoder
        cond = [attention_contexts.transpose(1, 2), perc_loudness, f0, energy]
        if global_cond is not None:
            cond.append(global_cond)
        cond = torch.cat(cond, dim=1)
        z, log_s_sum, logdet_w_sum = self.decoder(gt_mels.clone(), cond)
                                    #   [B, n_mel, dec_T], [B, dec_T, enc_dim] # Series of Flows
        
        outputs = {
            "melglow": [z    , log_s_sum    , logdet_w_sum    ],
            "durglow": [dur_z, dur_log_s_sum, dur_logdet_w_sum],
            "varglow": [variance_z, variance_log_s_sum, variance_logdet_w_sum],
            "sylps"  : [pred_sylps, sylps],
        }
        return outputs
    
    def update_device(self, *inputs):
        target_device = next(self.parameters()).device
        target_float_dtype = next(self.parameters()).dtype
        outputs = []
        for input in inputs:
            if type(input) == Tensor:# move all Tensor types to GPU
                if input.dtype == torch.float32:
                    outputs.append(input.to(target_device, target_float_dtype))# convert float to half if required.
                else:
                    outputs.append(input.to(target_device                    ))# leave Long / Bool unchanged in datatype
            else:
                outputs.append(input)
        return outputs
    
    def inference(self,
            text:             Tensor,            #  LongTensor[B, enc_T]
            speaker_ids:      Tensor,            #  LongTensor[B]
            torchmoji_hidden: Tensor,            # FloatTensor[B, embed] 
            sylps:         Optional[Tensor]=None,# FloatTensor[B]        or None
            text_lengths:  Optional[Tensor]=None,#  LongTensor[B]        or None
            durations:     Optional[Tensor]=None,# FloatTensor[B, enc_T] or None
            perc_loudness: Optional[Tensor]=None,# FloatTensor[B]        or None
            f0:            Optional[Tensor]=None,# FloatTensor[B, dec_T] or None
            energy:        Optional[Tensor]=None,# FloatTensor[B, dec_T] or None
            mel_sigma: float=1.0, dur_sigma: float=1.0, var_sigma: float=1.0):
        assert not self.training, "model must be in eval() mode"
        
        # move Tensors to GPU (if not already there)
        text, speaker_ids, torchmoji_hidden, sylps, text_lengths, durations, perc_loudness, f0, energy = self.update_device(text, speaker_ids, torchmoji_hidden, sylps, text_lengths, durations, perc_loudness, f0, energy)
        B, enc_T = text.shape
        
        if text_lengths is None:
            text_lengths = torch.ones((B,)).to(text)*enc_T
        assert text_lengths is not None
        
        melenc_outputs = self.mel_encoder(gt_mels, output_lengths, speaker_ids=speaker_ids) if (self.mel_encoder is not None and not self.melenc_ignore) else None# [B, dec_T, melenc_dim]
        
        embedded_text = self.embedding(text).transpose(1, 2)#    [B, embed, sequence]
        encoder_outputs, pred_sylps = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_ids)# [B, enc_T, enc_dim]
        
        assert sylps is not None # needs to be updated with pred_sylps soon ^TM
        
        memory = [encoder_outputs,]
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, enc_T, 1)
            memory.append(embedded_speakers)# [B, enc_T, enc_dim]
        if sylps is not None:
            sylps = sylps[:, None, None]# [B] -> [B, 1, 1]
            sylps = sylps.repeat(1, enc_T, 1)
            memory.append(sylps)# [B, enc_T, enc_dim]
        if torchmoji_hidden is not None:
            emotion_embed = torchmoji_hidden.unsqueeze(1)# [B, C] -> [B, 1, C]
            emotion_embed = self.torchmoji_linear(emotion_embed)# [B, 1, in_C] -> [B, 1, out_C]
            emotion_embed = emotion_embed.repeat(1, enc_T, 1)
            memory.append(emotion_embed)#   [B, enc_T, enc_dim]
        memory = torch.cat(memory, dim=2)# [[B, enc_T, enc_dim], [B, enc_T, speaker_dim]] -> [B, enc_T, enc_dim+speaker_dim]
        
        # DurationGlow
        mask = get_mask_from_lengths(text_lengths)# [B, T]
        ln_enc_durations = self.duration_glow.infer(memory.transpose(1, 2), sigma=dur_sigma)
                                                #  ([B, enc_dim, enc_T]   ,                )
        enc_durations = self.lbn_duration.inverse(ln_enc_durations, mask)[:, 0]# [B, 2, enc_T] -> [B, enc_T]
        #enc_durations = ln_enc_durations.exp()[:, 0]# [B, 2, enc_T] -> [B, enc_T]
        
        attention_contexts = get_attention_from_lengths(memory, enc_durations, text_lengths)
                           #                -> [B, dec_T, enc_dim]
        B, dec_T, enc_dim = attention_contexts.shape
        
        if perc_loudness is None:
            perc_loudness = torch.zeros(B, 1, dec_T, device=attention_contexts.device, dtype=attention_contexts.dtype)
        else:
            perc_loudness = perc_loudness[:, None, None].repeat(1, 1, dec_T)
                                # [B]        -> [B, 1, dec_T]
        
        if f0 is None:
            f0 = torch.zeros(B, 1, dec_T, device=attention_contexts.device, dtype=attention_contexts.dtype)
        else:
            f0 = f0.unsqueeze(1)# [B, dec_T] -> [B, 1, dec_T]
        if energy is None:
            energy = torch.zeros(B, 1, dec_T, device=attention_contexts.device, dtype=attention_contexts.dtype)
        else:
            energy = energy.unsqueeze(1)# [B, dec_T] -> [B, 1, dec_T]
        
        incomplete_variances = torch.cat( (attention_contexts.transpose(1, 2),
                                perc_loudness, f0, energy), dim=1)# -> [B, C, dec_T]
        variances = self.variance_inpainter.infer(incomplete_variances, sigma=var_sigma)
        variances = variances.chunk(2, dim=1)[0]# [B, 3, dec_T]
        f0     = variances[:, 1, :]
        energy = variances[:, 2, :]
        
        global_cond = None
        if self.melenc_enable: # take all current info, and produce global cond tokens which can be randomly sampled from later
            global_cond = torch.randn(B, n_tokens)# [B, n_tokens]
        
        # Decoder
        cond = [attention_contexts.transpose(1, 2), variances]
        if global_cond is not None:
            cond.append(global_cond)
        cond = torch.cat(cond, dim=1)
        spect = self.decoder.infer(cond, sigma=mel_sigma)
        
        outputs = {
            "spect":              spect,
            "enc_frames": enc_durations,
            "f0":                    f0,
            "energy":            energy,
        }
        return outputs
