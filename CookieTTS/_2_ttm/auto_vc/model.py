from math import sqrt, ceil
import random
import numpy as np
from numpy import finfo

import torch
from torch import nn
from torch.nn import functional as F

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict

from CookieTTS.utils.model.layers import ConvNorm, LinearNorm
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame, freeze_grads

drop_rate = 0.5

def load_model(hparams, device=None):
    model = AutoVC(hparams)
    if torch.cuda.is_available() and device != 'cpu':
        model = model.cuda()
    return model


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.bottleneck_dim = hparams.bottleneck_dim
        self.speaker_encoder_dim = hparams.speaker_encoder_dim
        self.freq = hparams.freq
        
        convolutions = []
        for i in range(hparams.n_enc_layers):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.n_mel_channels+self.speaker_encoder_dim if i==0 else hparams.enc_conv_dim,
                         hparams.enc_conv_dim,
                         kernel_size=5, stride=1,
                         padding=2, causal=getattr(hparams, 'use_causal_convs', False),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.enc_conv_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(hparams.enc_conv_dim, self.bottleneck_dim, 2, batch_first=True, bidirectional=True)
    
    def forward(self, mel, c_org):# [B, n_mel, mel_T], [B, embed]
        c_org = c_org.unsqueeze(-1).expand(-1, -1, mel.size(-1))
        mel = torch.cat((mel, c_org), dim=1)
        
        for conv in self.convolutions:
            mel = F.relu(conv(mel))
        mel = mel.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(mel)
        out_forward  = outputs[:, :, :self.bottleneck_dim]# [B, mel_T, C//2]
        out_backward = outputs[:, :, self.bottleneck_dim:]# [B, mel_T, C//2]
        
        codes = []
        for i in range(0, outputs.size(1)-self.freq+1, self.freq):# .append( [[B, C], ...]*mel_T//freq )
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))
        
        return codes


class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.bottleneck_dim = hparams.bottleneck_dim
        self.speaker_encoder_dim  = hparams.speaker_encoder_dim
        self.decoder_conv_dim  = hparams.decoder_conv_dim
        
        self.lstm1 = nn.LSTM(self.bottleneck_dim*2+self.speaker_encoder_dim, self.decoder_conv_dim, 1, batch_first=True)
        
        convolutions = []
        for i in range(hparams.decoder_n_conv_layers):
            conv_layer = nn.Sequential(
                ConvNorm(self.decoder_conv_dim,
                         self.decoder_conv_dim,
                         kernel_size=5, stride=1,
                         padding=2, causal=getattr(hparams, 'use_causal_convs', False),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(self.decoder_conv_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(self.decoder_conv_dim, hparams.decoder_lstm_dim, hparams.decoder_n_lstm_layers, batch_first=True)
        
        self.linear_projection = LinearNorm(hparams.decoder_lstm_dim, hparams.n_mel_channels)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        
        for i in range(0, 5):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.n_mel_channels if i == 0 else hparams.postnet_embedding_dim,
                             hparams.n_mel_channels if i == 4 else hparams.postnet_embedding_dim,
                             kernel_size=5, stride=1,
                             padding=2, causal=getattr(hparams, 'use_causal_convs', False),
                             dilation=1, w_init_gain='linear' if i == 4 else 'tanh'),
                    nn.BatchNorm1d(
                             hparams.n_mel_channels if i == 4 else hparams.postnet_embedding_dim))
            )
    
    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        
        x = self.convolutions[-1](x)
        
        return x


class AutoVC(nn.Module):
    """Generator network."""
    def __init__(self, hparams):
        super(AutoVC, self).__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self, gt_mel, c_org, c_trg):# gt_mel[B, n_mel, mel_T]
                                            # c_org[B, embed] = original speaker embed
                                            # c_trg[B, embed] =   target speaker embed
        
        codes = self.encoder(gt_mel, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        # [[B, C],]*mel_T//freq -> [ [B, C] -> [B, freq, C] for x in codes] -> [B, mel_T, C]
        code_exp = torch.cat([code.unsqueeze(1).expand(-1,ceil(gt_mel.size(2)/len(codes)),-1) for code in codes], dim=1)
        if code_exp.shape[1] != gt_mel.shape[2]:
            code_exp = code_exp[:, :gt_mel.shape[2]]
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,gt_mel.size(2),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs).transpose(1, 2)
        
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        
        outputs = {
            "pred_mel": mel_outputs,        # [B, n_mel, mel_T]
    "pred_mel_postnet": mel_outputs_postnet,# [B, n_mel, mel_T]
    "bottleneck_codes": torch.cat(codes, dim=-1),
        }
        return outputs