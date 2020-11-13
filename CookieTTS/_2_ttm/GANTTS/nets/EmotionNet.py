import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import weight_norm as norm
import numpy as np
import module as mm
from CookieTTS.utils.model.layers import ConvNorm, LinearNorm

EPSILON = 1e-6 # smallest value that will not result in inf in fp16

class ReferenceEncoder(nn.Module):
    """
    Reference Encoder.
    6 convs + GRU + FC
    :param conv_filters: List[int, int, ...]
    :param rnn_dim: Scalar.
    :param conv_act_func: Conv2d activation function
    :param out_activation_fn: output Linear activation function
    """
    def __init__(self, hparams, conv_filters, rnn_dim, bias=False, conv_act_func=torch.relu, out_activation_fn=torch.tanh, drop_rate=0.):
        super(ReferenceEncoder, self).__init__()
        # ref_enc_filters
        
        channels = [1] + conv_filters + [rnn_dim]
        self.convs = nn.ModuleList([
            mm.Conv2d(channels[c], channels[c+1], 3, stride=2, bn=True, bias=bias, activation_fn=conv_act_func, drop_rate=drop_rate)
            for c in range(len(channels)-1)
        ]) # [B, dec_T/r, 128]
        self.gru = nn.GRU(rnn_dim*2, rnn_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
        )
        self.activation_fn = out_activation_fn
    
    def forward(self, x, hidden=None):
        """
        :param x: (Batch, 1, dec_T, n_mel) Tensor. Mel Spectrogram
        :param hidden: Tensor. initial hidden state for gru
        Returns:
            y_: (Batch, 1, Embedding) Reference Embedding
        """
        y_ = x.transpose(1, 2).unsqueeze(1) # [B, n_mel, dec_T] -> [B, 1, dec_T, n_mel]
        
        for i in range(len(self.convs)):
            y_ = self.convs[i](y_)
        # [B, C, dec_T//64, n_mel//64]
        
        y_ = y_.transpose(1, 2) # [B, C, dec_T//64, n_mel//64] -> [B, dec_T//64, C, n_mel//64]
        shape = y_.shape
        y_ = y_.contiguous().view(shape[0], shape[1], shape[2]*shape[3]) # [B, dec_T//64, C, n_mel//64] -> [B, dec_T//64, C*n_mel//64] merge last 2 dimensions
        
        y_ = self.gru(y_, hidden)[1]# [1, B, T_Embed]
        
        y_ = self.fc(y_.squeeze(0)) # [1, B, T_Embed] -> [B, T_Embed]
        
        if self.activation_fn is not None:
            y_ = self.activation_fn(y_)
        return y_.unsqueeze(1) # (Batch, 1, Embedding)


class EmotionNet(nn.Module):
    def __init__(self, hparams):
        super(EmotionNet, self).__init__()
        self.unknown_id = len(hparams.emotion_classes)
        
        self.ref_enc = ReferenceEncoder(hparams, hparams.emotionnet_ref_enc_convs,
                                               hparams.emotionnet_ref_enc_rnn_dim,
                                               hparams.emotionnet_ref_enc_use_bias,
                                               drop_rate=hparams.emotionnet_ref_enc_droprate,)
        
        input_dim = hparams.speaker_embedding_dim + hparams.emotionnet_ref_enc_rnn_dim + hparams.emotionnet_RNN_dim
        self.classifier_layer = LinearNorm(input_dim, len(hparams.emotion_classes))
        
        input_dim = input_dim + len(hparams.emotion_classes)
        self.classifier_layer_dropout = hparams.emotionnet_classifier_layer_dropout
        self.latent_layer = LinearNorm(input_dim, hparams.emotionnet_latent_dim*2)
        
        self.encoder_outputs_dropout = hparams.emotionnet_encoder_outputs_dropout
        self.text_rnn = nn.GRU(hparams.encoder_LSTM_dim, hparams.emotionnet_RNN_dim, batch_first=True)
    
    def reparameterize(self, mu, logvar, rand_sample=None):
        # use for VAE sampling
        if rand_sample or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, gt_mels, speaker_embed, encoder_outputs, text_lengths=None, emotion_id=None, emotion_onehot=None):# [B]
        ref = self.ref_enc(gt_mels)# [B, 1, Embed]
        speaker_embed = speaker_embed[:, None]# [B, embed] -> [B, 1, embed]
        
        if self.training and self.encoder_outputs_dropout > 0.0:
            encoder_outputs = F.dropout(encoder_outputs, p=self.encoder_outputs_dropout, training=self.training, inplace=False)
        
        if text_lengths is not None:
            encoder_outputs = nn.utils.rnn.pack_padded_sequence(encoder_outputs, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        encoder_output = self.text_rnn(encoder_outputs)[1].transpose(0, 1)# [B, enc_T, enc_dim] -> [1, B, enc_dim] -> [B, 1, enc_dim]
        
        cat_inputs = torch.cat((ref, speaker_embed, encoder_output), dim=2)# [B, 1, dim]
        
        if self.training and self.classifier_layer_dropout > 0.0:
            cat_inputs = F.dropout(cat_inputs, p=self.classifier_layer_dropout, training=self.training, inplace=True)
        prob_energies = self.classifier_layer(cat_inputs)# [B, 1, n_class]
        zs = F.log_softmax(prob_energies, dim=2)         # [B, 1, n_class]
        
        # use GT emotion labels where supervision is possible
        ss_zs = zs.new_empty(zs.shape)
        ss_zs[emotion_id==self.unknown_id] = zs[emotion_id==self.unknown_id]
        ss_zs[emotion_id!=self.unknown_id] = (emotion_onehot[:, None][emotion_id!=self.unknown_id]+EPSILON).log()
        
        latent_inputs = torch.cat((cat_inputs, ss_zs), dim=2)# [B, 1, dim]
        zu_params = self.latent_layer(latent_inputs)# [B, 1, 2*lat_dim]
        assert not torch.isinf(zu_params).any()
        zu_mu, zu_logvar = zu_params.chunk(2, dim=2)# [B, 1, 2*lat_dim] -> [B, 1, lat_dim], [B, 1, lat_dim]
        zu = self.reparameterize(zu_mu, zu_logvar)
        
        return zs, zu, zu_mu, zu_logvar, zu_params
    
    def infer_rand(self, sylps):# [B]
        #syl_zu = torch.zeros_like(sylps, layout=(sylps.shape[0],)).normal_()[:, None]# [B] -> [B, 1]
        return syl_zu
    
    def infer_controlled(self, x, mu=0.0):# [B], int
        #syl_zu = torch.ones(x.shape[0])[:, None] * mu# [B] -> [B, 1]
        return syl_zu
    
    def infer_auto(self, sylps, rand_sampling=False):# [B]
        #ln_sylps = sylps.log()
        #sylps_cat = torch.stack((sylps, ln_sylps), dim=1)# [B, 2]
        #syl_res = self.seq_layers(sylps_cat)
        #syl_params = sylps_cat + self.res_weight * syl_res# [B, 2]
        #syl_mu = syl_params[:, 0]    # [B]
        #syl_logvar = syl_params[:, 1]# [B]
        #syl_zu = self.reparameterize(syl_mu, syl_logvar, rand_sampling)# [B]
        #syl_zu = syl_zu[:, None]# [B] -> [B, 1]
        return syl_zu