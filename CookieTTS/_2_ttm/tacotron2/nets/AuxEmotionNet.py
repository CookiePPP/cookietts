import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieTTS.utils.model.layers import ConvNorm, LinearNorm


class AuxEmotionNet(nn.Module):
    def __init__(self, hparams):
        super(AuxEmotionNet, self).__init__()
        layers = []
        for i, dim in enumerate(hparams.auxemotionnet_layer_dims):
            last_layer = (i+1 == len(hparams.auxemotionnet_layer_dims))
            in_dim = out_dim = dim
            if i == 0:
                in_dim = hparams.torchMoji_attDim
            if last_layer:
                out_dim = hparams.torchMoji_attDim
            layers.append( LinearNorm(in_dim, out_dim) )
            if not last_layer:
                layers.append( nn.LeakyReLU(negative_slope=0.05, inplace=True) )
        self.seq_layers = nn.Sequential( *layers )
        
        self.n_classes = len(hparams.emotion_classes)
        input_dim = hparams.speaker_embedding_dim + hparams.torchMoji_attDim + hparams.auxemotionnet_RNN_dim
        self.classifier_layer_dropout = hparams.auxemotionnet_classifier_layer_dropout
        self.latent_classifier_layer = LinearNorm(input_dim, hparams.emotionnet_latent_dim*2+self.n_classes)
        
        self.encoder_outputs_dropout = hparams.auxemotionnet_encoder_outputs_dropout
        self.text_rnn = nn.GRU(hparams.encoder_LSTM_dim, hparams.auxemotionnet_RNN_dim, batch_first=True)
    
    def reparameterize(self, mu, logvar, rand_sample=None):
        # use for VAE sampling
        if rand_sample or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, torchmoji_hidden, speaker_embed, encoder_outputs, text_lengths=None):# [B]
        ref = self.seq_layers(torchmoji_hidden[:, None])# [B, 1, Embed]
        speaker_embed = speaker_embed[:, None]# [B, embed] -> [B, 1, embed]
        
        if self.training and self.encoder_outputs_dropout > 0.0:
            encoder_outputs = F.dropout(encoder_outputs, p=self.encoder_outputs_dropout, training=self.training, inplace=False)
        
        if text_lengths is not None:
            encoder_outputs = nn.utils.rnn.pack_padded_sequence(encoder_outputs, text_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)
        encoder_output = self.text_rnn(encoder_outputs)[1].transpose(0, 1)# [B, enc_T, enc_dim] -> [1, B, enc_dim] -> [B, 1, enc_dim]
        
        cat_inputs = torch.cat((ref, speaker_embed, encoder_output), dim=2)# [B, 1, dim]
        
        if self.training and self.classifier_layer_dropout > 0.0:
            cat_inputs = F.dropout(cat_inputs, p=self.classifier_layer_dropout, training=self.training, inplace=True)
        cat_energies = self.latent_classifier_layer(cat_inputs)# [B, 1, n_class]
        prob_energies = cat_energies[:,:,:self.n_classes]# [B, 1, n_class]
        aux_zu_params = cat_energies[:,:,self.n_classes:]# [B, 1, 2*lat_dim]
        aux_zs = F.log_softmax(prob_energies, dim=2)     # [B, 1, n_class]
        
        aux_zu_mu, aux_zu_logvar = aux_zu_params.chunk(2, dim=2)# [B, 1, 2*lat_dim] -> [B, 1, lat_dim], [B, 1, lat_dim]
        return aux_zs, aux_zu_mu, aux_zu_logvar, aux_zu_params
    
    def infer_rand(self, sylps):# [B]
        syl_zu = torch.zeros_like(sylps, layout=(sylps.shape[0],)).normal_()[:, None]# [B] -> [B, 1]
        return syl_zu
    
    def infer_controlled(self, x, mu=0.0):# [B], int
        syl_zu = torch.ones(x.shape[0])[:, None] * mu# [B] -> [B, 1]
        return syl_zu
    
    def infer_auto(self, sylps, rand_sampling=False):# [B]
        ln_sylps = sylps.log()
        sylps_cat = torch.stack((sylps, ln_sylps), dim=1)# [B, 2]
        syl_res = self.seq_layers(sylps_cat)
        syl_params = sylps_cat + self.res_weight * syl_res# [B, 2]
        syl_mu = syl_params[:, 0]    # [B]
        syl_logvar = syl_params[:, 1]# [B]
        syl_zu = self.reparameterize(syl_mu, syl_logvar, rand_sampling)# [B]
        syl_zu = syl_zu[:, None]# [B] -> [B, 1]
        return syl_zu