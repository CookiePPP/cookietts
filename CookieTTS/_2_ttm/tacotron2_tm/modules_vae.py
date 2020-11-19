# https://github.com/jinhan/tacotron2-vae/blob/master/modules.py
# jinhan/tacotron2-vae

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from CookieTTS.utils.model.layers import LinearNorm
#from CoordConv import CoordConv2d


class ReferenceEncoder(nn.Module):
    '''
    inputs --- [B, Ty/r, n_mels*r]  mels
    outputs --- [B, res_enc_gru_size]
    '''
    
    def __init__(self, hparams):
        super().__init__()
        assert hparams.res_enc_gru_dim%2==0, 'res_enc_gru_dim must be even'
        K = len(hparams.res_enc_filters)
        filters = [1] + hparams.res_enc_filters
        # 첫번째 레이어로 CoordConv를 사용하는 것이 positional 정보를 잘 보존한다고 함. https://arxiv.org/pdf/1811.02122.pdf
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels=filters[0],
                           out_channels=filters[0 + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)))
        self.convs.extend([nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(1,K)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hparams.res_enc_filters[i]) for i in range(K)])
        
        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=hparams.res_enc_filters[-1] * out_channels,
                          hidden_size=hparams.res_enc_gru_dim,
                          batch_first=True)
        self.n_mels   = hparams.n_mel_channels
        self.n_tokens = hparams.res_enc_n_tokens
        self.post_fc  = nn.Linear(hparams.res_enc_gru_dim,    hparams.res_enc_n_tokens*2)
        self.embed_fc = nn.Linear(hparams.res_enc_n_tokens, hparams.res_enc_embed_dim )
    
    def forward(self, inputs, input_lengths=None, rand_sampling=True):
        B = inputs.size(0)
        out = inputs.contiguous().view(B, 1, -1, self.n_mels)  # [B, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.leaky_relu(out, negative_slope=0.05, inplace=True)  # [B, 128, Ty//2^K, n_mels//2^K]
        
        out = out.transpose(1, 2)  # [B, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        B = out.size(0)
        out = out.contiguous().view(B, T, -1)  # [B, Ty//2^K, 128*n_mels//2^K]
        
        if False and input_lengths is not None:
            # pytorch tensor are not reversible, hence the conversion
            input_lengths = input_lengths.cpu().numpy()
            out = nn.utils.rnn.pack_padded_sequence(out, input_lengths, batch_first=True, enforce_sorted=False)
            # [B, T, C]
        
        out = self.gru(out)[1].squeeze(0)# -> [1, B, C]
        
        out = self.post_fc(out)# [1, B, C] -> [B, 2*n_tokens]
        
        mu     = out.chunk(2, dim=1)[0]
        logvar = out.chunk(2, dim=1)[1]
        zr = self.reparameterize(mu, logvar, rand_sampling)# [B, 2*n_tokens] -> [B, n_tokens]
        embed = self.embed_fc(zr)# [B, n_tokens] -> [B, embed]
        
        return embed, zr, mu, logvar, out
    
    def prior(self, x, std=0.0):
        zr = torch.randn(x.shape[0], self.n_tokens, device=x.device, dtype=next(self.parameters()).dtype) * std
        embed = self.embed_fc(zr)# [B, n_tokens] -> [B, embed]
        return embed
    
    def reparameterize(self, mu, logvar, rand_sampling=False):
        if self.training or rand_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
