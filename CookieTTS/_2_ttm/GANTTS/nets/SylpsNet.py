import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieTTS.utils.model.layers import ConvNorm, LinearNorm


class SylpsNet(nn.Module):
    def __init__(self, hparams):
        super(SylpsNet, self).__init__()
        layers = []
        for i, dim in enumerate(hparams.sylpsnet_layer_dims):
            last_layer = (i+1 == len(hparams.sylpsnet_layer_dims))
            in_dim = out_dim = dim
            if i == 0:
                in_dim = 2
            if last_layer:
                out_dim = 1
            layers.append( LinearNorm(in_dim, out_dim) )
            if not last_layer:
                layers.append( nn.LeakyReLU(negative_slope=0.05, inplace=True) )
        self.seq_layers = nn.Sequential( *layers )
        self.res_weight = nn.Parameter( torch.tensor(0.01) )
    
    def reparameterize(self, mu, logvar, rand_sample=None):
        # use for VAE sampling
        if rand_sample or self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, sylps):# [B]
        ln_sylps = sylps.log()
        sylps_cat = torch.stack((sylps, ln_sylps), dim=1)# [B, 2]
        syl_res = self.seq_layers(sylps_cat)
        syl_params = sylps_cat + self.res_weight * syl_res# [B, 2]
        syl_mu = syl_params[:, 0]    # [B]
        syl_logvar = syl_params[:, 1]# [B]
        syl_zu = self.reparameterize(syl_mu, syl_logvar)# [B]
        syl_zu = syl_zu[:, None]# [B] -> [B, 1]
        return syl_zu, syl_mu, syl_logvar
    
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