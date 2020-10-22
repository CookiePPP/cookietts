import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d

def glow_loss(z, log_s_sum, logdet_w_sum, output_lengths, sigma):
    dec_T = output_lengths.max()
    
    B = z.shape[0]
    z = z.view(z.shape[0], -1, dec_T).float()
    log_s_sum = log_s_sum.view(B, -1, dec_T)
    B, z_channels, dec_T = z.shape
    
    n_elems = (output_lengths.float().sum()*z_channels)
    
    # remove paddings before loss calc
    mask = get_mask_from_lengths(output_lengths)[:, None, :] # [B, 1, T] BoolTensor
    mask = mask.expand(B, z_channels, dec_T)# [B, z_channels, T] BoolTensor
    
    z = torch.masked_select(z, mask)
    loss_z = ((z.pow(2).sum()) / sigma)/n_elems # mean z (over all elements)
    
    log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
    loss_s = -log_s_sum.float().sum()/n_elems
    
    loss_w = -logdet_w_sum.float().sum()/(z_channels*dec_T)
    
    loss = loss_z+loss_w+loss_s
    return loss, loss_z, loss_w, loss_s

class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.n_group = hparams.n_group
        sigma = hparams.sigma
        self.sigma2 = sigma ** 2
        self.sigma2_2 = self.sigma2 * 2
        self.dg_sigma2_2 = (hparams.dg_sigma**2)*2
        self.DurGlow_enable = hparams.DurGlow_enable
        self.MelGlow_loss_scalar = hparams.MelGlow_loss_scalar
        self.DurGlow_loss_scalar = hparams.DurGlow_loss_scalar
        self.VarGlow_loss_scalar = hparams.VarGlow_loss_scalar
    
    def forward(self, model_output, targets, loss_scalars):
        # loss scalars
        MelGlow_ls = loss_scalars['MelGlow_ls'] if loss_scalars['MelGlow_ls'] is not None else self.MelGlow_loss_scalar
        DurGlow_ls = loss_scalars['DurGlow_ls'] if loss_scalars['DurGlow_ls'] is not None else self.DurGlow_loss_scalar
        VarGlow_ls = loss_scalars['VarGlow_ls'] if loss_scalars['VarGlow_ls'] is not None else self.VarGlow_loss_scalar
        
        # loss_func
        mel_target, text_lengths, output_lengths, perc_loudness_target, f0_target, energy_target, sylps_target = targets
        B, n_mel, dec_T = mel_target.shape
        enc_T = text_lengths.max()
        output_lengths_float = output_lengths.float()
        
        loss_dict = {}
        
        # Spectrogram Loss / DecoderGlow Loss
        if True:
            mel_z, log_s_sum, logdet_w_sum = model_output['melglow']
            
            # remove paddings before loss calc
            mask = get_mask_from_lengths(output_lengths)[:, None, :] # [B, 1, T] BoolTensor
            mask = mask.expand(mask.size(0), mel_target.size(1), mask.size(2))# [B, n_mel, T] BoolTensor
            
            n_elems = (output_lengths_float.sum() * n_mel)
            mel_z = torch.masked_select(mel_z, mask)
            dec_loss_z = ((mel_z.pow(2).sum()) / self.sigma2_2)/n_elems # mean z (over all elements)
            
            log_s_sum = log_s_sum.view(B, -1, dec_T)
            log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
            dec_loss_s = -log_s_sum.sum()/(n_elems)
            
            dec_loss_w = -logdet_w_sum.sum()/(n_mel*dec_T)
            
            dec_loss_d = dec_loss_z+dec_loss_w+dec_loss_s
            loss = dec_loss_d*MelGlow_ls
            del mel_z, log_s_sum, logdet_w_sum, mask, n_elems
            loss_dict["Decoder_Loss_Z"] = dec_loss_z
            loss_dict["Decoder_Loss_W"] = dec_loss_w
            loss_dict["Decoder_Loss_S"] = dec_loss_s
            loss_dict["Decoder_Loss_Total"] = dec_loss_d
        
        # DurationGlow Loss
        if True:#self.DurGlow_enable:
            dur_z, log_s_sum, logdet_w_sum = model_output['durglow']
            _ = glow_loss(dur_z, log_s_sum, logdet_w_sum, text_lengths, self.dg_sigma2_2)
            dur_loss_d, dur_loss_z, dur_loss_w, dur_loss_s = _
            #z_channels = 2
            #dur_z = dur_z.view(dur_z.shape[0], z_channels, -1)# [B, z_channels, T]
            #
            ## remove paddings before loss calc
            #mask = get_mask_from_lengths(text_lengths)[:, None, :]#[B, 1, T] BoolTensor
            #mask = mask.expand(mask.size(0), z_channels, mask.size(2))# [B, z_channels, T] BoolTensor
            #n_elems = (text_lengths.sum() * z_channels)
            #
            #dur_z = torch.masked_select(dur_z, mask)
            #dur_loss_z = ((dur_z.pow(2).sum()) / self.dg_sigma2_2)/n_elems # mean z (over all elements)
            #
            #log_s_sum = log_s_sum.view(B, -1, enc_T)
            #log_s_sum = torch.masked_select(log_s_sum, mask[:, :log_s_sum.shape[1], :])
            #dur_loss_s = -log_s_sum.sum()/(n_elems)
            #
            #dur_loss_w = -logdet_w_sum.sum()/(z_channels*enc_T)
            #
            #dur_loss_d = dur_loss_z+dur_loss_w+dur_loss_s
            if self.DurGlow_loss_scalar:
                loss = loss + dur_loss_d*DurGlow_ls
            del dur_z, log_s_sum, logdet_w_sum
            loss_dict["Duration_Loss_Z"] = dur_loss_z
            loss_dict["Duration_Loss_W"] = dur_loss_w
            loss_dict["Duration_Loss_S"] = dur_loss_s
            loss_dict["Duration_Loss_Total"] = dur_loss_d
        
        # VarianceGlow Loss
        if True:
            z, log_s_sum, logdet_w_sum = model_output['varglow']
            z_channels = 6
            z = z.view(z.shape[0], z_channels, -1)
            
            # remove paddings before loss calc
            mask = get_mask_from_lengths(output_lengths)[:, None, :]#   [B, 1, T] BoolTensor
            mask = mask.expand(mask.size(0), z_channels, mask.size(2))# [B, n_mel, T] BoolTensor
            n_elems = (output_lengths_float.sum() * z_channels)
            
            z = torch.masked_select(z, mask)
            var_loss_z = ((z.pow(2).sum()) / self.sigma2_2)/n_elems # mean z (over all elements)
            
            log_s_sum = log_s_sum.view(B, -1, dec_T)
            log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
            var_loss_s = -log_s_sum.sum()/(n_elems)
            
            var_loss_w = -logdet_w_sum.sum()/(z_channels*dec_T)
            
            var_loss_d = var_loss_z+var_loss_w+var_loss_s
            loss = loss + var_loss_d*VarGlow_ls
            del z, log_s_sum, logdet_w_sum, mask, n_elems, z_channels
            loss_dict["Variance_Loss_Z"] = var_loss_z
            loss_dict["Variance_Loss_W"] = var_loss_w
            loss_dict["Variance_Loss_S"] = var_loss_s
            loss_dict["Variance_Loss_Total"] = var_loss_d
        
        loss_dict["loss"] = loss
        return loss_dict
