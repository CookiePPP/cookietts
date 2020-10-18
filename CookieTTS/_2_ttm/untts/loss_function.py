import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d

class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.n_group = hparams.n_group
        sigma = hparams.sigma
        self.sigma2 = sigma ** 2
        self.sigma2_2 = self.sigma2 * 2
        self.dg_sigma2_2 = (hparams.dg_sigma**2)*2
        self.DurGlow_enable = hparams.DurGlow_enable
        self.DurGlow_loss_scalar = hparams.DurGlow_loss_scalar
        self.VarGlow_loss_scalar = hparams.VarGlow_loss_scalar
    
    def forward(self, model_output, targets):
        mel_target, text_lengths, output_lengths, perc_loudness_target, f0_target, energy_target, sylps_target = targets
        batch_size, n_mel_channels, frames = mel_target.shape
        output_lengths_float = output_lengths.float()
        
        # Spectrogram Loss / DecoderGlow Loss
        if True:
            mel_z, log_s_sum, logdet_w_sum = model_output['melglow']
            
            # remove paddings before loss calc
            mask = get_mask_from_lengths(output_lengths)[:, None, :] # [B, 1, T] BoolTensor
            mask = mask.expand(mask.size(0), mel_target.size(1), mask.size(2))# [B, n_mel, T] BoolTensor
            
            n_elems = (output_lengths_float.sum() * n_mel_channels)
            mel_z = torch.masked_select(mel_z, mask)
            dec_loss_z = ((mel_z.pow(2).sum()) / self.sigma2_2)/n_elems # mean z (over all elements)
            
            log_s_sum = log_s_sum.view(batch_size, -1, frames)
            log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
            dec_loss_s = -log_s_sum.sum()/(n_elems)
            
            dec_loss_w = -logdet_w_sum.sum()/(n_mel_channels*frames)
            
            loss = dec_loss_d = dec_loss_z+dec_loss_w+dec_loss_s
            del mel_z, log_s_sum, logdet_w_sum, mask, n_elems
        
        # DurationGlow Loss
        if self.DurGlow_enable:
            dur_z, log_s_sum, logdet_w_sum = model_output['durglow']
            dur_z = dur_z.view(dur_z.shape[0], 2, -1)
            if True:
                # remove paddings before loss calc
                mask = get_mask_from_lengths(text_lengths)[:, None, :] # [B, 1, T] BoolTensor
                mask = mask.expand(mask.size(0), 2, mask.size(2))# [B, 2, T] BoolTensor
            chars = text_lengths.max()
            n_elems = (text_lengths.sum() * 2)
            
            dur_z = torch.masked_select(dur_z, mask)
            dur_loss_z = ((dur_z.pow(2).sum()) / self.dg_sigma2_2)/n_elems # mean z (over all elements)
            
            log_s_sum = log_s_sum.view(batch_size, -1, chars)
            log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
            dur_loss_s = -log_s_sum.sum()/(n_elems)
            
            dur_loss_w = -logdet_w_sum.sum()/(2*chars)
            
            dur_loss_d = dur_loss_z+dur_loss_w+dur_loss_s
            if self.DurGlow_loss_scalar:
                loss = loss + dur_loss_d*self.DurGlow_loss_scalar
            del dur_z, log_s_sum, logdet_w_sum, mask, n_elems, chars
        
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
            
            log_s_sum = log_s_sum.view(batch_size, -1, frames)
            log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
            var_loss_s = -log_s_sum.sum()/(n_elems)
            
            var_loss_w = -logdet_w_sum.sum()/(z_channels*frames)
            
            var_loss_d = var_loss_z+var_loss_w+var_loss_s
            loss = loss + var_loss_d*self.VarGlow_loss_scalar
            del z, log_s_sum, logdet_w_sum, mask, n_elems, z_channels
        
        loss_dict = {
            "loss": loss,
            "Decoder_Loss_Z": dec_loss_z,
            "Decoder_Loss_W": dec_loss_w,
            "Decoder_Loss_S": dec_loss_s,
            "Decoder_Loss_Total": dec_loss_d,
            "Duration_Loss_Z": dur_loss_z,
            "Duration_Loss_W": dur_loss_w,
            "Duration_Loss_S": dur_loss_s,
            "Duration_Loss_Total": dur_loss_d,
            "Variance_Loss_Z": var_loss_z,
            "Variance_Loss_W": var_loss_w,
            "Variance_Loss_S": var_loss_s,
            "Variance_Loss_Total": var_loss_d,
        }
        
        return loss_dict
