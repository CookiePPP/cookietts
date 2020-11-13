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

@torch.jit.script
def NormalLLLoss(mu, logvar, target):
    loss = ((mu-target).pow(2)/logvar.exp())+logvar
    if True:
        loss = loss.mean()
    else:
        pass
    return loss

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
        self.Sylps_loss_scalar   = hparams.Sylps_loss_scalar
    
    def forward(self, model_output, targets, loss_scalars):
        # loss scalars
        MelGlow_ls = loss_scalars['MelGlow_ls'] if loss_scalars['MelGlow_ls'] is not None else self.MelGlow_loss_scalar
        DurGlow_ls = loss_scalars['DurGlow_ls'] if loss_scalars['DurGlow_ls'] is not None else self.DurGlow_loss_scalar
        VarGlow_ls = loss_scalars['VarGlow_ls'] if loss_scalars['VarGlow_ls'] is not None else self.VarGlow_loss_scalar
        Sylps_ls   = loss_scalars['Sylps_ls'  ] if loss_scalars['Sylps_ls'  ] is not None else self.Sylps_loss_scalar
        
        # loss_func
        mel_target, text_lengths, output_lengths, perc_loudness_target, f0_target, energy_target, sylps_target, voiced_mask, char_f0, char_voiced, char_energy, *_ = targets
        B, n_mel, dec_T = mel_target.shape
        enc_T = text_lengths.max()
        output_lengths_float = output_lengths.float()
        
        loss_dict = {}
        
        # Decoder / MelGlow Loss
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
            assert not (torch.isnan(loss) | torch.isinf(loss)).any(), 'Inf/NaN Loss at MelGlow Latents'
        
        # CVarGlow Loss
        if True:
            z, log_s_sum, logdet_w_sum = model_output['cvarglow']
            _ = glow_loss(z, log_s_sum, logdet_w_sum, text_lengths, self.dg_sigma2_2)
            cvar_loss_d, cvar_loss_z, cvar_loss_w, cvar_loss_s = _
            
            if self.DurGlow_loss_scalar:
                loss = loss + cvar_loss_d*DurGlow_ls
            del z, log_s_sum, logdet_w_sum
            loss_dict["CVar_Loss_Z"] = cvar_loss_z
            loss_dict["CVar_Loss_W"] = cvar_loss_w
            loss_dict["CVar_Loss_S"] = cvar_loss_s
            loss_dict["CVar_Loss_Total"] = cvar_loss_d
            assert not (torch.isnan(loss) | torch.isinf(loss)).any(), 'Inf/NaN Loss at CVarGlow Latents'
        
        # FramGlow Loss
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
            assert not (torch.isnan(loss) | torch.isinf(loss)).any(), 'Inf/NaN Loss at VarGlow Latents'
        
        # Sylps Loss
        if True:
            enc_global_outputs, sylps = model_output['sylps']# [B, 2], [B]
            mu, logvar = enc_global_outputs.transpose(0, 1)[:2, :]# [2, B]
            
            loss_dict["zSylps_Loss"] = NormalLLLoss(mu, logvar, sylps)# [B], [B], [B] -> [B]
            loss = loss + loss_dict["zSylps_Loss"]*Sylps_ls
            del mu, logvar, enc_global_outputs, sylps
            assert not (torch.isnan(loss) | torch.isinf(loss)).any(), 'Inf/NaN Loss at Pred Sylps'
        
        # Perceived Loudness Loss
        if True:
            enc_global_outputs, perc_loudness = model_output['perc_loud']# [B, 2], [B]
            mu, logvar = enc_global_outputs.transpose(0, 1)[2:4, :]# [2, B]
            
            loss_dict["zPL_Loss"] = NormalLLLoss(mu, logvar, perc_loudness)# [B], [B], [B] -> [B]
            loss = loss + loss_dict["zPL_Loss"]*Sylps_ls
            del mu, logvar, enc_global_outputs, perc_loudness
            assert not (torch.isnan(loss) | torch.isinf(loss)).any(), 'Inf/NaN Loss at Pred Perceived Loudness'
        
        loss_dict["loss"] = loss
        return loss_dict
