import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math
from CookieTTS.utils.model.utils import get_mask_from_lengths, alignment_metric, get_first_over_thresh
from typing import Optional


# https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_tacotron2.py#L28-L156
class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """
    def __init__(self, sigma=0.4, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None
    
    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None
    
    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        B, mel_T, enc_T = self.guided_attn_masks.shape
        losses = self.guided_attn_masks * att_ws[:, :mel_T, :enc_T]
        loss = torch.sum(losses.masked_select(self.masks)) / torch.sum(olens) # get mean along B and mel_T
        if self.reset_always:
            self._reset_masks()
        return loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = ilens.shape[0]
        max_ilen = int(ilens.max().item())
        max_olen = int(olens.max().item())
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen, device=olen.device), torch.arange(ilen, device=ilen.device))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
        """
        in_masks = get_mask_from_lengths(ilens)  # (B, T_in)
        out_masks = get_mask_from_lengths(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

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
        
        # Gate Loss
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        
        self.spec_MSE_weight     = hparams.spec_MSE_weight
        self.spec_MFSE_weight    = hparams.spec_MFSE_weight
        self.postnet_MSE_weight  = hparams.postnet_MSE_weight
        self.postnet_MFSE_weight = hparams.postnet_MFSE_weight
        self.masked_select = hparams.masked_select
        
        self.gate_loss_weight = hparams.gate_loss_weight
        
        # KL Scheduler Params
        if False:
            self.anneal_function = 'logistic'
            self.lag = ''
            self.k = 0.00025
            self.x0 = 1000
            self.upper = 0.005
        elif False:
            self.anneal_function = 'constant'
            self.lag = None
            self.k = None
            self.x0 = None
            self.upper = 0.5 # weight
        else:
            self.anneal_function = 'cycle'
            self.lag = 50#     dead_steps
            self.k = 7950#   warmup_steps
            self.x0 = 10000#  cycle_steps
            self.upper = 1.0 # aux weight
            assert (self.lag+self.k) <= self.x0
        
        self.sylps_kld_weight = hparams.sylps_kld_weight # SylNet KDL Weight
        self.sylps_MSE_weight = hparams.sylps_MSE_weight
        self.sylps_MAE_weight = hparams.sylps_MAE_weight
        
        self.diag_att_weight  = hparams.diag_att_weight
        self.guided_att = GuidedAttentionLoss(sigma=hparams.DiagonalGuidedAttention_sigma)
    
    def vae_kl_anneal_function(self, anneal_function, lag, step, k, x0, upper):
        if anneal_function == 'logistic': # https://www.desmos.com/calculator/neksnpgtmz
            return float(upper/(upper+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            if step > lag:
                return min(upper, step/x0)
            else:
                return 0
        elif anneal_function == 'cycle':
            return min(1,(max(0,(step%x0)-lag))/k) * upper
        elif anneal_function == 'constant':
            return upper or 0.001
    
    def colate_losses(self, loss_dict, loss_scalars, loss=None):
        for k, v in loss_dict.items():
            loss_scale = getattr(self, f'{k}_weight', 1.0)
            loss_scale = loss_scalars[f'{k}_weight'] if (f'{k}_weight' in loss_scalars and loss_scalars[f'{k}_weight'] is not None) else loss_scale
            if loss is not None:
                loss += v*loss_scale
            else:
                loss = v*loss_scale
        loss_dict['loss'] = loss
        return loss_dict
    
    def file_losses(self, loss_dict):
        
        return 
    
    def forward(self, pred, gt, loss_scalars):
        
        loss_dict = {}
        file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        for i in range(B):
            current_time = time.time()
            if gt['audiopath'][i] not in file_losses:
                file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        if True:
            pred_mel_postnet = pred['pred_mel_postnet']
            pred_mel         = pred['pred_mel']
            gt_mel           =   gt['gt_mel']
            
            B, n_mel, mel_T = gt_mel.shape
            
            mask = get_mask_from_lengths(gt['mel_lengths'])
            mask = mask.expand(gt_mel.size(1), *mask.shape).permute(1, 0, 2)
            
            # spectrogram / decoder loss
            pred_mel = torch.masked_select(pred_mel, mask)
            gt_mel   = torch.masked_select(gt_mel, mask)
            spec_SE = nn.MSELoss(reduction='none')(pred_mel, gt_mel)
            loss_dict['spec_MSE'] = spec_SE.mean()
            
            losses = spec_SE.split([x*n_mel for x in gt['mel_lengths'].cpu()])
            for i in range(B):
                audiopath = gt['audiopath'][i]
                file_losses[audiopath]['spec_MSE'] = losses[i].mean().item()
            
            # postnet
            pred_mel_postnet.masked_fill_(~mask, 0.0)
            pred_mel_postnet = torch.masked_select(pred_mel_postnet, mask)
            loss_dict['postnet_MSE'] = nn.MSELoss()(pred_mel_postnet, gt_mel)
            
            # squared by frame, mean postnet
            mask = get_mask_from_lengths(gt['mel_lengths']).unsqueeze(-1)# -> [B, mel_T] -> [B, mel_T, 1]
            
            spec_AE = nn.L1Loss(reduction='none')(pred['pred_mel'], gt['gt_mel']).transpose(1, 2)# -> [B, mel_T, n_mel]
            spec_AE = spec_AE.masked_select(mask).view(gt['mel_lengths'].sum(), n_mel)# -> [B*mel_T, n_mel]
            loss_dict['spec_MFSE'] = (spec_AE * spec_AE.mean(dim=1, keepdim=True)).mean()# multiply by frame means (similar to square op from MSE) and get the mean of the losses
            
            post_AE = nn.L1Loss(reduction='none')(pred['pred_mel_postnet'], gt['gt_mel']).transpose(1, 2)# -> [B, mel_T, n_mel]
            post_AE = post_AE.masked_select(mask).view(gt['mel_lengths'].sum(), n_mel)# -> [B*mel_T, n_mel]
            loss_dict['postnet_MFSE'] = (post_AE * post_AE.mean(dim=1, keepdim=True)).mean()# multiply by frame means (similar to square op from MSE) and get the mean of the losses
        
        if True: # gate/stop loss
            gate_target =  gt['gt_gate_logits'].view(-1, 1)
            gate_out = pred['pred_gate_logits'].view(-1, 1)
            loss_dict['gate_loss'] = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)
            del gate_target, gate_out
        
        if True: # SylpsNet loss
            syl_mu     = pred['pred_sylps_mu']
            syl_logvar = pred['pred_sylps_logvar']
            loss_dict['sylps_kld'] = -0.5 * (1 + syl_logvar - syl_logvar.exp() - syl_mu.pow(2)).sum()/B
            del syl_mu, syl_logvar
        
        if True: # Pred Sylps loss
            pred_sylps = pred['pred_sylps'].squeeze(1)# [B, 1] -> [B]
            sylps_target = gt['gt_sylps']
            loss_dict['sylps_MAE'] = nn.L1Loss()(pred_sylps, sylps_target)
            loss_dict['sylps_MSE'] = nn.MSELoss()(pred_sylps, sylps_target)
            del pred_sylps, sylps_target
        
        if True:# Diagonal Attention Guiding
            alignments     = pred['alignments']
            text_lengths   = gt['text_lengths']
            output_lengths = gt['mel_lengths']
            pres_prev_state= gt['pres_prev_state']
            loss_dict['diag_att'] = self.guided_att(alignments[pres_prev_state==0.0],
                                                  text_lengths[pres_prev_state==0.0],
                                                output_lengths[pres_prev_state==0.0])
            del alignments, text_lengths, output_lengths
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        with torch.no_grad():# get Avg Max Attention and Diagonality Metrics
            
            atd = alignment_metric(pred['alignments'], gt['text_lengths'], gt['mel_lengths'])
            diagonalitys, avg_prob, char_max_dur, char_min_dur, char_avg_dur, p_missing_enc = atd.values()
            
            loss_dict['diagonality']       = diagonalitys.mean()
            loss_dict['avg_max_attention'] = avg_prob.mean()
            
            for i in range(B):
                audiopath = gt['audiopath'][i]
                file_losses[audiopath]['avg_max_attention'] =      avg_prob[i].cpu().item()
                file_losses[audiopath]['att_diagonality'  ] =  diagonalitys[i].cpu().item()
                file_losses[audiopath]['p_missing_enc']     = p_missing_enc[i].cpu().item()
                file_losses[audiopath]['char_max_dur']      =  char_max_dur[i].cpu().item()
                file_losses[audiopath]['char_min_dur']      =  char_min_dur[i].cpu().item()
                file_losses[audiopath]['char_avg_dur']      =  char_avg_dur[i].cpu().item()
            
            pred_gate = pred['pred_gate_logits'].sigmoid()
            pred_gate[:, :5] = 0.0
            # Get inference alignment scores
            pred_mel_lengths = get_first_over_thresh(pred_gate, 0.7)
            atd = alignment_metric(pred['alignments'], gt['text_lengths'], pred_mel_lengths)
            atd = {k: v.cpu() for k, v in atd.items()}
            diagonalitys, avg_prob, char_max_dur, char_min_dur, char_avg_dur, p_missing_enc = atd.values()
            scores = []
            for i in range(B):
                # factors that make up score
                weighted_score = avg_prob[i].item() # general alignment quality
                diagonality_punishment = max( diagonalitys[i].item()-1.10, 0) * 0.25 # speaking each letter at a similar pace.
                max_dur_punishment     = max( char_max_dur[i].item()-60.0, 0) * 0.005# getting stuck on same letter for 0.5s
                min_dur_punishment     = max(0.00-char_min_dur[i].item(),  0) * 0.5  # skipping single enc outputs
                avg_dur_punishment     = max(3.60-char_avg_dur[i].item(),  0)        # skipping most enc outputs
                mis_dur_punishment     = max(p_missing_enc[i].item()-0.08, 0) if gt['text_lengths'][i] > 12 and gt['mel_lengths'][i] < gt['mel_lengths'].max()*0.75 else 0.0 # skipping some percent of the text
                
                weighted_score -= (diagonality_punishment+max_dur_punishment+min_dur_punishment+avg_dur_punishment+mis_dur_punishment)
                scores.append(weighted_score)
                file_losses[audiopath]['att_score'] = weighted_score
            scores = torch.tensor(scores)
            scores[torch.isnan(scores)] = scores[~torch.isnan(scores)].mean()
            loss_dict['weighted_score'] = scores.to(pred['alignments'].device).mean()
        
        return loss_dict, file_losses

