import time
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import inspect
import torch.distributed as dist
from typing import Optional
import numpy as np
import math
from CookieTTS.utils.model.utils import get_mask_from_lengths, alignment_metric, get_first_over_thresh, freeze_grads
from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d
from CookieTTS._2_ttm.GANSpeech2.model import discriminator_loss

@torch.jit.script
def viterbi(log_prob_matrix, text_lengths, mel_lengths, pad_mag:float=1e12):
    B, L, T = log_prob_matrix.size()# [B, txt_T, mel_T]
    log_beta = torch.ones(B, L, T, device=log_prob_matrix.device, dtype=log_prob_matrix.dtype)*(-pad_mag)
    log_beta[:, 0, 0] = log_prob_matrix[:, 0, 0]
    
    for t in range(1, T):
        prev_step = torch.cat([log_beta[:, :, t-1:t], F.pad(log_beta[:, :, t-1:t], (0,0,1,-1), value=-pad_mag)], dim=-1).max(dim=-1)[0]
        log_beta[:, :, t] = prev_step+log_prob_matrix[:, :, t]
    
    curr_rows = text_lengths-1
    curr_cols = mel_lengths-1
    path = [curr_rows*1.0]
    for _ in range(T-1):
        is_go = log_beta[torch.arange(B), (curr_rows-1).to(torch.long), (curr_cols-1).to(torch.long)]\
                 > log_beta[torch.arange(B), (curr_rows).to(torch.long), (curr_cols-1).to(torch.long)]
        curr_rows = F.relu(curr_rows-1.0*is_go+1.0)-1.0
        curr_cols = F.relu(curr_cols-1+1.0)-1.0
        path.append(curr_rows*1.0)
    
    path.reverse()
    path = torch.stack(path, -1)
    
    indices = torch.arange(path.max()+1).view(1,1,-1).to(path) # 1, 1, L
    align = (indices==path.unsqueeze(-1)).to(path) # B, T, L
    
    for i in range(align.size(0)):
        pad= T-int(mel_lengths[i].item())
        align[i] = F.pad(align[i], (0,0,-pad,pad))
    
    return align.transpose(1,2)# [B, txt_T, mel_T]

# https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_tacotron2.py#L28-L156
class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """
    def __init__(self, sigma=0.5, pow=1.5, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.pow = pow
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
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma, self.pow)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma, pow):
        """Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen, device=olen.device), torch.arange(ilen, device=ilen.device))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        at_mask = 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))
        return at_mask.pow(pow)/at_mask.max()

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


class Loss(nn.Module):
    def __init__(self, h):
        super(Loss, self).__init__()
        self.rank       = h.rank
        self.n_gpus     = h.n_gpus
        self.n_frames_per_step = h.n_frames_per_step
        self.pos_weight = torch.tensor(h.gate_pos_weight)
        self.d_acc_weight   = 0.0
        self.d_acc_r_weight = 0.0
        self.d_acc_g_weight = 0.0
        
        self.guided_att = GuidedAttentionLoss()
        
        for thresh in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            setattr(self, f'ga_att_attscore_gt_{thresh*100.:.0f}_weight', 0.0)
        
        self.fast_discriminator_mode = h.fast_discriminator_mode
        self.GAN_enable = h.GAN_enable
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def colate_losses(self, loss_dict, loss_scalars, loss=None, loss_key_append=''):
        for k, v in loss_dict.items():
            loss_scale = loss_scalars.get(f'{k}_weight', None)
            
            if loss_scale is None:
                loss_scale = getattr(self, f'{k}_weight', None)
            
            if loss_scale is None:
                loss_scale = 1.0
                print(f'{k} is missing loss weight')
            
            if True:#loss_scale > 0.0: # with optimizer(..., inputs=params) this is no longer improves backprop performance.
                new_loss = v*loss_scale
                if new_loss > 40.0 or math.isnan(new_loss) or math.isinf(new_loss):
                    print(f'{k} reached {v}.')
                if loss is not None:
                    loss = loss + new_loss
                else:
                    loss = new_loss
            if False and self.rank == 0:
                print(f'{k:20} {loss_scale:05.2f} {loss:05.2f} {loss_scale*v:+010.6f}', v)
        loss_dict['loss'+loss_key_append] = loss or None
        return loss_dict
    
    def d_forward(self, iteration, model, pred, gt, loss_scalars, loss_params, file_losses=None):
        loss_dict = {}
        if file_losses is None:
            file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        
        #with torch.no_grad():
        #    current_time = time.time()
        #    for i in range(B):
        #        if gt['audiopath'][i] not in file_losses:
        #            file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        alignment = gt.get('alignment', None)
        if alignment is None:
            alignment = pred['hard_alignments']
        
        if self.GAN_enable:
            if self.fast_discriminator_mode:
                dur_loss = discriminator_loss(pred['pred_dur_realness'], pred['gt_dur_realness'], gt['text_lengths'])
                f0s_loss = discriminator_loss(pred['pred_f0s_realness'], pred['gt_f0s_realness'], gt[ 'mel_lengths'])
                mel_loss = discriminator_loss(pred['pred_mel_realness'], pred['gt_mel_realness'], gt[ 'mel_lengths'])
            else:
                logf0s_mask = (gt['gt_frame_voiceds'] > 0.5) & (pred['pred_voiced'] > 0.5)# [B, f0s_dim, mel_T]
                
                gt  [  'gt_logf0s_gated'] =   gt['gt_frame_logf0s']*logf0s_mask# [B, f0s_dim, mel_T]
                pred['pred_logf0s_gated'] = pred[    'pred_logf0s']*logf0s_mask# [B, f0s_dim, mel_T]
                dtype = next(model.parameters()).dtype
                
                _ = model.discriminator.discriminator_loss(gt, pred['encoder_outputs'].detach().to(dtype),
                                                           pred['pred_durlogf0voiced'].detach().to(dtype),
                                                           pred['pred_logf0s_gated'].detach().to(dtype),
                                                           pred['pred_mel'].detach().to(dtype),
                                                           alignment.detach())
                dur_loss, f0s_loss, mel_loss, (
                 (pred_dur_realness, gt_dur_realness),
                 (pred_f0s_realness, gt_f0s_realness),
                 (pred_mel_realness, gt_mel_realness)
                ) = _
            loss_dict['d_class_dur'] = dur_loss
            loss_dict['d_class_f0s'] = f0s_loss
            loss_dict['d_class_mel'] = mel_loss
            #loss_dict['d_acc_r']   = (real_fakeness>=0.5).masked_select(r_mask.transpose(1, 2)).sum()/r_mask.sum()
            #loss_dict['d_acc_g']   = (fake_fakeness<=0.5).masked_select(r_mask.transpose(1, 2)).sum()/r_mask.sum()
            #loss_dict['d_acc']   = ((real_fakeness>0.5).masked_select(r_mask.transpose(1, 2)).sum() + (fake_fakeness<=0.5).masked_select(r_mask.transpose(1, 2)).sum())/(2*r_mask.sum())
        
        for lossk, active_thresh in {k.replace('_active_thresh', ''): v for k, v in loss_params.items() if '_active_thresh' in k}.items():
            if lossk in loss_dict.keys() and loss_dict[lossk] < active_thresh:
                loss_dict[lossk] = loss_dict[lossk].detach()+loss_dict[lossk].mul(0.0)
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars, loss_key_append='_d')
        return loss_dict, file_losses
    
    def g_forward(self, iteration, model, pred, gt, loss_scalars, loss_params, file_losses=None, save_alignments=False):
        if file_losses is None:
            file_losses = {}
        loss_dict = {}
        
        alignment = gt.get('alignment', None)
        if alignment is None:
            alignment = pred['hard_alignments']
        
        if 'pred_mel_a' in pred:# Spectrogram Loss
            mask = get_mask_from_lengths(gt['mel_lengths']).unsqueeze(1)
            pred_mel = pred['pred_mel_a'].float()
            gt_mel   =   gt['gt_mel']    .float()
            numel = gt['mel_lengths'].sum()*gt_mel.shape[1]
            
            loss_dict['ga_mel_MAE'] = F. l1_loss(pred_mel, gt_mel, reduction='sum')/numel
            loss_dict['ga_mel_MSE'] = F.mse_loss(pred_mel, gt_mel, reduction='sum')/numel
        
        if 'pred_gate' in pred:# Gate Loss
            gt_gate = gt['gt_gate_logits'][:, :pred['pred_gate'].shape[1]]# [B, mel_T]
            gt_gate[:, -1] = (gt_gate.sum(dim=1) == 0.0).float()          # [B, mel_T][:, -1] = [B]
            
            gt_gate   = gt_gate          .masked_select(mask[:, 0, :pred['pred_gate'].shape[1]])# [B*mel_T]
            pred_gate = pred['pred_gate'].masked_select(mask[:, 0, :pred['pred_gate'].shape[1]])# [B*mel_T]
            loss_dict['ga_gate_BCE'] = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(pred_gate, gt_gate)
            del gt_gate, pred_gate
        
        if 'soft_alignments' in pred:# Diagonal Alignment Loss
            dec_lengths = (gt['mel_lengths']//self.n_frames_per_step).clamp(max=pred['soft_alignments'].shape[1])
            not_truncated  = ~(gt['pres_prev_state'] | gt['cont_next_iter'])
            loss_dict['ga_att_loss'] = self.guided_att(pred['soft_alignments'][not_truncated], gt['text_lengths'][not_truncated], dec_lengths[not_truncated])
        
        if 'soft_alignments' in pred and loss_scalars['ga_hard_att_MSE_weight'] > 0.0 or loss_scalars['ga_hard_att_MSE_weight'] > 0.0:# Hard Attention Loss
            dec_lengths = (gt['mel_lengths']//self.n_frames_per_step).clamp(max=pred['soft_alignments'].shape[1])
            att_mask_e = get_mask_from_lengths(gt['text_lengths'])# [B, txt_T]
            att_mask_d = get_mask_from_lengths(dec_lengths) # [B, mel_T]
            att_mask = att_mask_e.unsqueeze(1) * att_mask_d.unsqueeze(2)# [B, txt_T] * [B, mel_T] -> [B, mel_T, txt_T]
            
            loss_dict['ga_hard_att_MSE'] = (F.mse_loss(pred['soft_alignments'], pred['hard_alignments'], reduction='none')*att_mask).sum()/gt['mel_lengths'].sum()
            loss_dict['ga_hard_att_MAE'] = (F. l1_loss(pred['soft_alignments'], pred['hard_alignments'], reduction='none')*att_mask).sum()/gt['mel_lengths'].sum()
        
        if 'soft_alignments' in pred:# Attention Metrics (AvgMaxAttention, Diagonality, MaxDuration, ...)
            alignments = pred['soft_alignments']
            if not any(loss_scalars.get(x+'_weight', 0.0)>0.0 for x in ['g_att_diagonality','g_att_top1_avg_prob','g_att_top2_avg_prob','g_att_top3_avg_prob','g_att_avg_max_dur','g_att_avg_min_dur','g_att_avg_avg_dur','g_att_avg_missing_enc_p','g_att_avg_attscore']):
                alignments = alignments.detach()# detach alignments if none of the loss terms are being used for gradients
            
            atd = alignment_metric(alignments, input_lengths=gt['text_lengths'], output_lengths=dec_lengths, enc_min_thresh=0.7/self.n_frames_per_step)
            loss_dict['ga_att_diagonality']   = atd["diagonalitys"]     .mean()
            loss_dict['ga_att_top1_avg_prob'] = atd["avg_prob"]         .mean()
            loss_dict['ga_att_top2_avg_prob'] = atd["top2_avg_prob"]    .mean()
            loss_dict['ga_att_top3_avg_prob'] = atd["top3_avg_prob"]    .mean()
            loss_dict['ga_att_avg_max_dur']   = atd["encoder_max_dur"]  .mean()
            loss_dict['ga_att_avg_min_dur']   = atd["encoder_min_dur"]  .mean()
            loss_dict['ga_att_avg_avg_dur']   = atd["encoder_avg_dur"]  .mean()
            loss_dict['ga_att_avg_missing_enc_p'] = atd["p_missing_enc"].mean()
            
            # get attention score
            zero = torch.tensor(0.0, device=atd["diagonalitys"].device)
            diagonality_punishment = torch.maximum(torch.maximum(atd["diagonalitys"]-1.20, 1.00-atd["diagonalitys"]), zero) * 0.5 # speaking at a inconsistent pace
            max_dur_punishment     = torch.maximum((atd["encoder_max_dur"]-(60/self.n_frames_per_step)), zero) * 0.005# getting stuck on same letter for 0.5s
            mis_dur_punishment     = torch.maximum(atd["p_missing_enc"]-0.07, zero)# skipping some portion of the text
            mis_dur_punishment[gt['text_lengths'] < 12] = 0.0
            
            loss_dict['ga_att_avg_attscore'] = atd["top2_avg_prob"].clone() -(diagonality_punishment +max_dur_punishment +mis_dur_punishment)
            
            # get rate of score < [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            # saved as ['g_att_attscore_gt_40', 'g_att_attscore_gt_50', 'g_att_attscore_gt_60', ...]
            for thresh in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                loss_dict[f'ga_att_attscore_gt_{thresh*100.:.0f}'] = (loss_dict['ga_att_avg_attscore'] > thresh).sum()/loss_dict['ga_att_avg_attscore'].numel()
            
            loss_dict['ga_att_avg_attscore'] = loss_dict['ga_att_avg_attscore'].mean()
        
        if 'pred_dur' in pred:# Char Duration Loss
            text_mask = get_mask_from_lengths(gt['text_lengths'])
            pred_dur = pred['pred_dur'].squeeze(1)# [B, 1, txt_T]
            gt_char_dur = alignment.detach().sum(dim=1)# [B, mel_T, txt_T] -> [B, txt_T]
            loss_dict['g_dur_MAE'] = (F. l1_loss(pred_dur, gt_char_dur, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
            loss_dict['g_dur_MSE'] = (F.mse_loss(pred_dur, gt_char_dur, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
        
        if 'pred_char_logf0' in pred:# Char Pitch Loss
            logf0_mask =   gt['gt_char_voiced']>0.01
            text_mask = get_mask_from_lengths(gt['text_lengths'])
            pred_logf0 = pred['pred_char_logf0'].squeeze(1)# [B, 1, txt_T]
            gt_logf0   =   gt[  'gt_char_logf0'].squeeze(1)# [B, 1, txt_T]
            loss_dict['g_logf0_MAE'] = (F. l1_loss(pred_logf0, gt_logf0, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
            loss_dict['g_logf0_MSE'] = (F.mse_loss(pred_logf0, gt_logf0, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
        
        if 'pred_char_voiced' in pred:# Char Voiced Loss
            text_mask = get_mask_from_lengths(gt['text_lengths'])
            pred_voiced = pred['pred_char_voiced'].float().squeeze(1)# [B, 1, txt_T]
            gt_voiced   =   gt[  'gt_char_voiced'].float()
            loss_dict['g_voiced_MAE'] = (            F. l1_loss(pred_voiced, gt_voiced, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
            loss_dict['g_voiced_MSE'] = (            F.mse_loss(pred_voiced, gt_voiced, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
            loss_dict['g_voiced_BCE'] = (F.binary_cross_entropy(pred_voiced, gt_voiced, reduction='none')*text_mask).sum(dtype=torch.float)/text_mask.sum()
        
        if 'pred_logf0s' in pred:# Frame F0s Loss
            loss_dict['g_logf0s_MAE'] = (F. l1_loss(pred['pred_logf0s'], gt['gt_frame_logf0s'], reduction='none')*gt['gt_frame_voiceds']).sum(dtype=torch.float)/(gt['gt_frame_voiceds'].sum()*gt['gt_frame_f0s'].shape[1])
            loss_dict['g_logf0s_MSE'] = (F.mse_loss(pred['pred_logf0s'], gt['gt_frame_logf0s'], reduction='none')*gt['gt_frame_voiceds']).sum(dtype=torch.float)/(gt['gt_frame_voiceds'].sum()*gt['gt_frame_f0s'].shape[1])
        
        if 'pred_voiced' in pred:# Frame Voiced Loss
            mel_mask = get_mask_from_lengths(gt['mel_lengths']).unsqueeze(1)
            pred_voiceds = pred['pred_voiced'     ].float()# [B, n_f0s, mel_T]
            gt_voiceds   =   gt['gt_frame_voiceds'].float()# [B, n_f0s, mel_T]
            loss_dict['g_voiceds_MAE'] = (            F. l1_loss(pred_voiceds, gt_voiceds, reduction='none')*mel_mask).sum(dtype=torch.float)/(mel_mask.sum()*gt['gt_frame_voiceds'].shape[1])
            loss_dict['g_voiceds_MSE'] = (            F.mse_loss(pred_voiceds, gt_voiceds, reduction='none')*mel_mask).sum(dtype=torch.float)/(mel_mask.sum()*gt['gt_frame_voiceds'].shape[1])
            loss_dict['g_voiceds_BCE'] = (F.binary_cross_entropy(pred_voiceds, gt_voiceds, reduction='none')*mel_mask).sum(dtype=torch.float)/(mel_mask.sum()*gt['gt_frame_voiceds'].shape[1])
        
        if 'pred_mel' in pred:# Mel Loss
            mel_mask = get_mask_from_lengths(gt['mel_lengths']).unsqueeze(1)
            loss_dict['g_mel_MAE'] = (F. l1_loss(pred['pred_mel'], gt['gt_mel'], reduction='none')*mel_mask).sum(dtype=torch.float)/(mel_mask.sum()*gt['gt_mel'].shape[1])
            loss_dict['g_mel_MSE'] = (F.mse_loss(pred['pred_mel'], gt['gt_mel'], reduction='none')*mel_mask).sum(dtype=torch.float)/(mel_mask.sum()*gt['gt_mel'].shape[1])
        
        if self.GAN_enable:
            dtype = next(model.parameters()).dtype
            
            if self.fast_discriminator_mode:
                logf0s_mask = (gt['gt_frame_voiceds'] > 0.5) & (pred['pred_voiced'] > 0.5)# [B, f0s_dim, mel_T]
                
                gt  [  'gt_logf0s_gated'] =   gt['gt_frame_logf0s']*logf0s_mask# [B, f0s_dim, mel_T]
                pred['pred_logf0s_gated'] = pred[    'pred_logf0s']*logf0s_mask# [B, f0s_dim, mel_T]
                dtype = next(model.parameters()).dtype
            
            args = (gt, pred['encoder_outputs'].detach().to(dtype),
                        pred['pred_durlogf0voiced'].to(dtype),
                        pred['pred_logf0s_gated'].to(dtype),
                        pred['pred_mel'].to(dtype), alignment.detach())
            if self.fast_discriminator_mode:
                _ = model.generator_loss(*args)
            else:
                with freeze_grads(model.discriminator):
                    _ = model.generator_loss(*args)
            dur_loss, f0s_loss, mel_loss, (
             (pred_dur_realness, gt_dur_realness),
             (pred_f0s_realness, gt_f0s_realness),
             (pred_mel_realness, gt_mel_realness)
            ) = _
            loss_dict['g_class_dur'] = dur_loss
            loss_dict['g_class_f0s'] = f0s_loss
            loss_dict['g_class_mel'] = mel_loss
            if self.fast_discriminator_mode:
                pred['pred_dur_realness'], pred['gt_dur_realness'] = pred_dur_realness, gt_dur_realness
                pred['pred_f0s_realness'], pred['gt_f0s_realness'] = pred_f0s_realness, gt_f0s_realness
                pred['pred_mel_realness'], pred['gt_mel_realness'] = pred_mel_realness, gt_mel_realness
        
        for lossk, active_thresh in {k.replace('_active_thresh', ''): v for k, v in loss_params.items() if '_active_thresh' in k}.items():
            if lossk in loss_dict.keys() and loss_dict[lossk] < active_thresh:
                loss_dict[lossk] = loss_dict[lossk].detach()+loss_dict[lossk].mul(0.0)
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        not_truncated  = ~(gt['pres_prev_state'] | gt['cont_next_iter'])
        for i, not_trunc in enumerate(not_truncated):
            if not_trunc and save_alignments:
                mel_length  = gt['mel_lengths'    ][i]
                text_length = gt['text_lengths'   ][i]
                align = alignment[i, :mel_length, :text_length]# -> [mel_T, txt_T]
                
                audiopath = gt['audiopath'][i]
                is_arpa   = gt['arpa'][i]
                alignpath = os.path.splitext(audiopath)[0]+('_palign.pt' if is_arpa else '_galign.pt')
                torch.save(align.detach().clone().half(), alignpath)
        
        return loss_dict, file_losses

