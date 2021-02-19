import time
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist

from typing import Optional

import math
import numpy as np

from CookieTTS.utils.model.utils import get_mask_from_lengths, alignment_metric, get_first_over_thresh, freeze_grads
from CookieTTS._2_ttm.untts.model import MaskedBatchNorm1d
from piqa.ssim import SSIM

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


# https://discuss.pytorch.org/t/fastest-way-of-converting-a-real-number-to-a-one-hot-vector-representing-a-bin/21578/2
def indexes_to_one_hot(indexes, num_classes=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    orig_shape = indexes.shape
    indexes = indexes.type(torch.int64).view(-1, 1)
    num_classes = num_classes if num_classes is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], num_classes, device=indexes.device).scatter_(1, indexes, 1)
    one_hots = one_hots.view(*orig_shape, -1)
    return one_hots

def get_class_durations(text, alignment, n_symbols):
    # get duration of each symbol. zero for symbols that don't exist in this transcript.
    durations = alignment.sum(1)# [B, mel_T, txt_T] -> [B, txt_T]
    sym_text_onehot = indexes_to_one_hot(text, num_classes=n_symbols)# -> [B, txt_T, n_symbols]
    char_durs = durations.unsqueeze(1) @ sym_text_onehot# [B, 1, txt_T] @ [B, txt_T, n_symbols] -> [B, 1, n_symbols]
    char_durs = char_durs.squeeze(1)# [B, 1, n_symbols] -> [B, n_symbols]
    return char_durs

class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.rank       = hparams.rank
        self.n_frames_per_frame = getattr(hparams, 'n_frames_per_frame', 1)
        self.n_fpf_per_elem     = getattr(hparams, 'n_fpf_per_elem', False)
        
        self.n_symbols  = hparams.n_symbols
        self.n_speakers = hparams.n_speakers
        
        # Gate Loss
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        
        self.SSIM = SSIM(window_size=11, sigma=1.5, n_channels=1, reduction='mean')
        
        self.spec_MSE_weight     = hparams.spec_MSE_weight
        self.spec_MFSE_weight    = hparams.spec_MFSE_weight
        self.postnet_MSE_weight  = hparams.postnet_MSE_weight
        self.postnet_MFSE_weight = hparams.postnet_MFSE_weight
        self.masked_select = hparams.masked_select
        
        self.use_res_enc         = hparams.use_res_enc
        self.use_res_enc_dis     = hparams.use_res_enc_dis
        self.res_enc_kld_weight  = hparams.res_enc_kld_weight
        self.res_enc_gMSE_weight = hparams.res_enc_gMSE_weight
        
        self.use_dbGAN  = hparams.use_dbGAN
        self.use_InfGAN = hparams.use_InfGAN
        
        self.prenet_use_code_loss = getattr(hparams, 'prenet_use_code_loss')#, False)
        if self.prenet_use_code_loss:
            self.prenet_dim = hparams.prenet_dim
            self.gt_code_bn = MaskedBatchNorm1d(hparams.prenet_dim, momentum=0.05, eval_only_momentum=False, affine=False).cuda()
            self.pr_code_bn = MaskedBatchNorm1d(hparams.prenet_dim, momentum=0.05, eval_only_momentum=False, affine=False).cuda()
        
        self.gate_loss_weight = hparams.gate_loss_weight
        
        self.sylps_kld_weight = hparams.sylps_kld_weight # SylNet KLD Weight
        self.sylps_MSE_weight = hparams.sylps_MSE_weight
        self.sylps_MAE_weight = hparams.sylps_MAE_weight
        
        self.diag_att_weight  = hparams.diag_att_weight
        self.guided_att = GuidedAttentionLoss(sigma=hparams.DiagonalGuidedAttention_sigma)
        
        self.HiFiGAN_enable       = getattr(hparams, 'HiFiGAN_enable',       False)
        self.HiFiGAN_segment_size = getattr(hparams, 'HiFiGAN_segment_size', 16384)
        self.HiFiGAN_batch_size   = getattr(hparams, 'HiFiGAN_batch_size'  ,     2)
        self.hop_length           = hparams.hop_length
    
    def best_frames(self, pred_mels, gt_mel, mask, frame_mean=True):# [B, n_mel*n_frames, mel_T], [B, n_mel, mel_T]
        """Output best frames from a set of predicted mels and the target mel."""
        if self.n_frames_per_frame > 1:
            B, n_mel_n_frames, mel_T = pred_mels.shape
            pred_mels = pred_mels.chunk(self.n_frames_per_frame, dim=1)# [[B, n_mel, mel_T],]*n_frames
            aerr_mels = []
            for i in range(self.n_frames_per_frame):
                aerr_mel = F.l1_loss(pred_mels[i].detach(), gt_mel.detach(), reduction='none')
                if frame_mean:
                    aerr_mel = aerr_mel.mean(dim=1, keepdim=True)
                aerr_mels.append(aerr_mel)# -> [B, n_mel, mel_T]
            
            ae_index = aerr_mels[0].new_zeros(aerr_mels[0].shape)
            pred_mel = pred_mels[0]
            aerr_mel = aerr_mels[0]
            for i in range(1, self.n_frames_per_frame):
                pred_mel = pred_mel.where(aerr_mel < aerr_mels[i], pred_mels[i])
                aerr_mel = aerr_mel.where(aerr_mel < aerr_mels[i], aerr_mels[i])
                ae_index[~(aerr_mel < aerr_mels[i])] = float(i)
            
            masked_ae_index = ae_index.masked_select(mask)
            weights = []
            for i in range(self.n_frames_per_frame):
                p_best = float((masked_ae_index==float(i)).sum().item())/masked_ae_index.numel()
                weights.append((1/(p_best+1e-6))/self.n_frames_per_frame)
                if p_best == 0.:
                    print(f"Warning! p_best of projection {i} has no good outputs. If this happens constantly then this model is dead.")
                    continue
            
            ae_weights = aerr_mels[0].new_zeros(aerr_mels[0].shape)
            for i in range(self.n_frames_per_frame):
                ae_weights[ae_index==float(i)] = weights[i]
        else:
            pred_mel   = pred_mels
            ae_index   = pred_mels.new_zeros(pred_mels[0].shape)
            ae_weights = pred_mels.new_ones (pred_mels[0].shape)
            if frame_mean:
                ae_index   = ae_index  [:, 0:1, :]
                ae_weights = ae_weights[:, 0:1, :]
        return pred_mel, ae_index, ae_weights# [B, n_mel, mel_T], [B, n_mel, mel_T], [B, n_mel, mel_T]
    
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
            loss_scale = loss_scalars.get(f'{k}_weight', None)
            if loss_scale is None:
                if 'all_losses' in loss_scalars:
                    loss_scale = loss_scalars.get('all_losses')
                else:
                    loss_scale = getattr(self, f'{k}_weight', 1.0)
                    print(f'{k} is missing loss weight')
            if loss_scale > 0.0:
                new_loss = v*loss_scale
                if new_loss > 1000.0:
                    print(f'{k} reached {v}.')
                if loss is not None:
                    loss = loss + new_loss
                else:
                    loss = new_loss
            if False and self.rank == 0:
                print(f'{k:20} {loss_scale:05.2f} {loss:05.2f} {loss_scale*v:+010.6f}', v)
        loss_dict['loss'] = loss
        return loss_dict
    
    def forward(self, model, pred, gt, loss_scalars, resGAN=None, dbGAN=None, infGAN=None, hifiGAN=None,):
        loss_dict = {}
        file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        tfB = B//(model.decoder.half_inference_mode+1)
        for i in range(tfB):
            current_time = time.time()
            if gt['audiopath'][i] not in file_losses:
                file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        if True:
            pred_mel_postnet = pred['pred_mel_postnet']# [B, n_mel, mel_T]
            pred_mels        = pred['pred_mel']        # [B, n_mel*n_frames, mel_T]
            gt_mel           =   gt['gt_mel']          # [B, n_mel, mel_T]
            mel_lengths      =   gt['mel_lengths']     # [B]
            
            # fill padded with zeros
            mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)# [B, 1, mel_T]
            pred_mel_postnet.masked_fill_(~mask, 0.0)
            pred_mels       .masked_fill_(~mask, 0.0)
            
            # check for nan/infs
            with torch.no_grad():
                pred_mels       .masked_fill_(torch.isnan(pred_mels       ) | torch.isinf(pred_mels       ), 0.0)# debug?
                pred_mel_postnet.masked_fill_(torch.isnan(pred_mel_postnet) | torch.isinf(pred_mel_postnet), 0.0)# debug?
                assert not torch.isnan(pred_mels).any(), 'mel has NaNs'
                assert not torch.isinf(pred_mels).any(), 'mel has Infs'
                assert not torch.isnan(pred_mel_postnet).any(), 'mel has NaNs'
                assert not torch.isinf(pred_mel_postnet).any(), 'mel has Infs'
            
            # split tensor into teacher-forced items only if required
            if model.decoder.half_inference_mode:
                pred_mel_postnet = pred_mel_postnet.chunk(2, dim=0)[0]
                pred_mels        = pred_mels       .chunk(2, dim=0)[0]
                gt_mel           = gt_mel          .chunk(2, dim=0)[0]
                mel_lengths      = mel_lengths     .chunk(2, dim=0)[0]
                mask             = mask            .chunk(2, dim=0)[0]
            B, n_mel, mel_T = gt_mel.shape
            
            teacher_force_till = loss_scalars.get('teacher_force_till',   0)
            p_teacher_forcing  = loss_scalars.get('p_teacher_forcing' , 1.0)
            if p_teacher_forcing == 0.0 and teacher_force_till > 1:
                gt_mel           = gt_mel          [:, :, :teacher_force_till]
                pred_mels        = pred_mels       [:, :, :teacher_force_till]
                pred_mel_postnet = pred_mel_postnet[:, :, :teacher_force_till]
                mel_lengths      = mel_lengths.clamp(max=teacher_force_till)
            
            # get best combo of frames if using multiframe outputs
            pred_mel, ae_index, ae_weights = self.best_frames(pred_mels, gt_mel, mask, frame_mean=not self.n_fpf_per_elem)# -> [B, n_mel, mel_T]
            pred['pred_mel_b'] = pred_mel
            
            # spectrogram / decoder loss
            gt_mels_selected = gt_mel.repeat(1, self.n_frames_per_frame, 1).masked_select(mask)
            pred_mels_selected = pred_mels.masked_select(mask)
            loss_dict['specs_MSE' ] = F.mse_loss      (pred_mels_selected, gt_mels_selected)
            loss_dict['specs_MSAE'] = F.smooth_l1_loss(pred_mels_selected, gt_mels_selected)
            
            pred_mel_selected   = torch.masked_select(pred_mel,   mask)# -> [B*n_mel*mel_T]
            gt_mel_selected     = torch.masked_select(gt_mel,     mask)# -> [B*n_mel*mel_T]
            ae_weights_selected = torch.masked_select(ae_weights, mask)# -> [B*n_mel*mel_T]
            spec_SE = nn.MSELoss(reduction='none')(pred_mel_selected, gt_mel_selected)
            loss_dict['spec_MSE'] = spec_SE.mean()
            
            spec_SAE = nn.SmoothL1Loss(reduction='none')(pred_mel_selected, gt_mel_selected)
            loss_dict['spec_MSAE'] = spec_SAE.mean()
            
            losses = spec_SE.split([x*n_mel for x in mel_lengths.cpu()])
            for i in range(tfB):
                audiopath = gt['audiopath'][i]
                file_losses[audiopath]['spec_MSE'] = losses[i].mean().item()
            
            # weighted multi-frame MSE
            numfr = mel_lengths.sum().item()
            loss_dict['spec_MSE_MF' ] = (spec_SE .view(numfr, n_mel)*ae_weights_selected.view(numfr, -1)).mean()
            loss_dict['spec_MSAE_MF'] = (spec_SAE.view(numfr, n_mel)*ae_weights_selected.view(numfr, -1)).mean()
            
            # SSIM
            #self.SSIM.to(gt_mel)
            #loss_dict['spec_SSIM']    = 1.0-self.SSIM(pred_mel        .unsqueeze(1), gt_mel.unsqueeze(1))# [B, 1, n_mel, mel_T]
            #loss_dict['postnet_SSIM'] = 1.0-self.SSIM(pred_mel_postnet.unsqueeze(1), gt_mel.unsqueeze(1))# [B, 1, n_mel, mel_T]
            
            # postnet
            pred_mel_postnet_selected = torch.masked_select(pred_mel_postnet, mask)
            loss_dict['postnet_MSE' ] = nn.MSELoss()(pred_mel_postnet_selected, gt_mel_selected)
            loss_dict['postnet_MSAE'] = nn.SmoothL1Loss()(pred_mel_postnet_selected, gt_mel_selected)
            
            # squared by frame, mean postnet
            mask = mask.transpose(1, 2)[:, :, :1]# [B, mel_T, n_mel] -> [B, mel_T, 1]
            
            spec_AE = nn.L1Loss(reduction='none')(pred_mel, gt_mel).transpose(1, 2)# -> [B, mel_T, n_mel]
            spec_AE = spec_AE.masked_select(mask).view(numfr, n_mel)               # -> [B *mel_T, n_mel]
            spec_FSE = spec_AE * spec_AE.mean(dim=1, keepdim=True)# multiply by frame means (similar to square op from MSE) 
            loss_dict['spec_MFSE'] = spec_FSE.mean()              # and get the mean of the losses
            
            loss_dict['spec_MFSE_MF'] = (spec_FSE.view(numfr, n_mel)*ae_weights_selected.view(numfr, -1)).mean()
            
            post_AE = nn.L1Loss(reduction='none')(pred_mel_postnet, gt_mel).transpose(1, 2)# -> [B, mel_T, n_mel]
            post_AE = post_AE.masked_select(mask).view(mel_lengths.sum(), n_mel)# -> [B*mel_T, n_mel]
            loss_dict['postnet_MFSE'] = (post_AE * post_AE.mean(dim=1, keepdim=True)).mean()# multiply by frame means (similar to square op from MSE) and get the mean of the losses
            del gt_mel, spec_AE, post_AE,#pred_mel_postnet, pred_mel
        
        if self.n_frames_per_frame > 1:
            mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)# [B, 1, mel_T]
            
            mel_outputs_logprob = pred['mel_outputs_logprob']# [B, n_frames*(n_mel OR 1), mel_T]
            mel_outputs_logprob = torch.stack(mel_outputs_logprob.chunk(self.n_frames_per_frame, dim=1), dim=-1)# [B, n_frames*(n_mel OR 1), mel_T] -> [B, (n_mel OR 1), mel_T, n_frames]
            mel_outputs_logprob = mel_outputs_logprob.masked_select(mask.unsqueeze(-1)).view(-1, self.n_frames_per_frame)# [B, (n_mel OR 1), mel_T, n_frames] -> [B*(n_mel OR 1)*mel_T, n_frames]
            ae_index_selected = ae_index.masked_select(mask).round().long()# [B*(n_mel OR 1)*mel_T]
            
            loss_dict['MF_CE_loss'] = nn.CrossEntropyLoss()(mel_outputs_logprob, ae_index_selected)# [B*(n_mel OR 1)*mel_T, n_frames], [B*(n_mel OR 1)*mel_T]
            
            #mel_outputs_prob = F.softmax(mel_outputs_logprob, dim=1)# <-- how to calculate the probabilities
        
        if True: # gate/stop loss
            gate_target =   gt['gt_gate_logits'  ]
            gate_out    = pred['pred_gate_logits']
            if model.decoder.half_inference_mode:
                gate_target = gate_target.chunk(2, dim=0)[0]
                gate_out    = gate_out   .chunk(2, dim=0)[0]
            gate_target = gate_target.view(-1, 1)
            gate_out    = gate_out   .view(-1, 1)
            
            loss_dict['gate_loss'] = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)
            del gate_target, gate_out
        
        if True: # SylpsNet loss
            syl_mu     = pred['pred_sylps_mu']
            syl_logvar = pred['pred_sylps_logvar']
            if model.decoder.half_inference_mode:
                syl_logvar  = syl_logvar.chunk(2, dim=0)[0]
                syl_mu      = syl_mu    .chunk(2, dim=0)[0]
            loss_dict['sylps_kld'] = -0.5 * (1 + syl_logvar - syl_logvar.exp() - syl_mu.pow(2)).sum()/B
            del syl_mu, syl_logvar
        
        if True: # Pred Sylps loss
            pred_sylps = pred['pred_sylps'].squeeze(1)# [B, 1] -> [B]
            sylps_target = gt['gt_sylps']
            if model.decoder.half_inference_mode:
                pred_sylps      = pred_sylps     .chunk(2, dim=0)[0]
                sylps_target    = sylps_target   .chunk(2, dim=0)[0]
            loss_dict['sylps_MAE'] =  nn.L1Loss()(pred_sylps, sylps_target)
            loss_dict['sylps_MSE'] = nn.MSELoss()(pred_sylps, sylps_target)
            del pred_sylps, sylps_target
        
        if 'mdn_loss' in pred:
            safe_mdn_ids = ~(gt['pres_prev_state'] | gt['cont_next_iter']) & (pred['mdn_loss'].detach() < 1e3)# BoolTensor[B]
            loss_dict['mdn_loss'] = pred['mdn_loss'][safe_mdn_ids].mean()
        
        if ('pred_logdur' in pred) and ("mdn_alignment" in pred):
            pred_logdur = pred['pred_logdur'].squeeze(-1)# [B, txt_T]
            mask = get_mask_from_lengths(gt['text_lengths'])# [B, txt_T]
            loss_dict['dur_loss'] = F.mse_loss(pred['mdn_alignment'].sum(dim=1).detach().clamp(min=0.5).log().masked_fill(~mask, 0.0),
                                                                                                  pred_logdur.masked_fill(~mask, 0.0))# [B, txt_T], [B, txt_T] -> []
        
        if 'mdn_alignment' in pred:
            gt_alignments   = pred['mdn_alignment'][safe_mdn_ids]
            pred_alignments = pred['alignments'   ][safe_mdn_ids]
            abs_err = F.mse_loss(pred_alignments, gt_alignments, reduction='none')# [B, mel_T, txt_T]
            loss_dict['gt_align_mse'] = abs_err.sum(2).mean()
        
        if True:# Diagonal Attention Guiding
            alignments     = pred['alignments'  ]
            text_lengths   =   gt['text_lengths']
            output_lengths =   gt['mel_lengths' ]
            not_truncated  = ~(gt['pres_prev_state'] | gt['cont_next_iter'])
            if model.decoder.half_inference_mode:
                alignments      = alignments    .chunk(2, dim=0)[0]
                text_lengths    = text_lengths  .chunk(2, dim=0)[0]
                output_lengths  = output_lengths.chunk(2, dim=0)[0]
                not_truncated   = not_truncated .chunk(2, dim=0)[0]
            if not_truncated.sum():
                loss_dict['diag_att'] = self.guided_att(alignments[not_truncated],
                                                      text_lengths[not_truncated],
                                                    output_lengths[not_truncated])
            else:
                loss_dict['diag_att'] = alignments.new_tensor(0.028)
            del alignments, text_lengths, output_lengths, not_truncated
        
        if self.use_res_enc and resGAN is not None:# Residual Encoder KL Divergence Loss
            mu, logvar, mulogvar = pred['res_enc_pkg']
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_dict['res_enc_kld'] = kl_loss
            
            if self.use_res_enc_dis:
                # discriminator attempts to predict the letters and speakers using the residual latent space,
                #  the generator attempts to increase the discriminators loss so the latent space will lose speaker and dur info
                #   making it more likely the latent contains information relating to background noise conditions and
                #    other features more relavent to human interests.
                with torch.no_grad():
                    gt_speakers   = gt['speaker_id_onehot'].float() # [B, n_speakers]
                    gt_sym_durs   = get_class_durations(gt['text'], pred['alignments'].detach(), self.n_symbols)# [B, n_symbols]
                out = resGAN.discriminator(mulogvar)# learns to predict the speaker and
                B = out.shape[0]
                pred_sym_durs, pred_speakers = out.squeeze(-1).split([self.n_symbols, self.n_speakers], dim=1) # amount of 'a','b','c','.', etc sounds that are in the audio.
                                                                          # if there isn't a 'd' sound in the transcript, then d will be 0.0
                                                                          # if there are multiple 'a' sounds, their durations are summed.
                pred_speakers = torch.nn.functional.softmax(pred_speakers, dim=1)
                loss_dict['res_enc_gMSE'] = (nn.MSELoss(reduction='sum')(pred_sym_durs, gt_sym_durs.mean(dim=1, keepdim=True))*0.0001 + nn.MSELoss(reduction='sum')(pred_speakers, gt_speakers.mean(dim=1, keepdim=True)))/B
                
                resGAN.gt_speakers = gt_speakers
                resGAN.gt_sym_durs = gt_sym_durs
                del gt_speakers, gt_sym_durs, pred_sym_durs, pred_speakers
            
            del mu, logvar, kl_loss, mulogvar
        
        if 1 and model.training and self.use_dbGAN and dbGAN is not None:
            pred_mel_postnet = pred['pred_mel_postnet'].unsqueeze(1)# -> [  B, 1, n_mel, mel_T]
            speaker_embed    = pred['speaker_embed']# -> [B, embed]
            if model.decoder.half_inference_mode:
                pred_mel_postnet = pred_mel_postnet.chunk(2, dim=0)[0]# -> [tfB, 1, n_mel, mel_T]
                speaker_embed    = speaker_embed   .chunk(2, dim=0)[0]# -> [tfB, embed]
            B, _, n_mel, mel_T = pred_mel_postnet.shape
            mels = torch.cat((pred_mel.unsqueeze(1), pred_mel_postnet), dim=0).float()# [2*B, 1, n_mel, mel_T]
            with torch.no_grad():
                assert not (torch.isnan(mels) | torch.isinf(mels)).any(), 'NaN or Inf value found in computation'
            
#            if False:
#                pred_fakeness = checkpoint(dbGAN.discriminator, mels, speaker_id.repeat(2)).squeeze(1)# -> [2*B, mel_T//?]
#            else:
            pred_fakeness = dbGAN.discriminator(mels, speaker_embed.repeat(2, 1)).squeeze(1)# -> [2*B, mel_T//?]
            pred_fakeness, postnet_fakeness = pred_fakeness.chunk(2, dim=0)# -> [B, mel_T//?], [B, mel_T//?]
            
            tfB, post_mel_T = pred_fakeness.shape
            real_label = torch.ones(tfB, post_mel_T, device=pred_mel.device, dtype=pred_mel.dtype)*-1.0# [B]
            loss_dict['dbGAN_gLoss'] = F.mse_loss(pred_fakeness, real_label)*0.5 + F.mse_loss(postnet_fakeness, real_label)*0.5
            with torch.no_grad():
                assert not torch.isnan(loss_dict['dbGAN_gLoss']), 'dbGAN loss is NaN'
                assert not torch.isinf(loss_dict['dbGAN_gLoss']), 'dbGAN loss is Inf'
            del mels, real_label, pred_fakeness, postnet_fakeness, pred_mel_postnet, speaker_embed
        
        if self.use_InfGAN and infGAN is not None and model.decoder.half_inference_mode:
            with torch.no_grad():
                pred_gate = pred['pred_gate_logits'].chunk(2, dim=0)[1].sigmoid()
                pred_gate[:, :5] = 0.0
                # Get inference alignment scores
                pred_mel_lengths = get_first_over_thresh(pred_gate, 0.5)
                pred_mel_lengths.clamp_(max=mel_T)
                pred['pred_mel_lengths'] = pred_mel_lengths
                mask = get_mask_from_lengths(pred_mel_lengths, max_len=mel_T).unsqueeze(1)# [B, 1, mel_T]
            
            tfB = pred_gate.shape[0]
            with freeze_grads(model.decoder.prenet):
                args = infGAN.merge_inputs(model, pred, gt, tfB, mask)# [B/2, mel_T, embed]
            
            if infGAN.training and infGAN.gradient_checkpoint:
                inf_infness = checkpoint(infGAN.discriminator, *args).squeeze(1)# -> [B/2, mel_T]
            else:
                inf_infness = infGAN.discriminator(*args).squeeze(1)# -> [B/2, mel_T]
            
            tf_label = torch.ones(tfB, device=pred_gate.device, dtype=pred_gate.dtype)[:, None].expand(tfB, mel_T)*-1.# [B/2]
            loss_dict['InfGAN_gLoss'] = 2.*F.mse_loss(inf_infness, tf_label)
        
        if "var_mu" in pred:# KLD for Variational Encoder
            mu, logvar = pred["var_mu"], pred["var_logvar"]
            loss_dict['VE_KLD'] = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.prenet_use_code_loss:# Encoded Prenet Loss
            dtype = next(model.decoder.prenet.parameters()).dtype
            if model.decoder.half_inference_mode:
                gt_mel           =   gt['gt_mel'       ].chunk(2, dim=0)[0].to(dtype)
                mel_lengths      =   gt['mel_lengths'  ].chunk(2, dim=0)[0].long()
                speaker_embed    = pred['speaker_embed'].chunk(2, dim=0)[0].to(dtype)
            else:
                gt_mel           =   gt['gt_mel'       ].to(dtype)
                mel_lengths      =   gt['mel_lengths'  ].long()
                speaker_embed    = pred['speaker_embed'].to(dtype)
            
            mask = get_mask_from_lengths(mel_lengths, max_len=gt_mel.shape[2]).transpose(0, 1).unsqueeze(-1)# -> [mel_T, B, 1]
            tfB, n_mel, mel_T = gt_mel.shape
            with freeze_grads(model.decoder.prenet):
                with torch.no_grad():
                    with torch.random.fork_rng(devices=[0,]):
                        gt_pn_code = model.decoder.prenet(gt_mel.permute(2, 0, 1), speaker_embed, disable_dropout=True).detach()# [B, n_mel, mel_T] -> [mel_T, B, prenet_dim]
                    gt_pn_code = torch.masked_select(gt_pn_code, mask).view(-1, self.prenet_dim, 1)# [mel_T*B, n_mel, 1]
                    gt_pn_code = self.gt_code_bn(gt_pn_code)# [mel_T, B, prenet_dim]
            with torch.random.fork_rng(devices=[0,]):
                pred_pn_code = model.decoder.prenet(pred_mel.to(dtype).permute(2, 0, 1), speaker_embed, disable_dropout=True)# [B, n_mel, mel_T] -> [mel_T, B, prenet_dim]
            pred_pn_code = torch.masked_select(pred_pn_code, mask).view(-1, self.prenet_dim, 1)# [mel_T*B, n_mel, 1]
            pred_pn_code = self.pr_code_bn(pred_pn_code)
            loss_dict['prenet_code_MAE'] = F.l1_loss(gt_pn_code, pred_pn_code)
        
        if self.HiFiGAN_enable:
            # get (indexes) items in batch to use for HiFiGAN
            # items much have length greater than hifigan segment length
            # items must have native sampling rate higher than 38KHz
            mel_seg_len = (self.HiFiGAN_segment_size//self.hop_length)+1
            indexes = (gt['mel_lengths'] >= mel_seg_len) & (gt['sampling_rate'] >= hifiGAN.STFT.mel_fmax*2.)
            indexes = indexes & (indexes.cumsum(dim=0) <= self.HiFiGAN_batch_size)
            
            pred['hifigan_indexes'] = indexes # save for later
            if indexes.sum() >= self.HiFiGAN_batch_size:
                mel_lengths = gt['mel_lengths'][indexes]
                max_start = mel_lengths.min()-mel_seg_len
                start_ind = 0
                if max_start:
                    start_ind = random.randint(0, max_start)
                
                #pred['hifigan_inputs'] = pred['pred_mel_postnet'] # debug, please ignore/delete/tell me if this gets commited after 25th Dec
                pred['hifigan_inputs'  ] = pred['hifigan_inputs'][indexes][:, :,              start_ind:(start_ind+mel_seg_len)]
                gt[  'hifigan_gt_audio'] =   gt['gt_audio'      ][indexes][:, self.hop_length*start_ind:(start_ind+mel_seg_len)*self.hop_length].unsqueeze(1)
                
                pred['hifigan_pred_audio'] = hifiGAN.generator(pred['hifigan_inputs'])
                pred['hifigan_pred_mel'  ] = hifiGAN.STFT.mel_spectrogram(pred['hifigan_pred_audio'].squeeze(1))
                with torch.no_grad():
                    gt['hifigan_gt_mel'  ] = hifiGAN.STFT.mel_spectrogram(  gt['hifigan_gt_audio'  ].squeeze(1))
                
                hifiGAN.generator_loss(gt['hifigan_gt_audio'], pred['hifigan_pred_audio'],
                                       gt['hifigan_gt_mel'  ], pred['hifigan_pred_mel'  ], loss_dict)
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        with torch.no_grad():# get Avg Max Attention and Diagonality Metrics
            atd = alignment_metric(pred['alignments'].detach().clone(), gt['text_lengths'], gt['mel_lengths'])
            diagonalitys, avg_prob, char_max_dur, char_min_dur, char_avg_dur, p_missing_enc = atd.values()
            
            loss_dict['diagonality']       = diagonalitys.mean()
            loss_dict['avg_max_attention'] = avg_prob.mean()
            
            for i in range(tfB):
                audiopath = gt['audiopath'][i]
                file_losses[audiopath]['avg_max_attention'] =      avg_prob[i].cpu().item()
                file_losses[audiopath]['att_diagonality']   =  diagonalitys[i].cpu().item()
                file_losses[audiopath]['p_missing_enc']     = p_missing_enc[i].cpu().item()
                file_losses[audiopath]['char_max_dur']      =  char_max_dur[i].cpu().item()
                file_losses[audiopath]['char_min_dur']      =  char_min_dur[i].cpu().item()
                file_losses[audiopath]['char_avg_dur']      =  char_avg_dur[i].cpu().item()
                
                if 0:
                    diagonality_path = f'{os.path.splitext(audiopath)[0]}_diag.pt'
                    torch.save(diagonalitys[i].detach().clone().cpu(), diagonality_path)
                    
                    avg_prob_path = f'{os.path.splitext(audiopath)[0]}_avgp.pt'
                    torch.save(    avg_prob[i].detach().clone().cpu(), avg_prob_path   )
            
            pred_gate = pred['pred_gate_logits'].sigmoid()
            pred_gate[:, :5] = 0.0
            # Get inference alignment scores
            pred_mel_lengths = get_first_over_thresh(pred_gate, 0.7)
            atd = alignment_metric(pred['alignments'].detach().clone(), gt['text_lengths'], pred_mel_lengths)
            atd = {k: v.cpu() for k, v in atd.items()}
            diagonalitys, avg_prob, char_max_dur, char_min_dur, char_avg_dur, p_missing_enc = atd.values()
            scores = []
            for i in range(tfB):
                audiopath = gt['audiopath'][i]
                
                # factors that make up score
                weighted_score = avg_prob[i].item() # general alignment quality
                diagonality_punishment = max( diagonalitys[i].item()-1.10, 0) * 0.25 # speaking each letter at a similar pace.
                max_dur_punishment     = max( char_max_dur[i].item()-60.0, 0) * 0.005# getting stuck on same letter for 0.5s
                min_dur_punishment     = max(0.00-char_min_dur[i].item(),  0) * 0.5  # skipping single enc outputs
                avg_dur_punishment     = max(3.60-char_avg_dur[i].item(),  0)        # skipping most enc outputs
                mis_dur_punishment     = max(p_missing_enc[i].item()-0.08, 0)        # skipping some percent of the text
                
                if True:
                    weighted_score -= max_dur_punishment
                if gt['text_lengths'][i] > 12 and gt['mel_lengths'][i] < gt['mel_lengths'].max()*0.75:
                    weighted_score -= mis_dur_punishment
                if not (gt['pres_prev_state'][i] or gt['cont_next_iter'][i]):
                    weighted_score -= (diagonality_punishment+min_dur_punishment+avg_dur_punishment)
                scores.append(weighted_score)
                file_losses[audiopath]['att_score'] = weighted_score
            scores = torch.tensor(scores)
            loss_dict['weighted_score'] = scores.to(pred['alignments'].device).mean()
        
        return loss_dict, file_losses

