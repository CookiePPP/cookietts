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

# https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_TacoSpeech.py#L28-L156
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
    loss = (F.mse_loss(mu, target, reduction='none')/logvar.exp())+logvar
    if True:
        loss = loss.mean()
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


class TacoSpeechLoss(nn.Module):
    def __init__(self, hparams):
        super(TacoSpeechLoss, self).__init__()
        self.memory_efficient = hparams.memory_efficient
        
        self.rank       = hparams.rank
        self.n_gpus     = hparams.n_gpus
        
        self.n_symbols  = hparams.n_symbols
        self.n_speakers = hparams.n_speakers
        
        self.sylps_MSE_weight = hparams.sylps_MSE_weight
        self.sylps_MAE_weight = hparams.sylps_MAE_weight
        
        self.diag_att_weight  = hparams.diag_att_weight
        self.guided_att = GuidedAttentionLoss(sigma=hparams.DiagonalGuidedAttention_sigma)
        
        self.HiFiGAN_enable       = getattr(hparams, 'HiFiGAN_enable',       False)
        self.HiFiGAN_segment_size = getattr(hparams, 'HiFiGAN_segment_size', 16384)
        self.HiFiGAN_batch_size   = getattr(hparams, 'HiFiGAN_batch_size'  ,     2)
        self.hop_length           = hparams.hop_length
        
        self.HiFiGAN_g_msd_class_weight      = 0.0
        self.HiFiGAN_g_mpd_class_weight      = 0.0
        self.HiFiGAN_g_msd_featuremap_weight = 0.0
        self.HiFiGAN_g_mpd_featuremap_weight = 0.0
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
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
                loss_scale = getattr(self, f'{k}_weight', None)
            
            if loss_scale is None:
                loss_scale = 1.0
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
    
    def forward(self, iteration, model, pred, gt, loss_scalars, hifiGAN=None,):
        self.train(model.training)
        loss_dict = {}
        file_losses = {}# dict of {"audiofile": {"spec_MSE": spec_MSE, "avg_prob": avg_prob, ...}, ...}
        
        B, n_mel, mel_T = gt['gt_mel'].shape
        #tfB = B//(model.decoder.half_inference_mode+1)
        #for i in range(tfB):
        #    current_time = time.time()
        #    if gt['audiopath'][i] not in file_losses:
        #        file_losses[gt['audiopath'][i]] = {'speaker_id_ext': gt['speaker_id_ext'][i], 'time': current_time}
        
        if True:
            pred_mel    = pred['pred_mel']   # [B, n_mel*n_frames, mel_T]
            gt_mel      =   gt['gt_mel']     # [B, n_mel, mel_T]
            
            # fill padded with zeros
            mel_mask = get_mask_from_lengths(gt['mel_lengths']).unsqueeze(1)# [B, 1, mel_T]
            pred_mel.masked_fill_(~mel_mask, 0.0)
           #gt_mel  .masked_fill_(~mel_mask, 0.0)# padded area is already zero so I just commented this out.
            
            # check for nan/infs
            with torch.no_grad():
                #pred_mel.masked_fill_(torch.isnan(pred_mel) | torch.isinf(pred_mel), 0.0)# debug?
                assert not torch.isnan(pred_mel).any(), 'mel has NaNs'
                assert not torch.isinf(pred_mel).any(), 'mel has Infs'
            
            # spectrogram / decoder loss
            scalar = mel_mask.numel()/mel_mask.sum()
            loss_dict['decoder_MAE'] = F. l1_loss(pred_mel, gt_mel)*scalar
            loss_dict['decoder_MSE'] = F.mse_loss(pred_mel, gt_mel)*scalar
        
        if True: # Pred Sylps loss
            pred_sylps = pred['pred_sylps'].squeeze(1)# [B, 1] -> [B]
            sylps_target = gt['gt_sylps']
            loss_dict['sylps_MAE'] = F. l1_loss(pred_sylps, sylps_target)
            loss_dict['sylps_MSE'] = F.mse_loss(pred_sylps, sylps_target)
            del pred_sylps, sylps_target
        
        if 'varpred_pred_fesvd' in pred:
            gt_fesvd = pred['bn_fesvd']
            pr_fesvd = pred['varpred_pred_fesvd']# [B, 8, txt_T]
            text_mask = get_mask_from_lengths(gt['text_lengths']).unsqueeze(1)# [B, 1, mel_T]
            pr_fesvd = pr_fesvd.masked_fill(~text_mask, 0.0)
            gt_fesvd = gt_fesvd.masked_fill(~text_mask, 0.0)
            scalar = text_mask.numel()/text_mask.sum()
            loss_dict['fesvd_MAE'] = F. l1_loss(pr_fesvd, gt_fesvd)*scalar
            loss_dict['fesvd_MSE'] = F.mse_loss(pr_fesvd, gt_fesvd)*scalar
        
        if 'varpred_hidden_pred' in pred:
            h_c_mulogvar = pred['varpred_hidden_pred']# [B, n_lstm, 4*lstm_dim]
            h_c_gt       = pred['varpred_hidden_gt'  ]# [B, n_lstm, 2*lstm_dim]
            h_c_mu, h_c_logvar = h_c_mulogvar.chunk(2, dim=2)# [B, n_lstm, 4*lstm_dim] -> [B, n_lstm, 2*lstm_dim], [B, n_lstm, 2*lstm_dim]
            loss_dict['varpred_hdn_NLL'] = NormalLLLoss(h_c_mu, h_c_logvar, h_c_gt.detach())
        
        if 'pred_frame_fesv' in pred:
            pred_fesv = pred['pred_frame_fesv']
            pred_fesv = pred_fesv.transpose(1, 2)# [B, 7, mel_T]
            
            gt_fesv   = pred['bngt_fesv']      # [B, 7, mel_T]
            voice_flag =  gt['gt_frame_voiced']# [B, mel_T]
            mask = gt_fesv.new_ones(*gt_fesv.shape)
            mask[:, 1, :] = voice_flag
            pred_fesv = pred_fesv*mask
            gt_fesv   = gt_fesv  *mask
            
            mel_mask = get_mask_from_lengths(gt['mel_lengths']).unsqueeze(1)# [B, 1, mel_T]
            scalar = mel_mask.numel()/mel_mask.sum()
            loss_dict['f_fesvd_MAE'] = F. l1_loss(pred_fesv, gt_fesv)*scalar
            loss_dict['f_fesvd_MSE'] = F.mse_loss(pred_fesv, gt_fesv)*scalar
        
        if 'mdn_loss' in pred:
            safe_mdn_ids = ~(gt['pres_prev_state'] | gt['cont_next_iter']) & (pred['mdn_loss'].detach() < 1e3)# BoolTensor[B]
            loss_dict['mdn_loss'] = pred['mdn_loss'][safe_mdn_ids].mean()
        
        if ('pred_logdur' in pred) and ("mdn_alignment" in pred):
            pred_logdur = pred['pred_logdur'].squeeze(-1)# [B, txt_T]
            text_mask = get_mask_from_lengths(gt['text_lengths'])# [B, txt_T]
            loss_dict['dur_loss'] = F.mse_loss(pred['mdn_alignment'].sum(dim=1).detach().clamp(min=0.5).log().masked_fill(~text_mask, 0.0),
                                                                                                  pred_logdur.masked_fill(~text_mask, 0.0))# [B, txt_T], [B, txt_T] -> []
        
        if 'mdn_alignment' in pred:# Diagonal Attention Guiding
            alignments     = pred['mdn_alignment']
            text_lengths   =   gt['text_lengths']
            output_lengths =   gt['mel_lengths' ]
            not_truncated  = ~(gt['pres_prev_state'] | gt['cont_next_iter'])
            if not_truncated.sum():
                loss_dict['diag_att'] = self.guided_att(alignments[not_truncated],
                                                      text_lengths[not_truncated],
                                                    output_lengths[not_truncated])
            else:
                loss_dict['diag_att'] = alignments.new_tensor(0.028)
            del alignments, text_lengths, output_lengths, not_truncated
        
        if self.HiFiGAN_enable:
            # get (indexes) items in batch to use for HiFiGAN
            # items much have length greater than hifigan segment length
            # items must have native sampling rate higher than 38KHz
            mel_seg_len = (self.HiFiGAN_segment_size//self.hop_length)+1
            indexes = (gt['mel_lengths'] >= mel_seg_len) & (gt['sampling_rate'] >= hifiGAN.STFT.mel_fmax*2.)
            indexes = indexes & (indexes.cumsum(dim=0) <= self.HiFiGAN_batch_size)
            
            # get min n_indexes of all graphics cards
            min_n_indexes = indexes.sum()
            if min_n_indexes.eq(0):
                print(f"on rank {self.rank} n_indexes = {min_n_indexes}")
            if self.n_gpus > 1:
                dist.all_reduce(min_n_indexes, op=dist.ReduceOp.MIN)
            pred['hifigan_enabled'] = min_n_indexes>0
            pred['hifigan_indexes'] = indexes # save for later
            if min_n_indexes:
                mel_lengths = gt['mel_lengths'][indexes]
                max_start = mel_lengths.min()-mel_seg_len
                start_ind = 0
                if max_start:
                    start_ind = random.randint(0, max_start)
                
                #pred['hifigan_inputs'] = pred['pred_mel_postnet'] # debug, please ignore/delete/tell me if this gets commited after 25th Dec
                pred['hifigan_inputs'  ] = pred['hifigan_inputs'][indexes][:, :,              start_ind:(start_ind+mel_seg_len)]
                gt[  'hifigan_gt_audio'] =   gt['gt_audio'      ][indexes][:, self.hop_length*start_ind:(start_ind+mel_seg_len)*self.hop_length].unsqueeze(1)
                
                pred['hifigan_pred_audio'] = self.maybe_cp(hifiGAN.generator.__call__, *(pred['hifigan_inputs'],))
                pred['hifigan_pred_mel'] = self.maybe_cp(hifiGAN.STFT.mel_spectrogram_with_grad, *(pred['hifigan_pred_audio'].squeeze(1),))
                with torch.no_grad():
                    gt['hifigan_gt_mel'] = hifiGAN.STFT.mel_spectrogram(gt['hifigan_gt_audio'].squeeze(1))
                
                hifiGAN.generator_loss(gt['hifigan_gt_audio'], pred['hifigan_pred_audio'],
                                       gt['hifigan_gt_mel'  ], pred['hifigan_pred_mel'  ], loss_dict)
            else:
                print("WARNING: No valid inputs for HiFiGAN Generator!")
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        with torch.no_grad():
            if iteration > 12000 and 'mdn_alignment' in pred:
                not_truncated  = ~(gt['pres_prev_state'] | gt['cont_next_iter'])
                for i, not_trunc in enumerate(not_truncated):
                    if not_trunc:
                        mel_length  = gt['mel_lengths'][i]
                        text_length = gt['text_lengths'][i]
                        alignment = pred['mdn_alignment'][i, :mel_length, :text_length]# -> [mel_T, txt_T]
                        
                        audiopath = gt['audiopath'][i]
                        is_arpa   = gt['arpa'][i]
                        alignpath = os.path.splitext(audiopath)[0]+('_palign.pt' if is_arpa else '_galign.pt')
                        torch.save(alignment.detach().clone().half(), alignpath)
        
        return loss_dict, file_losses

