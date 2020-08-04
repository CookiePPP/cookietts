import torch
from torch import nn
import torch.nn.functional as F
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d

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
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
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
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = get_mask_from_lengths(ilens)  # (B, T_in)
        out_masks = get_mask_from_lengths(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

def AttentionPathLoss(attention_scores, output_lengths, text_lengths):
    """
    Calculates/minimizes difference between adjacent alignment frames.
    *Should* prompt the model to attend to text in order rather than skipping around the text.
    """
    # [B, n_layers, dec_T, enc_T]
    if len(attention_scores.shape) == 4:
        attention_scores = attention_scores.mean(dim=(1,))# [B, n_layers, dec_T, enc_T] -> [B, dec_T, enc_T]
    B, dec_T, enc_T = attention_scores.shape              # [B, dec_T, enc_T]
    mask = ~get_mask_3d(output_lengths, text_lengths)     # [B, dec_T, enc_T]
    dec_shifted_attention_scores = torch.cat((attention_scores[:, 0:1], attention_scores[:, :-1]), dim=1) # Shift right by one [B, dec_T, enc_T]
    dec_enc_shifted_attention_scores = F.pad(dec_shifted_attention_scores, (1,0))[:, :, :-1]# Shift right and up by one [B, dec_T, enc_T]
    attention_scores = attention_scores.masked_fill(mask, 0.0) # [B, dec_T, enc_T]
    dec_shifted_attention_scores.masked_fill_(mask, 0.0)       # [B, dec_T, enc_T]
    dec_enc_shifted_attention_scores.masked_fill_(mask, 0.0)   # [B, dec_T, enc_T]
    
    att_loss = (((attention_scores - dec_shifted_attention_scores)**2) * ((attention_scores - dec_enc_shifted_attention_scores)**2)).sum() / (B*dec_T) # [B, dec_T, enc_T]
    return att_loss

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
        self.len_pred_loss_weight = hparams.len_pred_loss_weight
        self.pos_att_len_pred_att_weight = hparams.pos_att_len_pred_att_weight
        if hparams.pos_att_guided_attention:
            self.guided_att = GuidedAttentionLoss(sigma=hparams.pos_att_guided_attention_sigma)
            self.pos_att_guided_attention_alpha = hparams.pos_att_guided_attention_alpha
        
        self.att_path_loss = hparams.att_path_loss
        self.att_path_loss_weight = hparams.att_path_loss_weight
    
    def forward(self, model_output, targets):
        mel_target, gate_target, output_lengths, text_lengths, *_ = targets
        mel_out, attention_scores, log_s_sum, logdet_w_sum, pred_output_lengths, pred_output_lengths_std, dur_z, dur_log_s_sum, dur_logdet_w_sum, len_pred_attention = model_output
        batch_size, n_mel_channels, frames = mel_target.shape
        
        output_lengths_float = output_lengths.float()
        mel_out = mel_out.float()
        log_s_sum = log_s_sum.float()
        logdet_w_sum = logdet_w_sum.float()
        
        # Length Loss
        len_pred_loss = torch.nn.MSELoss()(pred_output_lengths.log(), output_lengths_float.log())
        
        # Spectrogram Loss
        if True:
            # remove paddings before loss calc
            mask = get_mask_from_lengths(output_lengths)[:, None, :] # [B, 1, T] BoolTensor
            mask = mask.expand(mask.size(0), mel_target.size(1), mask.size(2))# [B, n_mel, T] BoolTensor
        n_elems = (output_lengths_float.sum() * n_mel_channels)
        mel_out = torch.masked_select(mel_out, mask)
        loss_z = ((mel_out.pow(2).sum()) / self.sigma2_2)/n_elems # mean z (over all elements)
        
        loss_w = -logdet_w_sum.sum()/(n_mel_channels*frames)
        
        log_s_sum = log_s_sum.view(batch_size, -1, frames)
        log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
        loss_s = -log_s_sum.sum()/(n_elems)
        
        loss = loss_z+loss_w+loss_s+(len_pred_loss*self.len_pred_loss_weight)
        
        # Durations Loss
        if self.DurGlow_enable:
            dur_z = dur_z.view(dur_z.shape[0], 2, -1)
            if True:
                # remove paddings before loss calc
                mask = get_mask_from_lengths(text_lengths)[:, None, :] # [B, 1, T] BoolTensor
                mask = mask.expand(mask.size(0), 2, mask.size(2))# [B, 2, T] BoolTensor
            chars = text_lengths.max()
            n_elems = (text_lengths.sum() * 2)
            
            dur_z = torch.masked_select(dur_z, mask)
            dur_loss_z = ((dur_z.pow(2).sum()) / self.dg_sigma2_2)/n_elems # mean z (over all elements)
            
            dur_log_s_sum = dur_log_s_sum.view(batch_size, -1, chars)
            dur_log_s_sum = torch.masked_select(dur_log_s_sum , mask[:, :dur_log_s_sum.shape[1], :])
            dur_loss_s = -dur_log_s_sum.sum()/(n_elems)
            
            dur_loss_w = -dur_logdet_w_sum.sum()/(2*chars)
            
            if self.DurGlow_loss_scalar:
                loss = loss + (dur_loss_z+dur_loss_w+dur_loss_s) * self.DurGlow_loss_scalar
        else:
            dur_loss_z=dur_loss_w=dur_loss_s = None
        
        
        if hasattr(self, 'guided_att'):               # (optional) Diagonal Guided Attention Loss
            att_sc_shape = attention_scores.shape
            if len(att_sc_shape) == 4:
                attention_scores = attention_scores.view(att_sc_shape[0]*att_sc_shape[1], *att_sc_shape[2:])# [B, n_layers, dec_T, enc_T] -> [B*n_layers, dec_T, enc_T]
                text_lengths = text_lengths.repeat_interleave(att_sc_shape[1], dim=0)
                output_lengths = output_lengths.repeat_interleave(att_sc_shape[1], dim=0)
            att_loss = self.guided_att(attention_scores, text_lengths, output_lengths)
            #att_loss = ((att_loss / 0.006)**4)*0.006 # rescaling loss to be much more aggressive over 0.008 (otherwise, complete alignment collapse only increases loss a small amount and model does not recover)
            loss = loss + (att_loss * self.pos_att_guided_attention_alpha)
        elif len_pred_attention is not None:          # (optional) Duration Predictor Attention Loss
            att_sc_shape = attention_scores.shape
            if len(att_sc_shape) == 4:
                attention_scores = attention_scores.mean(dim=(1,))# [B, n_layers, dec_T, enc_T] -> [B, dec_T, enc_T]
            mask = get_mask_3d(output_lengths, text_lengths)
            len_pred_attention = torch.masked_select(len_pred_attention, mask)# [B, dec_T, enc_T] -> [B*-1]
            attention_scores = torch.masked_select(attention_scores, mask)    # [B, dec_T, enc_T] -> [B*-1]
            att_loss = torch.nn.MSELoss()(attention_scores, len_pred_attention)
            loss = loss + (att_loss * self.pos_att_len_pred_att_weight)
        else:
            att_loss = None
        
        if self.att_path_loss:
            att_loss = AttentionPathLoss(attention_scores, output_lengths, text_lengths)
            loss = loss + (att_loss * self.att_path_loss_weight)
        
        return loss, len_pred_loss, loss_z, loss_w, loss_s, att_loss, dur_loss_z, dur_loss_w, dur_loss_s
