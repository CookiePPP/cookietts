import torch
from torch import nn
from CookieTTS.utils.model.utils import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.n_group = hparams.n_group
        sigma = hparams.sigma
        self.sigma2 = sigma ** 2
        self.sigma2_2 = self.sigma2 * 2
    
    def forward(self, model_output, targets):
        mel_target, gate_target, output_lengths, *_ = targets
        mel_out, pred_output_lengths, log_s_sum, logdet_w_sum = model_output
        batch_size, n_mel_channels, frames = mel_target.shape
        
        output_lengths = output_lengths.float()
        mel_out = mel_out.float()
        log_s_sum = log_s_sum.float()
        log_s_sum = log_s_sum.view(batch_size, -1, frames)
        logdet_w_sum = logdet_w_sum.float()
        
        assert not torch.isnan(output_lengths).any(), 'output_lengths has NaN values.'
        assert not torch.isnan(pred_output_lengths).any(), 'pred_output_lengths has NaN values.'
        len_pred_loss = torch.nn.MSELoss()(pred_output_lengths.log(), output_lengths.log())
        assert not torch.isnan(len_pred_loss).any(), 'len_pred_loss has NaN values.'
        
        # remove paddings before loss calc
        mask = get_mask_from_lengths(output_lengths)
        mask = mask.expand(mel_target.size(1), mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        mel_out = torch.masked_select(mel_out, mask)
        log_s_sum = torch.masked_select(log_s_sum , mask[:, :log_s_sum.shape[1], :])
        
        # [B, T] -> [B]
        #loss = (mel_out.pow(2).sum()) / self.sigma2_2 - logdet_w_sum.sum() - log_s_sum.sum()
        #assert not torch.isnan(loss).any(), 'loss has NaN values.'
        
        #loss /= (output_lengths.sum() * n_mel_channels)
        
        loss_z = ((mel_out.pow(2).sum()) / self.sigma2_2)/(output_lengths.sum() * n_mel_channels) # mean z (over all elements)
        assert not torch.isnan(loss_z).any(), 'loss_z has NaN values.'
        
        logdet_w_sum = logdet_w_sum * output_lengths.mean()/output_lengths.max() # scale to remove padding
        loss_w = -logdet_w_sum.sum()/(batch_size * frames * (n_mel_channels/self.n_group)) # mean logdet_w
        assert not torch.isnan(loss_w).any(), 'loss_w has NaN values.'
        
        loss_s = -log_s_sum.sum()/(output_lengths.sum() * n_mel_channels)
        assert not torch.isnan(loss_s).any(), 'loss_s has NaN values.'
        
        loss = loss_z+loss_w+loss_s
        assert not torch.isnan(loss).any(), 'loss has NaN values.'
        
        loss = loss + len_pred_loss
        assert not torch.isnan(loss).any(), 'loss has NaN values.'
        return loss, len_pred_loss, loss_z, loss_w, loss_s
