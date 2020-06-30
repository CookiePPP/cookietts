import torch
from torch import nn
from utils import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        self.loss_func = hparams.loss_func
        self.masked_select = hparams.masked_select
    
    def forward(self, model_output, targets):
        mel_target, gate_target, output_lengths, *_ = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        
        # remove paddings before loss calc
        if self.masked_select:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(mel_target.size(1), mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            mel_target = torch.masked_select(mel_target, mask)
            mel_out = torch.masked_select(mel_out, mask)
            mel_out_postnet = torch.masked_select(mel_out_postnet, mask)
        
        if self.loss_func == 'MSELoss':
            mel_loss = nn.MSELoss()(mel_out, mel_target) + \
                nn.MSELoss()(mel_out_postnet, mel_target)
        elif self.loss_func == 'SmoothL1Loss':
            mel_loss = nn.SmoothL1Loss()(mel_out, mel_target) + \
                nn.SmoothL1Loss()(mel_out_postnet, mel_target)
        
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)
        return mel_loss + gate_loss, gate_loss
