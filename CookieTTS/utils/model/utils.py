import numpy as np
from scipy.io.wavfile import read
import torch
from typing import Optional


class freeze_grads():
    def __init__(self, submodule):
        self.submodule = submodule
    
    def __enter__(self):
        self.require_grads = []
        for param in self.submodule.parameters():
            self.require_grads.append(param.requires_grad)
            param.requires_grad = False
    
    def __exit__(self, type, value, traceback):
        for i, param in enumerate(self.submodule.parameters()):
            param.requires_grad = self.require_grads[i]


@torch.jit.script
def get_mask_from_lengths(lengths: torch.Tensor, max_len:int = 0):
    if max_len == 0:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1))
    return mask

@torch.jit.script
def get_mask_3d(widths, heights, max_w: Optional[torch.Tensor] = None, max_h: Optional[torch.Tensor] = None):
    device = widths.device
    B = widths.shape[0]
    if max_w is None:
        max_w = torch.max(widths)
    if max_h is None:
        max_h = torch.max(heights)
    seq_w = torch.arange(0, max_w, device=device) # [max_w]
    seq_h = torch.arange(0, max_h, device=device)# [max_h]
    mask_w = (seq_w.unsqueeze(0) < widths.unsqueeze(1)).to(torch.bool) # [1, max_w] < [B, 1] -> [B, max_w]
    mask_h = (seq_h.unsqueeze(0) < heights.unsqueeze(1)).to(torch.bool)# [1, max_h] < [B, 1] -> [B, max_h]
    mask = (mask_w.unsqueeze(2) & mask_h.unsqueeze(1))# [B, max_w, 1] & [B, 1, max_h] -> [B, max_w, max_h]
    return mask# [B, max_w, max_h]


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    batch_size = lengths.size(0)
    max_len = int(torch.max(lengths).item())
    mask = get_mask_from_lengths(lengths)
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate)
    dropped_mels = (mels * (~drop_mask).unsqueeze(1) +
                    global_mean[None, :, None] * drop_mask.unsqueeze(1))
    return dropped_mels


def get_first_over_thresh(x, threshold):
    """Takes [B, T] and outputs first T over threshold for each B (output.shape = [B])."""
    device = x.device
    x = x.clone().cpu().float() # using CPU because GPU implementation of argmax() splits tensor into 32 elem chunks, each chunk is parsed forward then the outputs are collected together... backwards
    x[:,-1] = threshold # set last to threshold just incase the output didn't finish generating.
    x[x>threshold] = threshold
    if int(''.join(torch.__version__.split('+')[0].split('.'))) < 170:
        return ( (x.size(1)-1)-(x.flip(dims=(1,)).argmax(dim=1)) ).to(device).int()
    else:
        return x.argmax(dim=1).to(device).int()


def alignment_metric(alignments, input_lengths=None, output_lengths=None, enc_min_thresh=0.7, average_across_batch=False):
    alignments = alignments.transpose(1,2) # [B, dec, enc] -> [B, enc, dec]
    # alignments [batch size, x, y]
    # input_lengths [batch size] for len_x
    # output_lengths [batch size] for len_y
    if input_lengths == None:
        input_lengths =  torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[1]-1) # [B] # 147
    if output_lengths == None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[2]-1) # [B] # 767
    batch_size = alignments.size(0)
    optimums = torch.sqrt(input_lengths.double().pow(2) + output_lengths.double().pow(2)).view(batch_size)
    
    # [B, enc, dec] -> [B, dec], [B, dec]
    values, cur_idxs = torch.max(alignments, 1) # get max value in column and location of max value
    
    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:,0][:,None], cur_idxs[:,:-1]), dim=1) # shift entire tensor by one.
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5) # [B, dec]
    dist.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=dist.size(1)), 0.0) # set dist of padded to zero
    dist = dist.sum(dim=(1)) # get total dist for each B
    diagonalitys = (dist + 1.4142135)/optimums # dist / optimal dist
    
    alignments.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=alignments.size(2))[:,None,:], 0.0)
    attm_enc_total = torch.sum(alignments, dim=2)# [B, enc, dec] -> [B, enc]
    
    # calc max  encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 0.0)
    encoder_max_focus = attm_enc_total.max(dim=1)[0] # [B, enc] -> [B]
    
    # calc mean encoder durations (with padding ignored)
    encoder_avg_focus = attm_enc_total.mean(dim=1)   # [B, enc] -> [B]
    encoder_avg_focus *= (attm_enc_total.size(1)/input_lengths.float())
    
    # calc min encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 1.0)
    encoder_min_focus = attm_enc_total.min(dim=1)[0] # [B, enc] -> [B]
    
    # calc average max attention (with padding ignored)
    values.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=values.size(1)), 0.0) # because padding
    avg_prob = values.mean(dim=1)
    avg_prob *= (alignments.size(2)/output_lengths.float()) # because padding
    
    # calc portion of encoder durations under min threshold
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), float(1e3))
    p_missing_enc = (torch.sum(attm_enc_total < enc_min_thresh, dim=1)) / input_lengths.float()
    
    if average_across_batch:
        diagonalitys      = diagonalitys     .mean()
        encoder_max_focus = encoder_max_focus.mean()
        encoder_min_focus = encoder_min_focus.mean()
        encoder_avg_focus = encoder_avg_focus.mean()
        avg_prob          = avg_prob         .mean()
        p_missing_enc     = p_missing_enc    .mean()
    
    output = {}
    output["diagonalitys"     ] = diagonalitys
    output["avg_prob"         ] = avg_prob
    output["encoder_max_focus"] = encoder_max_focus
    output["encoder_min_focus"] = encoder_min_focus
    output["encoder_avg_focus"] = encoder_avg_focus
    output["p_missing_enc"]     = p_missing_enc
    return output
