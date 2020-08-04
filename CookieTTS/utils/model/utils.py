import numpy as np
from scipy.io.wavfile import read
import torch
from typing import Optional


def get_mask_from_lengths(lengths: torch.Tensor, max_len = None):
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
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