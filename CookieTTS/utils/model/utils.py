import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1))
    return mask


def get_mask_3d(widths, heights):
    max_w = torch.max(widths).item()
    max_h = torch.max(heights).item()
    mask = torch.zeros(widths.size(0), max_w, max_h, device=widths.device)
    for i in range(widths.size(0)):
        mask[i,:widths[i],:heights[i]] = 1
    return mask==1


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