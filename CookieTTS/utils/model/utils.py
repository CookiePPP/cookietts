import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.autograd import Function

# Taken From https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
class GradScale(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_ if type(alpha_) is torch.Tensor else torch.tensor(alpha_, requires_grad=False))
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * alpha_
        return grad_input, None
grad_scale = GradScale.apply

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
    seq_w = torch.arange(0, max_w, device=device)# [max_w]
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


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate, soft_mask=False, local_mean=False, local_mean_range=5):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate).unsqueeze(1)# [B, 1, mel_T]
    
    if local_mean:
        def padidx(i):
            pad = (i+1)//2
            return (pad, -pad) if i%2==0 else (-pad, pad)
        mel_mean = sum([F.pad(mels.detach(), padidx(i), mode='replicate') for i in range(local_mean_range)])/local_mean_range# [B, n_mel, mel_T]
    else:
        if len(global_mean.shape) == 1:
            mel_mean = global_mean.unsqueeze(0) #    [n_mel] -> [B, n_mel]
        if len(mel_mean.shape) == 2:
            mel_mean = mel_mean.unsqueeze(-1)# [B, n_mel] -> [B, n_mel, mel_T]
    
    dropped_mels = (mels * ~drop_mask) + (mel_mean * drop_mask)
    if soft_mask:
        rand_mask = torch.rand(dropped_mels.shape[0], 1, dropped_mels.shape[2], device=mels.device, dtype=mels.dtype)
        rand_mask_inv = (1.-rand_mask)
        dropped_mels = (dropped_mels*rand_mask) + (mels*rand_mask_inv)
    return dropped_mels


def get_first_over_thresh(x, threshold):
    """Takes [B, T] and outputs indexes of first element over threshold for each vector in B (output.shape = [B])."""
    device = x.device
    x = x.clone().cpu().float() # using CPU because GPU implementation of argmax() splits tensor into 32 elem chunks, each chunk is parsed forward then the outputs are collected together... backwards
    x[:,-1] = threshold # set last to threshold just incase the output didn't finish generating.
    x[x>threshold] = threshold
    if int(''.join(torch.__version__.split('+')[0].split('.'))) < 170:# old pytorch < 1.7 did argmax backwards on CPU and even more broken on GPU.
        return ( (x.size(1)-1)-(x.flip(dims=(1,)).argmax(dim=1)) ).to(device).int()
    else:
        return x.argmax(dim=1).to(device).int()


def alignment_metric(alignments, input_lengths=None, output_lengths=None, enc_min_thresh=0.7, average_across_batch=False, adjacent_topk=True):
    """
    Diagonality = Network distance / Euclidean Distance
    https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2F1476-072X-7-7/MediaObjects/12942_2007_Article_201_Fig2_HTML.jpg
    
    Avg Max Attention = Average of Maximum Attention tokens at each timestep - Roughly equivalent to confidence between text and audio in TTS.
    Avg Top2 Attention = Average of Top 2 Attention tokens summed at each timestep. Can be better than Max because some symbols like "," and " " blend together. The model will not be better by learning the difference between "," and " " in the docoder so there's no reason to incentivize it.
    Avg Top3 Attention = Average of Top 3 Attention tokens summed at each timestep. Not tested, I'm guessing this also correlates with stability but I don't know how well.
    
    Encoder Max duration = Maximum timesteps spent on a single encoder token. If too much time is spent on a single token then that normally means the TTS model has gotten stuck on a single phoneme.
    Encoder Min duration = Minimum timesteps spent on a single encoder token. Can correlate with missing some letters or mis-pronouncing a word. The correlation is weak however so not really recommended for most models.
    Encoder Avg duration = Average timesteps spent on all (non-padded) encoder tokens. This value is equivalent to the speaking rate.
    
    p_missing_enc = Fraction of encoder tokens that had less summed alignment than enc_min_thresh. Used to identify if parts of the text were skipped.
    """
    alignments = alignments.transpose(1, 2).clone()# [B, dec, enc] -> [B, enc, dec]
    # alignments [batch size, x, y]
    # input_lengths [batch size] for len_x
    # output_lengths [batch size] for len_y
    if input_lengths == None:
        input_lengths =  torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[1]-1) # [B] # 147
    if output_lengths == None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[2]-1) # [B] # 767
    batch_size = alignments.size(0)
    euclidean_distance = torch.sqrt(input_lengths.double().pow(2) + output_lengths.double().pow(2)).view(batch_size)
    
    # [B, enc, dec] -> [B, dec], [B, dec]
    max_values, cur_idxs = torch.max(alignments, 1) # get max value in column and location of max value
    
    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:,0][:,None], cur_idxs[:,:-1]), dim=1)# shift entire tensor by one.
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5) # [B, dec]
    dist.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=dist.size(1)), 0.0)# set dist of padded to zero
    dist = dist.sum(dim=(1)) # get total Network distance for each alignment
    diagonalitys = (dist + 1.4142135)/euclidean_distance # Network distance / Euclidean dist
    
    alignments.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=alignments.size(2))[:,None,:], 0.0)
    attm_enc_total = torch.sum(alignments, dim=2)# [B, enc, dec] -> [B, enc]
    
    # calc max  encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 0.0)
    encoder_max_dur = attm_enc_total.max(dim=1)[0] # [B, enc] -> [B]
    
    # calc mean encoder durations (with padding ignored)
    encoder_avg_dur = attm_enc_total.mean(dim=1)   # [B, enc] -> [B]
    encoder_avg_dur *= (attm_enc_total.size(1)/input_lengths.float())
    
    # calc min encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 1.0)
    encoder_min_dur = attm_enc_total.min(dim=1)[0] # [B, enc] -> [B]
    
    # calc average max attention (with padding ignored)
    max_values.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=max_values.size(1)), 0.0) # because padding
    avg_prob = max_values.mean(dim=1)
    avg_prob *= (alignments.size(2)/output_lengths.float()) # because padding
    
    # calc average top2 attention (with padding ignored)
    if adjacent_topk:
        alignment_summed = alignments + F.pad(alignments, (0, 0, 1, -1,))# [B, enc, dec]
        top_vals = torch.max(alignment_summed, dim=1)[0]# -> [B, dec]
    else:
        top_vals = torch.topk(alignments, k=2, dim=1, largest=True, sorted=True)[0].sum(dim=1)# [B, enc, dec] -> [B, dec]
    top2_avg_prob = top_vals.mean(dim=1)
    top2_avg_prob *= (alignments.size(2)/output_lengths.float()) # because padding
    
    # calc average top3 attention (with padding ignored)
    if adjacent_topk:
        alignment_summed = alignments + F.pad(alignments, (0, 0, 1, -1,)) + F.pad(alignments, (0, 0, -1, 1,))# [B, enc, dec]
        top_vals = torch.max(alignment_summed, dim=1)[0]# -> [B, dec]
    else:
        top_vals = torch.topk(alignments, k=3, dim=1, largest=True, sorted=True)[0].sum(dim=1)# [B, enc, dec] -> [B, dec]
    top3_avg_prob = top_vals.mean(dim=1)
    top3_avg_prob *= (alignments.size(2)/output_lengths.float()) # because padding
    
    # calc portion of encoder durations under min threshold
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), float(1e3))
    p_missing_enc = (torch.sum(attm_enc_total < enc_min_thresh, dim=1)) / input_lengths.float()
    
    if average_across_batch:
        diagonalitys    = diagonalitys   .mean()
        avg_prob        = avg_prob       .mean()
        top2_avg_prob   = top2_avg_prob  .mean()
        top3_avg_prob   = top3_avg_prob  .mean()
        encoder_max_dur = encoder_max_dur.mean()
        encoder_min_dur = encoder_min_dur.mean()
        encoder_avg_dur = encoder_avg_dur.mean()
        p_missing_enc   = p_missing_enc  .mean()
    
    output = {}
    output["diagonalitys"] = diagonalitys
    output["avg_prob"     ] = avg_prob
    output["top2_avg_prob"] = top2_avg_prob
    output["top3_avg_prob"] = top3_avg_prob
    output["encoder_max_dur"] = encoder_max_dur
    output["encoder_min_dur"] = encoder_min_dur
    output["encoder_avg_dur"] = encoder_avg_dur
    output["p_missing_enc"] = p_missing_enc
    return output


# taken from https://stackoverflow.com/a/30024601
import time
class elapsed_timer(object):
    def __init__(self, msg=''):
        self.msg = msg
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, typ, value, traceback):
        print(f'{self.msg} took {time.time()-self.start}s')
