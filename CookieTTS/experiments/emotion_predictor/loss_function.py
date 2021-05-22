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


class Loss(nn.Module):
    def __init__(self, h):
        super(Loss, self).__init__()
        self.rank       = h.rank
        self.n_gpus     = h.n_gpus
        self.class_NCE_weight = 1.0
    
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
    
    def g_forward(self, iteration, model, pr, gt, loss_scalars, loss_params, file_losses=None, save_alignments=False):
        if file_losses is None:
            file_losses = {}
        loss_dict = {}
        
        if True:# Classification Loss
            loss_dict['class_NCE'] = nn.CrossEntropyLoss()(pr['class_probs'], gt['gt_emotion_id'])
        
        #################################################################
        ## Colate / Merge the Losses into a single tensor with scalars ##
        #################################################################
        loss_dict = self.colate_losses(loss_dict, loss_scalars)
        
        return loss_dict, file_losses

