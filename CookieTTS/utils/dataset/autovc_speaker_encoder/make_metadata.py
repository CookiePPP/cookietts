"""
Generate speaker embeddings and metadata for training
"""
from .model_bl import D_VECTOR
from collections import OrderedDict
import os
import numpy as np
import torch

len_crop = 128

def get_speaker_encoder():
    speaker_encoder = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
    c_checkpoint = torch.load(os.path.join(os.path.split(__file__)[0], '3000000-BL.ckpt'))
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    speaker_encoder.load_state_dict(new_state_dict)
    del c_checkpoint, new_state_dict
    return speaker_encoder