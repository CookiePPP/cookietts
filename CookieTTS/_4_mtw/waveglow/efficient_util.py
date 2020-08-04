import os
from torch import nn


def remove_weight_norms(m):
    if hasattr(m, 'weight_g'):
        nn.utils.remove_weight_norm(m)


def add_weight_norms(m):
    if hasattr(m, 'weight'):
        nn.utils.weight_norm(m)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
