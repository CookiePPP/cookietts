import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# "Gated Convolutional Neural Networks for Domain Adaptation"
#  https://arxiv.org/pdf/1905.06906.pdf
@torch.jit.script
def GTU(input_a, input_b, n_channels: int):
    """Gated Tanh Unit (GTU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GTRU(input_a, input_b, n_channels: int):# saves significant VRAM and runs faster, unstable for first 150K~ iters. (test a difference layer initialization?)
    """Gated[?] Tanh ReLU Unit (GTRU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.relu(in_act[:, n_channels:, :], inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GLU(input_a, input_b, n_channels: int):
    """Gated Linear Unit (GLU)"""
    in_act = input_a+input_b
    l_act = in_act[:, :n_channels, :]
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = l_act * s_act
    return acts

# Random units I wanted to try
@torch.jit.script
def TTU(input_a, input_b, n_channels: int):
    """Tanh Tanh Unit (TTU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    t_act2 = torch.tanh(in_act[:, n_channels:, :])
    acts = t_act * t_act2
    return acts

@torch.jit.script
def STU(input_a, input_b, n_channels: int):
    """SeLU Tanh Unit (STU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.nn.functional.selu(in_act[:, n_channels:, :], inplace=True)
    acts = t_act * s_act
    return acts

@torch.jit.script
def GTSU(input_a, input_b, n_channels: int):
    """Gated TanhShrink Unit (GTSU)"""
    in_act = input_a+input_b
    t_act = torch.nn.functional.tanhshrink(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def SPTU(input_a, input_b, n_channels: int):
    """Softplus Tanh Unit (SPTU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.nn.functional.softplus(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GSIU(input_a, input_b, n_channels: int):
    """Gated Sinusoidal Unit (GSIU)"""
    in_act = input_a+input_b
    t_act = torch.sin(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GSIRU(input_a, input_b, n_channels: int):
    """Gated SIREN Unit (GSIRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = t_act * s_act
    return acts

@torch.jit.script
def GTSRU(input_a, input_b, n_channels: int):
    """Gated[?] TanhShrink ReLU Unit (GTSRU)"""
    in_act = input_a+input_b
    t_act = torch.nn.functional.tanhshrink(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.relu(in_act[:, n_channels:, :], inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GSIRRU(input_a, input_b, n_channels: int): # best and fastest converging unit, uses a lot of VRAM.
    """Gated[?] SIREN ReLU Unit (GSIRRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.relu(in_act[:, n_channels:, :], inplace=False)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GSIRLRU(input_a, input_b, n_channels: int):
    """Gated[?] SIREN Leaky ReLU Unit (GSIRLRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.leaky_relu(in_act[:, n_channels:, :], negative_slope=0.01, inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GSIRRLRU(input_a, input_b, n_channels: int):
    """Gated[?] SIREN Randomized Leaky ReLU Unit (GSIRRLRU)"""
    in_act = input_a+input_b
    in_act[:, :n_channels, :].detach().mul_(16) # modify tensor WITHOUT telling autograd.
    t_act = torch.sin(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.rrelu(in_act[:, n_channels:, :], lower=0.01, upper=0.1, inplace=True)
    acts = t_act * r_act
    return acts

@torch.jit.script
def GTLRU(input_a, input_b, n_channels: int):
    """Gated[?] Tanh Leaky ReLU Unit (GTLRU)"""
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    r_act = torch.nn.functional.leaky_relu(in_act[:, n_channels:, :], negative_slope=0.01, inplace=True)
    acts = t_act * r_act
    return acts


def get_gate_func(gated_unit_str):
    if   gated_unit_str.upper() == 'GTU':
        return GTU
    elif gated_unit_str.upper() == 'GTRU':
        return GTRU
    elif gated_unit_str.upper() == 'GTLRU':
        return GTLRU
    elif gated_unit_str.upper() == 'GLU':
        return GLU
    elif gated_unit_str.upper() == 'TTU':
        return TTU
    elif gated_unit_str.upper() == 'STU':
        return STU
    elif gated_unit_str.upper() == 'GTSU':
        return GTSU
    elif gated_unit_str.upper() == 'SPTU':
        return SPTU
    elif gated_unit_str.upper() == 'GSIU':
        return GSIU
    elif gated_unit_str.upper() == 'GSIRU':
        return GSIRU
    elif gated_unit_str.upper() == 'GTSRU':
        return GTSRU
    elif gated_unit_str.upper() == 'GSIRRU':
        return GSIRRU
    elif gated_unit_str.upper() == 'GSIRLRU':
        return GSIRLRU
    elif gated_unit_str.upper() == 'GSIRRLRU':
        return GSIRRLRU
    else:
        raise Exception("gated_unit is invalid\nOptions are ('GTU','GTRU','GLU').")