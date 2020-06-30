import torch_xla
import torch_xla.core.xla_model as xm

def to_gpu(x):
    x = x.to(xm.xla_device())
    return x