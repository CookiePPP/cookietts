
def to_gpu(x):
    x = x.cuda(non_blocking=True)
    return x