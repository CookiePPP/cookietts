import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_grads
from torch.autograd import Function, set_grad_enabled, grad, gradcheck
import numpy as np

from functools import reduce
from operator import mul

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(zw, zf):
    t_act = torch.tanh(zw)
    s_act = torch.sigmoid(zf)
    acts = t_act * s_act
    return acts


class WaveFlowCoupling(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()
        self.memory_efficient = memory_efficient
        self.WN = transform_type(**kwargs)
    
    def forward(self, z, spect, speaker_ids):
        # pass first value through since that n_group does not have the prev sample information.
        z_shifted = z[:, :-1] # last value not needed (as there is nothing left to infer, thus no loss to be applied if used as input)
        
        if self.memory_efficient:
            log_s, t = checkpoint_grads(self.WN.__call__, z_shifted, spect, speaker_ids)
        else:
            log_s, t = self.WN(z_shifted, spect, speaker_ids)
        #assert not torch.isnan(log_s).any(), "log_s has NaN elements"
        #assert not torch.isnan(t).any(), "t has NaN elements"
        audio_out = torch.cat((z[:,:1], (z[:,1:] * log_s.exp()) + t), dim=1) # ignore changes to first sample since that conv has only padded information (which will not happen during inference)
        #assert not torch.isnan(audio_out).any(), "audio_out has NaN elements"
        return audio_out, log_s
    
    def inverse(self, audio_out, spect, speaker_ids):
        if hasattr(self, 'efficient_inverse'):
            print("gradient checkpointing not implemented for WaveFlowCoupling module inverse"); raise NotImplementedError
            #z, log_s = self.efficient_inverse(audio_out, spect, speaker_ids, self.WN, *self.param_list)
            #audio_out.storage().resize_(0)
            return z, log_s
        else:
            z = [audio_out[:,0:1],] # z is used as output tensor # initial sample # [B, 1, T//n_group]
            
            audio_queues = [None,]*self.WN.n_layers # create blank audio queue to flag for conv-queue
            spect_queues = [None,]*self.WN.n_layers # create blank spect queue to flag for conv-queue
            
            # conv queue with autoregressive initialized with zeros
            n_group = audio_out.shape[1]
            for i in range(n_group-1):# [0,1,2...22]
                audio_samp = z[-1]# just generated sample [B, 1, T//n_group]
                next_audio_samp = audio_out[:,i+1:i+2]# [B, n_group, T//n_group] -> [B, 1, T//n_group]
                
                acts, audio_queues, spect_queues = self.WN(audio_samp, spect, speaker_ids, audio_queues=audio_queues, spect_queues=spect_queues) # get next sample
                
                log_s, t = acts[:,:,-1:] # [2, B, 1, T//n_group] -> [B, 1, 1], [B, 1, 1]
                z.append( (next_audio_samp - t) / log_s.exp() ) # [B, 1, 1] save predicted next sample
            z = torch.cat(z, dim=1)# [B, n_group, T//n_group]
            return z, -log_s


class AffineCouplingBlock(nn.Module):
    def __init__(self,
                 transform_type,
                 memory_efficient=True,
                 **kwargs):
        super().__init__()
        
        self.WN = transform_type(**kwargs)
        if memory_efficient:
            self.efficient_forward = AffineCouplingFunc.apply
            self.efficient_inverse = InvAffineCouplingFunc.apply
            self.param_list = list(self.WN.parameters())
    
    def forward(self, z, spect, speaker_ids):
        if hasattr(self, 'efficient_forward'):
            audio_out, log_s = self.efficient_forward(z, spect, speaker_ids, self.WN, *self.param_list)
            z.storage().resize_(0)
            return audio_out, log_s
        else:
            audio_0, audio_1 = z.chunk(2, 1)
            audio_0_out = audio_0
            log_s, t = self.WN(audio_0, spect, speaker_ids)
            audio_1_out = audio_1 * log_s.exp() + t
            audio_out = torch.cat((audio_0_out, audio_1_out), 1)
            return audio_out, log_s
    
    def inverse(self, audio_out, spect, speaker_ids):
        if hasattr(self, 'efficient_inverse'):
            z, log_s = self.efficient_inverse(audio_out, spect, speaker_ids, self.WN, *self.param_list)
            audio_out.storage().resize_(0)
            return z, log_s
        else:
            audio_0_out, audio_1_out = audio_out.chunk(2, 1)
            audio_0 = audio_0_out
            log_s, t = self.WN(audio_0_out, spect, speaker_ids)
            audio_1 = (audio_1_out - t) / log_s.exp()
            z = torch.cat((audio_0, audio_1), 1)
            return z, -log_s


class AffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, z, spect, speaker_ids, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            audio_0, audio_1 = z.chunk(2, 1)
            audio_0, audio_1 = audio_0.contiguous(), audio_1.contiguous()
        
            log_s, t = F(audio_0, spect, speaker_ids)
            audio_1_out = audio_1 * log_s.exp() + t
            audio_0_out = audio_0
            audio_out = torch.cat((audio_0_out, audio_1_out), 1)
        
        ctx.save_for_backward(z.data, spect, speaker_ids, audio_out)
        return audio_out, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        z, spect, speaker_ids, audio_out = ctx.saved_tensors
        
        audio_0_out, audio_1_out = audio_out.chunk(2, 1)
        audio_0_out, audio_1_out = audio_0_out.contiguous(), audio_1_out.contiguous()
        dza, dzb = z_grad.chunk(2, 1)
        dza, dzb = dza.contiguous(), dzb.contiguous()
        
        with set_grad_enabled(True):
            audio_0 = audio_0_out
            audio_0.requires_grad = True
            log_s, t = F(audio_0, spect, speaker_ids)
        
        with torch.no_grad():
            s = torch.exp(log_s).half() # exp not implemented for fp16 therefore this is cast to fp32 by Nvidia/Apex
            audio_1 = (audio_1_out - t) / s # s is fp32 therefore audio_1 is cast to fp32.
            z.storage().resize_(reduce(mul, audio_1.shape) * 2) # z is fp16
            if z.dtype == torch.float16: # if z is fp16, cast audio_0 and audio_1 back to fp16.
              torch.cat((audio_0.half(), audio_1.half()), 1, out=z)#fp16  # .contiguous()
            else:
              torch.cat((audio_0, audio_1), 1, out=z) #fp32  # .contiguous()
            #z.copy_(xout)  # .detach()
        
        with set_grad_enabled(True):
            param_list = [audio_0] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [spect]
            if ctx.needs_input_grad[2]:
                param_list += [speaker_ids]
            dtsdxa, *dw = grad(torch.cat((log_s, t), 1), param_list,
                               grad_outputs=torch.cat((dzb * audio_1 * s + log_s_grad, dzb), 1))
            
            dxa = dza + dtsdxa
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
            if ctx.needs_input_grad[2]:
                *dw, ds = dw
            else:
                ds = None
        
        return (dx, dy, ds, None) + tuple(dw)


class InvAffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, audio_out, spect, speaker_ids, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            audio_0_out, audio_1_out = audio_out.chunk(2, 1)
            audio_0_out, audio_1_out = audio_0_out.contiguous(), audio_1_out.contiguous()
            
            log_s, t = F(audio_0_out, spect, speaker_ids)
            audio_1 = (audio_1_out - t) / log_s.exp()
            audio_0 = audio_0_out
            z = torch.cat((audio_0, audio_1), 1)
        
        ctx.save_for_backward(audio_out.data, spect, speaker_ids, z)
        return z, -log_s
    
    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        audio_out, spect, speaker_ids, z = ctx.saved_tensors
        
        audio_0, audio_1 = z.chunk(2, 1)
        audio_0, audio_1 = audio_0.contiguous(), audio_1.contiguous()
        dxa, dxb = x_grad.chunk(2, 1)
        dxa, dxb = dxa.contiguous(), dxb.contiguous()
        
        with set_grad_enabled(True):
            audio_0_out = audio_0
            audio_0_out.requires_grad = True
            log_s, t = F(audio_0_out, spect, speaker_ids)
            s = log_s.exp()
        
        with torch.no_grad():
            audio_1_out = audio_1 * s + t
            
            audio_out.storage().resize_(reduce(mul, audio_1_out.shape) * 2)
            torch.cat((audio_0_out, audio_1_out), 1, out=audio_out)
            #audio_out.copy_(zout)
        
        with set_grad_enabled(True):
            param_list = [audio_0_out] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [spect]
            if ctx.needs_input_grad[2]:
                param_list += [speaker_ids]
            dtsdza, *dw = grad(torch.cat((-log_s, -t / s), 1), param_list,
                               grad_outputs=torch.cat((dxb * audio_1_out / s.detach() + log_s_grad, dxb), 1))
            
            dza = dxa + dtsdza
            dzb = dxb / s.detach()
            dz = torch.cat((dza, dzb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
            if ctx.needs_input_grad[2]:
                *dw, ds = dw
            else:
                ds = None
            
        return (dz, dy, ds, None) + tuple(dw)


class InvertibleConv1x1(nn.Conv1d):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c, memory_efficient=False):
        super().__init__(c, c, 1, bias=False) # init as nn.Conv1d(c, c, kernel_size=1, stride=1) 
        
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.weight.data = W
        
        if memory_efficient:
            self.efficient_forward = Conv1x1Func.apply
            #self.efficient_inverse = InvConv1x1Func.apply # memory efficient Inverse is not needed and it's making fp16 more complicated so I'm ignoring it for now.
    
    def forward(self, z):
        if hasattr(self, 'efficient_forward'):
            audio_out, log_det_W = self.efficient_forward(z, self.weight)
            z.storage().resize_(0)
            return audio_out, log_det_W
        else:
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * self.weight.squeeze().float().slogdet()[1] # should fix nan logdet
            audio_out = super().forward(z)
            return audio_out, log_det_W
    
    def inverse(self, audio_out):
        W = self.weight.squeeze()
        if not hasattr(self, 'W_inverse'):
            W_inverse = W.float().inverse().unsqueeze(-1)
            if audio_out.dtype == torch.float16:
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        
        if hasattr(self, 'efficient_inverse'):
            z, log_det_W = self.efficient_inverse(audio_out, self.weight)
            audio_out.storage().resize_(0)
            return z, log_det_W
        else:
            log_det_W = None
            #*_, n_of_groups = audio_out.shape
            #log_det_W = -n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            z = F.conv1d(audio_out, self.W_inverse, bias=None, stride=1, padding=0)
            return z, log_det_W


class Conv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, weight):
        with torch.no_grad():
            *_, n_of_groups = z.shape
            if weight.dtype == torch.float16:
                log_det_W = n_of_groups * weight.squeeze().float().slogdet()[1].half()
            else:
                log_det_W = n_of_groups * weight.squeeze().slogdet()[1]
            audio_out = F.conv1d(z, weight)
        
        ctx.save_for_backward(z.data, weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            if weight.dtype == torch.float16:
                inv_weight = weight.squeeze().float().inverse().half()
            else:
                inv_weight = weight.squeeze().inverse()
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, inv_weight.unsqueeze(-1))
            
            dx = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ z.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups
        
        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, inv_weight, weight):
        with torch.no_grad():
            squ_weight = weight.squeeze()
            *_, n_of_groups = z.shape
            if squ_weight.dtype == torch.float16:
              log_det_W = -squ_weight.float().slogdet()[1].half() * n_of_groups
              audio_out = F.conv1d(z, squ_weight.inverse().unsqueeze(-1).half())
            else:
              log_det_W = -squ_weight.slogdet()[1] * n_of_groups
              audio_out = F.conv1d(z, squ_weight.inverse().unsqueeze(-1))
        
        ctx.save_for_backward(z.data, weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, weight)
            
            weight = weight.squeeze()
            weight_T = weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                 z.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups
        
        return dx, dinvw.unsqueeze(-1)


def permute_channels(x, height=None, reverse=False, bipart=False, shift=False, inverse_shift=False):
    if height is None:
        height = x.shape[1]
    h_permute = list(range(height))
    if bipart and reverse:
        half = len(h_permute)//2
        h_permute = h_permute[:half][::-1] + h_permute[half:][::-1] # reverse H halfs [0,1,2,3,4,5,6,7] -> [3,2,1,0] + [7,6,5,4] -> [3,2,1,0,7,6,5,4]
    elif reverse:
        h_permute = h_permute[::-1] # reverse entire H [0,1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1,0]
    if shift:
        h_permute = [h_permute[-1],] + h_permute[:-1] # shift last H into first position [0,1,2,3,4,5,6,7] -> [7,0,1,2,3,4,5,6]
    elif inverse_shift:
        h_permute = h_permute[1:] + [h_permute[0],]   # shift first H into last position [0,1,2,3,4,5,6,7] -> [1,2,3,4,5,6,7,0]
    return x[:,h_permute]


class PermuteHeight():
    """
    The layer outputs the permuted channel dim, and a placeholder log determinant.
    class.inverse() performs the permutation in reverse.
    """
    def __init__(self, n_remaining_channels, k, n_flows, sigma):
        assert n_flows%2==0, "PermuteHeight requires even n_flows"
        self.sigma = sigma
        self.const = -(0.5 * np.log(2 * np.pi) + np.log(self.sigma))/n_flows
        
        # b) we reverse Z(7), Z(6), Z(5), Z(4) over the height dimension as before, but bipartition Z(3), Z(2), Z(1), Z(0) in the middle of the height dimension then reverse each part respectively.
        if k%4 in (2,3): # Flows (2,3, 6,7, 10,11)
            self.bipart_reverse = True
            self.reverse = True
        else:            # Flows (0,1, 4,5, 8,9)
            self.bipart_reverse = False
            self.reverse = True
    
    def __call__(self, z):
        B, C, n_of_groups = z.shape
        audio_out = permute_channels(z, height=C, reverse=self.reverse, bipart=self.bipart_reverse)
        log_det_W = torch.tensor(n_of_groups*self.const, device=audio_out.device)
        return audio_out, log_det_W
    
    def inverse(self, audio_out):
        z = permute_channels(audio_out, reverse=self.reverse, bipart=self.bipart_reverse)
        log_det_W = None
        return z, log_det_W