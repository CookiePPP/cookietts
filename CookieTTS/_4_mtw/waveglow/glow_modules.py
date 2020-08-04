import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, set_grad_enabled, grad, gradcheck

from functools import reduce
from operator import mul


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
        
        #W = torch.randn(c, c).qr()[0]
        #self.weight.data = W[..., None] # init layer
        
        if memory_efficient:
            self.efficient_forward = Conv1x1Func.apply
            self.efficient_inverse = InvConv1x1Func.apply
    
    def forward(self, z):
        if hasattr(self, 'efficient_forward'):
            audio_out, log_det_W = self.efficient_forward(z, self.weight)
            z.storage().resize_(0)
            return audio_out, log_det_W
        else:
            batch_size, group_size, n_of_groups = z.size()
            log_det_W = n_of_groups * self.weight.squeeze().slogdet()[1]  # should fix nan logdet
            audio_out = super().forward(z)
            return audio_out, log_det_W

    def inverse(self, audio_out):
        if hasattr(self, 'efficient_inverse'):
            z, log_det_W = self.efficient_inverse(audio_out, self.weight)
            audio_out.storage().resize_(0)
            return z, log_det_W
        else:
            weight = self.weight.squeeze()
            *_, n_of_groups = audio_out.shape
            log_det_W = -n_of_groups * weight.slogdet()[1]  # should fix nan logdet
            z = F.conv1d(audio_out, weight.inverse().unsqueeze(-1))
            return z, log_det_W

class Conv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, weight):
        with torch.no_grad():
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * weight.squeeze().slogdet()[1]
            #log_det_W = n_of_groups * weight.squeeze().float().slogdet()[1].half()
            audio_out = F.conv1d(z, weight)
        
        ctx.save_for_backward(z.data, weight, audio_out)
        return audio_out, log_det_W
    
    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape
        
        with torch.no_grad():
            inv_weight = weight.squeeze().inverse()
            #inv_weight = weight.squeeze().float().inverse().half()
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, inv_weight.unsqueeze(-1))
            
            dx = F.conv1d(z_grad, weight[..., 0].t().unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight.shape[0], -1) @ z.transpose(1, 2).contiguous().view(
                -1, weight.shape[1])
            dw += inv_weight.t() * log_det_W_grad * n_of_groups
        
        return dx, dw.unsqueeze(-1)


class InvConv1x1Func(Function):
    @staticmethod
    def forward(ctx, z, inv_weight):
        with torch.no_grad():
            sqr_inv_weight = inv_weight.squeeze()
            *_, n_of_groups = z.shape
            log_det_W = n_of_groups * -sqr_inv_weight.slogdet()[1]
            audio_out = F.conv1d(z, sqr_inv_weight.inverse().unsqueeze(-1))

        ctx.save_for_backward(z.data, inv_weight, audio_out)
        return audio_out, log_det_W

    @staticmethod
    def backward(ctx, z_grad, log_det_W_grad):
        z, inv_weight, audio_out = ctx.saved_tensors
        *_, n_of_groups = audio_out.shape

        with torch.no_grad():
            z.storage().resize_(reduce(mul, audio_out.shape))
            z[:] = F.conv1d(audio_out, inv_weight)

            inv_weight = inv_weight.squeeze()
            weight_T = inv_weight.inverse().t()
            dx = F.conv1d(z_grad, weight_T.unsqueeze(-1))
            dw = z_grad.transpose(0, 1).contiguous().view(weight_T.shape[0], -1) @ \
                 z.transpose(1, 2).contiguous().view(-1, weight_T.shape[1])
            dinvw = - weight_T @ dw @ weight_T
            dinvw -= weight_T * log_det_W_grad * n_of_groups

        return dx, dinvw.unsqueeze(-1)


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
    
    def forward(self, audio, spect):
        if hasattr(self, 'efficient_forward'):
            audio, log_s = self.efficient_forward(audio, spect, self.WN, *self.param_list)
            #audio.storage().resize_(0)
            return audio, log_s
        else:
            audio_0, audio_1 = audio.chunk(2, 1)
            t, log_s = self.WN(audio_0, spect)
            audio_1 = audio_1 * log_s.exp() + t
            audio = torch.cat((audio_0, audio_1), 1)
            return audio, log_s
    
    def inverse(self, z, spect):
        if hasattr(self, 'efficient_inverse'):
            z, log_s = self.efficient_inverse(z, spect, self.WN, *self.param_list)
            #z.storage().resize_(0)
            return z, log_s
        else:
            audio_0, audio_1 = z.chunk(2, 1)
            t, log_s = self.WN(audio_0, spect)
            audio_1 = (audio_1 - t) / log_s.exp()
            z = torch.cat((audio_0, audio_1), 1)
            return z, -log_s


class AffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, z, spect, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            audio_0, audio_1 = audio.chunk(2, 1)
            audio_0, audio_1 = audio_0.contiguous(), audio_1.contiguous()
            
            t, log_s = F(audio_0, spect)
            audio_1_out = audio_1 * log_s.exp() + t
            audio_0_out = audio_0
            audio_out = torch.cat((audio_0_out, audio_1_out), 1)
        
        ctx.save_for_backward(audio.data, spect, audio_out)
        return audio_out, log_s

    @staticmethod
    def backward(ctx, z_grad, log_s_grad):
        F = ctx.F
        z, spect, audio_out = ctx.saved_tensors
        
        audio_0_out, audio_1_out = audio_out.chunk(2, 1)
        audio_0_out, audio_1_out = audio_0_out.contiguous(), audio_1_out.contiguous()
        dza, dzb = z_grad.chunk(2, 1)
        dza, dzb = dza.contiguous(), dzb.contiguous()
        
        with set_grad_enabled(True):
            audio_0 = audio_0_out
            audio_0.requires_grad = True
            t, log_s = F(audio_0, spect)
        
        with torch.no_grad():
            s = log_s.exp()
            audio_1 = (audio_1_out - t) / s
            z.storage().resize_(reduce(mul, audio_1.shape) * 2)
            torch.cat((audio_0, audio_1), 1, out=z) #fp32  # .contiguous()
            #torch.cat((audio_0.half(), audio_1.half()), 1, out=z)#fp16  # .contiguous()
            #z.copy_(xout)  # .detach()
        
        with set_grad_enabled(True):
            param_list = [audio_0] + list(F.parameters())
            if ctx.needs_input_grad[1]:
                param_list += [spect]
            dtsdxa, *dw = grad(torch.cat((log_s, t), 1), param_list,
                               grad_outputs=torch.cat((dzb * audio_1 * s + log_s_grad, dzb), 1))
        
            dxa = dza + dtsdxa
            dxb = dzb * s
            dx = torch.cat((dxa, dxb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
        
        return (dx, dy, None) + tuple(dw)


class InvAffineCouplingFunc(Function):
    @staticmethod
    def forward(ctx, audio_out, spect, F, *F_weights):
        ctx.F = F
        with torch.no_grad():
            audio_0_out, audio_1_out = audio_out.chunk(2, 1)
            audio_0_out, audio_1_out = audio_0_out.contiguous(), audio_1_out.contiguous()
            
            t, log_s = F(audio_0_out, spect)
            audio_1 = (audio_1_out - t) / log_s.exp()
            audio_0 = audio_0_out
            z = torch.cat((audio_0, audio_1), 1)
        
        ctx.save_for_backward(audio_out.data, spect, z)
        return z, -log_s
    
    @staticmethod
    def backward(ctx, x_grad, log_s_grad):
        F = ctx.F
        audio_out, spect, z = ctx.saved_tensors
        
        audio_0, audio_1 = z.chunk(2, 1)
        audio_0, audio_1 = audio_0.contiguous(), audio_1.contiguous()
        dxa, dxb = x_grad.chunk(2, 1)
        dxa, dxb = dxa.contiguous(), dxb.contiguous()
        
        with set_grad_enabled(True):
            audio_0_out = audio_0
            audio_0_out.requires_grad = True
            t, log_s = F(audio_0_out, spect)
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
            dtsdza, *dw = grad(torch.cat((-log_s, -t / s), 1), param_list,
                               grad_outputs=torch.cat((dxb * audio_1_out / s.detach() + log_s_grad, dxb), 1))
            
            dza = dxa + dtsdza
            dzb = dxb / s.detach()
            dz = torch.cat((dza, dzb), 1)
            if ctx.needs_input_grad[1]:
                *dw, dy = dw
            else:
                dy = None
        return (dz, dy, None) + tuple(dw)

