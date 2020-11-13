import torch
import torch.nn.functional as F

# PreEmpthasis & DeEmpthasis taken from https://github.com/AppleHolic/pytorch_sound/blob/master/pytorch_sound/models/sound.py#L64
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 3, 'The number of dimensions of input tensor must be 3!'
        # reflect padding to match lengths of in/out
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter)


# "Efficient" Loss Function
class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1., loss_empthasis=0.0, elementwise_mean=True):
        super().__init__()
        self.sigma2 = sigma ** 2
        self.sigma2_2 = self.sigma2 * 2
        self.mean = elementwise_mean
        if loss_empthasis > 0.0:
            print("'loss_empthasis' is depreciated.")
    
    def forward(self, model_outputs):
        z, logdet, logdet_w_sum, log_s_sum = model_outputs # [B, ...], [B]
        
        z = z.float()
        logdet = logdet.float()
        
        # [B, T] -> [B]
        loss = z.pow(2).sum(1) / self.sigma2_2
        loss = loss - logdet # safe original
        loss /= z.size(1) # average by segment length
        loss = loss.mean() # average by batch_size
        return loss
