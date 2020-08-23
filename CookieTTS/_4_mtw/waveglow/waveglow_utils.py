import torch
import torch.nn.functional as F

# PreEmpthasis & DeEmpthasis taken from https://github.com/AppleHolic/pytorch_sound/blob/master/pytorch_sound/models/sound.py#L64
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.flipped_filter = torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
    
    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        input = input.unsqueeze(1)# [B, T] -> [B, 1, T]
        # reflect padding to match lengths of in/out
        input = F.pad(input, (1, 0), 'reflect')
        self.flipped_filter = self.flipped_filter.to(input)
        return F.conv1d(input, self.flipped_filter).squeeze(1)# [B, 1, T] -> [B, T]

# This one runs slow AF, use scipy.signal on CPU instead.
class InversePreEmphasis(torch.nn.Module):
    """
    Implement Inverse Pre-emphasis by using RNN to boost up inference speed.
    """
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.rnn = torch.nn.RNN(1, 1, 1, bias=False, batch_first=True)
        # use originally on that time
        self.rnn.weight_ih_l0.data.fill_(1)
        # multiply coefficient on previous output
        self.rnn.weight_hh_l0.data.fill_(self.coef)

    def forward(self, input: torch.tensor) -> torch.tensor:
        x, _ = self.rnn(input.transpose(1, 2))
        return x.transpose(1, 2)