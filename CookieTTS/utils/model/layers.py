import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple, Optional
from CookieTTS.utils.model.utils import get_mask_from_lengths


class DynamicConvolutionAttention(nn.Module):
    """A first attempt at making this Attention."""
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 dynamic_filter_num, dynamic_filter_len): # default 8, 21)
        super(DynamicConvolutionAttention, self).__init__()
        self.dynamic_filter_len = dynamic_filter_len
        self.dynamic_filter_num = dynamic_filter_num
        
        # static
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim, out_bias=True)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        
        # dynamic
        self.dynamic_filter = torch.nn.Sequential(
            LinearNorm(attention_rnn_dim, attention_dim, bias=True, w_init_gain='tanh'), # attention_rnn_dim -> attention dim
            nn.Tanh(),#nn.LeakyReLU(negative_slope=0.1),#nn.ReLU(),#nn.Tanh(),
            LinearNorm(attention_dim, dynamic_filter_num*dynamic_filter_len, bias=False, w_init_gain='tanh'), # filter_num * filter_length
            )
        self.vg = LinearNorm(dynamic_filter_num, attention_dim, bias=True)
        
        # prior
        self.prior_filter = self.get_prior_filter(dynamic_filter_num, dynamic_filter_len).to("cuda")
        
        # misc
        self.score_mask_value = -float("inf")
    
    def get_prior_filter(self, dynamic_filter_num, dynamic_filter_len):
        assert dynamic_filter_len == 21, "Only filter_len of 21 is currently supported" # I don't know how to calcuate this one atm so here's a set of premade values I found on their Reddit post
        prior_filters = torch.tensor( [0.7400209, 0.07474979, 0.04157422, 0.02947039, 0.023170564, 0.019321883, 0.016758798, 0.014978543, 0.013751862, 0.013028075, 0.013172861] ) # [filter_len-10]
        prior_filters = prior_filters.flip(dims=(0,)) # [filter_len-10] -> [filter_len-10]
        prior_filters = prior_filters[None, None, :] # [filter_len-10] -> [1, 1, filter_len-10]
        return prior_filters
    
    def get_alignment_energies(self, attention_RNN_state,
                               attention_weights_cat):
        """
        PARAMS
        ------
        attention_RNN_state: attention rnn last output [B, dim]       ## decoder output (batch, n_mel_channels * n_frames_per_step)
        attention_weights_cat: prev and cumulative att weights (B, 2, enc_time)
        
        RETURNS
        -------
        alignment (batch, enc_time)
        """
        verbose = 0 # debug
        # get Static filter intermediate value
        processed = self.location_layer(attention_weights_cat) # [B, 2, enc_T] -> [B, attention_n_filters, enc_T] -> [B, enc_T, attention_dim] # take prev+cumulative att weights, send through conv -> linear
        if verbose: print("1 processed.shape =", processed.shape) # [16, 90, 1]
        
        # get Dynamic filter intermediate value(s)
        prev_att = attention_weights_cat[:, 0, :][:, :, None] # [B, 2, enc_T] -> [B, enc_T] -> [B, enc_T, 1]
        dynamic_filt = self.dynamic_filter(attention_RNN_state) # [B, AttRNN_dim] -> [B, attention_dim] -> [B, 1, attention_dim] -> [B, 1, dynamic_filter_num*dynamic_filter_len]
        dynamic_filt = dynamic_filt.view([-1, self.dynamic_filter_len, self.dynamic_filter_num]) # [B, 1, dynamic_filter_num*dynamic_filter_len] -> [B, dynamic_filter_len, dynamic_filter_num]
        if verbose: print("1 prev_att.shape =", prev_att.shape) # [16, 90, 1]
        if verbose: print("1 dynamic_filt.shape =", dynamic_filt.shape) # [16, 21, 8]
        
        if True: # calc dynamic energies from matmul with dynamic filter
            # "stack previous alignments into matrices" # https://www.reddit.com/r/MachineLearning/comments/dmo0z1/r_attenchilada_locationrelative_attention/f6vtkmk/
            prev_att_stacked = prev_att.repeat(1,1,self.dynamic_filter_len)
            dynamic = prev_att_stacked @ dynamic_filt # [B, enc_T, dynamic_filter_len] @ [B, dynamic_filter_len, dynamic_filter_num] -> [B, enc_T, dynamic_filter_num]
            if True: # extra linear?
                dynamic = self.vg(dynamic) # [B, enc_T, dynamic_filter_num] -> [B, enc_T, attention_dim]
                pass
        else:  # calc dynamic engeries from F.conv1d with dynamic filter
            dynamic_filt = dynamic_filt.permute(2,0,1)[:,:,None,:] # [B, dynamic_filter_len, dynamic_filter_num] -> [dynamic_filter_num,B                 ,1, dynamic_filter_len]
                                                                                                                   #(out_channels      ,in_channels/groups,kH,kW                )
            if verbose: print("1.9 dynamic_filt.shape =", dynamic_filt.shape) # [8, 24, 1, 21]
            shape = dynamic_filt.shape
            dynamic_filt = dynamic_filt.reshape(shape[0]*shape[1], 1, 1, shape[-1]) # [dynamic_filter_num,B,1, dynamic_filter_len] -> [dynamic_filter_num*B,1,1, dynamic_filter_len]
            prev_att_shaped = prev_att[None, ...].permute(0, 1, 3, 2) # [B, enc_T, 1] -> [1, B, enc_T, 1] -> [1        ,B          ,1 ,enc_T]
                                                                                                            #(minibatch,in_channels,iH,iW   )
            if verbose: print("2 prev_att_shaped.shape =", prev_att_shaped.shape) # [1, 24, 1, 65] -> [24, 8, 1, 65]
            if verbose: print("2 dynamic_filt.shape =", dynamic_filt.shape) # [8, 24, 1, 21]
            if False:
                padd = (self.dynamic_filter_len-1)//2
                dynamic = torch.nn.functional.conv2d(prev_att_shaped, dynamic_filt, bias=None, stride=1, padding=(0,padd), dilation=1, groups=prev_att_shaped.size(1))# [1, B, 1, enc_T] -> [1, B*dyna_f_num, 1, enc_T]
            else:
                padd = self.dynamic_filter_len - 1
                prev_att_shaped = F.pad(prev_att_shaped, (padd, 0)) # [1, 1, B, enc_T] -> [1, 1, B, padd+enc_T]
                dynamic = torch.nn.functional.conv2d(prev_att_shaped, dynamic_filt, bias=None, stride=1, padding=0, dilation=1, groups=prev_att_shaped.size(1))# [1, B, 1, enc_T] -> [1, B*dyna_f_num, 1, enc_T]
            if verbose: print("2 dynamic.shape =", dynamic.shape) # [1, 8, 1, 65]
            dynamic = dynamic.view(shape[1], shape[0], -1).transpose(1,2) # [1, B*dyna_f_num, 1, enc_T] -> [B, dyna_f_num, enc_T] -> [B, enc_T, dyna_f_num]
        
        if verbose: print("2.1 dynamic.shape =", dynamic.shape) # [1, 8, 1, 65]
        
        # I don't currently know how the Dynamic and Static energies are meant to interact (I can't tell from the paper).
        if True: # first try addition
            energies = self.v( torch.tanh( processed + dynamic ) ) # [B, enc_T, attention_dim] -> [B, enc_T, 1] # mix them
        elif True: # then try concatentation
            proc = torch.cat( (processed, dynamic), dim=2) # [B, enc_T, dynamic_filter_num] + [B, enc_T, attention_dim] -> [B, enc_T, dynamic_filter_num+attention_dim]
            energies = self.v( torch.tanh( proc ) ) # [B, enc_T, attention_dim] -> [B, enc_T, 1] # mix them
        elif True: # then try adding seperated energies
            static_energies = self.v( torch.tanh( processed ) ) # [B, enc_T, attention_dim] -> [B, enc_T, 1] # mix them
            if verbose: print("static_energies.shape =", static_energies.shape)
            dynamic_energies = self.vg( torch.tanh(dynamic) ) # [B, enc_T, dynamic_filter_num] -> [B, enc_T, 1] # mix them
            if verbose: print("dynamic_energies.shape =", dynamic_energies.shape)
            energies = static_energies + dynamic_energies # [B, enc_T, 1] + [B, enc_T, 1] -> [B, enc_T, 1]
        else:
            pass
        
        if False: # add the Prior filter
            padd = self.dynamic_filter_len - 11
            prev_att = F.pad(prev_att.transpose(1,2), (padd, 0)) # [B, enc_T, 1] -> [B, 1, enc_T] -> [B, 1, enc_T+padd]
            prior_energy = F.conv1d(prev_att, self.prior_filter.to(prev_att.dtype)) # [B, 1, enc_T+padd] -> [B, 1, enc_T]
            prior_energy = (prior_energy.clamp(min=1e-6)).log() # [B, enc_T, 1] clamp min value so log doesn't underflow
            #prior_energy = prior_energy.clamp(min=1e-6)
            #prior_energy = prior_energy.squeeze(-1) # [B, enc_T, 1] -> [B, enc_T]
            
            if verbose: print("3 energies.shape =", energies.shape) #
            if verbose: print("3 prior_energy.shape =", prior_energy.shape) #
            
            energies += prior_energy.transpose(1,2) # [B, enc_T, 1]
        
        return energies.squeeze(-1) # [B, enc_T, 1] -> [B, enc_T] # squeeze blank dim

    def forward(self, attention_RNN_state, attention_weights_cat, memory, mask, attention_weights=None):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        if attention_weights is None:
            alignment = self.get_alignment_energies(
                attention_RNN_state, attention_weights_cat) # outputs [B, enc_T]
            
            if mask is not None:
                alignment.data.masked_fill_(mask, self.score_mask_value)
            
            attention_weights = F.softmax(alignment, dim=1) # softmax
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # unsqueeze, bmm
        attention_context = attention_context.squeeze(1) # squeeze
        
        return attention_context, attention_weights


class GMMAttention(nn.Module): # Experimental from NTT123
    def __init__(self, num_mixtures, attention_layers, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size, hparams):
        super(GMMAttention, self).__init__()
        self.num_mixtures = num_mixtures
        self.normalize_attention_input = hparams.normalize_attention_input
        self.delta_min_limit = hparams.delta_min_limit
        self.delta_offset = hparams.delta_offset
        self.lin_bias = hparams.lin_bias
        self.initial_gain = hparams.initial_gain
        lin = nn.Linear(attention_dim, 3*num_mixtures, bias=self.lin_bias)
        lin.weight.data.mul_(0.01)
        if self.lin_bias:
            lin.bias.data.mul_(0.008)
            lin.bias.data.sub_(2.0)
        
        if attention_layers == 1:
            self.F = nn.Sequential(
                    LinearNorm(attention_rnn_dim, attention_dim, bias=True, w_init_gain=self.initial_gain),
                    nn.Tanh(),
                    lin)
        elif attention_layers == 2:
            self.F = nn.Sequential(
                    LinearNorm(attention_rnn_dim, attention_dim, bias=True, w_init_gain=self.initial_gain),
                    LinearNorm(attention_dim, attention_dim, bias=False, w_init_gain='tanh'),
                    nn.Tanh(),
                    lin)
        else:
            print(f"attention_layers invalid, valid values are... 1, 2\nCurrent Value {attention_layers}")
            raise
        
        self.score_mask_value = 0 # -float("inf")
        
        self.register_buffer('pos', torch.arange(
            0, 2000, dtype=torch.float).view(1, -1, 1).data)
    
    
    def get_alignment_energies(self, attention_hidden_state, memory, previous_location):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        memory: encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        
        RETURNS
        -------
        alignment (batch, max_time)
        """
        if self.normalize_attention_input:
            attention_hidden_state = attention_hidden_state.tanh()
        w, delta, scale = self.F(attention_hidden_state.unsqueeze(1)).chunk(3, dim=-1)
        delta = delta.sigmoid()#*1.0 # normalize 0.0 - 1.0,
        if self.delta_min_limit:
            delta = delta.clamp(min=self.delta_min_limit) # supposed to be fine with autograd but not 100% confident.
        if self.delta_offset:
            delta = delta + self.delta_offset
        loc = previous_location + delta
        scale = scale.sigmoid() * 2 + 1
        
        if True: # I don't know anything about this but both versions exist
            pos = self.pos[:, :memory.shape[1], :]
            z1 = torch.erf((loc-pos+0.5)*scale)
            z2 = torch.erf((loc-pos-0.5)*scale)
            z = (z1 - z2)*0.5
            w = torch.sigmoid(w) #w = torch.softmax(w, dim=-1) # not sure which to use
        else:
            std = torch.nn.functional.softplus(scale + 5) + 1e-5
            pos = self.pos[:, :memory.shape[1], :]
            z1 = torch.tanh((loc-pos+0.5) / std)
            z2 = torch.tanh((loc-pos-0.5) / std)
            z = (z1 - z2)*0.5
            w = torch.softmax(w, dim=-1) + 1e-5
        
        z = torch.bmm(z, w.squeeze(1).unsqueeze(2)).squeeze(-1) # [B, enc_T, num_mixtures] @ [B, num_mixtures, 1] -> [B, enc_T]
        # z = z.sum(dim=-1)
        return z, loc
    
    def forward(self, attention_hidden_state, memory, previous_location, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment, loc = self.get_alignment_energies(attention_hidden_state, memory, previous_location)
        
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        
        #attention_weights = alignment # without softmax
        attention_weights = F.softmax(alignment, dim=1) # with softmax
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights, loc


from torch.nn.modules.rnn import LSTMCell
class LSTMCellWithZoneout(nn.LSTMCell):# taken from https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/tacotron2/decoder.html
                                           # and modified with dropout and to use LSTMCell as base
    """ZoneOut Cell module.
    
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.

    Examples:
        >>> lstm = LSTMCellWithZoneout(16, 32, zoneout=0.5)

    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305

    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch

    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, dropout: float=0.0, zoneout: float=0.0, inplace=True):
        """Initialize zone out cell module.
        
        Args:
            cell (torch.nn.Module): Pytorch recurrent cell module
                e.g. `torch.nn.Module.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.
        
        """
        super().__init__(input_size, hidden_size, bias)
        self.zoneout_rate = zoneout
        self.dropout_rate = dropout
        self.inplace      = inplace
        if zoneout > 1.0 or zoneout < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")
        if dropout > 1.0 or dropout < 0.0:
            raise ValueError("dropout probability must be in the range from 0.0 to 1.0.")
    
    def forward(self, inputs: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        
        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).

        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).

        """
        next_hidden = super(LSTMCellWithZoneout, self).forward(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        next_hidden = self._dropout(hidden, next_hidden, self.dropout_rate)
        return next_hidden
    
    def _dropout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._dropout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )
        
        return F.dropout(next_h, prob, self.training, self.inplace)
    
    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


# adapted from https://github.com/pytorch/pytorch/blob/1f40f2a1723fd3b9302bea3829f2570c8e0e7e94/benchmarks/fastrnns/custom_lstms.py#L184-L211
class SegLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args, dropout:float=0.0, zoneout:float=0.1, dropout_inplace:bool=True, **cell_kwargs):
        super(SegLSTMLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)
        self.zoneout_rate = zoneout
        self.dropout_rate = dropout
        self.inplace      = dropout_inplace
        if zoneout > 1.0 or zoneout < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")
        if dropout > 1.0 or dropout < 0.0:
            raise ValueError("dropout probability must be in the range from 0.0 to 1.0.")
    
    @jit.script_method
    def _dropout(self, h:Tensor, next_h:Tensor, prob:float):
        return F.dropout(next_h, prob, self.training, self.inplace)
    
    @jit.script_method
    def _zoneout(self, h:Tensor, next_h:Tensor, prob:float):
        if self.training:
            mask = h.new_empty(h.shape[0], h.shape[1]).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h
    
    @jit.script_method
    def get_mask_from_duration(self, durs: Tensor):
        padded_states      = durs.eq(0).sum(1)   # [B]         <- number of txt with zero duration.
        forward_reset_idxs = durs.cumsum(dim=1)-1# [B, txt_T]  <- indexes to reset the hidden states.
        
        mel_lengths = durs.sum(1)#[B]
        
        seq = torch.arange(durs.sum(1).max(), device=durs.device)[None, :, None]# [B, mel_T, 1]
        reset_mask = (seq==forward_reset_idxs.unsqueeze(1)).any(dim=2)# [B, mel_T] <- reset indexes as a mask. True == Reset, False = Retain
        retain_mask = ~reset_mask# [B, mel_T]
        
        pad_mask = get_mask_from_lengths(padded_states)# [B, zeroed duration]
        pad_zero = pad_mask*.0                         # [B, zeroed duration]
        return reset_mask.t(), retain_mask.t(), pad_mask.t(), pad_zero.t(), mel_lengths
    
    @jit.script_method
    def forward(self, input: Tensor,               # [x_T, B, in_dim]
                       durs: Tensor,               # [y_T, B]
                      state: Optional[Tuple[Tensor, Tensor]]=None,# ([B, cell_dim], [B, cell_dim])
                 reset_mask: Optional[Tensor]=None,# [x_T, B]
                retain_mask: Optional[Tensor]=None,# [x_T, B]
                   pad_mask: Optional[Tensor]=None,# [?_T, B]
                   pad_zero: Optional[Tensor]=None,# [?_T, B]
                mel_lengths: Optional[Tensor]=None,# [B]
               ) -> Tuple[Tensor, List[Tensor], Tensor]:
        inputs = input.unbind(0)# [x_T, B, in_dim] -> [[B, in_dim],]*x_T
        assert len(inputs) == durs.sum(0).max(), f'got max length of {durs.sum(0).max().item()}, expected {len(inputs)}'
        
        if reset_mask is None or retain_mask is None or pad_mask is None or pad_zero is None or mel_lengths is None:
            reset_mask, retain_mask, pad_mask, pad_zero, mel_lengths = self.get_mask_from_duration(durs.transpose(0, 1))# [txt_T, B] -> [B, txt_T]
        assert  reset_mask.shape[0] == len(inputs), f'reset_mask has len of {reset_mask.shape[0]}, expected {len(inputs)}'
        assert retain_mask.shape[0] == len(inputs), f'retain_mask has len of {retain_mask.shape[0]}, expected {len(inputs)}'
        assert  reset_mask.shape[1] == input[0].shape[0]# reset_mask  [x_T, B]
        assert retain_mask.shape[1] == input[0].shape[0]# retain_mask [x_T, B]
        assert    pad_mask.shape[1] == input[0].shape[0]# pad_mask    [?_T, B]
        assert    pad_zero.shape[1] == input[0].shape[0]# pad_zero    [?_T, B]
        
        if state is None:
            blank_state = input.new_zeros(inputs[0].shape[0], self.cell.hidden_size)
            state = (blank_state, blank_state)
        assert state is not None
        
        mask = ~get_mask_from_lengths(mel_lengths).transpose(0, 1)# [x_T, B]
        retain_mask_ = (~(reset_mask | mask)).unsqueeze(2).unbind(0)# [x_T, B] -> [[B, 1],]*x_T
        reset_mask_  =   (reset_mask | mask) .unsqueeze(2).unbind(0)# [x_T, B] -> [[B, 1],]*x_T
        
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            prev_state = state
            state = self.cell(inputs[i], state)# -> ([B, cell_dim], [B, cell_dim])
            state = (self._zoneout(prev_state[0], state[0], self.zoneout_rate), self._zoneout(prev_state[1], state[1], self.zoneout_rate))
            state = (self._dropout(prev_state[0], state[0], self.dropout_rate), self._dropout(prev_state[1], state[1], self.dropout_rate))
            outputs += [state[0]]# [B, cell_dim]
            if reset_mask_[i].any():
                state = (state[0]*retain_mask_[i], state[1]*retain_mask_[i])# reset hidden + cell at end of segments
        outputs = torch.stack(outputs)# [x_T, B, cell_dim]
        outputs *= get_mask_from_lengths(mel_lengths).transpose(0, 1).unsqueeze(-1)# [B, x_T]
        
        # [mel_T, B, cell_dim], [?_T, B] -> [mel_T, B, cell_dim], [?_T, B, cell_dim] -> [mel_T+?_T, B, cell_dim]
        segout = torch.cat((outputs, pad_zero.unsqueeze(2).repeat(1, 1, outputs.shape[2])), dim=0)# [mel_T+?_T, B, cell_dim]
        
        # [mel_T, B], [?_T, B] -> [mel_T+?_T, B, cell_dim]
        segout_mask = torch.cat((reset_mask, pad_mask), dim=0).unsqueeze(2).repeat(1, 1, segout.shape[2])
        
        segout = segout.transpose(0, 1)[segout_mask.transpose(0, 1)].view(durs.shape[1], durs.shape[0], -1)# [B, y_T, cell_dim]
        
        return outputs, state, segout# [x_T, B, cell_dim], ([B, cell_dim], [B, cell_dim]), [B, y_T, cell_dim]


# adapted from https://github.com/pytorch/pytorch/blob/1f40f2a1723fd3b9302bea3829f2570c8e0e7e94/benchmarks/fastrnns/custom_lstms.py#L184-L211
class SegReverseLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args, dropout:float=0.0, zoneout:float=0.1, dropout_inplace:bool=True, **cell_kwargs):
        super(SegReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)
        self.zoneout_rate = zoneout
        self.dropout_rate = dropout
        self.inplace      = dropout_inplace
        if zoneout > 1.0 or zoneout < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")
        if dropout > 1.0 or dropout < 0.0:
            raise ValueError("dropout probability must be in the range from 0.0 to 1.0.")
    
    @jit.script_method
    def _dropout(self, h:Tensor, next_h:Tensor, prob:float):
        return F.dropout(next_h, prob, self.training, self.inplace)
    
    @jit.script_method
    def _zoneout(self, h:Tensor, next_h:Tensor, prob:float):
        if self.training:
            mask = h.new_empty(h.shape[0], h.shape[1]).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h
    
    @jit.script_method
    def get_mask_from_duration(self, durs: Tensor):
        padded_states      = durs.eq(0).sum(1)   # [B]         <- number of txt with zero duration.
        
        mel_lengths = durs.sum(1)#[B]
        n_padded_frames = mel_lengths.max()-mel_lengths# [B]
        
        rev_reset_idxs = (durs.flip(1).cumsum(dim=1)-1).flip(1)+n_padded_frames.unsqueeze(1)# [B, txt_T]  <- indexes to reset the hidden states.
        rev_reset_idxs.masked_fill_(durs.eq(0), -1)
        
        seq = torch.arange(durs.sum(1).max(), device=durs.device)[None, :, None]# [B, mel_T, 1]
        rev_reset_mask = (seq==rev_reset_idxs.unsqueeze(1)).any(dim=2)# [B, mel_T] <- reset indexes as a mask. True == Reset, False = Retain
        rev_retain_mask = ~rev_reset_mask# [B, mel_T]
        
        pad_mask = get_mask_from_lengths(padded_states)# [B, zeroed duration]
        pad_zero = pad_mask*.0                         # [B, zeroed duration]
        return rev_reset_mask.t(), rev_retain_mask.t(), pad_mask.t(), pad_zero.t(), mel_lengths
    
    @jit.script_method
    def forward(self, input: Tensor,               # [x_T, B, in_dim]
                       durs: Tensor,               # [y_T, B]
                      state: Optional[Tuple[Tensor, Tensor]]=None,# ([B, cell_dim], [B, cell_dim])
                 reset_mask: Optional[Tensor]=None,# [x_T, B]
                retain_mask: Optional[Tensor]=None,# [x_T, B]
                   pad_mask: Optional[Tensor]=None,# [?_T, B]
                   pad_zero: Optional[Tensor]=None,# [?_T, B]
                mel_lengths: Optional[Tensor]=None,# [B]
               ) -> Tuple[Tensor, List[Tensor], Tensor]:
        inputs = input.flip(0).unbind(0)# [x_T, B, in_dim] -> [[B, in_dim],]*x_T
        assert len(inputs) == durs.sum(0).max(), f'got max length of {durs.sum(0).max().item()}, expected {len(inputs)}'
        
        if reset_mask is None or retain_mask is None or pad_mask is None or pad_zero is None or mel_lengths is None:
            reset_mask, retain_mask, pad_mask, pad_zero, mel_lengths = self.get_mask_from_duration(durs.transpose(0, 1))# [txt_T, B] -> [B, txt_T]
        assert  reset_mask.shape[0] == len(inputs), f'reset_mask has len of {reset_mask.shape[0]}, expected {len(inputs)}'
        assert retain_mask.shape[0] == len(inputs), f'retain_mask has len of {retain_mask.shape[0]}, expected {len(inputs)}'
        assert  reset_mask.shape[1] == input[0].shape[0]# reset_mask  [x_T, B]
        assert retain_mask.shape[1] == input[0].shape[0]# retain_mask [x_T, B]
        assert    pad_mask.shape[1] == input[0].shape[0]# pad_mask    [?_T, B]
        assert    pad_zero.shape[1] == input[0].shape[0]# pad_zero    [?_T, B]
        
        if state is None:
            blank_state = input.new_zeros(inputs[0].shape[0], self.cell.hidden_size)
            state = (blank_state, blank_state)
        assert state is not None
        
        mask = ~get_mask_from_lengths(mel_lengths).transpose(0, 1)# [x_T, B]
        retain_mask_ = (~(reset_mask | mask)).unsqueeze(2).unbind(0)# [x_T, B] -> [[B, 1],]*x_T
        reset_mask_  =   (reset_mask | mask) .unsqueeze(2).unbind(0)# [x_T, B] -> [[B, 1],]*x_T
        
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            prev_state = state
            state = self.cell(inputs[i], state)# -> ([B, cell_dim], [B, cell_dim])
            state = (self._zoneout(prev_state[0], state[0], self.zoneout_rate), self._zoneout(prev_state[1], state[1], self.zoneout_rate))
            state = (self._dropout(prev_state[0], state[0], self.dropout_rate), self._dropout(prev_state[1], state[1], self.dropout_rate))
            outputs += [state[0]]# [B, cell_dim]
            if reset_mask_[i].any():
                state = (state[0]*retain_mask_[i], state[1]*retain_mask_[i])# reset hidden + cell at end of segments
        outputs = torch.stack(outputs)# [x_T, B, cell_dim]
        outputs *= get_mask_from_lengths(mel_lengths).transpose(0, 1).flip(0).unsqueeze(-1)# [B, x_T] -> [x_T, B, 1]
        
        # [mel_T, B, cell_dim], [?_T, B] -> [mel_T, B, cell_dim], [?_T, B, cell_dim] -> [mel_T+?_T, B, cell_dim]
        segout = torch.cat((pad_zero.unsqueeze(2).repeat(1, 1, outputs.shape[2]), outputs), dim=0)# [mel_T+?_T, B, cell_dim]
        
        # [mel_T, B], [?_T, B] -> [mel_T+?_T, B, cell_dim]
        segout_mask = torch.cat((pad_mask, reset_mask), dim=0).unsqueeze(2).repeat(1, 1, segout.shape[2])
        
        segout = segout.transpose(0, 1)[segout_mask.transpose(0, 1)].view(durs.shape[1], durs.shape[0], -1)# [B, y_T, cell_dim]
        
        return outputs.flip(0), state, segout.flip(1)# [x_T, B, cell_dim], ([B, cell_dim], [B, cell_dim]), [B, y_T, cell_dim]


class SegLSTMWithZoneout(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int=1, bias: bool=True, batch_first:bool=False, dropout: float=0.0, bidirectional:bool=False, zoneout: float=0.0, inplace=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.bias        = bias
        self.batch_first = batch_first
        
        self.forward_cell = []
        for i in range(num_layers):  
            self.forward_cell.append(SegLSTMLayer(nn.LSTMCell, input_size, hidden_size, bias, dropout=dropout, zoneout=zoneout, dropout_inplace=inplace))
        self.forward_cell = nn.ModuleList(self.forward_cell)
        
        if bidirectional:
            self.reverse_cell = []
            for i in range(num_layers):
                self.reverse_cell.append(SegReverseLSTMLayer(nn.LSTMCell, input_size, hidden_size, bias, dropout=dropout, zoneout=zoneout, dropout_inplace=inplace))
            self.reverse_cell = nn.ModuleList(self.reverse_cell)
    
    def forward(self, x: Tensor, durs: Tensor):# [x_T, B, D], [x_T, B]
        forward_args = self.forward_cell[0].get_mask_from_duration(durs.transpose(0, 1))
        if hasattr(self, 'reverse_cell'):
            reverse_args = self.reverse_cell[0].get_mask_from_duration(durs.transpose(0, 1))
        
        for i in range(self.num_layers):
            x, (h, c), seg_x = self.forward_cell[i](x, durs, None, *forward_args)
            if hasattr(self, 'reverse_cell'):
                rev_x, (rev_h, rev_c), rev_seg_x = self.reverse_cell[i](x, durs, None, *reverse_args)
                x = x + rev_x
        
        if hasattr(self, 'reverse_cell'):
            seg_x = seg_x + rev_seg_x
            h = h + rev_h
            c = c + rev_c
        return x, (h, c), seg_x


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear', dropout=0.):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = dropout
        
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        x = self.linear_layer(x)
        if self.training and self.dropout > 0.:
            x = F.dropout(x, p=self.dropout)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', dropout=0., causal=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        
        self.dropout = dropout
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=0 if causal else padding,
                                    dilation=dilation, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        self.causal_pad = (kernel_size-1)*dilation if causal else 0
    
    def maybe_pad(self, signal, pad_right=False):
        if self.causal_pad:
            if pad_right:
                signal = F.pad(signal, (0, self.causal_pad))
            else:
                signal = F.pad(signal, (self.causal_pad, 0))
        return signal
    
    def forward(self, signal):
        conv_signal = self.conv(self.maybe_pad(signal))
        if self.training and self.dropout > 0.:
            conv_signal = F.dropout(conv_signal, p=self.dropout)
        return conv_signal


class ConvNorm2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', dropout=0.):
        super(ConvNorm2D, self).__init__()
        self.dropout = dropout
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=1, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        if self.training and self.dropout > 0.:
            conv_signal = F.dropout(conv_signal, p=self.dropout)
        return conv_signal


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout_val = dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        if self.dropout_val > 0.:
            out = self.dropout(out)
        return out
