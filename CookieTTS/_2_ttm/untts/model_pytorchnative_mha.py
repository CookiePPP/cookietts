from math import sqrt
import numpy as np
from numpy import finfo
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from CookieTTS.utils.model.layers import ConvNorm, ConvNorm2D, LinearNorm
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, get_mask_3d
from CookieTTS._2_ttm.untts.fastpitch.length_predictor import TemporalPredictor
from CookieTTS._2_ttm.untts.fastpitch.transformer import PositionalEmbedding
from CookieTTS._2_ttm.untts.waveglow.glow import FlowDecoder
from CookieTTS._2_ttm.untts.waveglow.durglow import DurationGlow

drop_rate = 0.5

def load_model(hparams):
    model = UnTTS(hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None, attention=None, t_max=None, t_min=None):
        if attention is None:
            dk = query.size()[-1]
            scores = query.matmul(key.transpose(-2, -1)) / sqrt(dk)# [B*n_head, dec_T, enc_dim//n_head] @ [B*n_head, enc_T, enc_dim//n_head].t() -> [B*n_head, dec_T, enc_T]
            if (t_min is not None) and (t_max is not None):
                scores_shape = scores.shape
                T = torch.rand((scores_shape[0], scores_shape[1], 1), device=scores.device, dtype=scores.dtype) * (t_max-t_min) + t_min # [B*n_head, dec_T, enc_T] temperature
                scores = scores/T# [B*n_head, dec_T, enc_T]/[B*n_head, dec_T, 1]
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -65500.0)# [B*n_head, dec_T, enc_T]
            attention = F.softmax(scores, dim=-1) # softmax along enc dim
        attention_context = attention.matmul(value)# [B*n_head, dec_T, enc_T] @ [B*n_head, enc_T, enc_dim//n_head] -> [B*n_head, dec_T, enc_dim//n_head]
        return attention_context, attention# [B*n_head, dec_T, enc_dim//n_head], [B*n_head, dec_T, enc_T]

# https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
    
    def forward(self, q, k, v, mask=None, attention_override=None, t_max=None, t_min=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)# [B, dec_T, enc_dim], [B, enc_T, enc_dim], [B, enc_T, enc_dim]
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        
        q = self._reshape_to_batches(q)# [B, dec_T, enc_dim] -> [B*n_head, dec_T, enc_dim//n_head]
        k = self._reshape_to_batches(k)# [B, enc_T, enc_dim] -> [B*n_head, enc_T, enc_dim//n_head]
        v = self._reshape_to_batches(v)# [B, enc_T, enc_dim] -> [B*n_head, enc_T, enc_dim//n_head]
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)# [B, dec_T, enc_T] -> [B*n_head, dec_T, enc_T]
        if attention_override is not None and attention_override.shape[0] != q.shape[0]:
            attention_override = attention_override.repeat(self.head_num, 1, 1)
        y, attention_scores = ScaledDotProductAttention()(q, k, v, mask, attention=attention_override, t_max=t_max, t_min=t_min)
        y = self._reshape_from_batches(y)# [B*n_head, dec_T, enc_dim//n_head] -> [B, dec_T, enc_dim]
        
        att_shape = attention_scores.shape
        attention_scores = attention_scores.view(att_shape[0]//self.head_num, self.head_num, *att_shape[1:])
        # [B*n_head, dec_T, enc_T] -> [B, n_head, dec_T, enc_T]
        
        y = self.linear_o(y)# [B, dec_T, enc_dim]
        if self.activation is not None:
            y = self.activation(y)
        return y, attention_scores# [B, dec_T, enc_dim], [B, n_head, dec_T, enc_T]
    
    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)
    
    def _reshape_to_batches(self, x):# [B, enc_T, enc_dim] -> [B*n_head, enc_T, enc_dim//n_head]
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)
    
    def _reshape_from_batches(self, x):# [B*n_head, enc_T, enc_dim//n_head] -> [B, enc_T, enc_dim]
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)
    
    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class MoBoAlignerAttention(nn.Module):
    def __init__(self, hparams):
        super(MoBoAlignerAttention, self).__init__()
    
    def get_T(self, scores, min=0.1, max=1.0):
        # get uniform distribution between min and max
        T = torch.rand(scores.shape, device=scores.device, dtype=scores.dtype) * (max-min) + min # temperature
        return T
    
    @torch.jit.script
    def MoBoSearch(scores, D: int):
    #def MoBoSearch(self, scores, D: int):
        #scores [B, dec_T, enc_T]
        a = scores*0
        a[:, 0, :] = 0   # Eq (4)  if i == 0: Assign 1.0
        a[:, 0, 0] = 1.0 # Eq (4)       else: Assign 0.0
        B = scores*0
        J = scores.shape[1]-1# dec max index
        I = scores.shape[2]-1# enc max index
        a_len = scores.shape
        
        kw = D-1 # width of window to sum over
        out_channels = scores.shape[2]
        energy_sum = F.conv1d(F.pad(scores.transpose(1, 2), (0,kw-1)), torch.ones(out_channels, 1, kw, device=scores.device, dtype=scores.dtype), groups=out_channels).transpose(1, 2)# [B, J, I] -> [B, J, I] summed in windows across J dim
        cond_prob = scores / (energy_sum+1e-8) # Eq (3)
        assert torch.isnan(cond_prob).sum() < 1, 'assertion failure'
        
        a_rray = []
        cond_prob_ = F.pad(cond_prob, (0,0,D,0))# [B, J, I]
        for i in range(scores.shape[2]):# enc
            # get ai,j = P(Bi = j)
            a_ = F.pad(a_rray[-1] if len(a_rray) else a[:, :, max(i-1, 0)], (D,0))# [B, J, I]
            #a[:, :, i] = torch.stack([ a_[:, k:a_len[1]+k, max(i-1, 0)]*cond_prob_[:,k:a_len[1]+k, max(i-1, 0)] for k in range(D-1)], dim=0).sum((0,))
            
            input = a_ * cond_prob_[:, :, max(i-1, 0)]# [B, J]
            a_rray.append( torch.stack([ input[:, k:a_len[1]+k] for k in range(D-1)], dim=0).sum((0,)) )
        a = torch.stack(a_rray, dim=2)
        assert torch.isnan(a).sum() < 1, 'assertion failure'
        
        # 5.4s to Calc B using this method :/
        #B = jitloop(B, a, cond_prob, scores, D, J, scores.size(1))
        for j in range(scores.shape[1]):# dec
            for k in range(max(j-D, 0), j-1):
                # get P(Bi-1 = k)
                stop_prob = a[:, k, :] # [B, I]
                
                # get P(Bi >= j|Bi-1 = k)
                cond_sum = cond_prob[:,j:min(k+D,J),:].sum((1,))# [B, I]
                #cond_sum = cond_sums[:,j,:]
                
                # get Bi,j = P(Bi-1 < j <= Bi)
                B[:,j,:] += (stop_prob * cond_sum)# [B, I]*[B, I] -> [B, I]
        return B# [B, dec_T, enc_T] return the boundary probs
    
    def forward(self, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=None, max_temp=1.5, max_D=20):
        assert melenc_outputs is not None, 'Mel Encoder required for MoBo Attention'
        batch_size, enc_T, enc_dim = encoder_outputs.shape# [B, enc_T, enc_dim]
        key = encoder_outputs  # [B, enc_T, att_dim]
        query = melenc_outputs # [B, dec_T, att_dim]
        
        # get Scaled Dot Product Energies
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / sqrt(dk)# [B, dec_T, att_dim] @ [B, enc_T, att_dim].t() -> [B, dec_T, enc_T]
        s_alignment = F.softmax(scores, dim=2)# [B, dec_T, enc_T]
        assert not torch.isnan(scores).any()
        G = -(-(torch.rand(scores.shape, device=scores.device, dtype=scores.dtype)*0.9+0.0001).log()).log() # Gumbel noise # does the paper use Log10 or Natural Log? # This produces inf very easily
        assert not torch.isnan(G).any()
        assert not torch.isinf(G).any()
        T = self.get_T(scores[:,0].unsqueeze(1), min=0.5, max=max_temp)# [B, 1, enc_T] # note, max_temp should anneal as training continues
        assert not torch.isnan(T).any()
        assert not torch.isinf(T).any()
        assert not (T==0.0).any()
        #scores = (scores+G)/T# [B, dec_T, enc_T] # used to replace Gumbel softmax
        scores = (scores)/T# [B, dec_T, enc_T] # used to replace Gumbel softmax
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        scores = scores.float().exp()
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        
        if cond_lens is not None:
            mask = get_mask_3d(output_lengths, cond_lens)        # [B, dec_T, enc_T]
            scores = scores.masked_fill(mask == 0, 0.0)#-65500.0)# [B, dec_T, enc_T]
        B = self.MoBoSearch(scores, max_D).to(encoder_outputs)# [B, dec_T, enc_T]
        
        #B = F.softmax(B, dim=2)
        assert not torch.isnan(B).any()
        assert not torch.isinf(B).any()
        att_out = B @ encoder_outputs# [B, dec_T, enc_T] @ [B, enc_T, att_dim] -> [B, dec_T, att_dim]
        return att_out, torch.stack((s_alignment, B), dim=1)# [B, 2, dec_T, enc_T]

class LenPredictorAttention(nn.Module):
    def __init__(self, hparams):
        super(LenPredictorAttention, self).__init__()
    
    def forward(self, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=None):
        B, enc_T, enc_dim = encoder_outputs.shape# [Batch Size, Text Length, Encoder Dimension]
        dec_T = output_lengths.max().item()# Length of Spectrogram
        
        encoder_lengths = encoder_lengths# [B, enc_T]
        encoder_outputs = encoder_outputs# [B, enc_T, enc_dim]
        
        start_pos = torch.zeros(encoder_lengths.shape[0], device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B]
        attention_pos = torch.arange(dec_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype).expand(B, dec_T)# [B, dec_T, enc_T]
        attention = torch.zeros(B, dec_T, enc_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B, dec_T, enc_T]
        for enc_inx in range(encoder_lengths.shape[1]):
            dur = encoder_lengths[:, enc_inx]# [B]
            end_pos = start_pos + dur# [B]
            if cond_lens is not None: # if last char, extend till end of decoder sequence
                mask = (cond_lens == (enc_inx+1))# [B]
                if mask.any():
                    end_pos.masked_fill_(mask, dec_T)
            
            att = (attention_pos>=start_pos.unsqueeze(-1).repeat(1, dec_T)) & (attention_pos<end_pos.unsqueeze(-1).repeat(1, dec_T))
            attention[:, :, enc_inx][att] = 1.# set predicted duration values to positive
            
            start_pos = start_pos + dur # [B]
        if cond_lens is not None:
            attention = attention * get_mask_3d(output_lengths, cond_lens)
        return attention.matmul(encoder_outputs), attention# [B, dec_T, enc_T] @ [B, enc_T, enc_dim] -> [B, dec_T, enc_dim], [B, dec_T, enc_T]


@torch.jit.script
def basic_att_jit_forward(encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, mask, cond_lens, act_linear_w, location_conv_w, padding: int):
    # Inputs:
    # GT_mel          [B, n_mel, dec_T]
    # encoder_outputs [B, enc_T, enc_dim]
    
    # Outputs:
    # att_out    [B, dec_T, enc_dim]
    # att_scores [B, dec_T, enc_T]]
    
    B, enc_T, enc_dim = encoder_outputs.shape
    B, dec_T, att_dim = melenc_outputs.shape
    
    att_score = torch.zeros(B, 2, enc_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B, 2, enc_T]
    att_score_cum = torch.zeros(B, 1, enc_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B, 1, enc_T]
    acts = melenc_outputs# [B, att_dim, dec_T] (causal?) Conv/RNN to learn modifier for location
    att_out = torch.zeros(B, 1, enc_dim, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
    att_outs = []
    att_scores = []
    
    for di in range(dec_T):
        melenc = acts[:, di:di+1]# [B, dec_T, att_dim] -> [B, 1, att_dim]
        #act = F.linear(torch.cat((act, att_out), dim=2), act_linear_w)# [B, 1, att_dim+enc_dim] -> [B, 1, att_dim]
        
        att_out = att_out.repeat(1, enc_T, 1) # [B, 1, att_dim] -> [B, enc_T, att_dim]
        melenc = melenc.repeat(1, enc_T, 1)   # [B, 1, att_dim] -> [B, enc_T, att_dim]
        #loc_input = torch.cat((melenc.transpose(1, 2), att_out.transpose(1, 2), att_score), dim=1)# [B, enc_T, att_dim].t(), [B, 2, enc_T] -> [B, att_dim+2, enc_T]
        loc_layer_out = F.conv1d(att_score, location_conv_w, padding=padding)# [B, 2, enc_T] -> [B, att_dim, enc_T]
        att_weights = (melenc.transpose(1, 2) + att_out.transpose(1, 2) + loc_layer_out).tanh()# [B, att_dim, enc_T]
        att_weights = F.linear(att_weights.transpose(1, 2), act_linear_w).transpose(1, 2)# [B, att_dim, enc_T] -> [B, 1, enc_T]
        
        att_score = F.softmax(att_weights, dim=2)# [B, 1, enc_T] Softmax along Encoder
        
        if mask is not None:
            att_score.masked_fill_(mask, 0.0)
            #att_score = att_score * mask# [B, 1, enc_T]
        
        att_out = att_score @ encoder_outputs # [B, 1, enc_T] @ [B, enc_T, enc_dim] -> [B, 1, enc_dim]
        att_outs.append(att_out)
        att_scores.append(att_score)
        
        att_score_cum = att_score_cum + att_score# [B, 1, enc_T]
        att_score = torch.cat((att_score, att_score_cum), dim=1)# cat([B, 1, enc_T], [B, 1, enc_T]) -> [B, 2, enc_T]
    
    att_outs = torch.cat(att_outs, dim=1)# arr of [B, enc_dim] -> [B, dec_T, enc_dim]
    att_scores = torch.cat(att_scores, dim=1)# arr of [B, 1, enc_T] -> [B, dec_T, enc_T
    return att_outs, att_scores

class BasicAttention(nn.Module):
    def __init__(self, hparams):
        super(BasicAttention, self).__init__()
        self.act_linear = LinearNorm(hparams.pos_att_dim, 1)
        
        self.padding = int((hparams.bas_att_location_kw - 1) / 2)
        self.location_conv = ConvNorm(2, hparams.pos_att_dim,
                                      kernel_size=hparams.bas_att_location_kw,
                                      padding=self.padding, bias=False, stride=1,
                                      dilation=1)
    
    def forward(self, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=None):
        if output_lengths is not None:# masking for batches
            dec_mask = get_mask_from_lengths(output_lengths).unsqueeze(2)# [B, dec_T, 1]
            melenc_outputs = melenc_outputs * dec_mask# [B, dec_T, enc_dim] * [B, dec_T, 1] -> [B, dec_T, enc_dim]
        mask = get_mask_from_lengths(cond_lens).unsqueeze(1) if (cond_lens is not None) else None # [B, 1, enc_T]
        att_outs, att_scores = basic_att_jit_forward(encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, mask, cond_lens, self.act_linear.linear_layer.weight, self.location_conv.conv.weight, self.padding)
        if output_lengths is not None:
            att_outs = att_outs * dec_mask# [B, dec_T, enc_dim] * [B, dec_T, 1]
        return att_outs, att_scores


class GMMAttention(nn.Module): # Experimental from NTT123
    def __init__(self, hparams):
        super(GMMAttention, self).__init__()
        
        self.att_dim = hparams.pos_att_dim
        input_dim = hparams.pos_att_dim + self.att_dim*2 # MelEnc output + Att Out (Recurrent/AR) + Next Encoder Out
        
        # Attention LSTM keeps track of position much better than just Linears
        self.attrnn_dim = hparams.gmm_att_attrnn_dim
        self.attention_rnn = LSTMCellWithZoneout(
            input_dim, # input_size
            self.attrnn_dim,# hidden_size,
            zoneout_prob=hparams.gmm_att_attrnn_zoneout)
        
        self.num_mixtures = hparams.gmm_att_num_mixtures
        self.delta_min_limit = hparams.gmm_att_delta_min_limit
        self.delta_offset = hparams.gmm_att_delta_offset
        self.lin_bias = hparams.gmm_att_lin_bias
        lin = nn.Linear(hparams.pos_att_dim, 3*self.num_mixtures, bias=self.lin_bias)
        lin.weight.data.mul_(0.01)
        if self.lin_bias:
            w_bias, delta_bias, std_bias= torch.chunk(lin.bias.data, 3, dim=0)
            w_bias.fill_(0.0)     # 
            delta_bias.fill_(0.0) # initial movement per frame
            std_bias.fill_(3.0)  # initial standard deviation used for the alignment
        
        layers = []
        layers.append( LinearNorm(hparams.gmm_att_attrnn_dim, self.att_dim, bias=True, w_init_gain='tanh') )
        for i in range(hparams.gmm_att_n_layers-1):
            layers.append( nn.LeakyReLU(negative_slope=0.1) )
            layers.append( LinearNorm(self.att_dim, self.att_dim, bias=False, w_init_gain='tanh') )
        layers.append( nn.Tanh() )
        layers.append( lin )
        
        self.F = nn.Sequential(*layers)
        
        
        self.score_mask_value = 0 # -float("inf")
        
        self.register_buffer('pos', torch.arange(
            0, 2000, dtype=torch.float).view(1, -1, 1).data)
    
    
    @torch.jit.script
    def gmm_jit(input, memory, pos, loc, delta_min_limit: float, delta_offset: float):
    #def gmm_jit(self, input, memory, pos, delta_min_limit: float, delta_offset: float):
        
        input_chunks = input.chunk(3, dim=-1)# [B, T, 3*n_mix] -> [B, T, n_mix], [B, T, n_mix], [B, T, n_mix]
        w = input_chunks[0]
        delta_ = input_chunks[1]
        std_ = input_chunks[2]
        
        w = torch.softmax(w, dim=-1) # [B, dec_T, n_mix] # If using softmax here, softmax is not needed on the output
        #w = w.sigmoid()              # [B, dec_T, n_mix]
        #w = w.exp()                  # [B, dec_T, n_mix]
        #w = F.softplus(w)            # [B, dec_T, n_mix]
        
        #std_ = std_.sigmoid()*.95+.05 # [B, dec_T, n_mix]
        std_ = std_.sigmoid()         # [B, dec_T, n_mix]
        #std_ = F.softplus(std_)      # [B, dec_T, n_mix]
        
        #delta_ = delta_.sigmoid()#* 2                 # [B, dec_T, n_mix] Remove ability to move backwards in the text
        delta_ = delta_.sigmoid() + 0.005             # [B, dec_T, n_mix] Remove ability to move backwards in the text
        #delta_ = F.softplus(delta_) - 0.1             # [B, 1, n_mix] Remove ability to move backwards in the text
        #delta_ = torch.nn.functional.softplus(delta_) # [B, 1, n_mix] Remove ability to move backwards in the text
        
        if delta_min_limit > 0.0:
            delta_ = delta_.clamp(min=delta_min_limit) # supposed to be fine with autograd but not 100% confident.
        if delta_offset > 0.0:
            delta_ = delta_ + delta_offset
        
        assert not torch.any(loc != loc), "loc has NaN elements"
        assert not torch.any(std_ != std_), "std_ has NaN elements"
        assert not torch.any(delta_ != delta_), "delta_ has NaN elements"
        B, enc_T, _ = memory.shape
        B, dec_T, n_mix = w.shape
        if dec_T > 1:
            z_ = []
            for di in range(dec_T):# for di in range dec_T
                std   =   std_[:,di:di+1,:]# [B, dec_T, n_mix] -> [B, 1, n_mix]
                delta = delta_[:,di:di+1,:]# [B, dec_T, n_mix] -> [B, 1, n_mix]
                
                loc = loc + delta          # [B, 1, n_mix]
                
                z = torch.tanh(((loc.unsqueeze(3)-pos)/std.unsqueeze(3)).permute(3, 0, 1, 2))# ([1, enc_T, 1, 1]-[1, enc_T, 1, 2]) / ([B, 1, n_mix, 1]) -> [B, enc_T, n_mix], [B, enc_T, n_mix]
                z = (z[1] - z[0])*0.5          # [B, enc_T, n_mix]
                z_.append( z )
            z = torch.stack(z_, dim=1)     # [B, dec_T, enc_T, n_mix]
        else:
            std   =   std_[:,0:1,:]        # [B, dec_T, n_mix] -> [B, 1, n_mix]
            delta = delta_[:,0:1,:]        # [B, dec_T, n_mix] -> [B, 1, n_mix]
            
            loc = loc + delta              # [B, 1, n_mix]
            
            z = torch.tanh(((loc.unsqueeze(3)-pos)/std.unsqueeze(3)).permute(3, 0, 1, 2))# ([1, enc_T, 1, 1]-[1, enc_T, 1, 2]) / ([B, 1, n_mix, 1]) -> [B, enc_T, n_mix], [B, enc_T, n_mix]
            z = ((z[1] - z[0])*0.5).unsqueeze(1)# [B, 1, enc_T, n_mix]
        
        assert not torch.any(w != w), "w has NaN elements"
        assert not torch.any(z != z), "z has NaN elements"
        
        z = z.view(B*dec_T, enc_T, n_mix)       # [B, dec_T, enc_T, n_mix] -> [B*dec_T, enc_T, n_mix]
        w = w.view(B*dec_T, n_mix).unsqueeze(-1)# [B, dec_T, n_mix] -> [B*dec_T, n_mix, 1]
        z = torch.bmm(z, w)                     # [B*dec_T, enc_T, n_mix] @ [B*dec_T, n_mix, 1] -> [B*dec_T, enc_T, 1]
        z = z.view(B, dec_T, enc_T)             # [B*dec_T, enc_T, 1] -> [B, dec_T, enc_T]
        
        return z, loc
    
    def get_alignment_energies(self, query, memory, pos, loc):
        """
        PARAMS
        ------
        query:  decoder output  (batch, T_out, channels)
        memory: encoder outputs (B, T_in, attention_dim)
        
        RETURNS
        -------
        attention_output (batch, T_out, attention_dim)
        alignment (batch, max_time)
        """
        #query = query.tanh()
        query = query.squeeze(1) # [B, dec_T, C] -> [B, C] # This Attention RNN will only work when attending to single time-steps :/
        self.attention_hidden, self.attention_cell = self.attention_rnn(query, (self.attention_hidden, self.attention_cell))# [B, dec_T, C] -> [B, dec_T, attrnn_dim]
        F_query = self.attention_hidden.unsqueeze(1)
        
        assert not torch.isnan(F_query).any(), "F_query has NaN elements"
        assert not torch.isinf(F_query).any(), "F_query has inf elements"
        out = self.F(F_query)# [B, dec_T, attrnn_dim] -> [B, dec_T, 3*num_mixtures]
        assert not torch.isnan(out).any(), "out has NaN elements"
        assert not torch.isinf(out).any(), "out has inf elements"
        z, loc = self.gmm_jit(out, memory, pos, loc, self.delta_min_limit, self.delta_offset)
        assert not torch.isnan(z).any(), "z has NaN elements"
        assert not torch.isinf(z).any(), "z has inf elements"
        return z, loc
    
    def forward_(self, encoder_outputs, melenc_outputs, encoder_lengths, pos, loc, mask=None):# [B, seq_len, dim], int, [B]
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment, loc = self.get_alignment_energies(query=melenc_outputs, memory=encoder_outputs, pos=pos, loc=loc)# ([B, dec_T, att_dim], [B, enc_T, enc_dim]) -> [B, dec_T, enc_T]
        
        if False: # if use_softmax:
            if mask is not None:
                alignment.data.masked_fill_(mask, -65500.0)   # [B, dec_T, enc_T]
            attention_weights = F.softmax(alignment, dim=2)# [B, dec_T, enc_T] Softmax over enc_T
        else:
            if mask is not None:
                alignment.data.masked_fill_(mask, 0.0)   # [B, dec_T, enc_T]
            attention_weights = alignment
        next_attention_weights = torch.roll(attention_weights, shifts=1, dims=2)# [B, dec_T, enc_T] Shift the attention forward by one, give the network information extra information about the next frame
        attention_context = attention_weights @ encoder_outputs           # [B, dec_T, enc_T] @ [B, enc_T, att_dim] -> [B, dec_T, att_dim]
        next_attention_context = next_attention_weights @ encoder_outputs # [B, dec_T, enc_T] @ [B, enc_T, att_dim] -> [B, dec_T, att_dim]
        
        return attention_context, attention_weights, next_attention_context, loc # [B, dec_T, att_dim], [B, dec_T, enc_T]
    
    def reset_lstm(self, bsz): # set / reset the LSTM internal state.
        self.attention_hidden = Variable(self.attention_rnn.weight_ih.data.new( # attention hidden state
            bsz, self.attrnn_dim).zero_())
        self.attention_cell = Variable(self.attention_rnn.weight_ih.data.new(   # attention cell state
            bsz, self.attrnn_dim).zero_())
    
    def forward(self, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=None):# [B, seq_len, dim], int, [B]
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        pos = torch.arange(encoder_outputs.shape[1], device=encoder_outputs.device, dtype=encoder_outputs.dtype)[None, :, None]# [1, enc_T, 1]
        pos = torch.stack((pos+0.5, pos-0.5), dim=3)# [1, enc_T, 1, 2]
        
        B, enc_T, enc_dim = encoder_outputs.shape
        attention_context = torch.zeros(B, 1, self.att_dim, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        next_attention_context = encoder_outputs[:, 0:1, :]# [B, 1, att_dim]
        loc = torch.zeros(B, 1, self.num_mixtures, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# [B, 1, n_mix]
        loc = loc - 0.1
        self.reset_lstm(B)
        mask = ~get_mask_3d(output_lengths, cond_lens)# [B, dec_T, enc_T]
        attention_contexts_ = []
        attention_weights_ = []
        cond_lens_min1 = (cond_lens-1)[:, None, None]# [B, 1, 1]
        for di in range(melenc_outputs.shape[1]):
            melenc_outputs_ = melenc_outputs[:, di:di+1]                  # [B, 1, att_dim]
            mask_ = mask[:, di:di+1]
            
            query = torch.cat((melenc_outputs_, attention_context, next_attention_context), dim=2)# cat([B, 1, att_dim], [B, 1, att_dim], [B, 1, att_dim]) -> [B, 1, att_dim*3]
            attention_context, attention_weights, next_attention_context, loc = self.forward_(encoder_outputs, query, encoder_lengths, pos, loc, mask=mask_)# [B, 1, att_dim], [B, 1, att_dim]
            loc = loc - (loc - cond_lens_min1).clamp(min=0.0)# [B, 1, n_mix] - ([B, 1, n_mix] - [B, 1, 1]) -> [B, 1, n_mix] Clamp any location trying to go off the end of the text.
            attention_contexts_.append(attention_context)
            attention_weights_.append(attention_weights)
        
        attention_weights = torch.cat(attention_weights_, dim=1)   # [B, dec_T, enc_T]
        attention_context = torch.cat(attention_contexts_, dim=1)  # [B, dec_T, att_dim]
        assert not torch.isnan(attention_context).any(), "attention_context has NaN elements"
        assert not torch.isinf(attention_context).any(), "attention_context has inf elements"
        
        mask = ~get_mask_from_lengths(output_lengths).unsqueeze(-1)# [B, dec_T, 1]
        attention_context.data.masked_fill_(mask, 0.0)        # [B, dec_T, att_dim]
        
        return attention_context, attention_weights # [B, dec_T, att_dim], [B, dec_T, enc_T]


@torch.jit.script
def GLU(input_a, n_channels_int):
    in_act = input_a
    t_act = in_act[:, :n_channels_int, :]
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

class ConvolutionBlock(nn.Module): # Module concept taken from https://arxiv.org/pdf/1905.08459.pdf
    def __init__(self, hparams):
        super(ConvolutionBlock, self).__init__()
        self.pos_att_dim = hparams.pos_att_dim
        self.dropout = nn.Dropout(0.5)
        self.pos_att_conv_block_groups = hparams.pos_att_conv_block_groups
        self.conv = nn.Conv1d(self.pos_att_dim, self.pos_att_dim*2, 1, groups=self.pos_att_conv_block_groups or 1)
    
    def forward(self, x):
        x = self.dropout(x).transpose(1, 2)
        act = self.conv(x)
        x = GLU(act, torch.tensor(self.pos_att_dim)) + x
        return x.transpose(1, 2)

class PositionalAttention(nn.Module):
    def __init__(self, hparams):
        super(PositionalAttention, self).__init__()
        self.head_num = hparams.pos_att_head_num
        self.pos_enc_dim = hparams.pos_att_dim
        self.frames_per_char_tf_chance = hparams.pos_att_use_duration_predictor_teacher_forcing
        
        self.positional_embedding = PositionalEmbedding(self.pos_enc_dim, inv_freq=hparams.pos_att_inv_freq, range_scaler=hparams.pos_att_step_size, range_shifter=0.0)
        
        self.multi_head_attentions = nn.ModuleList()
        self.convolution_blocks = nn.ModuleList()
        for i in range(hparams.n_mha_layers):
            multi_head_attention = torch.nn.MultiheadAttention(self.pos_enc_dim,
                                                               self.head_num, dropout=0.1, bias=True,
                                                               add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
            self.multi_head_attentions.append(multi_head_attention)
            conv_block = ConvolutionBlock(hparams)
            self.convolution_blocks.append(conv_block)
        
        self.poc_enc_q = hparams.pos_att_positional_encoding_for_query
        self.pos_enc_k = hparams.pos_att_positional_encoding_for_key
        self.pos_enc_v = hparams.pos_att_positional_encoding_for_value
        if self.pos_enc_k or self.pos_enc_v:
            self.enc_positional_embedding = PositionalEmbedding(self.pos_enc_dim, inv_freq=hparams.pos_att_enc_inv_freq, range_scaler=hparams.pos_att_enc_step_size, range_shifter=0.0, use_external_scaler=hparams.pos_att_use_duration_predictor_for_scalar, learn_ext_scalar=hparams.pos_att_duration_predictor_learn_scalar)
        
        if hparams.rezero_pos_enc_kv:
            self.rezero_kv = nn.Parameter(torch.ones(1)*0.04)
        if hparams.rezero_pos_enc_q:
            self.rezero_q = nn.Parameter(torch.ones(1)*0.04)
        
        if hparams.pos_enc_positional_embedding_kv: # learned positional encoding
            self.pos_embedding_kv_max = 400
            self.pos_embedding_kv = nn.Embedding(self.pos_embedding_kv_max, self.pos_enc_dim)
        if hparams.pos_enc_positional_embedding_q: # learned positional encoding
            self.pos_embedding_q_max = 18000
            self.pos_embedding_q = nn.Embedding(self.pos_embedding_q_max, self.pos_enc_dim)
        
        if (hparams.pos_att_t_min is not None) and (hparams.pos_att_t_max is not None):
            self.t_min = hparams.pos_att_t_min
            self.t_max = hparams.pos_att_t_max
    
    def forward(self, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=None):# [B, seq_len, dim], int, [B]
        batch_size, enc_T, enc_dim = encoder_outputs.shape# [B, enc_T, enc_dim]
        
        ######################################
        # get Query from Positional Encoding #
        ######################################
        if hasattr(self, 'positional_embedding') or hasattr(self, 'pos_embedding_q'):
            dec_T_max = int(output_lengths.max().item())
            dec_pos_emb = torch.arange(0, dec_T_max, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# + trandint
            if hasattr(self, 'pos_embedding_q'):
                dec_pos_emb = self.pos_embedding_q(dec_pos_emb.clamp(0, self.pos_embedding_q_max).long())[None, ...].repeat(batch_size, 1, 1)# [B, enc_T, enc_dim]
            elif hasattr(self, 'positional_embedding'):
                dec_pos_emb = self.positional_embedding(dec_pos_emb, bsz=batch_size)# [B, dec_T, enc_dim]
            if hasattr(self, 'rezero_q'):
                dec_pos_emb = dec_pos_emb * self.rezero_q
            if melenc_outputs is None:
                melenc_outputs = dec_pos_emb.new_zeros(dec_pos_emb.shape)
            if self.poc_enc_q:
                melenc_outputs = melenc_outputs + dec_pos_emb
        if output_lengths is not None:# masking for batches
            dec_mask = get_mask_from_lengths(output_lengths).unsqueeze(2)# [B, dec_T, 1]
            melenc_outputs = melenc_outputs * dec_mask# [B, dec_T, enc_dim] * [B, dec_T, 1] -> [B, dec_T, enc_dim]
        q = melenc_outputs# [B, dec_T, enc_dim]
        
        ######################################
        # get Key/Value from Encoder Outputs #
        ######################################
        k = v = encoder_outputs# [B, enc_T, enc_dim]
        if hasattr(self, 'enc_positional_embedding') or hasattr(self, 'pos_embedding_kv'):# (optional) add position encoding to Encoder outputs
            enc_pos_emb = torch.arange(0, enc_T, device=encoder_outputs.device, dtype=encoder_outputs.dtype)# + trandint
            if hasattr(self, 'pos_embedding_kv'):
                enc_pos_emb = self.pos_embedding_kv(enc_pos_emb.clamp(0, self.pos_embedding_kv_max).long())[None, ...].repeat(batch_size, 1, 1)# [B, enc_T, enc_dim]
            elif hasattr(self, 'enc_positional_embedding'):
                pred_output_lengths = encoder_lengths.sum((1,))# [B, enc_T] -> [B]
                pred_frames_per_char = pred_output_lengths/cond_lens# [B]/[B] -> [B]
                if self.training:# randomly pick between gt and predicted frames per char (during inference both values are the same and this will have no effect)
                    gt_frames_per_char = output_lengths.to(pred_frames_per_char)/cond_lens# [B]/[B] -> [B]
                    if self.frames_per_char_tf_chance == 1.0:
                        pred_frames_per_char = gt_frames_per_char
                    else:
                        pred_frames_per_char = torch.where(
                            torch.rand(*pred_frames_per_char.shape, device=pred_frames_per_char.device) < self.frames_per_char_tf_chance,
                            gt_frames_per_char, pred_frames_per_char)
                enc_pos_emb = self.enc_positional_embedding(enc_pos_emb, bsz=batch_size, range_scaler=pred_frames_per_char)# [B, enc_T, enc_dim]
            if hasattr(self, 'rezero_kv'):
                enc_pos_emb = enc_pos_emb * self.rezero_kv
            if self.pos_enc_k:
                k = k + enc_pos_emb
            if self.pos_enc_v:
                v = v + enc_pos_emb
        enc_mask = get_mask_from_lengths(cond_lens).unsqueeze(1).repeat(1, q.size(1), 1) if (cond_lens is not None) else None# [B, dec_T, enc_T]
        
        #######################################################################
        # Multi-Layer Positional Attention (Multi-head Dot-Product Attention) #
        #######################################################################
        q = q.transpose(0, 1)# [B, dec_T, enc_dim] -> [dec_T, B, enc_dim]
        k = k.transpose(0, 1)# [B, enc_T, enc_dim] -> [enc_T, B, enc_dim]
        v = v.transpose(0, 1)# [B, enc_T, enc_dim] -> [enc_T, B, enc_dim]
        
        if hasattr(self, 't_min'):# apply random temperature to Key values (which will be used for Attention inside MHA)
            T = torch.rand((1, k.shape[1], 1), device=k.device, dtype=k.dtype) * (self.t_max-self.t_min) + self.t_min # [1, B, enc_dim]
            k = k/T
        
        enc_mask = ~enc_mask[:, 0, :] if (cond_lens is not None) else None# [B, dec_T, enc_T] -> # [B, enc_T]
        if True:
            attn_mask = ~get_mask_3d(output_lengths, cond_lens).repeat_interleave(self.head_num, 0) if (cond_lens is not None) else None#[B*n_head, dec_T, enc_T]
            attn_mask = attn_mask.float() * -5500.0 if (cond_lens is not None) else None
        else:
            attn_mask = None
        
        output = q
        attention_scores = []
        for multi_head_attention_layer, conv_block in zip(self.multi_head_attentions, self.convolution_blocks):
            output_tmp, attention_score_ = multi_head_attention_layer(output, k, v, key_padding_mask=enc_mask, attn_mask=attn_mask)# [dec_T, B, enc_dim], [B, dec_T, enc_T]
            output_tmp = conv_block(output_tmp)
            output = output + output_tmp
            attention_score_ = attention_score_*get_mask_3d(output_lengths, cond_lens) if (cond_lens is not None) else attention_scores
            attention_scores.append(attention_score_)
        
        output = output.transpose(0, 1)# [dec_T, B, enc_dim] -> [B, dec_T, enc_dim]
        attention_scores = torch.stack(attention_scores, dim=1)# [B, n_layers, dec_T, enc_T]
        
        if output_lengths is not None:
            output = output * dec_mask# [B, dec_T, enc_dim] * [B, dec_T, 1]
        return output, attention_scores


class LSTMCellWithZoneout(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True, zoneout_prob=0.1):
        super().__init__(input_size, hidden_size, bias)
        self._zoneout_prob = zoneout_prob

    def forward(self, input, hx):
        old_h, old_c = hx
        new_h, new_c = super(LSTMCellWithZoneout, self).forward(input, hx)
        if self.training and self._zoneout_prob > 0.0:
            c_mask = torch.empty_like(new_c).bernoulli_(p=self._zoneout_prob).bool().data
            h_mask = torch.empty_like(new_h).bernoulli_(p=self._zoneout_prob).bool().data
            h = torch.where(h_mask, old_h, new_h)
            c = torch.where(c_mask, old_c, new_c)
            return h, c
        else:
            return new_h, new_c


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim, out_bias=False):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=out_bias, w_init_gain='tanh')
    
    def forward(self, attention_weights_cat): # [B, 2, enc]
        processed_attention = self.location_conv(attention_weights_cat) # [B, 2, enc] -> [B, n_filters, enc]
        processed_attention = processed_attention.transpose(1, 2) # [B, n_filters, enc] -> [B, enc, n_filters]
        processed_attention = self.location_dense(processed_attention) # [B, enc, n_filters] -> [B, enc, attention_dim]
        return processed_attention # [B, enc, attention_dim]


class MelEncoder(nn.Module):
    """MelEncoder module:
        - Three 1-d convolution banks
    """
    def __init__(self, hparams):
        super(MelEncoder, self).__init__() 
        self.melenc_speaker_embed_dim = hparams.melenc_speaker_embed_dim
        if self.melenc_speaker_embed_dim:
            self.melenc_speaker_embedding = nn.Embedding(
            hparams.n_speakers, self.melenc_speaker_embed_dim)
        
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.melenc_conv_hidden_dim = hparams.melenc_conv_dim
        self.output_dim = hparams.pos_att_dim
        self.drop_chance = hparams.melenc_drop_frame_rate
        
        convolutions = []
        for _ in range(hparams.melenc_n_layers):
            input_dim = hparams.n_mel_channels+self.melenc_speaker_embed_dim if (_ == 0) else self.melenc_conv_hidden_dim
            output_dim = self.output_dim if (_ == hparams.melenc_n_layers-1) else self.melenc_conv_hidden_dim
            conv_layer = nn.Sequential(
                ConvNorm(input_dim,
                         output_dim,
                         kernel_size=hparams.melenc_kernel_size, stride=1,
                         padding=int((hparams.melenc_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(output_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.LReLU = nn.LeakyReLU(negative_slope=0.01)
    
    def drop_frames(self, spect, drop_chance=0.0):
        if drop_chance > 0.0:# randomly set some frames to zeros
            B, n_mel, dec_T = spect.shape
            frames_to_keep = torch.rand(B, 1, dec_T, device=spect.device, dtype=spect.dtype) > drop_chance
            spect = spect * frames_to_keep
        return spect
    
    def forward(self, spect, output_lengths, speaker_ids=None, enc_drop_rate=0.2):
        spect = self.drop_frames(spect, self.drop_chance)
        
        if self.melenc_speaker_embed_dim:
            speaker_embedding = self.melenc_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, dec_T]
            speaker_embedding = speaker_embedding.repeat(1, 1, spect.size(2)) # extend across all encoder steps
            spect = torch.cat((spect, speaker_embedding), dim=1) # [B, embed, dec_T]
        
        for conv in self.convolutions:
            #spect = F.dropout(F.relu(conv(spect)), enc_drop_rate, self.training) # Normal ReLU
            spect = F.dropout(self.LReLU(conv(spect)), enc_drop_rate, self.training) # LeakyReLU
        
        spect = spect * get_mask_from_lengths(output_lengths).unsqueeze(1)# [B, dec_dim, dec_T] * [B, 1, dec_T] -> [B, dec_dim, enc_T]
        
        return spect.transpose(1, 2)# [B, dec_dim, dec_T] -> [B, dec_T, dec_dim]


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__() 
        self.encoder_speaker_embed_dim = hparams.encoder_speaker_embed_dim
        if self.encoder_speaker_embed_dim:
            self.encoder_speaker_embedding = nn.Embedding(
            hparams.n_speakers, self.encoder_speaker_embed_dim)
            std = sqrt(2.0 / (hparams.n_speakers + self.encoder_speaker_embed_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.encoder_speaker_embedding.weight.data.uniform_(-val, val)
        
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.encoder_conv_hidden_dim = hparams.encoder_conv_hidden_dim
        
        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            if _ == 0:
                if self.encoder_concat_speaker_embed == 'before_conv':
                    input_dim = hparams.symbols_embedding_dim+self.encoder_speaker_embed_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    input_dim = hparams.symbols_embedding_dim
                else:
                    raise NotImplementedError(f'encoder_concat_speaker_embed is has invalid value {hparams.encoder_concat_speaker_embed}, valid values are "before","inside".')
            else:
                input_dim = self.encoder_conv_hidden_dim
            
            if _ == (hparams.encoder_n_convolutions)-1: # last conv
                if self.encoder_concat_speaker_embed == 'before_conv':
                    output_dim = hparams.encoder_LSTM_dim
                elif self.encoder_concat_speaker_embed == 'before_lstm':
                    output_dim = hparams.encoder_LSTM_dim-self.encoder_speaker_embed_dim
            else:
                output_dim = self.encoder_conv_hidden_dim
            
            conv_layer = nn.Sequential(
                ConvNorm(input_dim,
                         output_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(output_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(hparams.encoder_LSTM_dim,
                            int(hparams.encoder_LSTM_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        self.LReLU = nn.LeakyReLU(negative_slope=0.01) # LeakyReLU
    
    def forward(self, text, text_lengths, speaker_ids=None, enc_drop_rate=0.2):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.repeat(1, 1, text.size(2)) # extend across all encoder steps
            if self.encoder_concat_speaker_embed == 'before_conv':
                text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            #text = F.dropout(F.relu(conv(text)), enc_drop_rate, self.training) # Normal ReLU
            text = F.dropout(self.LReLU(conv(text)), enc_drop_rate, self.training) # LeakyReLU
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        text = text.transpose(1, 2)
        
        # pytorch tensor are not reversible, hence the conversion
        text_lengths = text_lengths.cpu().numpy()
        text = nn.utils.rnn.pack_padded_sequence(
            text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(text)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        
        return outputs
    
    def inference(self, text, speaker_ids=None, text_lengths=None):
        if self.encoder_speaker_embed_dim:
            speaker_embedding = self.encoder_speaker_embedding(speaker_ids)[:, None].transpose(1,2) # [B, embed, sequence]
            speaker_embedding = speaker_embedding.repeat(1, 1, text.size(2))
            if self.encoder_concat_speaker_embed == 'before_conv':
                text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        for conv in self.convolutions:
            #text = F.dropout(F.relu(conv(text)), drop_rate, self.training) # Normal ReLU
            text = F.dropout(self.LReLU(conv(text)), drop_rate, self.training) # LeakyReLU
        
        if self.encoder_speaker_embed_dim and self.encoder_concat_speaker_embed == 'before_lstm':
            text = torch.cat((text, speaker_embedding), dim=1) # [B, embed, sequence]
        
        text = text.transpose(1, 2) # [B, embed, sequence] -> [B, sequence, embed]
        
        if text_lengths is not None:
            #text *= get_mask_from_lengths(text_lengths)[:, :, None]
            # pytorch tensor are not reversible, hence the conversion
            text_lengths = text_lengths.cpu().numpy()
            text = nn.utils.rnn.pack_padded_sequence(
                text, text_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(text) # -> [B, sequence, embed]
        
        if text_lengths is not None:
            #outputs *= get_mask_from_lengths(text_lengths)[:, :, None]
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)
        
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        if hparams.use_GMMAttention:
            self.positional_attention = GMMAttention(hparams)
        elif hparams.use_BasicAttention:
            self.positional_attention = BasicAttention(hparams)
        elif hparams.use_MoboAttention:
            self.positional_attention = MoBoAlignerAttention(hparams)
        elif hparams.use_duration_predictor_for_attention:
            self.positional_attention = LenPredictorAttention(hparams)
        else:
            self.positional_attention = PositionalAttention(hparams)
        self.melglow = FlowDecoder(hparams)
        if (hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim) != hparams.pos_att_dim: # this layer is not needed if the attention is same dimension as encoder outputs
            self.attention_pre = nn.Linear(hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim, hparams.pos_att_dim, bias=False)
    
    def forward(self, gt_mels, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, text_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        gt_mels: Decoder inputs i.e. mel-specs
        melenc_outputs: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        log_det: log deterimates of Affine Coupling + InvertibleConv
        """
        # Positional Attention
        if hasattr(self, 'attention_pre'):
            encoder_outputs = self.attention_pre(encoder_outputs)# Crush dimensions of input
        cond, attention_scores = self.positional_attention(encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=text_lengths)
        cond = cond.transpose(1, 2)# [B, enc_T, enc_dim] -> [B, enc_dim, dec_T] # Masked Multi-head Attention
        
        # Decode Spect into Z
        mel_outputs, log_s_sum, logdet_w_sum = self.melglow(gt_mels, cond)
        return mel_outputs, attention_scores, log_s_sum, logdet_w_sum

    def infer(self, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=None, speaker_ids=None, sigma=None):
        """ Decoder inference
        PARAMS
        ------
        cond: Encoder outputs
        
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        """
        # Positional Attention
        if hasattr(self, 'attention_pre'):
            encoder_outputs = self.attention_pre(encoder_outputs)# Crush dimensions of input
        cond, attention_scores = self.positional_attention(encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=cond_lens)
        cond = cond.transpose(1, 2)# [B, enc_T, enc_dim] -> [B, enc_dim, dec_T] # Masked Multi-head Attention
        
        # Decode Z into Spect
        mel_outputs = self.melglow.infer(cond, sigma=sigma)
        return mel_outputs, attention_scores


class UnTTS(nn.Module):
    def __init__(self, hparams):
        super(UnTTS, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.encoder_concat_speaker_embed = hparams.encoder_concat_speaker_embed
        self.speaker_embedding_dim = hparams.speaker_embedding_dim
        
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.mel_encoder = MelEncoder(hparams) if hparams.melenc_enable else None
        self.melenc_ignore = hparams.melenc_ignore_at_inference
        
        self.len_pred_input = hparams.len_pred_input.lower()
        if self.len_pred_input == 'encoder':
            len_input_dim = hparams.encoder_LSTM_dim+self.speaker_embedding_dim
        elif self.len_pred_input == 'embedding':
            len_input_dim = hparams.symbols_embedding_dim+self.speaker_embedding_dim
        self.length_predictor = TemporalPredictor(len_input_dim, hparams)
        self.len_pred_guiding_att = LenPredictorAttention(hparams) if hparams.pos_att_len_pred_guided_att else None
        
        self.duration_glow = DurationGlow(hparams) if hparams.DurGlow_enable else None
        self.decoder = Decoder(hparams)
        if self.speaker_embedding_dim:
            self.speaker_embedding = nn.Embedding(
                hparams.n_speakers, self.speaker_embedding_dim)
        
    def parse_batch(self, batch):
        text_padded, text_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states = batch
        text_padded = to_gpu(text_padded).long()
        text_lengths = to_gpu(text_lengths).long()
        output_lengths = to_gpu(output_lengths).long()
        speaker_ids = to_gpu(speaker_ids.data).long()
        mel_padded = to_gpu(mel_padded).float()
        max_len = torch.max(text_lengths.data).item() # used by loss func
        gate_padded = to_gpu(gate_padded).float() # used by loss func
        if torchmoji_hidden is not None:
            torchmoji_hidden = to_gpu(torchmoji_hidden).float()
        if preserve_decoder_states is not None:
            preserve_decoder_states = to_gpu(preserve_decoder_states).float()
        return (
            (text_padded, text_lengths, mel_padded, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states),
            (mel_padded, gate_padded, output_lengths, text_lengths))
            # returns ((x),(y)) as (x) for training input, (y) for ground truth/loss calc
    
    def mask_outputs(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            # [B, n_mel, steps]
            outputs[0].data.masked_fill_(mask, 0.0) # [B, n_mel, T]
        return outputs
    
    def forward(self, inputs):
        text, text_lengths, gt_mels, max_len, output_lengths, speaker_ids, torchmoji_hidden, preserve_decoder_states = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        assert not torch.isnan(text).any(), 'text has NaN values.'
        
        melenc_outputs = self.mel_encoder(gt_mels, output_lengths, speaker_ids=speaker_ids) if (self.mel_encoder is not None) else None# [B, dec_T, melenc_dim]
        assert melenc_outputs is None or not torch.isnan(melenc_outputs).any(), 'melenc_outputs has NaN values.'
        
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        assert not torch.isnan(embedded_text).any(), 'embedded_text has NaN values.'
        encoder_outputs = self.encoder(embedded_text, text_lengths, speaker_ids=speaker_ids) # [B, enc_T, enc_dim]
        assert not torch.isnan(encoder_outputs).any(), 'encoder_outputs has NaN values.'
        
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_speakers), dim=2) # [batch, enc_T, enc_dim]
        
        # predict length of each input
        enc_out_mask = get_mask_from_lengths(text_lengths).unsqueeze(-1)# [B, enc_T, 1]
        if self.len_pred_input == 'encoder':
            len_pred_input = encoder_outputs
        elif self.len_pred_input == 'embedding':
            len_pred_input = torch.cat((embedded_text.transpose(1, 2), embedded_speakers), dim=2)# [B, enc_T, enc_dim]
        encoder_lengths = self.length_predictor(len_pred_input, enc_out_mask)# [B, enc_T, enc_dim]
        encoder_lengths = encoder_lengths.clamp(0.001, 4096)# sum lengths (used to predict mel-spec length)
        pred_output_lengths = encoder_lengths.sum((1,))
        pred_output_lengths_std = torch.std(encoder_lengths, 1) if (False) else None
        assert pred_output_lengths_std is None or not torch.isnan(pred_output_lengths_std).any(), 'pred_output_lengths has NaN values.'
        
        len_pred_attention = self.len_pred_guiding_att(encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=text_lengths)[-1] if (self.len_pred_guiding_att is not None) else None
        
        # Decoder
        mel_outputs, attention_scores, log_s_sum, logdet_w_sum = self.decoder(gt_mels, encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, text_lengths) # [B, n_mel, dec_T], [B, dec_T, enc_dim] -> [B, n_mel, dec_T], [B] # Series of Flows
        
        # DurationGlow
        if self.duration_glow is not None:
            if len(attention_scores.shape) == 4:
                attention_scores = attention_scores[:, -1] # [B, n_layers, dec_T, enc_T] -> [B, dec_T, enc_T]
            enc_durations = attention_scores.sum((1,)).detach().clone()#[B, dec_T, enc_T] -> [B, enc_T]
            enc_durations = enc_durations[:, None].repeat(1, 2, 1)
            dur_z, dur_log_s_sum, dur_logdet_w_sum = self.duration_glow(enc_durations, encoder_outputs.transpose(1, 2).detach().clone())# [B, enc_T], [B, enc_dim, enc_T]
        else:
            dur_z=dur_log_s_sum=dur_logdet_w_sum = None
        
        return self.mask_outputs(
            [mel_outputs, attention_scores, log_s_sum, logdet_w_sum, pred_output_lengths, pred_output_lengths_std, dur_z, dur_log_s_sum, dur_logdet_w_sum, len_pred_attention],
            output_lengths)
    
    def inference(self, text, speaker_ids, gt_mels=None, output_lengths=None, text_lengths=None, sigma=1.0):
        if text_lengths is None:
            text_lengths = torch.ones((text.shape[0],)).to(text)*text.shape[1]
        
        melenc_outputs = self.mel_encoder(gt_mels, output_lengths, speaker_ids=speaker_ids) if (self.mel_encoder is not None and not self.melenc_ignore) else None# [B, dec_T, melenc_dim]
        
        embedded_text = self.embedding(text).transpose(1, 2) # [B, embed, sequence]
        encoder_outputs = self.encoder.inference(embedded_text, speaker_ids=speaker_ids) # [B, enc_T, enc_dim]
        
        if self.speaker_embedding_dim:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, embedded_speakers), dim=2) # [batch, enc_T, enc_dim]
        
        # predict length of each input
        enc_out_mask = get_mask_from_lengths(text_lengths).unsqueeze(-1)# [B, enc_T, 1]
        if self.len_pred_input == 'encoder':
            len_pred_input = encoder_outputs
        elif self.len_pred_input == 'embedding':
            len_pred_input = torch.cat((embedded_text.transpose(1, 2), embedded_speakers), dim=2)# [B, enc_T, enc_dim]
        encoder_lengths = self.length_predictor(len_pred_input, enc_out_mask)# [B, enc_T, enc_dim]
        encoder_lengths = encoder_lengths.clamp(0.001, 4096)# sum lengths (used to predict mel-spec length)
        pred_output_lengths = encoder_lengths.sum((1,))
        
        if output_lengths is None:
            output_lengths = pred_output_lengths.round().int()
        
        # Decoder
        mel_outputs, attention_scores = self.decoder.infer(encoder_outputs, melenc_outputs, encoder_lengths, output_lengths, cond_lens=text_lengths, sigma=sigma) # [B, dec_T, emb] -> [B, n_mel, dec_T] # Series of Flows
        
        return self.mask_outputs(
            [mel_outputs, attention_scores, None, None, None])
