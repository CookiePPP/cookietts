# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Source: https://github.com/NVIDIA/DeepLearningExamples/blob/4d808052904d9afaa1ff14579aa4bb25990cf5db/PyTorch/SpeechSynthesis/FastPitch/fastpitch/transformer.py#L249

import torch
import torch.nn as nn
import torch.nn.functional as F

from CookieTTS.utils.model.utils import get_mask_from_lengths


class PositionalEmbedding(nn.Module):
    def __init__(self, demb, inv_freq=10000, range_scaler=1.0, range_shifter=0.0, learn_inv_freq=False, learn_scaler=True, learn_shifter=True, use_external_scaler=False, learn_ext_scalar=True):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        self.inv_freq = 1 / (inv_freq ** (torch.arange(0.0, demb, 2.0) / demb))# [dim//2]
        if learn_inv_freq:
            self.inv_freq = nn.Parameter(self.inv_freq)
        
        self.range_scaler = torch.ones((1,))*float(range_scaler)
        if learn_scaler:
            self.range_scaler = nn.Parameter(self.range_scaler)
        
        self.range_shifter = torch.ones((1,))*float(range_shifter)
        if learn_shifter:
            self.range_shifter = nn.Parameter(self.range_shifter)
        
        if use_external_scaler:
            self.learn_ext_scalar = learn_ext_scalar
            if self.learn_ext_scalar:
                self.ext_scalar = nn.Parameter( torch.ones((1,))*0.5 ) # (optional) use initial 50% of external range_scaler
            else:
                self.ext_scalar = torch.ones((1,)) # (optional) use initial 50% of external range_scaler
    
    def forward(self, pos_seq, bsz=1, range_scaler=None):
        if hasattr(self, 'ext_scalar'):
            assert range_scaler is not None, 'Module initialized with use_external_scaler but no range_scaler input given'
            ext_sigmoid = self.ext_scalar.clamp(min=0.001, max=0.999)#.sigmoid()
            if self.learn_ext_scalar:
                range_scaler = (self.range_scaler*(1-ext_sigmoid)) + (range_scaler*(ext_sigmoid)) # [B] mix/merge the external and internal range_scaler
            
            pos_seq = pos_seq.expand(bsz, -1)[...,None]  # [dec_T]     -> [B, dec_T, 1]
            pos_seq = (pos_seq*range_scaler[:,None,None])# [B, dec_T, 1] * [B, 1, 1] -> [B, dec_T, 1]
            pos_seq = pos_seq+self.range_shifter
            inv_freq = self.inv_freq.expand(bsz,-1)[:,None,:]# [B, dim//2] -> [B, 1, dim//2]
            
            sinusoid_inp = pos_seq @ inv_freq.to(pos_seq)  # [B, dec_T, 1] @ [B, 1, dim//2] -> [B, dec_T, dim//2]
            pos_emb = torch.cat([sinusoid_inp.sin().to(pos_seq), sinusoid_inp.cos().to(pos_seq)], dim=-1)# [B, dec_T, dim//2] -> [B, dec_T, dim] OR [dec_T, dim//2] -> [dec_T, dim]
        else:
            pos_seq = (pos_seq*self.range_scaler)+self.range_shifter
            
            sinusoid_inp = torch.ger(pos_seq, self.inv_freq.to(pos_seq))
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)# [B, dec_T, dim//2] -> [B, dec_T, dim] OR [dec_T, dim//2] -> [dec_T, dim]
        
        if bsz is not None:
            return pos_emb.expand(bsz, -1, -1)
        else:
            return pos_emb


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.1,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)
    
    def _forward(self, inp, attn_mask=None):
        residual = inp
        
        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)
        
        n_head, d_head = self.n_head, self.d_head
        
        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=-1)
        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)
        
        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), d_head)
        
        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask, -float('inf'))
        
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)
        
        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)
        
        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        
        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)
        
        return output

    # disabled; slower
    def forward_einsum(self, h, attn_mask=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [bsz x n_head x qlen x klen]
        # attn_score = torch.einsum('ibnd,jbnd->bnij', (head_q, head_k))
        attn_score = torch.einsum('bind,bjnd->bnij', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            attn_score.masked_fill_(attn_mask[:, None, None, :], -float('inf'))

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # [bsz x n_head x qlen x klen] * [klen x bsz x n_head x d_head] 
        #     -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('bnij,bjnd->bind', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout,
                 **kwargs):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(d_model, d_inner, kernel_size, dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner, kernel_size,
                 dropout, dropatt, dropemb=0.0, embed_input=True, d_embed=None,
                 pre_lnorm=False):
        super(FFTransformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        if embed_input:
            self.word_emb = nn.Embedding(len(symbols), d_embed or d_model,
                                         padding_idx=pad_idx)
        else:
            self.word_emb = None

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(n_layer):
            self.layers.append(
                TransformerLayer(
                    n_head, d_model, d_head, d_inner, kernel_size, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

    def forward(self, dec_inp, seq_lens=None):
        if self.word_emb is None:
            inp = dec_inp
            mask = get_mask_from_lengths(seq_lens).unsqueeze(2)
        else:
            inp = self.word_emb(dec_inp)
            # [bsz x L x 1]
            mask = (dec_inp != pad_idx).unsqueeze(2)

        pos_seq = torch.arange(inp.size(1), device=inp.device, dtype=inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        out = self.drop(inp + pos_emb)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask
