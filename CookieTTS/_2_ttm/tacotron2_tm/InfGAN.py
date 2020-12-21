from math import sqrt
import random
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict

from CookieTTS.utils.model.utils import get_mask_from_lengths

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n_gpus
    return rt


class InfGANLSTMModule(nn.Module):
    def __init__(self, hparams, speaker_embedding_dim):
        super(InfGANModule, self).__init__()
        self.speaker_embedding_dim = speaker_embedding_dim
        self.use_lstm_hidden = True
        
        self.lstm = nn.LSTM(hparams.prenet_dim+self.speaker_embedding_dim,
                            hparams.InfGAN_LSTM_dim,
                            hparams.InfGAN_LSTM_n_layers,
                            batch_first=True,)
        
        post_input_dim = hparams.InfGAN_LSTM_dim*hparams.InfGAN_LSTM_n_layers if self.use_lstm_hidden else hparams.InfGAN_LSTM_dim
        self.post = nn.Sequential(OrderedDict([
                      ('linear1', nn.Linear(post_input_dim, hparams.InfGAN_LSTM_dim)),
                      ('tanh'   , nn.Tanh()),
                      ('linear2', nn.Linear(hparams.InfGAN_LSTM_dim, 1)),
                    ]))
    
    def forward(self, mels):                   # mels -> [B, mel_T, n_mel+spkr_embed]
        if self.use_lstm_hidden:
            pred_infness = self.lstm(mels)[1][0]                       # -> [n_layers, B, LSTM_dim]
            pred_infness = pred_infness.permute(1, 0, 2).contiguous()  # -> [B, n_layers, LSTM_dim]
            pred_infness = pred_infness.view(pred_infness.shape[0], -1)# -> [B, n_layers* LSTM_dim]
            pred_infness = self.post(pred_infness)                     # -> [B, 1]
        else:
            pred_infness = self.lstm(mels)[0].squeeze(1)# -> [B, mel_T, LSTM_dim]
            pred_infness = pred_infness.mean(dim=1)     # -> [B, LSTM_dim]
            pred_infness = self.post(pred_infness)      # -> [B, 1]
        return pred_infness.squeeze(1)              # -> [B] -> output

# RNN GAN loss on spectrogram to reduce unintelligiblitiy rate and improve stability.
# Should predict "inferenceness" by predicting whether the spectrogram came from teacher forcing or inference.
# The generator will then attempt to reduce inferenceness and increase teacher-forcedness outputs on the discriminator.
class LSTMInferenceGAN(nn.Module):
    def __init__(self, hparams):
        super(LSTMInferenceGAN, self).__init__()
        assert hparams.batch_size%2==0, 'Batch size must be even!'
        self.speaker_embedding_dim = hparams.speaker_embedding_dim if hparams.InfGAN_use_speaker else 0
        
        self.optimizer     = None
        self.discriminator = InfGANLSTMModule(hparams, self.speaker_embedding_dim)
        
        self.input_feature = getattr(hparams, 'InfGAN_input_feature', 'spect')
        if self.input_feature in ['spect','postnet']:
            self.input_dim = hparams.prenet_dim
        elif self.input_feature in ['attention_context',]:
            self.input_dim = None
        else:
            raise Exception(f'InfGAN Input Feature of {hparams.InfGAN_input_feature} is not valid.')
        self.fp16_run    = hparams.fp16_run
        self.n_gpus      = hparams.n_gpus
        self.gradient_checkpoint = hparams.gradient_checkpoint
    
    def state_dict(self):
        dict_ = {
        "discriminator_state_dict": self.discriminator.state_dict(),
            "optimzier_state_dict": self.optimizer.state_dict(),
        }
        return dict_
    
    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_file(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def load_state_dict(self, dict_):
        local_dict = self.discriminator.state_dict()
        new_dict   = dict_['discriminator_state_dict']
        local_dict.update({k: v for k,v in new_dict.items() if k in local_dict and local_dict[k].shape == new_dict[k].shape})
        n_missing_keys = len([k for k,v in new_dict.items() if not (k in local_dict and local_dict[k].shape == new_dict[k].shape)])
        self.discriminator.load_state_dict(local_dict)
        del local_dict, new_dict
        
        if n_missing_keys == 0:
            self.optimizer.load_state_dict(dict_['optimzier_state_dict'])
    
    def merge_inputs(self, tt2_model, pred_mel, speaker_ids, no_grad_speaker_embed=True):
        B, n_mel, mel_T = pred_mel.shape
        mels = pred_mel.new_empty(B, self.prenet_dim+self.speaker_embedding_dim, mel_T)
        
                                                          # [B, n_mel, mel_T] -> [B, prenet_dim, mel_T]
        pred_mel = pred_mel.to(next(tt2_model.decoder.prenet.parameters()).dtype)
        mels[:, :self.prenet_dim] = tt2_model.decoder.prenet(pred_mel.transpose(1, 2)).transpose(1, 2)
        if self.speaker_embedding_dim:
            if no_grad_speaker_embed:
                with torch.no_grad():
                    speaker_embed = tt2_model.speaker_embedding(speaker_ids.detach())# -> [B, embed]
            else:
                speaker_embed     = tt2_model.speaker_embedding(speaker_ids)# -> [B, embed]
            B, embed = speaker_embed.shape
            speaker_embed = speaker_embed[..., None].expand(B, embed, mel_T)# -> [B, embed, mel_T]
            mels[:, self.prenet_dim:] = speaker_embed
        mels = mels[:, :, 2:-2]#             [B, embed, mel_T] -> [B, embed, mel_T-4]
        mels = mels.transpose(1, 2).clone()# [B, embed, mel_T] -> [B, mel_T, embed]
        return mels
    
    def forward(self, tt2_model, pred, gt, reduced_loss_dict, loss_dict, loss_scalars):
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            pred_mel = pred['pred_mel_postnet'].detach() if self.input_feature == 'postnet' else pred['pred_mel'].detach()# [B, n_mel, mel_T]
            B, n_mel, mel_T = pred_mel.shape
            mels = self.merge_inputs(tt2_model, pred_mel, gt['speaker_id'])# [B, mel_T, embed]
        
        if self.training and self.gradient_checkpoint:
            pred_infness = checkpoint(self.discriminator, mels)# -> [B]
        else:
            pred_infness = self.discriminator(mels)# -> [B]
        tf_infness, inf_infness = pred_infness.chunk(2, dim=0)# [B] -> [B/2], [B/2]
        inf_label = torch.ones(B//2, device=pred_mel.device, dtype=pred_mel.dtype)    # [B/2]
        tf_label  = torch.ones(B//2, device=pred_mel.device, dtype=pred_mel.dtype)*-1.# [B/2]
        
        loss_dict['InfGAN_dLoss'] = F.mse_loss(inf_infness, inf_label) + F.mse_loss(tf_infness, tf_label)
        loss = loss_dict['InfGAN_dLoss'] * loss_scalars['InfGAN_dLoss_weight']
        
        reduced_loss_dict['InfGAN_dLoss'] = reduce_tensor(loss_dict['InfGAN_dLoss'].data, self.n_gpus).item() if self.n_gpus > 1 else loss_dict['InfGAN_dLoss'].item()
        
        if self.fp16_run:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        self.optimizer.step()


############################################
##  Causal Temporal Conv1d Inference GAN  ##
############################################
@torch.jit.script
def GLU(input_a, input_b: Optional[Tensor], n_channels: int):
    """Gated Linear Unit (GLU)"""
    in_act = input_a
    if input_b is not None:
        in_act += input_b
    l_act = in_act[:, :n_channels, :]
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    acts = l_act * s_act
    return acts


class CausalTemporalConv1d(nn.Module):
    def __init__(self, n_in_channels:int, cond_in_channels:int,
                n_out_channels:int      , n_channels    :int,
                n_layers      :int      , kernel_size   :int,
                seperable_conv:bool     , merge_res_skip:bool,
                res_skip      :bool=True, n_layers_dilations_w=None,
                summarize_output:bool=False, output_act_func=None,):
        super(CausalTemporalConv1d, self).__init__()
        assert(kernel_size % 2 == 1), 'kernel_size must be odd'
        assert(n_channels  % 2 == 0), 'n_channels must be even'
        assert (res_skip or merge_res_skip) or n_layers == 1, "Cannot remove res_skip without using merge_res_skip"
        self.n_layers   = n_layers
        self.n_channels = n_channels
        self.merge_res_skip = merge_res_skip
        self.summarize_output = summarize_output
        self.gated_unit = GLU
        
        self.in_layers       = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        self.cond_conv = nn.Conv1d(cond_in_channels, 2*n_channels, 1)
        
        start = nn.Conv1d(n_in_channels, n_channels, 1)
        start = nn.utils.weight_norm(start, name='weight')
        self.start = start
        
        if type(n_layers_dilations_w) == int:
            n_layers_dilations_w = [n_layers_dilations_w,]*n_layers # constant dilation if using int
            print("WARNING: Using constant dilation factor for WN in_layer dilation width.")
        self.padding = []
        for i in range(n_layers):
            dilation = 2**i if n_layers_dilations_w is None else n_layers_dilations_w[i]
            self.padding.append(kernel_size*dilation-dilation)
            if (not seperable_conv) or (kernel_size == 1):
                in_layer = nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                           dilation=dilation)
                in_layer = nn.utils.weight_norm(in_layer, name='weight')
            else:
                depthwise = nn.Conv1d(n_channels, n_channels, kernel_size,
                                    dilation=dilation, groups=n_channels)
                depthwise = nn.utils.weight_norm(depthwise, name='weight')
                pointwise = nn.Conv1d(n_channels, 2*n_channels, 1,
                                    dilation=dilation)
                pointwise = nn.utils.weight_norm(pointwise, name='weight')
                in_layer = torch.nn.Sequential(depthwise, pointwise)
            self.in_layers.append(in_layer)
            
            if res_skip:
                res_skip_channels = 2*n_channels if i<(n_layers-1) and not self.merge_res_skip else n_channels
                res_skip_layer = nn.Conv1d(n_channels, res_skip_channels, 1)
                res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
                self.res_skip_layers.append(res_skip_layer)
        
        end = nn.Conv1d(n_channels, n_out_channels, 1)
        end = nn.utils.weight_norm(end, name='weight')
        self.end = end
        self.output_act_func = output_act_func
    
    def causal_pad(self, inp, pad:int, pad_right=False):# [B, C, T]
        return F.pad(inp, (0, pad)) if pad_right else F.pad(inp, (pad, 0))
    
    def forward(self, inp, cond=None):# -> [B, indim, T]
        inp = self.start(inp)# -> [B, n_dim, T]
        
        if cond is not None or hasattr(self, 'cond_conv'):
            if len(cond.shape) == 2:
                cond = cond.unsqueeze(-1)# [B, C] -> [B, C, 1]
            
            cond = self.cond_conv(cond)# -> [B, C, 1]
        
        for i in range(self.n_layers):
            cpad_inp = self.causal_pad(inp, self.padding[i])
            acts = self.gated_unit(self.in_layers[i](cpad_inp), cond, self.n_channels)# -> [B, n_dim, T]
            
            res_skip_acts = self.res_skip_layers[i](acts) if ( hasattr(self, 'res_skip_layers') and len(self.res_skip_layers) ) else acts
            
            if i == 0:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):
                    inp  = inp + res_skip_acts[:,:self.n_channels ,:]
                    outp =       res_skip_acts[:, self.n_channels:,:]
                else:
                    outp = res_skip_acts
            else:
                if (not self.merge_res_skip) and (i < self.n_layers - 1):# if res_skip and not last layer
                    inp  = inp  + res_skip_acts[:,:self.n_channels ,:]
                    outp = outp + res_skip_acts[:, self.n_channels:,:]
                else:
                    outp = outp + res_skip_acts
        
        outp = self.end(outp)# -> [B, outdim, T]
        if self.output_act_func:
            outp = self.output_act_func(outp)
        if self.summarize_output:
            outp = outp.mean(dim=2).squeeze(1)# -> [B, outdim] or [B, 1] -> [B]
        return outp

# Should predict "inferenceness" by predicting whether the spectrogram came from teacher forcing or inference.
# The generator will then attempt to reduce inferenceness and increase teacher-forcedness outputs on the discriminator.
class TemporalInferenceGAN(nn.Module):
    def __init__(self, hparams):
        super(TemporalInferenceGAN, self).__init__()
        assert hparams.batch_size%2==0, 'Batch size must be even!'
        self.speaker_embedding_dim = hparams.speaker_embedding_dim if hparams.InfGAN_use_speaker else 0# speaker embeds
        self.use_spect   = hparams.InfGAN_use_spect  # decoder spectrogram
        self.use_postnet = hparams.InfGAN_use_postnet# postnet spectrogram
        self.use_DecRNN  = hparams.InfGAN_use_DecRNN #   decoderRNN hidden state(s)
        self.use_AttRNN  = hparams.InfGAN_use_AttRNN # AttentionRNN hidden state(s)
        self.use_context = hparams.InfGAN_use_context# Attention Contexts <- the current encoder outputs the attention is focused on
        
        self.input_dim = 0
        if self.use_spect  :# decoder spectrogram
            self.input_dim+=hparams.n_mel_channels
        if self.use_postnet:# postnet spectrogram
            self.input_dim+=hparams.n_mel_channels
        if self.use_DecRNN :#   decoderRNN hidden state(s)
            self.input_dim+=getattr(hparams, 'second_decoder_rnn_dim',   0) or getattr(hparams, 'decoder_rnn_dim'  , 0)
            if getattr(hparams, 'decoder_input_residual', False):
                self.input_dim+=hparams.prenet_dim
        if self.use_AttRNN :# AttentionRNN hidden state(s)
            self.input_dim+=getattr(hparams, 'second_attention_rnn_dim', 0) or getattr(hparams, 'attention_rnn_dim', 0)
        if self.use_context:# Attention Contexts <- the current encoder outputs the attention is focused on
            self.input_dim+=hparams.memory_bottleneck_dim if hparams.use_memory_bottleneck else \
                            hparams.encoder_LSTM_dim+hparams.speaker_embedding_dim+len(hparams.emotion_classes)+hparams.emotionnet_latent_dim+1
        
        self.optimizer     = None
        self.discriminator = CausalTemporalConv1d(
                n_in_channels    = self.input_dim,
                cond_in_channels = self.speaker_embedding_dim,
                n_out_channels   = 1,
                n_channels       = hparams.InfGAN_n_channels,
                n_layers         = hparams.InfGAN_n_layers,
                kernel_size      = hparams.InfGAN_kernel_size,
                seperable_conv   = hparams.InfGAN_seperable_conv,
                merge_res_skip   = hparams.InfGAN_merge_res_skip,
                res_skip         = hparams.InfGAN_res_skip,
                n_layers_dilations_w=None,
                summarize_output=False,
                output_act_func=nn.Tanh(),)
        self.fp16_run    = hparams.fp16_run
        self.n_gpus      = hparams.n_gpus
        self.gradient_checkpoint = hparams.gradient_checkpoint
    
    def state_dict(self):
        dict_ = {
        "discriminator_state_dict": self.discriminator.state_dict(),
            "optimzier_state_dict": self.optimizer.state_dict(),
        }
        return dict_
    
    def save_state_dict(self, path:str):
        torch.save(self.state_dict(), path)
    
    def load_state_dict_from_file(self, path:str):
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def load_state_dict(self, dict_):
        local_dict = self.discriminator.state_dict()
        new_dict   = dict_['discriminator_state_dict']
        local_dict.update({k: v for k,v in new_dict.items() if k in local_dict and local_dict[k].shape == new_dict[k].shape})
        n_missing_keys = len([k for k,v in new_dict.items() if not (k in local_dict and local_dict[k].shape == new_dict[k].shape)])
        self.discriminator.load_state_dict(local_dict)
        del local_dict, new_dict
        
        if 1 and n_missing_keys == 0:
            self.optimizer.load_state_dict(dict_['optimzier_state_dict'])
    
    def maybe_mask_fill(self, inp:Tensor, mask:Optional[Tensor]):
        if mask is not None:
            inp = inp.clone()
            inp.masked_fill_(~mask, 0.0)
        return inp
    
    def merge_inputs(self, tt2_model, pred:dict, gt:dict, tfB:int=0, mask:Optional[Tensor]=None,# mask[B, 1, mel_T] with True == non-padded areas
                                                                no_grad_speaker_embed:bool=True):
        out = []
        if self.use_spect  :# decoder spectrogram
            out.append(self.maybe_mask_fill(pred['pred_mel'][tfB:], mask))# [B, n_mel, mel_T]
        if self.use_postnet:# postnet spectrogram
            out.append(self.maybe_mask_fill(pred['pred_mel_postnet'][tfB:], mask))# [B, n_mel, mel_T]
        if self.use_DecRNN :#   decoderRNN hidden state(s)
            out.append(self.maybe_mask_fill(pred['decoder_rnn_output'][tfB:], mask))# [B, C, mel_T]
        if self.use_AttRNN :# AttentionRNN hidden state(s)
            out.append(self.maybe_mask_fill(pred['attention_rnn_output'][tfB:], mask))# [B, C, mel_T]
        if self.use_context:# Attention Contexts <- the current encoder outputs the attention is focused on    
            out.append(self.maybe_mask_fill(pred['attention_contexts'][tfB:], mask))# [B, memory_dim, mel_T]
        out = torch.cat(out, dim=1)# [[B, ?, mel_T], [B, ?, mel_T], ...] -> [B, C, mel_T]
        
        if self.speaker_embedding_dim:
            if no_grad_speaker_embed:
                with torch.no_grad():
                    speaker_embed = tt2_model.speaker_embedding(gt['speaker_id'][tfB:].detach())# -> [B, embed]
            else:
                speaker_embed     = tt2_model.speaker_embedding(gt['speaker_id'][tfB:])# -> [B, embed]
        return out, speaker_embed
    
    def forward(self, tt2_model, pred:dict, gt:dict, reduced_loss_dict:dict, loss_dict:dict, loss_scalars:dict):
        assert self.optimizer is not None
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            pred_mel = pred['pred_mel']
            B, n_mel, mel_T = pred_mel.shape
            mel_lengths = torch.cat((gt['mel_lengths'].chunk(2, dim=0)[0], pred['pred_mel_lengths']), dim=0)# [B]
            mask = get_mask_from_lengths(mel_lengths, max_len=mel_T).unsqueeze(1)# [B, 1, mel_T]
            mels, *embeds = self.merge_inputs(tt2_model, pred, gt, mask=mask)# [B, mel_T, embed]
        
        if self.training and self.gradient_checkpoint:
            pred_infness = checkpoint(self.discriminator, mels, *embeds).squeeze(1)# -> [B, mel_T]
        else:
            pred_infness = self.discriminator(mels, *embeds).squeeze(1)# -> [B, mel_T]
        tf_infness, inf_infness = pred_infness.chunk(2, dim=0)# [B, mel_T] -> [B/2, mel_T], [B/2, mel_T]
        inf_label = torch.ones(B//2, device=pred_mel.device, dtype=pred_mel.dtype)[:, None].expand(B//2, mel_T)    # [B/2, mel_T]
        tf_label  = torch.ones(B//2, device=pred_mel.device, dtype=pred_mel.dtype)[:, None].expand(B//2, mel_T)*-1.# [B/2, mel_T]
        
        loss_dict['InfGAN_dLoss'] = F.mse_loss(inf_infness, inf_label) + F.mse_loss(tf_infness, tf_label)
        loss = loss_dict['InfGAN_dLoss'] * loss_scalars['InfGAN_dLoss_weight']
        
        reduced_loss_dict['InfGAN_dLoss'] = reduce_tensor(loss_dict['InfGAN_dLoss'].data, self.n_gpus).item() if self.n_gpus > 1 else loss_dict['InfGAN_dLoss'].item()
        
        with torch.no_grad():
            tfB_mel_T = inf_infness.numel()# tfB * mel_T
            reduced_loss_dict['InfGAN_accuracy'] = ((inf_infness>0.0).sum()+(tf_infness<0.0).sum()).item()/(2*tfB_mel_T)
        
        if self.fp16_run:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        self.optimizer.step()
