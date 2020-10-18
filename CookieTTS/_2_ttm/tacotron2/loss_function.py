import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math
from CookieTTS.utils.model.utils import get_mask_from_lengths
from typing import Optional


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


# https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_tacotron2.py#L28-L156
class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """
    def __init__(self, sigma=0.4, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None
    
    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None
    
    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        B, dec_T, enc_T = self.guided_attn_masks.shape
        losses = self.guided_attn_masks * att_ws[:, :dec_T, :enc_T]
        loss = torch.sum(losses.masked_select(self.masks)) / torch.sum(olens) # get mean along B and dec_T
        if self.reset_always:
            self._reset_masks()
        return loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = ilens.shape[0]
        max_ilen = int(ilens.max().item())
        max_olen = int(olens.max().item())
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
        """
        in_masks = get_mask_from_lengths(ilens)  # (B, T_in)
        out_masks = get_mask_from_lengths(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

@torch.jit.script
def NormalLLLoss(mu, logvar, target):
    loss = ((mu-target).pow(2)/logvar.exp())+logvar
    if True:
        loss = loss.mean()
    else:
        pass
    return loss

class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        
        # Gate Loss
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        
        # Spectrogram Loss
        if hparams.LL_SpectLoss and any(bool(x) for x in [hparams.melout_MSE_scalar, hparams.melout_MAE_scalar, hparams.melout_SMAE_scalar, hparams.postnet_MSE_scalar, hparams.postnet_MAE_scalar, hparams.postnet_SMAE_scalar]):
            print("Warning! MSE, MAE and SMAE spectrogram losses will not be used when LL_SpectLoss is True")
        
        self.use_LL_Loss = hparams.LL_SpectLoss
        self.melout_LL_scalar  = hparams.melout_LL_scalar
        self.postnet_LL_scalar = hparams.postnet_LL_scalar
        self.melout_MSE_scalar  = hparams.melout_MSE_scalar
        self.melout_MAE_scalar  = hparams.melout_MAE_scalar
        self.melout_SMAE_scalar = hparams.melout_SMAE_scalar
        self.postnet_MSE_scalar  = hparams.postnet_MSE_scalar
        self.postnet_MAE_scalar  = hparams.postnet_MAE_scalar
        self.postnet_SMAE_scalar = hparams.postnet_SMAE_scalar
        self.adv_postnet_scalar = hparams.adv_postnet_scalar
        self.adv_postnet_reconstruction_weight = hparams.adv_postnet_reconstruction_weight
        self.dis_postnet_scalar = hparams.dis_postnet_scalar
        self.dis_spect_scalar   = hparams.dis_spect_scalar
        self.masked_select = hparams.masked_select
        
        # KL Scheduler Params
        if False:
            self.anneal_function = 'logistic'
            self.lag = ''
            self.k = 0.00025
            self.x0 = 1000
            self.upper = 0.005
        elif False:
            self.anneal_function = 'constant'
            self.lag = None
            self.k = None
            self.x0 = None
            self.upper = 0.5 # weight
        else:
            self.anneal_function = 'cycle'
            self.lag = 50#      dead_steps
            self.k = 7950#   warmup_steps
            self.x0 = 10000#   cycle_steps
            self.upper = 1.0 # aux weight
            assert (self.lag+self.k) <= self.x0
        
        # SylNet / EmotionNet / AuxEmotionNet Params
        self.n_classes = len(hparams.emotion_classes)
        
        self.zsClassificationNCELoss = hparams.zsClassificationNCELoss # EmotionNet Classification Loss (Negative Cross Entropy)
        self.zsClassificationMAELoss  = hparams.zsClassificationMAELoss  # EmotionNet Classification Loss (Mean Absolute Error)
        self.zsClassificationMSELoss  = hparams.zsClassificationMSELoss  # EmotionNet Classification Loss (Mean Squared Error)
                                        
        self.auxClassificationNCELoss = hparams.auxClassificationNCELoss # AuxEmotionNet NCE Classification Loss
        self.auxClassificationMAELoss = hparams.auxClassificationMAELoss # AuxEmotionNet MAE Classification Loss
        self.auxClassificationMSELoss = hparams.auxClassificationMSELoss # AuxEmotionNet MSE Classification Loss
        
        self.em_kl_weight   = hparams.em_kl_weight # EmotionNet KDL weight
        self.syl_KDL_weight = hparams.syl_KDL_weight # SylNet KDL Weight
        
        self.pred_sylpsMSE_weight = hparams.pred_sylpsMSE_weight # Encoder Pred Sylps MSE weight
        self.pred_sylpsMAE_weight = hparams.pred_sylpsMAE_weight # Encoder Pred Sylps MAE weight
        
        self.predzu_MSE_weight = hparams.predzu_MSE_weight # AuxEmotionNet Pred Zu MSE weight
        self.predzu_MAE_weight = hparams.predzu_MAE_weight # AuxEmotionNet Pred Zu MAE weight
        
        self.DiagonalGuidedAttention_scalar = hparams.DiagonalGuidedAttention_scalar
        self.guided_att = GuidedAttentionLoss(sigma=hparams.DiagonalGuidedAttention_sigma)
        
        # debug/fun
        self.AvgClassAcc = 0.0
    
    
    def vae_kl_anneal_function(self, anneal_function, lag, step, k, x0, upper):
        if anneal_function == 'logistic': # https://www.desmos.com/calculator/neksnpgtmz
            return float(upper/(upper+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            if step > lag:
                return min(upper, step/x0)
            else:
                return 0
        elif anneal_function == 'cycle':
            return min(1,(max(0,(step%x0)-lag))/k) * upper
        elif anneal_function == 'constant':
            return upper or 0.001
    
    
    def log_standard_categorical(self, p):
        """Calculates the cross entropy between a (one-hot) categorical vector
        and a standard (uniform) categorical distribution.
        Params:
            p: one-hot categorical distribution [B, n_classes]
        Returns:
            cross_entropy: [B]
        """
        # Uniform prior over y
        prior = F.softmax(torch.ones_like(p), dim=1) # [B, n_classes]
        prior.requires_grad = False
        
        cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1) # [B]
        
        return cross_entropy # [B]
    
    
    # -L(x,y), elbo for labeled data
    def _L(self, y, mu, logvar, beta = 1.0):
        B, d = mu.shape
        
        #KLD = ((beta*0.5)/B) * torch.sum(d + logvar - logvar.exp() - mu.pow(2))# "1 + logvar - logvar.exp()" <- keep variance close to 1.0
        KLD_ = (d + (logvar-logvar.exp()).sum()/B - mu.pow(2).sum()/B) # [] KL Divergence
        KLD = (beta/2)*KLD_
        
        loglik_y = -self.log_standard_categorical(y).sum()/B # [] log p(y)
        
        return -(loglik_y + KLD), -KLD_
    
    # -U(x), elbo for unlabeled data
    def _U(self, log_prob, mu, logvar, beta=1.0):
        B, d = mu.shape# [B, d]
        
        prob = torch.exp(log_prob) # [B, d] prediction from classifier # sums along d to 1.0, exp is equiv to softmax in this case.
        
        #Entropy of q(y|x)  =  H(q(y|x))
        H = -(prob * log_prob).sum(1).mean() # [B, d] -> [B] -> []
        
        # -L(x,y)
        KLD_ = (1 + (logvar-logvar.exp()) - mu.pow(2)).sum(1)# [B] KL Divergence with Normal Dist
        KLD = (beta/2) * KLD_
        
        # [B, n_classes] -> [B] constant, value same for all y since we have a uniform prior
        y = torch.zeros(1, self.n_classes, device=log_prob.device)
        y[:,0] = 1.
        loglik_y = -self.log_standard_categorical(y)
        
        _Lxy = loglik_y + KLD # [B] Categorical Loss + KL Divergence Loss
        
        # sum q(y|x) * -L(x,y) over y
        q_Lxy = (prob * _Lxy[:, None]).sum()/B # ([B, d] * [B, 1]).sum(1).mean() -> []
        
        KLD_bmean = KLD_.sum()/B
        return -(q_Lxy + H), -KLD_bmean # [] + [], []
    
    
    def forward(self, model_output, targets, criterion_dict, iter, em_kl_weight=None, DiagonalGuidedAttention_scalar=None):
        self.em_kl_weight = self.em_kl_weight if em_kl_weight is None else em_kl_weight
        self.DiagonalGuidedAttention_scalar = self.DiagonalGuidedAttention_scalar if DiagonalGuidedAttention_scalar is None else DiagonalGuidedAttention_scalar
        amp, n_gpus, model, model_d, hparams, optimizer, optimizer_d, grad_clip_thresh = criterion_dict.values()
        is_overflow = False
        grad_norm = 0.0
        
        mel_target, gate_target, output_lengths, text_lengths, emotion_id_target, emotion_onehot_target, sylps_target, preserve_decoder, *_ = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_out, mel_out_postnet, gate_out, alignments, pred_sylps, syl_package, em_package, aux_em_package, gan_package, *_ = model_output
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        
        Bsz, n_mel, dec_T = mel_target.shape
        
        unknown_id = self.n_classes
        supervised_mask = (emotion_id_target != unknown_id)# [B] BoolTensor
        unsupervised_mask = ~supervised_mask               # [B] BoolTensor
        
        # remove paddings before loss calc
        if self.masked_select:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(mel_target.size(1), mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            
            mel_target_not_masked = mel_target
            mel_target = torch.masked_select(mel_target, mask)
            if self.use_LL_Loss:
                mel_out, mel_logvar = mel_out.chunk(2, dim=1)
                if mel_out_postnet is not None:
                    mel_out_postnet, mel_logvar_postnet = mel_out_postnet.chunk(2, dim=1)
                mel_logvar = torch.masked_select(mel_logvar, mask)
                mel_logvar_postnet = torch.masked_select(mel_logvar_postnet, mask)
            
            mel_out_not_masked = mel_out
            mel_out = torch.masked_select(mel_out, mask)
            if mel_out_postnet is not None:
                mel_out_postnet_not_masked = mel_out_postnet
                mel_out_postnet = torch.masked_select(mel_out_postnet, mask)
        
        postnet_MSE = postnet_MAE = postnet_SMAE = postnet_LL = torch.tensor(0.)
        
        # spectrogram / decoder loss
        spec_MSE = nn.MSELoss()(mel_out, mel_target)
        spec_MAE = nn.L1Loss()(mel_out, mel_target)
        spec_SMAE = nn.SmoothL1Loss()(mel_out, mel_target)
        if mel_out_postnet is not None:
            postnet_MSE = nn.MSELoss()(mel_out_postnet, mel_target)
            postnet_MAE = nn.L1Loss()(mel_out_postnet, mel_target)
            postnet_SMAE = nn.SmoothL1Loss()(mel_out_postnet, mel_target)
        if self.use_LL_Loss:
            spec_LL = NormalLLLoss(mel_out, mel_logvar, mel_target)
            loss = (spec_LL*self.melout_LL_scalar)
            if mel_out_postnet is not None:
                postnet_LL = NormalLLLoss(mel_out_postnet, mel_logvar_postnet, mel_target)
            loss += (postnet_LL*self.postnet_LL_scalar)
        else:
            spec_LL = postnet_LL = torch.tensor(0.0, device=mel_out.device)
            loss = (spec_MSE*self.melout_MSE_scalar)
            loss += (spec_MAE*self.melout_MAE_scalar)
            loss += (spec_SMAE*self.melout_SMAE_scalar)
            loss += (postnet_MSE*self.postnet_MSE_scalar)
            loss += (postnet_MAE*self.postnet_MAE_scalar)
            loss += (postnet_SMAE*self.postnet_SMAE_scalar)
        
        if True: # gate/stop loss
            gate_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)
            loss += gate_loss
        
        if True: # SylpsNet loss
            sylzu, syl_mu, syl_logvar = syl_package
            sylKLD = -0.5 * (1 + syl_logvar - syl_logvar.exp() - syl_mu.pow(2)).sum()/Bsz
            loss += (sylKLD*self.syl_KDL_weight)
        
        if True: # Pred Sylps loss
            pred_sylps = pred_sylps.squeeze(1)# [B, 1] -> [B]
            sylpsMSE = nn.MSELoss()(pred_sylps, sylps_target)
            sylpsMAE = nn.L1Loss()(pred_sylps, sylps_target)
            loss += (sylpsMSE*self.pred_sylpsMSE_weight)
            loss += (sylpsMAE*self.pred_sylpsMAE_weight)
        
        if True: # EmotionNet loss
            zs, em_zu, em_mu, em_logvar, em_params = [x.squeeze(1) for x in em_package] # VAE-GST loss
            SupervisedLoss = ClassicationMAELoss = ClassicationMSELoss = ClassicationNCELoss = SupervisedKDL = UnsupervisedLoss = UnsupervisedKDL = torch.tensor(0)
            
            kl_scale = self.vae_kl_anneal_function(self.anneal_function, self.lag, iter, self.k, self.x0, self.upper)# outputs 0<s<1
            em_kl_weight = kl_scale*self.em_kl_weight
            
            if ( sum(supervised_mask) > 0): # if labeled data > 0:
                mu_labeled = em_mu[supervised_mask]
                logvar_labeled = em_logvar[supervised_mask]
                log_prob_labeled = zs[supervised_mask]
                y_onehot = emotion_onehot_target[supervised_mask]
                
                # -Elbo for labeled data (L(X,y))
                SupervisedLoss, SupervisedKDL = self._L(y_onehot, mu_labeled, logvar_labeled, beta=em_kl_weight)
                loss += SupervisedLoss
                
                # Add MSE/MAE Loss
                prob_labeled = log_prob_labeled.exp()
                ClassicationMAELoss = nn.L1Loss(reduction='sum')(prob_labeled, y_onehot)/Bsz
                loss += (ClassicationMAELoss*self.zsClassificationMAELoss)
                
                ClassicationMSELoss = nn.MSELoss(reduction='sum')(prob_labeled, y_onehot)/Bsz
                loss += (ClassicationMSELoss*self.zsClassificationMSELoss)
                
                # Add auxiliary classification loss q(y|x) # negative cross entropy
                ClassicationNCELoss = -torch.sum(y_onehot * log_prob_labeled, dim=1).mean()
                loss += (ClassicationNCELoss*self.zsClassificationNCELoss)
            
            if ( sum(unsupervised_mask) > 0): # if unlabeled data > 0:
                mu_unlabeled = em_mu[unsupervised_mask]
                logvar_unlabeled = em_logvar[unsupervised_mask]
                log_prob_unlabeled = zs[unsupervised_mask]
                
                # -Elbo for unlabeled data (U(x))
                UnsupervisedLoss, UnsupervisedKDL = self._U(log_prob_unlabeled, mu_unlabeled, logvar_unlabeled, beta=em_kl_weight)
                loss += UnsupervisedLoss
        
        if True: # AuxEmotionNet loss
            aux_zs, aux_em_mu, aux_em_logvar, aux_em_params = [x.squeeze(1) for x in aux_em_package]
            PredDistMSE = PredDistMAE = AuxClassicationMAELoss = AuxClassicationMSELoss = AuxClassicationNCELoss = torch.tensor(0)
            
            # pred em_zu dist param Loss
            PredDistMSE = nn.MSELoss()(aux_em_params, em_params)
            PredDistMAE = nn.L1Loss()( aux_em_params, em_params)
            loss += (PredDistMSE*self.predzu_MSE_weight + PredDistMAE*self.predzu_MAE_weight)
            
            # Aux Zs Classification Loss
            if ( sum(supervised_mask) > 0): # if labeled data > 0:
                log_prob_labeled = aux_zs[supervised_mask]
                prob_labeled = log_prob_labeled.exp()
                
                AuxClassicationMAELoss = nn.L1Loss(reduction='sum')(prob_labeled, y_onehot)/Bsz
                loss += (AuxClassicationMAELoss*self.auxClassificationMAELoss)
                
                AuxClassicationMSELoss = nn.MSELoss(reduction='sum')(prob_labeled, y_onehot)/Bsz
                loss += (AuxClassicationMSELoss*self.auxClassificationMSELoss)
                
                AuxClassicationNCELoss = -torch.sum(y_onehot * log_prob_labeled, dim=1).mean()
                loss += (AuxClassicationNCELoss*self.auxClassificationNCELoss)
        
        if True:# Diagonal Attention Guiding
            AttentionLoss = self.guided_att(alignments[preserve_decoder==0.0],
                                          text_lengths[preserve_decoder==0.0],
                                        output_lengths[preserve_decoder==0.0])
            loss += (AttentionLoss*self.DiagonalGuidedAttention_scalar)
        
        reduced_d_loss = reduced_avg_fakeness = avg_fakeness = 0.0
        GAN_Spect_MAE = adv_postnet_loss = torch.tensor(0.)
        if True and gan_package[0] is not None:
            real_labels = torch.zeros(mel_target_not_masked.shape[0], device=loss.device, dtype=loss.dtype)# [B]
            fake_labels = torch.ones( mel_target_not_masked.shape[0], device=loss.device, dtype=loss.dtype)# [B]
            
            mel_outputs_adv, speaker_embed, *_ = gan_package
            if self.masked_select:
                fill_mask = mel_target_not_masked == 0.0
                mel_outputs_adv = mel_outputs_adv.clone()
                mel_outputs_adv.masked_fill_(fill_mask, 0.0)
                mel_outputs_adv_masked = torch.masked_select(mel_outputs_adv, mask)
                mel_out_not_masked = mel_out_not_masked.clone()
                mel_out_not_masked.masked_fill_(fill_mask, 0.0)
                if mel_out_postnet is not None:
                    mel_out_postnet_not_masked = mel_out_postnet_not_masked.clone()
                    mel_out_postnet_not_masked.masked_fill_(fill_mask, 0.0)
            
            # spectrograms [B, n_mel, dec_T]
            # mel_target_not_masked
            # mel_out_not_masked
            # mel_out_postnet_not_masked
            # mel_outputs_adv
            
            speaker_embed = speaker_embed.unsqueeze(2).repeat(1, 1, dec_T)# [B, embed] -> [B, embed, dec_T]
            fake_pred_fakeness = model_d(mel_outputs_adv, speaker_embed.detach())# should speaker_embed be attached computational graph? Not sure atm
            avg_fakeness = fake_pred_fakeness.mean()# metric for Tensorboard
            # Tacotron2 Optimizer / Loss
            reduced_avg_fakeness = reduce_tensor(avg_fakeness.data, n_gpus).item() if hparams.distributed_run else avg_fakeness.item()
            
            adv_postnet_loss = nn.BCELoss()(fake_pred_fakeness, real_labels) # [B] -> [] calc loss to decrease fakeness of model
            GAN_Spect_MAE = nn.L1Loss()(mel_outputs_adv_masked, mel_target)
            if reduced_avg_fakeness > 0.4:
                loss += (adv_postnet_loss*self.adv_postnet_scalar)
                loss += (GAN_Spect_MAE*(self.adv_postnet_scalar*self.adv_postnet_reconstruction_weight))
        
        # Tacotron2 Optimizer / Loss
        if hparams.distributed_run:
            reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            reduced_gate_loss = reduce_tensor(gate_loss.data, n_gpus).item()
        else:
            reduced_loss = loss.item()
            reduced_gate_loss = gate_loss.item()
        
        if optimizer is not None:
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), grad_clip_thresh)
                is_overflow = math.isinf(grad_norm) or math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_thresh)
            
            optimizer.step()
        
        # (optional) Discriminator Optimizer / Loss
        if True and gan_package[0] is not None:
            if optimizer_d is not None:
                optimizer_d.zero_grad()
            
            # spectrograms [B, n_mel, dec_T]
            # mel_target_not_masked
            # mel_out_not_masked
            # mel_out_postnet_not_masked
            # mel_outputs_adv
            
            fake_pred_fakeness = model_d(mel_outputs_adv.detach(), speaker_embed.detach())
            fake_d_loss = nn.BCELoss()(fake_pred_fakeness, fake_labels)# [B] -> [] loss to increase distriminated fakeness of fake samples
            
            real_pred_fakeness = model_d(mel_target_not_masked.detach(), speaker_embed.detach())
            real_d_loss = nn.BCELoss()(real_pred_fakeness, real_labels)# [B] -> [] loss to decrease distriminated fakeness of real samples
            
            if self.dis_postnet_scalar and mel_out_postnet is not None:
                fake_pred_fakeness = model_d(mel_out_postnet_not_masked.detach(), speaker_embed.detach())
                fake_d_loss += self.dis_postnet_scalar * nn.BCELoss()(fake_pred_fakeness, fake_labels)# [B] -> [] loss to increase distriminated fakeness of fake samples
            
            if self.dis_spect_scalar:
                fake_pred_fakeness = model_d(mel_out_not_masked.detach(), speaker_embed.detach())
                fake_d_loss += self.dis_spect_scalar * nn.BCELoss()(fake_pred_fakeness, fake_labels)# [B] -> [] loss to increase distriminated fakeness of fake samples
            
            d_loss = (real_d_loss+fake_d_loss) * (self.adv_postnet_scalar*0.5)
            reduced_d_loss = reduce_tensor(d_loss.data, n_gpus).item() if hparams.distributed_run else d_loss.item()
            
            if optimizer_d is not None and reduced_avg_fakeness < 0.85:
                if hparams.fp16_run:
                    with amp.scale_loss(d_loss, optimizer_d) as scaled_loss:
                        scaled_loss.backward()
                else:
                    d_loss.backward()
                
                if hparams.fp16_run:
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer_d), grad_clip_thresh)
                    is_overflow = math.isinf(grad_norm_d) or math.isnan(grad_norm_d)
                else:
                    grad_norm_d = torch.nn.utils.clip_grad_norm_(
                        model_d.parameters(), grad_clip_thresh)
                
                optimizer_d.step()
        
        with torch.no_grad(): # debug/fun
            S_Bsz = supervised_mask.sum().item()
            U_Bsz = unsupervised_mask.sum().item()
            ClassicationAccStr = 'N/A'
            Top1ClassificationAcc = 0.0
            if S_Bsz > 0:
                Top1ClassificationAcc = (torch.argmax(log_prob_labeled.exp(), dim=1) == torch.argmax(y_onehot, dim=1)).float().sum().item()/S_Bsz # top-1 accuracy
                self.AvgClassAcc = self.AvgClassAcc*0.95 + Top1ClassificationAcc*0.05
                ClassicationAccStr = round(Top1ClassificationAcc*100, 2)
        
        print(
            "            Total loss = ", loss.item(), '\n',
            "             Spect LLL = ", spec_LL.item(), '\n',
            "     Postnet Spect LLL = ", postnet_LL.item(), '\n',
            "             Spect MSE = ", spec_MSE.item(), '\n',
            "             Spect MAE = ", spec_MAE.item(), '\n',
            "            Spect SMAE = ", spec_SMAE.item(), '\n',
            "     Postnet Spect MSE = ", postnet_MSE.item(), '\n',
            "     Postnet Spect MAE = ", postnet_MAE.item(), '\n',
            "    Postnet Spect SMAE = ", postnet_SMAE.item(),'\n',
            "              Gate BCE = ", gate_loss.item(), '\n',
            "                sylKLD = ", sylKLD.item(), '\n',
            "              sylpsMSE = ", sylpsMSE.item(), '\n',
            "              sylpsMAE = ", sylpsMAE.item(), '\n',
            "        SupervisedLoss = ", SupervisedLoss.item(), '\n',
            "         SupervisedKDL = ", SupervisedKDL.item(), '\n',
            "      UnsupervisedLoss = ", UnsupervisedLoss.item(), '\n',
            "       UnsupervisedKDL = ", UnsupervisedKDL.item(), '\n',
            "   ClassicationMSELoss = ", ClassicationMSELoss.item(), '\n',
            "   ClassicationMAELoss = ", ClassicationMAELoss.item(), '\n',
            "   ClassicationNCELoss = ", ClassicationNCELoss.item(), '\n',
            "AuxClassicationMSELoss = ", AuxClassicationMSELoss.item(), '\n',
            "AuxClassicationMAELoss = ", AuxClassicationMAELoss.item(), '\n',
            "AuxClassicationNCELoss = ", AuxClassicationNCELoss.item(), '\n',
            "      Predicted Zu MSE = ", PredDistMSE.item(), '\n',
            "      Predicted Zu MAE = ", PredDistMAE.item(), '\n',
            "     DiagAttentionLoss = ", AttentionLoss.item(), '\n',
            "       PredAvgFakeness = ", reduced_avg_fakeness, '\n',
            "          GeneratorMAE = ", GAN_Spect_MAE.item(), '\n',
            "         GeneratorLoss = ", adv_postnet_loss.item(), '\n',
            "     DiscriminatorLoss = ", reduced_d_loss/self.adv_postnet_scalar, '\n',
            "      ClassicationAcc  = ", ClassicationAccStr, '%\n',
            "      AvgClassicatAcc  = ", round(self.AvgClassAcc*100, 2), '%\n',
            "      Total Batch Size = ", Bsz, '\n',
            "      Super Batch Size = ", S_Bsz, '\n',
            "      UnSup Batch Size = ", U_Bsz, '\n',
            sep='')
        
        loss_terms = [
            [loss.item(), 1.0],
            [spec_MSE.item(), self.melout_MSE_scalar],
            [spec_MAE.item(), self.melout_MAE_scalar],
            [spec_SMAE.item(), self.melout_SMAE_scalar],
            [postnet_MSE.item(), self.postnet_MSE_scalar],
            [postnet_MAE.item(), self.postnet_MAE_scalar],
            [postnet_SMAE.item(), self.postnet_SMAE_scalar],
            [gate_loss.item(), 1.0],
            [sylKLD.item(), self.syl_KDL_weight],
            [sylpsMSE.item(), self.pred_sylpsMSE_weight],
            [sylpsMAE.item(), self.pred_sylpsMAE_weight],
            [SupervisedLoss.item(), 1.0],
            [SupervisedKDL.item(), em_kl_weight*0.5],
            [UnsupervisedLoss.item(), 1.0],
            [UnsupervisedKDL.item(), em_kl_weight*0.5],
            [ClassicationMSELoss.item(), self.zsClassificationMSELoss],
            [ClassicationMAELoss.item(), self.zsClassificationMAELoss],
            [ClassicationNCELoss.item(), self.zsClassificationNCELoss],
            [AuxClassicationMSELoss.item(), self.auxClassificationMSELoss],
            [AuxClassicationMAELoss.item(), self.auxClassificationMAELoss],
            [AuxClassicationNCELoss.item(), self.auxClassificationNCELoss],
            [PredDistMSE.item(), self.predzu_MSE_weight],
            [PredDistMAE.item(), self.predzu_MAE_weight],
            [Top1ClassificationAcc, 1.0],
            [reduced_avg_fakeness, 1.0],
            [adv_postnet_loss.item(), self.adv_postnet_scalar],
            [reduced_d_loss/self.adv_postnet_scalar, self.adv_postnet_scalar],
            ]
        return loss, gate_loss, loss_terms, reduced_loss, reduced_gate_loss, grad_norm, is_overflow