import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn
from CookieTTS.utils.model.utils import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        
        # Gate Loss
        self.pos_weight = torch.tensor(hparams.gate_positive_weight)
        
        # Spectrogram Loss
        self.melout_MSE_scalar = hparams.melout_MSE_scalar
        self.melout_MAE_scalar = hparams.melout_MAE_scalar
        self.melout_SMAE_scalar = hparams.melout_SMAE_scalar
        self.postnet_MSE_scalar = hparams.postnet_MSE_scalar
        self.postnet_MAE_scalar = hparams.postnet_MAE_scalar
        self.postnet_SMAE_scalar = hparams.postnet_SMAE_scalar
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
            self.lag = 500#   dead_steps
            self.k = 2500 #  warmup_steps
            self.x0 = 5000 # cycle_steps
            self.upper = 1.0 # aux weight
            assert (self.lag+self.k) <= self.x0
        
        # SylNet / EmotionNet / AuxEmotionNet Params
        self.n_classes = len(hparams.emotion_classes)
        
        self.zsClassificationNCELoss = 1.0 # EmotionNet Classification Loss (Negative Cross Entropy)
        self.zsClassificationMAELoss = 0.0 # EmotionNet Classification Loss (Mean Absolute Error)
        self.zsClassificationMSELoss = 1.0 # EmotionNet Classification Loss (Mean Squared Error)
        
        self.auxClassificationMAELoss = 1.0 # AuxEmotionNet MAE Classification Loss
        self.auxClassificationMSELoss = 0.0 # AuxEmotionNet MSE Classification Loss
        self.auxClassificationNCELoss = 1.0 # AuxEmotionNet NCE Classification Loss
        
        self.em_kl_weight   = 0.0005 # EmotionNet KDL weight
        self.syl_KDL_weight = 0.0020 # SylNet KDL Weight
        
        self.pred_sylpsMSE_weight = 0.005# Encoder Pred Sylps MSE weight
        self.pred_sylpsMAE_weight = 0.000# Encoder Pred Sylps MAE weight
        
        self.predzu_MSE_weight = 0.05 # AuxEmotionNet Pred Zu MSE weight
        self.predzu_MAE_weight = 0.00 # AuxEmotionNet Pred Zu MAE weight
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
    
    
    def forward(self, model_output, targets, iter):
        mel_target, gate_target, output_lengths, emotion_id_target, emotion_onehot_target, sylps_target, *_ = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_out, mel_out_postnet, gate_out, alignments, pred_sylps, syl_package, em_package, aux_em_package, *_ = model_output
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
            mel_target = torch.masked_select(mel_target, mask)
            mel_out = torch.masked_select(mel_out, mask)
            mel_out_postnet = torch.masked_select(mel_out_postnet, mask)
        
        spec_MSE = nn.MSELoss()(mel_out, mel_target)
        loss = (spec_MSE*self.melout_MSE_scalar)
        spec_MAE = nn.L1Loss()(mel_out, mel_target)
        loss += (spec_MAE*self.melout_MAE_scalar)
        spec_SMAE = nn.SmoothL1Loss()(mel_out, mel_target)
        loss += (spec_SMAE*self.melout_SMAE_scalar)
        
        postnet_MSE = nn.MSELoss()(mel_out_postnet, mel_target)
        loss += (postnet_MSE*self.postnet_MSE_scalar)
        postnet_MAE = nn.L1Loss()(mel_out_postnet, mel_target)
        loss += (postnet_MAE*self.postnet_MAE_scalar)
        postnet_SMAE = nn.SmoothL1Loss()(mel_out_postnet, mel_target)
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
        
        # debug/fun
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
            ]
        return loss, gate_loss, loss_terms