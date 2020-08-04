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
        self.loss_func = hparams.loss_func
        self.masked_select = hparams.masked_select
        
        # VAE
        self.vae_gst_loss = hparams.gst_vae_mode
        if False:
            self.anneal_function = 'logistic'
            self.lag = ''
            self.k = 0.00025
            self.x0 = 1000
            self.upper = 0.005
            self.kl_weight = 1.0
        else:
            self.anneal_function = 'cycle'
            self.lag = 1000#   dead_steps
            self.k = 4000 #  warmup_steps
            self.x0 = 10000 # cycle_steps
            self.upper = 1.0 # aux weight
            self.kl_weight = 1.0 # weight
            assert (self.lag+self.k) <= self.x0
        
        # Semi-Supervised VAE
        self.ss_vae_gst_loss = hparams.ss_vae_gst
        self.n_classes = len(hparams.vae_classes)
        self.vae_alpha = 1.0 # Classifier loss
        
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
            return 0.001
    
    
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
        
        print(
            f"sloglik_y = {loglik_y}",
            f"sKLD_ = {KLD_}", sep='\n')
        
        return loglik_y + KLD
    
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
        
        print(
            f"uloglik_y = {loglik_y.sum()/B}",
            f"uKLD_ = {KLD_.sum()/B}",
            f"_Lxy = {_Lxy}",
            f"H = {H}", sep='\n')
        
        return q_Lxy + H # [] + []
    
    def forward(self, model_output, targets, iter):
        mel_target, gate_target, output_lengths, emotion_id_target, emotion_onehot_target, *_ = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_out, mel_out_postnet, gate_out, alignments, gst_extra = model_output
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        
        Bsz, n_mel, dec_T = mel_target.shape
        
        # remove paddings before loss calc
        if self.masked_select:
            mask = get_mask_from_lengths(output_lengths)
            mask = mask.expand(mel_target.size(1), mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            mel_target = torch.masked_select(mel_target, mask)
            mel_out = torch.masked_select(mel_out, mask)
            mel_out_postnet = torch.masked_select(mel_out_postnet, mask)
        
        if self.loss_func == 'MSELoss':
            loss = nn.MSELoss()(mel_out, mel_target) + \
                nn.MSELoss()(mel_out_postnet, mel_target)
        elif self.loss_func == 'SmoothL1Loss':
            loss = nn.SmoothL1Loss()(mel_out, mel_target) + \
                nn.SmoothL1Loss()(mel_out_postnet, mel_target)
        
        if True: # gate/stop loss
            gate_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(gate_out, gate_target)
            loss += gate_loss
        
        if self.vae_gst_loss: # VAE-GST loss
            kl_weight = self.vae_kl_anneal_function(self.anneal_function, self.lag, iter, self.k, self.x0, self.upper) # outputs 0<y<1
            kl_weight = kl_weight * self.kl_weight
            if not self.ss_vae_gst_loss: # where y labels don't exist
                z, mu, logvar = gst_extra
                kl_loss = -0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2))# "1 + logvar - logvar.exp()" <- keep variance close to 1.0
                loss += kl_loss*kl_weight
            else: # Semi-Supervised Loss for case where y labels exist
                z, mu, logvar, zs = gst_extra# zu_sample, zu_mean, zu_logvar, pred_logprob
                emotion_id_target     # indexes
                emotion_onehot_target # one-hot
                
                unknown_id = self.n_classes
                supervised_mask = (emotion_id_target != unknown_id)# [B] BoolTensor
                unsupervised_mask = ~supervised_mask               # [B] BoolTensor
                
                SupervisedLoss = 0
                ClassicationLoss = 0
                if ( sum(supervised_mask) > 0): # if labeled data > 0:
                    mu_labeled = mu[supervised_mask]
                    logvar_labeled = logvar[supervised_mask]
                    y_onehot = emotion_onehot_target[supervised_mask]
                    
                    # -Elbo for labeled data (L(X,y))
                    SupervisedLoss = -self._L(y_onehot, mu_labeled, logvar_labeled, beta=kl_weight)
                    
                    # Add auxiliary classification loss q(y|x)
                    s_log_prob = zs[supervised_mask]
                    
                    # negative cross entropy
                    ClassicationLoss = -torch.sum(y_onehot * s_log_prob, dim=1).mean()
                
                UnsupervisedLoss = 0
                if ( sum(unsupervised_mask) > 0): # if unlabeled data > 0:
                    mu_unlabeled = mu[unsupervised_mask]
                    logvar_unlabeled = logvar[unsupervised_mask]
                    
                    # get q(y|x) from classifier
                    u_log_prob = zs[unsupervised_mask]
                    
                    # -Elbo for unlabeled data (U(x))
                    UnsupervisedLoss = -self._U(u_log_prob, mu_unlabeled, logvar_unlabeled, beta=kl_weight)
                
                # debug/fun
                S_Bsz = supervised_mask.sum().item()
                U_Bsz = unsupervised_mask.sum().item()
                ClassicationAccStr = 'N/A'
                if S_Bsz > 0:
                    ClassificationAcc = (torch.argmax(s_log_prob.exp(), dim=1) == torch.argmax(y_onehot, dim=1)).float().sum().item()/S_Bsz # top-1 accuracy
                    self.AvgClassAcc = self.AvgClassAcc*0.95 + ClassificationAcc*0.05
                    ClassicationAccStr = round(ClassificationAcc*100, 2)
                print(
                    "            loss = ", loss, '\n',
                    "  SupervisedLoss = ", SupervisedLoss, '\n',
                    "UnsupervisedLoss = ", UnsupervisedLoss, '\n',
                    "ClassicationLoss = ", ClassicationLoss, '\n',
                    "ClassicationAcc  = ", ClassicationAccStr, '%\n',
                    "AvgClassicatAcc  = ", round(self.AvgClassAcc*100, 2), '%\n',
                    "Total Batch Size = ", Bsz, '\n',
                    "Super Batch Size = ", S_Bsz, '\n',
                    "UnSup Batch Size = ", U_Bsz, '\n',
                    sep='')
                loss += SupervisedLoss + UnsupervisedLoss + (ClassicationLoss*self.vae_alpha)
                
        return loss, gate_loss
