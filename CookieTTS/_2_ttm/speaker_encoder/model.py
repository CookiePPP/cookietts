from math import sqrt, ceil
import random
import numpy as np
from numpy import finfo

import torch
from torch import nn
from torch.nn import functional as F

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict

from CookieTTS.utils.model.layers import ConvNorm, LinearNorm
from CookieTTS.utils.model.GPU import to_gpu
from CookieTTS.utils.model.utils import get_mask_from_lengths, dropout_frame, freeze_grads

drop_rate = 0.5

def load_model(hparams, device=None):
    model = SpeakerEnc(hparams)
    if torch.cuda.is_available() and device != 'cpu':
        model = model.cuda()
    return model


class SpeakerEnc(nn.Module):
    """Generator network."""
    def __init__(self, hparams):
        super(SpeakerEnc, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)
    
    def parse_batch(self, batch, device='cuda'):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self, gt_mel):# gt_mel[B, n_mel, mel_T]
        gt_mel = gt_mel.transpose(1, 2)# [B, n_mel, mel_T] -> [B, mel_T, n_mel]
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(gt_mel)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True)# [B, embed, -1]
        embeds_normalized = embeds.div(norm)
        return embeds_normalized# [B, embed]
    
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.
        :param embeds: the embeddings as a tensor of shape (B, 
        utterances_per_speaker, embed)
        :return: the similarity matrix as a tensor of shape (B,
        utterances_per_speaker, B)
        """
        B, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(B, utterances_per_speaker,
                                 B).to(self.loss_device)
        mask_matrix = 1 - np.eye(B, dtype=np.int)
        for j in range(B):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(B, B, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(B, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (B, 
        utterances_per_speaker, embed)
        :return: the loss and the EER for this batch of embeddings.
        """
        B, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((B * utterances_per_speaker, 
                                         B))
        ground_truth = np.repeat(np.arange(B), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, B, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer