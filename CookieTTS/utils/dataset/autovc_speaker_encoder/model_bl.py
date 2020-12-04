import torch
import torch.nn as nn
from .make_spect import get_spect
from CookieTTS.utils.dataset.utils import load_wav_to_torch

class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=80, dim_cell=768, dim_emb=256):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)
    
    def forward(self, x):
        self.lstm.flatten_parameters()            
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)
        return embeds_normalized
    
    def get_embed_from_path(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=16000)
        spec = get_spect(audio).float().unsqueeze(0)# [1, mel_T, n_mel]
        spec = spec.to(next(self.parameters()))
        if spec.shape[-1]%128:
            spec = torch.nn.functional.pad(spec, (0, 0, 0, spec.shape[-1]%128))
        embeds = []
        for i in range(0, spec.shape[-1], 128):
            embed = self(spec[:, :, i:i+128])# [1, 128, 80] -> [1, embed]
            embeds.append(embed)
            if i > 5:
                break
        embeds = torch.mean(torch.cat(embeds, dim=0), dim=0)# [1, embed]
        return embed.cpu().float().squeeze(0)# [embed]