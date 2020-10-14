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

# Source: https://github.com/NVIDIA/DeepLearningExamples/blob/4d808052904d9afaa1ff14579aa4bb25990cf5db/PyTorch/SpeechSynthesis/FastPitch/fastpitch/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieTTS.utils.model.layers import ConvReLUNorm


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, hparams):
        super(TemporalPredictor, self).__init__()
        filter_size = hparams.len_pred_filter_size
        kernel_size = hparams.len_pred_kernel_size
        dropout = hparams.len_pred_dropout
        n_layers = hparams.len_pred_n_layers
        
        self.layers = nn.Sequential(*[ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout) for i in range(n_layers)]
        )
        self.fc = nn.Linear(filter_size, 1, bias=True)

    def forward(self, enc_out, enc_out_mask=None):# [B, enc_T, dim]
        if enc_out_mask is not None:
            enc_out = enc_out * enc_out_mask
        
        enc_out = self.layers(enc_out.transpose(1, 2)).transpose(1, 2)
        enc_out = self.fc(enc_out)
        
        if enc_out_mask is not None:
            enc_out = enc_out * enc_out_mask
        return enc_out.squeeze(-1)# [B, enc_T]