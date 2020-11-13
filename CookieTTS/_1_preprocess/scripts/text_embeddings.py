# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
import json
import csv
import numpy as np
import os
from tqdm import tqdm

from CookieTTS.utils.torchmoji.sentence_tokenizer import SentenceTokenizer
from CookieTTS.utils.torchmoji.model_def import torchmoji_feature_encoding
from CookieTTS.utils.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from CookieTTS.utils.dataset.utils import load_filepaths_and_text

def write_hidden_states(path_quote_pairs, BATCH_SIZE=50, maxlen=120, name_extension='_tm', max_path_len=999):
    """
    Save torchMoji hidden states as `.npy` files for every path_quote pair in path_quote_pairs.
    
    RETURNS:
        new_path_arr: An array of the new `.npy` files paths, this will be the same len and order as the input array.
    """
    print(f'Tokenizing using dictionary from {VOCAB_PATH}')
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    
    st = SentenceTokenizer(vocabulary, maxlen)
    
    print(f'Loading model from {PRETRAINED_PATH}.')
    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    
    print('Running predictions with TorchMoji.')
    new_paths = []
    for i in tqdm(range(0, len(path_quote_pairs), BATCH_SIZE), total=len(list(range(0, len(path_quote_pairs), BATCH_SIZE))), smoothing=0.01):
        paths = [x[0] for x in path_quote_pairs[i:i+BATCH_SIZE]]
        texts = [x[1] for x in path_quote_pairs[i:i+BATCH_SIZE]]
        
        tokenized, _, _ = st.tokenize_sentences(texts)
        embedding = model(tokenized) # returns np array [B, Embed]
        for j in range(len(embedding)):
            filepath_without_ext = ".".join(paths[j].split(".")[:-1])
            path_path_len = min(len(filepath_without_ext), max_path_len)
            file_path_safe = filepath_without_ext[0:path_path_len]
            new_path = os.path.splitext(paths[j])[0]+'_tm.pt'
            np.save(new_path, embedding[j])
            new_paths.append(new_path)
    assert len(new_paths) == len(path_quote_pairs), f'new_path and path_quote_pairs have different length.\nTorchMoji likely failed to tokenize one of the inputs.\nlen(new_paths) = {len(new_paths)}\nlen(path_quote_pairs) = {len(path_quote_pairs)}'
    return new_paths
