# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import numpy as np
import os
from tqdm import tqdm

from utils.torchmoji.sentence_tokenizer import SentenceTokenizer
from utils.torchmoji.model_def import torchmoji_feature_encoding
from utils.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


INPUT_PATHS = [
    '/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/train_taca2.txt',
    '/media/cookie/Samsung 860 QVO/ClipperDatasetV2/filelists/validation_taca2.txt',
    ]
BATCH_SIZE = 50

# get dataset from text files
dataset = [j.split("|") for i in [files_to_list(x) for x in INPUT_PATHS] for j in i]
paths = [x[0] for x in dataset]
texts = [x[1] for x in dataset]

# remove filtered_chars from text
filtered_chars=["☺","␤"]
for i, text in enumerate(texts):
    for filtered_char in filtered_chars:
        texts[i] = texts[i].replace(filtered_char,"")

data = list(zip(paths,texts))

maxlen = 120

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_feature_encoding(PRETRAINED_PATH)
print(model)
print('Running predictions.')
for i in tqdm(range(0, len(data), BATCH_SIZE), total=len(list(range(0, len(data), BATCH_SIZE))), smoothing=0.01):
    paths = [x[0] for x in data[i:i+BATCH_SIZE]]
    texts = [x[1] for x in data[i:i+BATCH_SIZE]]
    #print(texts)
    tokenized, _, _ = st.tokenize_sentences(texts)
    embedding = model(tokenized) # returns np array [B, Embed]
    for j in range(len(embedding)):
        filepath_without_ext = ".".join(paths[j].split(".")[:-1])
        path_path_len = min(len(filepath_without_ext), 999)
        file_path_safe = filepath_without_ext[0:path_path_len]
        #if os.path.exists(file_path_safe+"_embed.npy"): os.remove(file_path_safe+"_embed.npy")
        np.save(file_path_safe + "_.npy", embedding[j])
        #tqdm.write(str(embedding[j]))
