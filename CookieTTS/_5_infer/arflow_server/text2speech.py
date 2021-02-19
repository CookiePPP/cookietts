import sys
import os
import numpy as np
import random
import time
from datetime import datetime

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import json
import re
import difflib
from glob import glob
from unidecode import unidecode
import nltk # sentence spliting
from nltk import sent_tokenize

from CookieTTS._2_ttm.arflow_hifigan.model import load_model
from CookieTTS._4_mtw.waveglow.denoiser import Denoiser
from CookieTTS.utils.text import text_to_sequence
from CookieTTS.utils.dataset.utils import load_filepaths_and_text
from CookieTTS.utils.model.utils import alignment_metric

def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).long()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.int64)
    mask = (ids < lengths.unsqueeze(1))
    return mask


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # list(chunks([0,1,2,3,4,5,6,7,8,9],2)) -> [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# generator for text splitting.
def parse_text_into_segments(texts, target_segment_len=120, split_at_quotes=True, split_at_newline=True):
    texts = (texts.strip()
                 .replace("  "," ")# remove double spaces
                 .replace("_" ," ")# remove _
                 .replace("*" , "")# remove *
                 .replace("> --------------------------------------------------------------------------","")
                 .replace("------------------------------------","")
            )
    assert len(texts)
    
    if split_at_quotes:
        # split text by quotes
        quo ='"' # nested quotes in list comprehension are hard to work with
        wsp =' '
        texts = [f'"{text.replace(quo,"").strip(wsp)}"' if i%2 else text.replace(quo,"").strip(wsp) for i, text in enumerate(unidecode(texts).split('"'))]
        assert len(texts)
    else:
        texts = [unidecode(texts),]
    
    if split_at_newline:
        texts = [text.lstrip(',.!? ') for textp in texts for text in textp.splitlines(True)]
    else:
        texts = [text.lstrip(',.!? ') for text in texts]
        assert len(texts)
    
    is_inside_quotes = False
    texts_out = []
    rev_texts = list(reversed(texts))
    while len(rev_texts):
        text = rev_texts.pop()# pop current segment to text_seg
        
        is_inside_quotes = bool(text.startswith('"'))
        end_line         = bool(text.endswith('\n'))
        end_paragraph    = bool(len(rev_texts) == 0 or (text.endswith('\n') and rev_texts[-1] == '\n'))
        if len(text.strip(' \n?!.;:*()[]"\'_@~#$%^&+=-|`')) == 0:# ensure that there is more than just symbols in the text segment.
            continue
        
        if (len(rev_texts) and
            len(text)+1+len(rev_texts[-1]) <= target_segment_len and
            ((not split_at_newline) or (not end_line)) and
            ((not split_at_quotes ) or ('"'  not in text))
           ):
            rev_texts[-1] = f'{text} {rev_texts[-1]}'
            continue
        if len(text) <= target_segment_len:
            text = text.strip('\n "')
            if text[-1] not in set(".,?!;:"):
                text+='.'
            texts_out.append(text)
        else:
            if any(x in text for x in set('.?!')):
                text_parts = sent_tokenize(text)
                tmp = ''
                j = 0
                for part in text_parts:
                    if j==0 or len(tmp)+1+len(part) <= target_segment_len:
                        tmp+=f' {part}'
                        j+=1
                    else:
                        break
                if len(tmp) <= target_segment_len:
                    text = (' '.join(text_parts[:j])).strip('\n "')
                    if text[-1] not in set(".,?!;:"):
                        text+='.'
                    texts_out.append(text)
                    if len(text_parts[j:]):
                        rev_texts.append(' '.join(text_parts[j:]))
                    continue
            if ',' in text:
                text_parts = text.split(',')
                tmp = ''
                j = 0
                for part in text_parts:
                    if j==0 or len(tmp)+1+len(part) <= target_segment_len:
                        tmp+=f' {part}'
                        j+=1
                    else:
                        break
                if len(tmp) <= target_segment_len:
                    text = (','.join(text_parts[:j])).strip('\n "')
                    if text[-1] not in set(".,?!;:"):
                        text+=','
                    texts_out.append(text)
                    if len(text_parts[j:]):
                        rev_texts.append(','.join(text_parts[j:]))
                    continue
            if ' ' in text:
                text_parts = [x for x in text.split(' ') if len(x.split())]
                tmp = ''
                j = 0
                for part in text_parts:
                    if j==0 or len(tmp)+1+len(part) <= target_segment_len:
                        tmp+=f' {part}'
                        j+=1
                    else:
                        break
                if len(tmp) <= target_segment_len:
                    text = (' '.join(text_parts[:j])).strip('\n "')
                    if text[-1] not in set(".,?!;:"):
                        text+=','
                    texts_out.append(text)
                    if len(text_parts[j:]):
                        rev_texts.append(' '.join(text_parts[j:]))
                    continue
                else:
                    print(f'[{tmp.lstrip()}]')
                    raise Exception('Found text segment over target length with no punctuation breaks. (no spaces, commas, periods, exclaimation/question points, colons, etc.)')
    texts = texts_out
    
    # remove " marks
    texts = [x.replace('"',"").lstrip() for x in texts]
    
    # remove empty text inputs
    texts = [x for x in texts if len(x.strip())]
    
    return texts


def get_first_over_thresh(x, threshold):
    """Takes [B, T] and outputs first T over threshold for each B (output.shape = [B])."""
    device = x.device
    x = x.clone().cpu().float() # using CPU because GPU implementation of argmax() splits tensor into 32 elem chunks, each chunk is parsed forward then the outputs are collected together... backwards
    x[:,-1] = threshold # set last to threshold just incase the output didn't finish generating.
    x[x>threshold] = threshold
    if int(''.join(torch.__version__.split('+')[0].split('.'))) < 170:
        return ( (x.size(1)-1)-(x.flip(dims=(1,)).argmax(dim=1)) ).to(device).int()
    else:
        return x.argmax(dim=1).to(device).int()


class T2S:
    def __init__(self, conf):
        self.conf = conf
        torch.set_grad_enabled(False)
        
        # load arflow
        self.ttm_current = self.conf['TTM']['default_model']
        assert self.ttm_current in self.conf['TTM']['models'].keys(), "arflow default model not found in config models"
        arflow_path = self.conf['TTM']['models'][self.ttm_current]['modelpath'] # get first available arflow
        self.arflow, self.ttm_hparams, self.ttm_sp_name_lookup, self.ttm_sp_id_lookup = self.load_arflow(arflow_path)
        
        # override since my checkpoints are still missing speaker names
        if self.conf['TTM']["models"][self.ttm_current]['use_speaker_ids_file_override']:
            speaker_ids_fpath = self.conf['TTM']["models"][self.ttm_current]['speaker_ids_file']
            self.ttm_sp_name_lookup = {name.strip(): self.ttm_sp_id_lookup[int(ext_id.strip())] for _, name, ext_id, *_ in load_filepaths_and_text(speaker_ids_fpath)}
        
        # load HiFi-GAN
        self.MTW_current = self.conf['MTW']['default_model']
        assert self.MTW_current in self.conf['MTW']['models'].keys(), "HiFi-GAN default model not found in config models"
        vocoder_path = self.conf['MTW']['models'][self.MTW_current]['modelpath']
        self.vocoder, self.MTW_conf = self.load_hifigan(vocoder_path)
        
        # load torchMoji
        self.tm_sentence_tokenizer, self.tm_torchmoji = self.load_torchmoji()
        
        # load arpabet/pronounciation dictionary
        dict_path = self.conf['dict_path']
        self.load_arpabet_dict(dict_path)
        
        # download nltk package for splitting text into sentences
        nltk.download('punkt')
        
        print("T2S Initialized and Ready!")
    
    
    def load_arpabet_dict(self, dict_path):
        print("Loading ARPAbet Dictionary... ", end="")
        self.arpadict = {}
        for line in reversed((open(dict_path, "r").read()).splitlines()):
            self.arpadict[(line.split(" ", 1))[0]] = (line.split(" ", 1))[1].strip()
        print("Done!")
    
    
    def ARPA(self, text, punc=r"!?,.;:␤#-_'\"()[]"):
        text = text.replace("\n"," ")
        out = ''
        for word in text.split(" "):
            end_chars = ''; start_chars = ''
            while any(elem in word for elem in punc) and len(word) > 1:
                if word[-1] in punc: end_chars = word[-1] + end_chars; word = word[:-1]
                elif word[0] in punc: start_chars = start_chars + word[0]; word = word[1:]
                else: break
            if word.upper() in self.arpadict.keys():
                word = "{" + str(self.arpadict[word.upper()]) + "}"
            out = (out + " " + start_chars + word + end_chars).strip()
        return out
    
    
    def load_torchmoji(self):
        """ Use torchMoji to score texts for emoji distribution.
        
        The resulting emoji ids (0-63) correspond to the mapping
        in emoji_overview.png file at the root of the torchMoji repo.
        
        Writes the result to a csv file.
        """
        import json
        import numpy as np
        import os
        from CookieTTS.utils.torchmoji.sentence_tokenizer import SentenceTokenizer
        from CookieTTS.utils.torchmoji.model_def import torchmoji_feature_encoding
        from CookieTTS.utils.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
        
        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        
        maxlen = 130
        texts = ["Testing!",]
        
        with torch.no_grad():
            # init model
            st = SentenceTokenizer(vocabulary, maxlen)
            torchmoji = torchmoji_feature_encoding(PRETRAINED_PATH)
        return st, torchmoji
    
    
    def get_torchmoji_hidden(self, texts):
        with torch.no_grad():
            tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences(texts) # input array [B] e.g: ["Test?","2nd Sentence!"]
            embedding = self.tm_torchmoji(tokenized) # returns np array [B, Embed]
        return embedding
    
    
    def load_hifigan(self, vocoder_path):
        print("Loading HiFi-GAN...")
        from CookieTTS._4_mtw.hifigan.models import load_model as load_hifigan_model
        vocoder, vocoder_config = load_hifigan_model(vocoder_path)
        vocoder.half()
        print("Done!")
        
        print("Clearing CUDA Cache... ", end='')
        torch.cuda.empty_cache()
        print("Done!")
        
        print('\n'*10)
        import gc # prints currently alive Tensors and Variables  # - And fixes the memory leak? I guess poking the leak with a stick is the answer for now.
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    pass#print(type(obj), obj.size())
            except:
                pass
        print('\n'*10)
        
        return vocoder, vocoder_config
    
    def update_hifigan(self, vocoder_name):
        print(f"Changing HiFi-GAN to {vocoder_name}")
        self.MTW_current = vocoder_name
        assert self.MTW_current in self.conf['MTW']['models'].keys(), f"HiFi-GAN model '{vocoder_name}' not found in config models"
        vocoder_path = self.conf['MTW']['models'][self.MTW_current]['modelpath']
        self.vocoder, self.MTW_conf = self.load_hifigan(vocoder_path)
    
    def load_arflow(self, arflow_path):
        """Loads arflow,
        Returns:
        - model
        - hparams
        - speaker_lookup
        """
        checkpoint = torch.load(arflow_path) # load file into memory
        print("Loading arflow... ", end="")
        checkpoint_hparams = checkpoint['hparams'   ] # get hparams
        checkpoint_dict    = checkpoint['state_dict'] # get state_dict
        
        model = load_model(checkpoint_hparams) # initialize the model
        model.load_state_dict(checkpoint_dict) # load pretrained weights
        _ = model.cuda().eval()#.half()
        print("Done")
        
        arflow_speaker_name_lookup = checkpoint['speaker_name_lookup'] # save speaker name lookup
        arflow_speaker_id_lookup   = checkpoint['speaker_id_lookup']   # save speaker_id lookup
        
        print(f"This arflow model has been trained for {checkpoint['iteration']} Iterations.")
        return model, checkpoint_hparams, arflow_speaker_name_lookup, arflow_speaker_id_lookup
    
    
    def update_mf(self, arflow_name):
        self.arflow, self.ttm_hparams, self.ttm_sp_name_lookup, self.ttm_sp_id_lookup = self.load_arflow(self.conf['TTM']['models'][arflow_name]['modelpath'])
        self.ttm_current = arflow_name
        
        if self.conf['TTM']['models'][arflow_name]['use_speaker_ids_file_override']:# (optional) override
            self.ttm_sp_name_lookup = {name.strip(): self.ttm_sp_id_lookup[int(ext_id.strip())] for _, name, ext_id, *_ in load_filepaths_and_text(self.conf['TTM']['models'][arflow_name]['speaker_ids_file'])}
    
    
    def get_closest_names(self, names):
        possible_names = list(self.ttm_sp_name_lookup.keys())
        validated_names = [difflib.get_close_matches(name, possible_names, n=2, cutoff=0.01)[0] for name in names] # change all names in input to the closest valid name
        return validated_names
    
    
    @torch.no_grad()
    def infer(self, text, speaker, use_arpabet, cat_silence_s,
              batch_size, max_attempts, max_duration_s, dyna_max_duration_s,
              textseg_len_target, split_nl, split_quo, multispeaker_mode, seed,
              global_sigma, char_sigma, frame_sigma,
              filename_prefix=None,
              status_updates=True, show_time_to_gen=True,
              target_score=0.75, absolutely_required_score=0.60, absolute_maximum_tries=256):
        # res_sigma=global_variance, mel_z_sigma=frame_variance, char_z_sigma=char_variance
        """
        PARAMS:
        ...
        
        ...
        """
        os.makedirs(self.conf["working_directory"], exist_ok=True)
        os.makedirs(self.conf["output_directory" ], exist_ok=True)
        
        if seed != '':
            torch.random.manual_seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
        
        # time to gen
        audio_len = 0.
        start_time = time.time()
        
        # Score Metric
        scores = []
        
        # Score Parameters
        diagonality_weighting = 0.5 # 'pacing factor', a penalty for clips where the model pace changes often/rapidly. # this thing does NOT work well for Rarity.
        max_focus_weighting   = 1.0 # 'stuck factor', a penalty for clips that spend execisve time on the same letter.
        min_focus_weighting   = 0.5 # 'miniskip factor', a penalty for skipping/ignoring single letters in the input text.
        avg_focus_weighting   = 1.0 # 'skip factor', a penalty for skipping very large parts of the input text
        
        # split the text into chunks (if applicable)
        texts = parse_text_into_segments(text, target_segment_len=textseg_len_target, split_at_quotes=split_quo, split_at_newline=split_nl)
        del text
        
        total_len = len(texts)
        
        # update arflow stopping params
        frames_per_second = float(self.ttm_hparams.sampling_rate/self.ttm_hparams.hop_length)
        self.arflow.max_decoder_steps = int(min(max([len(t) for t in texts]) * float(dyna_max_duration_s)*frames_per_second, float(max_duration_s)*frames_per_second))
        
        # find closest valid name(s)
        speaker_names = self.get_closest_names(speaker)
        
        # add a filename prefix to keep multiple requests seperate
        if not filename_prefix:
            filename_prefix = f'{datetime.now().strftime("%y_%m_%d-%H_%M_%S")}-{speaker_names[0][:20]}-{str(seed)}'
        
        # add output filename
        output_filename = f"{filename_prefix}"
        
        # pick how the batch will be handled
        simultaneous_texts = max(batch_size//max_attempts, 1)
        batch_size_per_text = min(batch_size, max_attempts)
        
        # for size merging
        running_fsize = 0
        fpaths = []
        out_count = 0
        
        # keeping track of stats for html/terminal
        show_inference_progress_start = time.time()
        all_best_scores = []
        continue_from = 0
        counter = 0
        total_specs = 0
        n_passes = 0
        
        text_batch_in_progress = []
        for text_index, text in enumerate(texts):
            if text_index < continue_from: print(f"Skipping {text_index}.\t",end=""); counter+=1; continue
            last_text = (text_index == (total_len-1)) # true if final text input
            
            # setup the text batches
            text_batch_in_progress.append(text)
            if (len(text_batch_in_progress) == simultaneous_texts) or last_text: # if text batch ready or final input
                text_batch = text_batch_in_progress
                text_batch_in_progress = []
            else:
                continue # if batch not ready, add another text
            
            self.arflow.max_decoder_steps = int(min(max([len(t) for t in text_batch]) * float(dyna_max_duration_s)*frames_per_second, float(max_duration_s)*frames_per_second))
            
            if multispeaker_mode == "not_interleaved": # non-interleaved
                batch_speaker_names = speaker_names * -(-simultaneous_texts//len(speaker_names))
                batch_speaker_names = batch_speaker_names[:simultaneous_texts]
            elif multispeaker_mode == "interleaved": # interleaved
                repeats = -(-simultaneous_texts//len(speaker_names))
                batch_speaker_names = [i for i in speaker_names for _ in range(repeats)][:simultaneous_texts]
            elif multispeaker_mode == "random": # random
                batch_speaker_names = [random.choice(speaker_names),] * simultaneous_texts
            elif multispeaker_mode == "cycle_next": # use next speaker for each text input
                def shuffle_and_return():
                    first_speaker = speaker_names[0]
                    speaker_names.append(speaker_names.pop(0))
                    return first_speaker
                batch_speaker_names = [shuffle_and_return() for i in range(simultaneous_texts)]
            else:
                raise NotImplementedError
            
            if 0:# (optional) use different speaker list for text inside quotes
                speaker_ids = [random.choice(speakers).split("|")[2] if ('"' in text) else random.choice(narrators).split("|")[2] for text in text_batch] # pick speaker if quotemark in text, else narrator
            text_batch  = [text.replace('"',"") for text in text_batch] # remove quotes from text
            
            if len(batch_speaker_names) > len(text_batch):
                batch_speaker_names = batch_speaker_names[:len(text_batch)]
                simultaneous_texts = len(text_batch)
            
            # get speaker_ids (arflow)
            arflow_speaker_ids = [self.ttm_sp_name_lookup[speaker] for speaker in batch_speaker_names]
            arflow_speaker_ids = torch.LongTensor(arflow_speaker_ids).cuda().repeat_interleave(batch_size_per_text)
            
            # get style input
            try:
                tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences(text_batch) # input array [B] e.g: ["Test?","2nd Sentence!"]
            except:
                raise Exception(f"TorchMoji failed to tokenize text:\n{text_batch}")
            try:
                embedding = self.tm_torchmoji(tokenized) # returns np array [B, Embed]
            except Exception as ex:
                print(f'Exception: {ex}')
                print(f"TorchMoji failed to process text:\n{text_batch}")
            style_input = torch.from_numpy(embedding).cuda().repeat_interleave(batch_size_per_text, dim=0)
            style_input = style_input.to(next(self.arflow.parameters()).dtype)
            
            if style_input.size(0) < (simultaneous_texts*batch_size_per_text):
                diff = -(-(simultaneous_texts*batch_size_per_text) // style_input.size(0))
                style_input = style_input.repeat(diff, 1)[:simultaneous_texts*batch_size_per_text]
            
            # check punctuation and add '.' if missing
            valid_last_char = '-,.?!;:' # valid final characters in texts
            text_batch = [text+'.' if (text[-1] not in valid_last_char) else text for text in text_batch]
            
            # parse text
            gtext_batch = text_batch
            if use_arpabet: # convert texts to ARPAbet (phonetic) versions.
                text_batch = [self.ARPA(text) for text in text_batch]
            
            # convert texts to number representation, pad where appropriate and move to GPU
            sequence_split = [torch.LongTensor(text_to_sequence(text, self.ttm_hparams.text_cleaners)) for text in text_batch] # convert texts to numpy representation
            text_lengths = torch.tensor([seq.size(0) for seq in sequence_split])
            max_len = text_lengths.max().item()
            sequence = torch.zeros(text_lengths.size(0), max_len).long() # create large tensor to move each text input into
            for i in range(text_lengths.size(0)): # move each text into padded input tensor
                sequence[i, :sequence_split[i].size(0)] = sequence_split[i]
            sequence = sequence.cuda().long().repeat_interleave(batch_size_per_text, dim=0) # move to GPU and repeat text
            text_lengths = text_lengths.cuda().long() # move to GPU
            
            # debug # Looks like pytorch 1.5 doesn't run contiguous on some operations the previous versions did.
            text_lengths = text_lengths.clone()
            sequence     = sequence.clone()
            
            if status_updates: mf_start=time.time(); print("Running arflow", end='')
            try:
                best_score = np.ones (simultaneous_texts) * -1e5
                tries      = np.zeros(simultaneous_texts)
                best_generations = [ 0]*simultaneous_texts
                best_score_str   = ['']*simultaneous_texts
                while np.amin(best_score) < target_score:
                    # run arflow
                    if status_updates: print("..", end='')
                    outputs = self.arflow.inference(sequence, text_lengths.repeat_interleave(batch_size_per_text, dim=0), arflow_speaker_ids, style_input, res_sigma=global_sigma, mel_z_sigma=frame_sigma, char_z_sigma=char_sigma)
                    batch_pred_mel = outputs['pred_mel']
                    
                    # metric for html side
                    n_passes   +=1# metric for html
                    total_specs+=batch_pred_mel.shape[0]
                    
                    # get alignment metrics for each item
                    output_lengths = outputs["mel_lengths"]
                    
                    # split batch into items
                    batch = list(zip(
                        output_lengths.split(1,dim=0),
                        batch_pred_mel.split(1,dim=0),))
                    
                    for j in range(simultaneous_texts): # process each set of text spectrograms seperately
                        start, end = (j*batch_size_per_text), ((j+1)*batch_size_per_text)
                        sametext_batch = batch[start:end] # seperate the full batch into pieces that use the same input text
                        assert len(sametext_batch) >= 1
                        
                        # process all items related to the j'th text input
                        for k, (output_length, pred_mel) in enumerate(sametext_batch):
                            # factors that make up score
                            weighted_score = 0.75
                            
                            score_str = (f"[{weighted_score:.3f}weighted_score]")
                            if torch.isnan(pred_mel).any() or weighted_score == float('nan'):
                                weighted_score = 1e-7
                            if weighted_score > best_score[j]:
                                best_score[j]       = weighted_score
                                best_score_str[j]   = score_str
                                best_generations[j] = [pred_mel, output_length]
                            tries[j]+=1
                            if np.amin(tries) >= max_attempts and np.amin(best_score) > absolutely_required_score:
                                raise StopIteration
                            if np.amin(tries) >= absolute_maximum_tries:
                                print(f"Absolutely required score not achieved in {absolute_maximum_tries} attempts - ", end='')
                                raise StopIteration
                    
                    #if np.amin(tries) < (max_attempts-1):
                    #    print(f'Minimum score of {np.amin(best_score)} is less than Target score of {target_score}. Retrying.')
                    #elif np.amin(best_score) < absolutely_required_score:
                    #    print(f"Minimum score of {np.amin(best_score)} is less than 'Absolutely Required score' of {absolutely_required_score}. Retrying.")
            except StopIteration:
                del batch
                if status_updates: print(f"\nDone in {time.time()-mf_start:.2f}s")
                pass
            
            assert not any([x==0 for x in best_generations]), 'arflow Failed to generate one of the texts after multiple attempts.'
            
            # logging
            all_best_scores.extend(best_score)
            
            # cleanup VRAM
            style_input = sequence = None
            
            # [[mel, melpost, gate, align], [mel, melpost, gate, align], [mel, melpost, gate, align]] -> [[mel, mel, mel], [melpost, melpost, melpost], [gate, gate, gate], [align, align, align]]
            batch_pred_mel = [x[0][0].T for x in best_generations]
            output_lengths            = torch.stack([x[1][0] for x in best_generations], dim=0)
            # pickup the best attempts from each input
            
            max_length = output_lengths.max()
            batch_pred_mel = torch.nn.utils.rnn.pad_sequence(batch_pred_mel, batch_first=True, padding_value=-11.52).transpose(1,2)[:,:,:max_length]
           #alignments_batch = torch.nn.utils.rnn.pad_sequence(alignments_batch, batch_first=True, padding_value=0)[:,:max_length,:]
            
            if status_updates:
                vo_start = time.time()
                print("Running Vocoder... ")
            
            self.vocoder_batch_size = 16
            
            # Run Vocoder
            vocoder_dtype = next(self.vocoder.parameters()).dtype
            audio_batch = []
            for i in range(0, len(batch_pred_mel), self.vocoder_batch_size):
                pred_mel_part_batch = batch_pred_mel[i:i+self.vocoder_batch_size]
                pred_mel_part_batch = pred_mel_part_batch + 1.2# increase volume cause vocoder is trained on peak normalized
                pred_mel_part_batch[pred_mel_part_batch < -11.+1.2] = -11.52
                audio_batch.extend( self.vocoder(pred_mel_part_batch.to(vocoder_dtype)).squeeze(1).cpu().split(1, dim=0) )# [b, T]
            # audio_batch shapes = [[1, T], [1, T], [1, T], [1, T], ...]
            
            if status_updates:
                print(f'Done in {time.time()-vo_start:.2f}s')
            
            if status_updates: sv_start=time.time(); print(f"Saving audio files to disk... ")
            
            # write audio files and any stats
            audio_bs = len(audio_batch)
            for j, audio in enumerate(audio_batch):
                # remove Vocoder padding
                audio_end = output_lengths[j] * self.MTW_conf['hop_size']
                audio     = audio[:, :audio_end]
                
                # remove arflow padding
                spec_end = output_lengths[j]
                
                # save audio
                filename = f"{filename_prefix}_{counter//300:04}_{counter:06}.wav"
                save_path = os.path.join(self.conf['working_directory'], filename)
                
                # add silence to clips (ignore last clip)
                if cat_silence_s:
                    cat_silence_samples = int(cat_silence_s*self.MTW_conf['sampling_rate'])
                    audio = F.pad(audio, (0, cat_silence_samples))
                
                # scale audio for int16 output
                audio = (audio.float() * 2**15).squeeze().numpy().astype('int16')
                
                # remove if already exists
                if os.path.exists(save_path):
                    print(f"File already found at [{save_path}], overwriting.")
                    os.remove(save_path)
                
                write(save_path, self.MTW_conf['sampling_rate'], audio)
                
                counter   += 1
                audio_len += audio_end.item()
                
                # ------ merge clips of 300 ------ #
                last_item = (j == audio_bs-1)
                if (counter % 300) == 0 or (last_text and last_item): # if 300th file or last item of last batch.
                    i = (counter-1)//300
                    # merge batch of 300 files together
                    print(f"Merging audio files {i*300} to {((i+1)*300)-1}... ", end='')
                    fpath = os.path.join(self.conf['working_directory'], f"{filename_prefix}_concat_{i:04}.wav")
                    files_to_merge = os.path.join(self.conf["working_directory"], f"{filename_prefix}_{i:04}_*.wav")
                    os.system(f'sox "{files_to_merge}" -b 16 "{fpath}"')
                    assert os.path.exists(fpath), f"'{fpath}' failed to generate."
                    del files_to_merge
                    
                    # delete the original 300 files
                    print("Cleaning up remaining temp files... ", end="")
                    tmp_files = [fp for fp in glob(os.path.join(self.conf['working_directory'], f"{filename_prefix}_{i:04}_*.wav")) if "output" not in fp]
                    _ = [os.remove(fp) for fp in tmp_files]
                    print("Done")
                    
                    # add merged file to final output(s)
                    fsize = os.stat(fpath).st_size
                    running_fsize += fsize
                    fpaths += [fpath,]
                    if ( running_fsize/(1024**3) > self.conf['output_maxsize_gb'] ) or (len(fpaths) > 300) or (last_text and last_item): # if (total size of fpaths is > 2GB) or (more than 300 inputs) or (last item of last batch): save as output
                        fpath_str = '"'+'" "'.join(fpaths)+'"' # chain together fpaths in string for SoX input
                        output_extension = self.conf['sox_output_ext']
                        if output_extension[0] != '.':
                            output_extension = f".{output_extension}"
                        out_name = f"{output_filename}-{out_count:02}{output_extension}"
                        out_path = os.path.join(self.conf['output_directory'], out_name)
                        os.system(f'sox {fpath_str} -b 16 "{out_path}"') # merge the merged files into final outputs. bit depth of 16 useful to stay in the 32bit duration limit
                        
                        if running_fsize >= (os.stat(out_path).st_size - 1024): # if output seems to have correctly generated.
                            print("Cleaning up merged temp files... ", end="") # delete the temp files and keep the output
                            _ = [os.remove(fp) for fp in fpaths]
                            print("Done")
                        
                        running_fsize = 0
                        out_count+=1
                        fpaths = []
                # ------ // merge clips of 300 // ------ #
                #end of writing loop
            
            if status_updates: print(f"Done in {time.time()-sv_start:.2f}s")
            
            show_inference_alignment_scores = True# Needs to be moved, workers cannot print to terminal. #bool(self.conf['terminal']['show_inference_alignment_scores'])
            for k, score in enumerate(best_score):
                scores+=[score,]
                if show_inference_alignment_scores:
                    print(f'Input_Str {k:02}: "{gtext_batch[k]}"\n'
                          f'Score_Str {k:02}: {best_score_str[k]}\n')
            
            if True:#self.conf['terminal']['show_inference_progress']:
                time_elapsed = time.time()-show_inference_progress_start
                time_per_clip = time_elapsed/(text_index+1)
                remaining_files = (total_len-(text_index+1))
                eta_finish = (remaining_files*time_per_clip)/60
                print(f"{text_index}/{total_len}, {eta_finish:.2f}mins remaining.")
                del time_per_clip, eta_finish, remaining_files, time_elapsed
            
            audio_seconds_generated = round(audio_len/self.MTW_conf['sampling_rate'],3)
            time_to_gen = round(time.time()-start_time,3)
            if show_time_to_gen:
                print(f"Generated {audio_seconds_generated}s of audio in {time_to_gen}s wall time - so far. (best of {tries.sum().astype('int')} tries this pass) ({audio_seconds_generated/time_to_gen:.2f}xRT) ({sum([x<0.6 for x in all_best_scores])/len(all_best_scores):.1%}Failure Rate)")
            
            print("\n") # seperate each pass
        
        scores    = np.stack(scores)
        avg_score = np.mean (scores)
        
        fail_rate = sum([x<0.6 for x in all_best_scores])/len(all_best_scores)
        rtf = audio_seconds_generated/time_to_gen# seconds of audio generated per second in real life
        
        out = {
                  "out_name": out_name,
               "time_to_gen": time_to_gen,
   "audio_seconds_generated": audio_seconds_generated,
               "total_specs": total_specs,
                  "n_passes": n_passes,
                 "avg_score": avg_score,
                       "rtf": rtf,
                 "fail_rate": fail_rate,
        }
        return out


def start_worker(config, device, request_queue, finished_queue):
    """
    Start a Text-2-Speech Worker.
    This process will check request_queue for requests and complete them till the host process is killed or something crashes.
    """
    t2s = T2S(conf['workers']) # initialize Text-2-Speech module. Loads models
    
    while queue_is_empty:
        time.sleep(0.1)# wait 100ms then check queue again...
