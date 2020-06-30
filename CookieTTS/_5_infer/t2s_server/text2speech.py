import sys
import os
import numpy as np
import random
import time
import argparse
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import json
import re
import difflib
from glob import glob
from unidecode import unidecode
import nltk # sentence spliting
from nltk import sent_tokenize

sys.path.append('../')
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from utils import load_filepaths_and_text

def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).long()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.int64)
    mask = (ids < lengths.unsqueeze(1))
    return mask

#@torch.jit.script # should work and be even faster, but makes it harder to debug and it's already fast enough right now
def alignment_metric(alignments, input_lengths=None, output_lengths=None, average_across_batch=False):
    alignments = alignments.transpose(1,2) # [B, dec, enc] -> [B, enc, dec]
    # alignments [batch size, x, y]
    # input_lengths [batch size] for len_x
    # output_lengths [batch size] for len_y
    if input_lengths == None:
        input_lengths =  torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[1]-1) # [B] # 147
    if output_lengths == None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[2]-1) # [B] # 767
    batch_size = alignments.size(0)
    optimums = torch.sqrt(input_lengths.double().pow(2) + output_lengths.double().pow(2)).view(batch_size)
    
    # [B, enc, dec] -> [B, dec], [B, dec]
    values, cur_idxs = torch.max(alignments, 1) # get max value in column and location of max value
    
    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:,0][:,None], cur_idxs[:,:-1]), dim=1) # shift entire tensor by one.
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5) # [B, dec]
    dist.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=dist.size(1)), 0.0) # set dist of padded to zero
    dist = dist.sum(dim=(1)) # get total dist for each B
    diagonalitys = (dist + 1.4142135)/optimums # dist / optimal dist
    
    alignments.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=alignments.size(2))[:,None,:], 0.0)
    attm_enc_total = torch.sum(alignments, dim=2)# [B, enc, dec] -> [B, enc]
    
    # calc max (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 0.0)
    encoder_max_focus = attm_enc_total.max(dim=1)[0] # [B, enc] -> [B]
    
    # calc mean (with padding ignored)
    encoder_avg_focus = attm_enc_total.mean(dim=1)   # [B, enc] -> [B]
    encoder_avg_focus *= (attm_enc_total.size(1)/input_lengths.float())
    
    # calc min (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask_from_lengths(input_lengths, max_len=attm_enc_total.size(1)), 1.0)
    encoder_min_focus = attm_enc_total.min(dim=1)[0] # [B, enc] -> [B]
    
    # calc average max attention (with padding ignored)
    values.masked_fill_(~get_mask_from_lengths(output_lengths, max_len=values.size(1)), 0.0) # because padding
    avg_prob = values.mean(dim=1)
    avg_prob *= (alignments.size(2)/output_lengths.float()) # because padding
    
    if average_across_batch:
        diagonalitys = diagonalitys.mean()
        encoder_max_focus = encoder_max_focus.mean()
        encoder_min_focus = encoder_min_focus.mean()
        encoder_avg_focus = encoder_avg_focus.mean()
        avg_prob = avg_prob.mean()
    return diagonalitys.cpu(), avg_prob.cpu(), encoder_max_focus.cpu(), encoder_min_focus.cpu(), encoder_avg_focus.cpu()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # list(chunks([0,1,2,3,4,5,6,7,8,9],2)) -> [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# generator for text splitting.
def parse_text_into_segments(texts, split_at_quotes=True, target_segment_length=200):
    """Swap speaker at every quote mark. Each split segment will have quotes around it (for information later on rather than accuracy to the original text)."""
    
    # split text by quotes
    quo ='"' # nested quotes in list comprehension are hard to work with
    texts = [f'"{text.replace(quo,"").strip()}"' if i%2 else text.replace(quo,"").strip() for i, text in enumerate(unidecode(texts).split('"'))]
    
    # clean up and remove empty texts
    def clean_text(text):
        text = text.strip()
        text = text.replace("\n"," ").replace("  "," ").replace("> --------------------------------------------------------------------------","").replace("------------------------------------","")
        return text
    texts = [clean_text(text) for text in texts if len(text.strip().replace('"','').strip()) or len(clean_text(text))]
    assert len(texts)
    
    # split text by sentences and add commas in where needed.
    def quotify(seg, text):
        if '"' in text:
            if seg[0] != '"': seg='"'+seg
            if seg[-1] != '"': seg+='"'
        return seg
    texts_tmp = []
    texts = [texts_tmp.extend([quotify(x.strip(), text) for x in sent_tokenize(text) if len(x.replace('"','').strip())]) for text in texts]
    texts = texts_tmp
    del texts_tmp
    assert len(texts)
    
    # merge neighbouring sentences
    quote_mode = False
    texts_output = []
    texts_segmented = ''
    texts_len = len(texts)
    for i, text in enumerate(texts):
        # split segment if quote swap
        if split_at_quotes and ('"' in text and quote_mode == False) or (not '"' in text and quote_mode == True):
            texts_segmented.replace('""','')
            texts_output.append(texts_segmented)
            texts_segmented=text
            quote_mode = not quote_mode
        
        # split segment if max length
        elif len(texts_segmented+text) > target_segment_length:
            texts_segmented.replace('""','')
            texts_output.append(texts_segmented)
            texts_segmented=text
        
        else: # continue adding to segment
            texts_segmented+= f' {text}'
    # add any remaining stuff.
    if len(texts_segmented):
        texts_output.append(texts_segmented)
    assert len(texts_output)
    
    return texts_output


def get_first_over_thresh(x, threshold):
    """Takes [B, T] and outputs first T over threshold for each B (output.shape = [B])."""
    device = x.device
    x = x.clone().cpu().float() # GPU implementation of argmax() splits tensor into 32 elem chunks, each chunk is parsed forward then the outputs are collected together... backwards
    x[:,-1] = threshold # set last to threshold just incase the output didn't finish generating.
    x[x>threshold] = threshold
    return ( (x.size(1)-1)-(x.flip(dims=(1,)).argmax(dim=1)) ).to(device).int()


class T2S:
    def __init__(self, conf):
        self.conf = conf
        
        # load Tacotron2
        self.ttm_current = self.conf['TTM']['default_model']
        assert self.ttm_current in self.conf['TTM']['models'].keys(), "Tacotron default model not found in config models"
        tacotron_path = self.conf['TTM']['models'][self.ttm_current]['modelpath'] # get first available Tacotron
        self.tacotron, self.ttm_hparams, self.ttm_sp_name_lookup, self.ttm_sp_id_lookup = self.load_tacotron2(tacotron_path)
        
        # load WaveGlow
        self.MTW_current = self.conf['MTW']['default_model']
        assert self.MTW_current in self.conf['MTW']['models'].keys(), "WaveGlow default model not found in config models"
        vocoder_path = self.conf['MTW']['models'][self.MTW_current]['modelpath'] # get first available waveglow
        vocoder_confpath = self.conf['MTW']['models'][self.MTW_current]['configpath']
        self.waveglow, self.MTW_denoiser, self.MTW_train_sigma, self.MTW_sp_id_lookup = self.load_waveglow(vocoder_path, vocoder_confpath)
        
        # load torchMoji
        if self.ttm_hparams.torchMoji_linear: # if Tacotron includes a torchMoji layer
            self.tm_sentence_tokenizer, self.tm_torchmoji = self.load_torchmoji()
        
        # override since my checkpoints are still missing speaker names
        if self.conf['TTM']['use_speaker_ids_file_override']:
            speaker_ids_fpath = self.conf['TTM']['speaker_ids_file']
            self.ttm_sp_name_lookup = {name: self.ttm_sp_id_lookup[int(ext_id)] for _, name, ext_id in load_filepaths_and_text(speaker_ids_fpath)}
        
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
        from torchmoji.sentence_tokenizer import SentenceTokenizer
        from torchmoji.model_def import torchmoji_feature_encoding
        from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
        
        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        
        maxlen = 130
        texts = ["Testing!",]
        
        with torch.no_grad():
            # init model
            st = SentenceTokenizer(vocabulary, maxlen, ignore_sentences_with_only_custom=True)
            torchmoji = torchmoji_feature_encoding(PRETRAINED_PATH)
        return st, torchmoji
    
    
    def get_torchmoji_hidden(self, texts):
        with torch.no_grad():
            tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences(texts) # input array [B] e.g: ["Test?","2nd Sentence!"]
            embedding = self.tm_torchmoji(tokenized) # returns np array [B, Embed]
        return embedding
    
    
    def is_ax(self, config):
        """Quickly check if a model uses the Ax WaveGlow core by what's available in the config file."""
        return True if 'upsample_first' in config.keys() else False
    
    
    def load_waveglow(self, vocoder_path, config_fpath):
        # Load config file
        with open(config_fpath) as f:
            data = f.read()
        config = json.loads(data)
        train_config = config["train_config"]
        data_config = config["data_config"]
        dist_config = config["dist_config"]
        vocoder_config = {
            **config["waveglow_config"], 
            'win_length': data_config['win_length'],
            'hop_length': data_config['hop_length']
        }
        print(vocoder_config)
        print(f"Config File from '{config_fpath}' successfully loaded.")
        
        # import the correct model core
        if self.is_ax(vocoder_config):
            from efficient_model_ax import WaveGlow
        else:
            if vocoder_config["yoyo"]:
                from efficient_model import WaveGlow
            else:
                from glow import WaveGlow
        
        # initialize model
        print(f"intializing WaveGlow model... ", end="")
        waveglow = WaveGlow(**vocoder_config).cuda()
        print(f"Done!")
        
        # load checkpoint from file
        print(f"loading WaveGlow checkpoint... ", end="")
        checkpoint = torch.load(vocoder_path)
        waveglow.load_state_dict(checkpoint['model']) # and overwrite initialized weights with checkpointed weights
        waveglow.cuda().eval().half() # move to GPU and convert to half precision
        print(f"Done!")
        
        print(f"initializing Denoiser... ", end="")
        denoiser = Denoiser(waveglow)
        print(f"Done!")
        vocoder_iters = checkpoint['iteration']
        print(f"WaveGlow trained for {vocoder_iters} iterations")
        speaker_lookup = checkpoint['speaker_lookup'] # ids lookup
        training_sigma = train_config['sigma']
        
        return waveglow, denoiser, training_sigma, speaker_lookup
    
    
    def update_wg(self, vocoder_name):
        self.waveglow, self.MTW_denoiser, self.MTW_train_sigma, self.MTW_sp_id_lookup = self.load_waveglow(self.conf['MTW']['models'][vocoder_name]['modelpath'], self.conf['MTW']['models'][vocoder_name]['configpath'])
        self.MTW_current = vocoder_name
    
    def load_tacotron2(self, tacotron_path):
        """Loads tacotron2,
        Returns:
        - model
        - hparams
        - speaker_lookup
        """
        checkpoint = torch.load(tacotron_path) # load file into memory
        print("Loading Tacotron... ", end="")
        checkpoint_hparams = checkpoint['hparams'] # get hparams
        checkpoint_dict = checkpoint['state_dict'] # get state_dict
        
        model = load_model(checkpoint_hparams) # initialize the model
        model.load_state_dict(checkpoint_dict) # load pretrained weights
        _ = model.cuda().eval().half()
        print("Done")
        tacotron_speaker_name_lookup = checkpoint['speaker_name_lookup'] # save speaker name lookup
        tacotron_speaker_id_lookup = checkpoint['speaker_id_lookup'] # save speaker_id lookup
        print(f"This Tacotron model has been trained for {checkpoint['iteration']} Iterations.")
        return model, checkpoint_hparams, tacotron_speaker_name_lookup, tacotron_speaker_id_lookup
    
    
    def update_tt(self, tacotron_name):
        self.model, self.ttm_hparams, self.ttm_sp_name_lookup, self.ttm_sp_id_lookup = self.load_tacotron2(self.conf['TTM']['models'][tacotron_name]['modelpath'])
        self.ttm_current = tacotron_name
        
        if self.conf['TTM']['use_speaker_ids_file_override']:# (optional) override
            self.ttm_sp_name_lookup = {name: self.ttm_sp_id_lookup[int(ext_id)] for _, name, ext_id in load_filepaths_and_text(self.conf['TTM']['speaker_ids_file'])}
    
    
    def get_MTW_sp_id_from_ttm_sp_names(self, names):
        """Get WaveGlow speaker ids from Tacotron2 named speaker lookup. (This should function should be removed once WaveGlow has named speaker support)."""
        ttm_model_ids = [self.ttm_sp_name_lookup[name] for name in names]
        reversed_lookup = {v: k for k, v in self.ttm_sp_id_lookup.items()}
        ttm_ext_ids = [reversed_lookup[int(speaker_id)] for speaker_id in ttm_model_ids]
        wv_model_ids = [self.MTW_sp_id_lookup[int(speaker_id)] for speaker_id in ttm_ext_ids]
        return wv_model_ids
    
    
    def get_closest_names(self, names):
        possible_names = list(self.ttm_sp_name_lookup.keys())
        validated_names = [difflib.get_close_matches(name, possible_names, n=2, cutoff=0.01)[0] for name in names] # change all names in input to the closest valid name
        return validated_names
    
    
    def infer(self, text, speaker_names, style_mode, textseg_mode, batch_mode, max_attempts, max_duration_s, batch_size, dyna_max_duration_s, use_arpabet, target_score, speaker_mode, cat_silence_s, textseg_len_target, gate_delay=4, gate_threshold=0.6, filename_prefix=None, status_updates=False, show_time_to_gen=True, end_mode='thresh', absolute_maximum_tries=4096, absolutely_required_score=-1e3):
        """
        PARAMS:
        ...
        gate_delay
            default: 4
            options: int ( 0 -> inf )
            info: a modifier for when spectrograms are cut off.
                  This would allow you to add silence to the end of a clip without an unnatural fade-out.
                  8 will give 0.1 seconds of delay before ending the clip.
                  If this param is set too high then the model will try to start speaking again
                  despite not having any text left to speak, therefore keeping it low is typical.
        gate_threshold
            default: 0.6
            options: float ( 0.0 -> 1.0 )
            info: used to control when Tacotron2 will stop generating new mel frames.
                  This will effect speed of generation as the model will generate
                  extra frames till it hits the threshold. This may be preferred if
                  you believe the model is stopping generation too early.
                  When end_mode == 'thresh', this param will also be used to decide
                  when the audio from the best spectrograms should be cut off.
        ...
        end_mode
            default: 'thresh'
            options: ['max','thresh']
            info: controls where the spectrograms are cut off.
                  'max' will cut the spectrograms off at the highest gate output, 
                  'thresh' will cut off spectrograms at the first gate output over gate_threshold.
        """
        assert end_mode in ['max','thresh'], f"end_mode of {end_mode} is not valid."
        assert gate_delay > -10, "gate_delay is negative."
        assert gate_threshold > 0.0, "gate_threshold less than 0.0"
        assert gate_threshold <= 1.0, "gate_threshold greater than 1.0"
        os.makedirs(self.conf["working_directory"], exist_ok=True)
        os.makedirs(self.conf["output_directory"], exist_ok=True)
        
        with torch.no_grad():
            # time to gen
            audio_len = 0
            start_time = time.time()
            
            # Score Metric
            scores = []
            
            # Score Parameters
            diagonality_weighting = 0.5 # 'pacing factor', a penalty for clips where the model pace changes often/rapidly. # this thing does NOT work well for Rarity.
            max_focus_weighting = 1.0   # 'stuck factor', a penalty for clips that spend execisve time on the same letter.
            min_focus_weighting = 1.0   # 'miniskip factor', a penalty for skipping/ignoring single letters in the input text.
            avg_focus_weighting = 1.0   # 'skip factor', a penalty for skipping very large parts of the input text
            
            # add a filename prefix to keep multiple requests seperate
            if not filename_prefix:
                filename_prefix = str(time.time())
            
            # add output filename
            output_filename = f"{filename_prefix}_output"
            
            # split the text into chunks (if applicable)
            if textseg_mode == 'no_segmentation':
                texts = [text,]
            elif textseg_mode == 'segment_by_line':
                texts = text.split("\n")
            elif textseg_mode == 'segment_by_sentence':
                texts = parse_text_into_segments(text, split_at_quotes=False, target_segment_length=textseg_len_target)
            elif textseg_mode == 'segment_by_sentencequote':
                texts = parse_text_into_segments(text, split_at_quotes=True, target_segment_length=textseg_len_target)
            else:
                raise NotImplementedError(f"textseg_mode of {textseg_mode} is invalid.")
            del text
            
            # cleanup for empty inputs.
            texts = [x.strip() for x in texts if len(x.strip())]
            
            total_len = len(texts)
            
            # update Tacotron stopping params
            frames_per_second = float(self.ttm_hparams.sampling_rate/self.ttm_hparams.hop_length)
            self.tacotron.decoder.gate_delay = int(gate_delay)
            self.tacotron.decoder.max_decoder_steps = int(min(max([len(t) for t in texts]) * float(dyna_max_duration_s)*frames_per_second, float(max_duration_s)*frames_per_second))
            self.tacotron.decoder.gate_threshold = float(gate_threshold)
            
            # find closest valid name(s)
            speaker_names = self.get_closest_names(speaker_names)
            
            # pick how the batch will be handled
            batch_size = int(batch_size)
            if batch_mode == "scaleup":
                simultaneous_texts = total_len
                batch_size_per_text = batch_size
            elif batch_mode == "nochange":
                simultaneous_texts = max(batch_size//max_attempts, 1)
                batch_size_per_text = min(batch_size, max_attempts)
            elif batch_mode == "scaledown":
                simultaneous_texts = total_len
                batch_size_per_text = -(-batch_size//total_len)
            else:
                raise NotImplementedError(f"batch_mode of {batch_mode} is invalid.")
            
            # for size merging
            running_fsize = 0
            fpaths = []
            out_count = 0
            
            # keeping track of stats for html/terminal
            show_inference_progress_start = time.time()
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
                
                self.tacotron.decoder.max_decoder_steps = int(min(max([len(t) for t in text_batch]) * float(dyna_max_duration_s)*frames_per_second, float(max_duration_s)*frames_per_second))
                
                if speaker_mode == "not_interleaved": # non-interleaved
                    batch_speaker_names = speaker_names * -(-simultaneous_texts//len(speaker_names))
                    batch_speaker_names = batch_speaker_names[:simultaneous_texts]
                elif speaker_mode == "interleaved": # interleaved
                    repeats = -(-simultaneous_texts//len(speaker_names))
                    batch_speaker_names = [i for i in speaker_names for _ in range(repeats)][:simultaneous_texts]
                elif speaker_mode == "random": # random
                    batch_speaker_names = [random.choice(speaker_names),] * simultaneous_texts
                elif speaker_mode == "cycle_next": # use next speaker for each text input
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
                
                # get speaker_ids (tacotron)
                tacotron_speaker_ids = [self.ttm_sp_name_lookup[speaker] for speaker in batch_speaker_names]
                tacotron_speaker_ids = torch.LongTensor(tacotron_speaker_ids).cuda().repeat_interleave(batch_size_per_text)
                
                # get speaker_ids (waveglow)
                vocoder_speaker_ids = self.get_MTW_sp_id_from_ttm_sp_names(batch_speaker_names)
                vocoder_speaker_ids = [self.MTW_sp_id_lookup[int(speaker_id)] for speaker_id in vocoder_speaker_ids]
                vocoder_speaker_ids = torch.LongTensor(vocoder_speaker_ids).cuda()
                
                # get style input
                if style_mode == 'mel':
                    mel = load_mel(audio_path.replace(".npy",".wav")).cuda().half()
                    style_input = mel
                elif style_mode == 'token':
                    pass
                    #style_input =
                elif style_mode == 'zeros':
                    style_input = None
                elif style_mode == 'torchmoji_hidden':
                    try:
                        tokenized, _, _ = self.tm_sentence_tokenizer.tokenize_sentences(text_batch) # input array [B] e.g: ["Test?","2nd Sentence!"]
                    except:
                        raise Exception(f"TorchMoji failed to tokenize text:\n{text_batch}")
                    try:
                        embedding = self.tm_torchmoji(tokenized) # returns np array [B, Embed]
                    except Exception as ex:
                        print(f'Exception: {ex}')
                        print(f"TorchMoji failed to process text:\n{text_batch}")
                        #raise Exception(f"text\n{text}\nfailed to process.")
                    style_input = torch.from_numpy(embedding).cuda().half().repeat_interleave(batch_size_per_text, dim=0)
                elif style_mode == 'torchmoji_string':
                    style_input = text_batch
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                
                if style_input.size(0) < (simultaneous_texts*batch_size_per_text):
                    diff = -(-(simultaneous_texts*batch_size_per_text) // style_input.size(0))
                    style_input = style_input.repeat(diff, 1)[:simultaneous_texts*batch_size_per_text]
                
                # check punctuation and add '.' if missing
                valid_last_char = '-,.?!;:' # valid final characters in texts
                text_batch = [text+'.' if (text[-1] not in valid_last_char) else text for text in text_batch]
                
                # parse text
                text_batch = [unidecode(text.replace("...",". ").replace("  "," ").strip()) for text in text_batch] # remove eclipses, double spaces, unicode and spaces before/after the text.
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
                sequence = sequence.clone()
                
                print("sequence.shape[0] =",sequence.shape[0]) # debug
                
                try:
                    best_score = np.ones(simultaneous_texts) * -9e9
                    tries      = np.zeros(simultaneous_texts)
                    best_generations = [0]*simultaneous_texts
                    best_score_str = ['']*simultaneous_texts
                    while np.amin(best_score) < target_score:
                        # run Tacotron
                        if status_updates: print("Running Tacotron2... ", end='')
                        mel_batch_outputs, mel_batch_outputs_postnet, gate_batch_outputs, alignments_batch = self.tacotron.inference(sequence, tacotron_speaker_ids, style_input=style_input, style_mode=style_mode, text_lengths=text_lengths.repeat_interleave(batch_size_per_text, dim=0))
                        
                        # metric for html side
                        n_passes+=1 # metric for html
                        total_specs+=mel_batch_outputs.shape[0]
                        
                        # get metrics for each item
                        if end_mode == 'thresh':
                            output_lengths = get_first_over_thresh(gate_batch_outputs, gate_threshold)
                        elif end_mode == 'max':
                            output_lengths = gate_batch_outputs.argmax(dim=1)
                        diagonality_batch, avg_prob_batch, enc_max_focus_batch, enc_min_focus_batch, enc_avg_focus_batch = alignment_metric(alignments_batch, input_lengths=text_lengths.repeat_interleave(batch_size_per_text, dim=0), output_lengths=output_lengths)
                        
                        # split batch into items
                        batch = list(zip(
                            mel_batch_outputs.split(1,dim=0),
                            mel_batch_outputs_postnet.split(1,dim=0),
                            gate_batch_outputs.split(1,dim=0),
                            alignments_batch.split(1,dim=0),
                            diagonality_batch,
                            avg_prob_batch,
                            enc_max_focus_batch,
                            enc_min_focus_batch,
                            enc_avg_focus_batch,))
                        
                        for j in range(simultaneous_texts): # process each set of text spectrograms seperately
                            start, end = (j*batch_size_per_text), ((j+1)*batch_size_per_text)
                            sametext_batch = batch[start:end] # seperate the full batch into pieces that use the same input text
                            
                            # process all items related to the j'th text input
                            for k, (mel_outputs, mel_outputs_postnet, gate_outputs, alignments, diagonality, avg_prob, enc_max_focus, enc_min_focus, enc_avg_focus) in enumerate(sametext_batch):
                                # factors that make up score
                                weighted_score =  avg_prob.item() # general alignment quality
                                diagonality_punishment = (max(diagonality.item(),1.20)-1.20) * 0.5 * diagonality_weighting  # speaking each letter at a similar pace.
                                max_focus_punishment = max((enc_max_focus.item()-40), 0) * 0.005 * max_focus_weighting # getting stuck on same letter for 0.6s
                                min_focus_punishment = max(0.25-enc_min_focus.item(),0) * min_focus_weighting # skipping single enc outputs
                                avg_focus_punishment = max(2.5-enc_avg_focus.item(), 0) * avg_focus_weighting # skipping most enc outputs
                                
                                weighted_score -= (diagonality_punishment + max_focus_punishment + min_focus_punishment + avg_focus_punishment)
                                score_str = f"{round(diagonality.item(),3)} {round(avg_prob.item()*100,2)}% {round(weighted_score,4)} {round(max_focus_punishment,2)} {round(min_focus_punishment,2)} {round(avg_focus_punishment,2)}|"
                                if weighted_score > best_score[j]:
                                    best_score[j] = weighted_score
                                    best_score_str[j] = score_str
                                    best_generations[j] = [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
                                tries[j]+=1
                                if np.amin(tries) >= max_attempts and np.amin(best_score) > (absolutely_required_score-1):
                                    raise StopIteration
                                if np.amin(tries) >= absolute_maximum_tries:
                                    print(f"Absolutely required score not achieved in {absolute_maximum_tries} attempts - ", end='')
                                    raise StopIteration
                        
                        if np.amin(tries) < (max_attempts-1):
                            print(f'Minimum score of {np.amin(best_score)} is less than Target score of {target_score}. Retrying.')
                        elif np.amin(best_score) < absolutely_required_score:
                            print(f"Minimum score of {np.amin(best_score)} is less than 'Absolutely Required score' of {absolutely_required_score}. Retrying.")
                except StopIteration:
                    del batch
                    if status_updates: print("Done")
                    pass
                
                # cleanup VRAM
                style_input = sequence = None
                
                # [[mel, melpost, gate, align], [mel, melpost, gate, align], [mel, melpost, gate, align]] -> [[mel, mel, mel], [melpost, melpost, melpost], [gate, gate, gate], [align, align, align]]
                mel_batch_outputs, mel_batch_outputs_postnet, gate_batch_outputs, alignments_batch = [x[0][0].T for x in best_generations], [x[1][0].T for x in best_generations], [x[2][0] for x in best_generations], [x[3][0] for x in best_generations]
                # pickup the best attempts from each input
                
                # stack best output arrays into tensors for WaveGlow
                gate_batch_outputs = torch.nn.utils.rnn.pad_sequence(gate_batch_outputs, batch_first=True, padding_value=0.0)
                
                # get duration(s)
                if end_mode == 'thresh':
                    max_lengths = get_first_over_thresh(gate_batch_outputs, gate_threshold)+gate_delay
                elif end_mode == 'max':
                    max_lengths = gate_batch_outputs.argmax(dim=1)+gate_delay
                max_length = torch.max(max_lengths)
                
                mel_batch_outputs = torch.nn.utils.rnn.pad_sequence(mel_batch_outputs, batch_first=True, padding_value=-11.6).transpose(1,2)[:,:,:max_length]
                mel_batch_outputs_postnet = torch.nn.utils.rnn.pad_sequence(mel_batch_outputs_postnet, batch_first=True, padding_value=-11.6).transpose(1,2)[:,:,:max_length]
                alignments_batch = torch.nn.utils.rnn.pad_sequence(alignments_batch, batch_first=True, padding_value=0)[:,:max_length,:]
                
                if status_updates:
                    print("Running WaveGlow... ", end='')
                # Run WaveGlow
                audio_batch = self.waveglow.infer(mel_batch_outputs_postnet, speaker_ids=vocoder_speaker_ids, sigma=self.MTW_train_sigma*0.95)
                audio_denoised_batch = self.MTW_denoiser(audio_batch, strength=0.0001).squeeze(1)
                print("audio_denoised_batch.shape =", audio_denoised_batch.shape) # debug
                if status_updates:
                    print('Done')
                
                
                # write audio files and any stats
                audio_bs = audio_batch.size(0)
                for j, (audio, audio_denoised) in enumerate(zip(audio_batch.split(1, dim=0), audio_denoised_batch.split(1, dim=0))):
                    # remove WaveGlow padding
                    audio_end = max_lengths[j] * self.ttm_hparams.hop_length
                    audio = audio[:,:audio_end]
                    audio_denoised = audio_denoised[:,:audio_end]
                    
                    # remove Tacotron2 padding
                    spec_end = max_lengths[j]
                    mel_outputs = mel_batch_outputs.split(1, dim=0)[j][:,:,:spec_end]
                    mel_outputs_postnet = mel_batch_outputs_postnet.split(1, dim=0)[j][:,:,:spec_end]
                    alignments = alignments_batch.split(1, dim=0)[j][:,:spec_end,:text_lengths[j]]
                    
                    # save audio
                    filename = f"{filename_prefix}_{counter//300:04}_{counter:06}.wav"
                    save_path = os.path.join(self.conf['working_directory'], filename)
                    
                    # add silence to clips (ignore last clip)
                    if cat_silence_s:
                        cat_silence_samples = int(cat_silence_s*self.ttm_hparams.sampling_rate)
                        audio = torch.nn.functional.pad(audio, (0, cat_silence_samples))
                    
                    # scale audio for int16 output
                    audio = (audio * 2**15).squeeze().cpu().numpy().astype('int16')
                    
                    # remove if already exists
                    if os.path.exists(save_path):
                        print(f"File already found at [{save_path}], overwriting.")
                        os.remove(save_path)
                    
                    if status_updates: print(f"Saving clip to [{save_path}]... ", end="")
                    write(save_path, self.ttm_hparams.sampling_rate, audio)
                    if status_updates: print("Done")
                    
                    counter+=1
                    audio_len+=audio_end
                    
                    # ------ merge clips of 300 ------ #
                    last_item = (j == audio_bs-1)
                    if (counter % 300) == 0 or (last_text and last_item): # if 300th file or last item of last batch.
                        i = (counter- 1)//300
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
                            out_name = f"{output_filename}_{out_count:02}{output_extension}"
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
                
                if True:#self.conf['show_inference_alignment_scores']:
                    for k, bs in enumerate(best_score):
                        print(f"Input_Str  {k}: '{text_batch[k]}'")
                        print(f"Best_Score {k}: {bs:0.4f}")
                        print(f"Score_Str  {k}: {best_score_str[k]}\n")
                
                for score in best_score:
                    scores+=[score,]
                
                if True:#self.conf['show_inference_progress']:
                    time_elapsed = time.time()-show_inference_progress_start
                    time_per_clip = time_elapsed/(text_index+1)
                    remaining_files = (total_len-(text_index+1))
                    eta_finish = (remaining_files*time_per_clip)/60
                    print(f"{text_index}/{total_len}, {eta_finish:.2f}mins remaining.")
                    del time_per_clip, eta_finish, remaining_files, time_elapsed
                
                audio_seconds_generated = round(audio_len.item()/self.ttm_hparams.sampling_rate,3)
                time_to_gen = round(time.time()-start_time,3)
                if show_time_to_gen:
                    print(f"Generated {audio_seconds_generated}s of audio in {time_to_gen}s wall time - so far. (best of {tries.sum().astype('int')} tries this pass)")
                
                print("\n") # seperate each pass
        
        scores = np.stack(scores)
        avg_score = np.mean(scores)
        
        return out_name, time_to_gen, audio_seconds_generated, total_specs, n_passes, avg_score