# install MFA (OS agnostic)

# download links
win_url = "https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_win64.zip"
lin_url = "https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz"

# download Montreal Forced Aligner
import os
import urllib.request
import subprocess
from sys import platform
from .extract_unknown import extract
from time import sleep
prev_wd = os.getcwd()
try:
    os.chdir(os.path.split(__file__)[0])
except:
    pass

if not os.path.exists('montreal-forced-aligner'):
    print('"montreal-forced-aligner" not found. Downloading...')
    sleep(0.10)
    if platform == "linux" or platform == "linux2":
            dl_url = lin_url
    elif platform == "darwin":
        raise NotImplementedError('MacOS not supported.')
    elif platform == "win32":
            dl_url = win_url
    dlname = dl_url.split("/")[-1]
    if not os.path.exists(dlname):
        urllib.request.urlretrieve(dl_url, dlname)
    assert os.path.exists(dlname), 'failed to download.'
    extract(dlname)
    sleep(0.10)
    os.unlink(dlname)
    
    # hotfix for MFA v1.0.1
    # Alternate Fix: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/149#issuecomment-621165857
    if platform == "linux":
        os.link( os.path.join('montreal-forced-aligner','lib/libpython3.6m.so.1.0'), os.path.join('montreal-forced-aligner','lib/libpython3.6m.so') )
    
    print("Done!")


binary_folder = os.path.abspath(os.path.join('montreal-forced-aligner','bin'))
os.chdir(prev_wd)


def get(text, word_phones, punc="!?,.;:␤#-_'\"()[]\n"):
    """Convert block of text into ARPAbet."""
    word_phones = [x for x in word_phones if len(x[0])]
    out = []
    for word in text.split(" "):
        end_chars = ''; start_chars = ''
        while any(elem in word for elem in punc) and len(word) > 1:
            if word[-1] in punc:
                end_chars = word[-1] + end_chars
                word = word[:-1]
            elif word[0] in punc:
                start_chars = start_chars + word[0]
                word = word[1:]
            else:
                break
        try:
            word = "{" + ' '.join(word_phones[0][1]) + "}"
            word_phones = word_phones[1:]
        except IndexError:
            pass
        out.append((start_chars + (word or '') + end_chars).rstrip())
    return ' '.join(out)


def get_arpa(quote, words, phonemes):
    
    # matchup words and phonemes using the timing info
    words_tmp = words
    phonemes_tmp = [x for x in phonemes if x['text'] not in ('sil','sp')]
    word_phones = []
    word_phones.append([])
    loops=0
    while len(words_tmp) and len(phonemes_tmp):
        wstart = words_tmp[0]['start']
        wend = words_tmp[0]['end']
        wtext = words_tmp[0]['text']
        if phonemes_tmp[0]['start'] >= wstart and phonemes_tmp[0]['end'] <= wend: # if phone within timeframe of word.
            word_phones[-1].append(phonemes_tmp[0]['text'])
            phonemes_tmp = phonemes_tmp[1:]
        elif phonemes_tmp[0]['end'] > wend: # else, if phone past end of current word.
            word_phones.append([]) # move on to the next word.
            words_tmp = words_tmp[1:]
        loops+=1
        if loops > 999:
            break
    
    # replace words in quote with aligned phonemes.
    word_phones = list(zip([x['text'] for x in words], word_phones))
    arpa_quote = get(quote, word_phones)
    return arpa_quote


def load_TextGrid(path, quote=None):
    """
    Turn Montreal Forced Aligner TextGrid output into dict.
    PARAMS:
        path: file path that will be loaded
    
    RETURNS:
        dict: a dict of the info from the TextGrid file. Example of returned structure below.
          e.g:
            dict = {
                "clip_start": 0,
                "clip_start": length_of_clip_in_seconds,
                "words_start": start_time_of_the_first_word,
                "words_end": end_time_of_the_last_word,
                "phone_start": start_time_of_the_first_phoneme,
                "phone_end": end_time_of_the_last_phoneme,
                "words": [
                    {
                        "start": start_of_this_first_word,
                        "end":   end_of_this_first_word,
                        "text":  name_of_this_first_word,
                    }, {
                        "start": start_of_this_2nd_word,
                        "end":   end_of_this_2nd_word,
                        "text":  name_of_this_2nd_word,
                    }
                ],
                "phones": [
                    {
                        "start": start_of_this_1st_phoneme,
                        "end":   end_of_this_1st_phoneme,
                        "text":  name_of_this_1st_phoneme,
                    }, {
                        "start": start_of_this_2nd_phoneme,
                        "end":   end_of_this_2nd_phoneme,
                        "text":  name_of_this_2nd_phoneme,
                    }
                ]
            }
    """
    text = open(path, 'r').read()
    assert text.startswith('File type = "ooTextFile"\nObject class = "TextGrid"\n'), 'Unsupported TextGrid filetype'
    text = text.split("\n")
    _dict = {}
    clip_start = float(text[3].split("=")[-1])
    clip_end = float(text[4].split("=")[-1])
    words_start = float(text[11].split("=")[-1])
    words_end = float(text[12].split("=")[-1])
    
    # text relating to word information
    end_word_list = 14+(int(text[13].split("=")[-1])*4)
    word_list = text[14:end_word_list]
    words = []
    for i, line in enumerate(word_list):
        if i%4==0:
            word = {}
        elif i%4==1: # xmin
            word['start'] = float(line.split("=")[-1])
        elif i%4==2: # xmax
            word['end'] = float(line.split("=")[-1])
        elif i%4==3: # text
            word['text'] = line.split("=")[-1].strip().strip('"')
            words.append(word)
    
    # text relating to phoneme information
    phone_list = text[end_word_list:]
    phone_start = float(phone_list[3].split("=")[-1])
    phone_end = float(phone_list[4].split("=")[-1])
    end_phone_list = end_word_list+(int(phone_list[5].split("=")[-1])*4)
    phones = []
    for i, line in enumerate(phone_list[6:6+end_phone_list]):
        if i%4==0:
            phone = {}
        elif i%4==1: # xmin
            phone['start'] = float(line.split("=")[-1])
        elif i%4==2: # xmax
            phone['end'] = float(line.split("=")[-1])
        elif i%4==3: # text
            phone['text'] = line.split("=")[-1].strip().strip('"')
            phones.append(phone)
    
    if quote is not None:
        # use aligned phoneme info to create ARPAbet transcript.
        _dict['arpabet_quote'] = get_arpa(quote, words, phones)
    
    # return data as dict
    _dict['clip_start'] = clip_start
    _dict['clip_end'] = clip_end
    _dict['words_start'] = words_start
    _dict['words_end'] = words_end
    _dict['phone_start'] = words_start
    _dict['phone_end'] = words_end
    _dict['words'] = words
    _dict['phones'] = phones
    return _dict


def force_align_path_quote_pairs(path_quotes, working_directory, dictionary_path, beam_width=10, n_jobs=4, dump_missing_vocab=False, quiet=False, ignore_exceptions=False, model_path="../pretrained_models/english.zip"):
    """
    Run Montreal Forced Aligner over an array of audio-text pairs and return phonetic/timing information.
    
    PARAMS:
        path_quotes: array of audiopath-quote pairs
                    e.g:
                    path_quotes = [
                        ['audio_0.wav', 'I am speaking the first transcript.'],
                        ['audio_1.wav', 'I am speaking the second transcript!'],
                    ]
        working_directory: THIS DIRECTORY/PATH MAY BE DELETED. This will temporarilly hold a renamed copy of the audio files since Montreal Forced Aligner (in my testing) crashes on filenames with spaces (and considers each new folder to be new speaker).
        dictionary_path: must contain a text file pronouncation dictionary.
        beam_width: Montreal Forced Aligner param, something about higher being better but slower. 10 is default, over 400 is apparently reasonable (there IS a perf hit).
            default: 10
        n_jobs: Processes to spawn when processing. Equal to the number of cores in your machine minus one is good enough.
            default: 4
        ignore_exceptions: (optional) use 'ignore exceptions' flag on Montreal Forced Aligner.
            default: False
        dump_missing_vocab: (optional) text file to dump (append) missing vocab.
            default: False
        model_path: (optional) the Montreal Forced Aligner model used for aligning.
            default: "../pretrained_models/english.zip"
                aka the pretrained model packaged with the binary.
    
    RETURNS:
        mfa_data: 
        missing_vocab: 
    """
    # get absolute paths
    abp = os.path.abspath
    path_quotes = [[abp(x[0]), x[1]] for x in path_quotes]
    working_directory = abp(working_directory)
    output_directory = abp(os.path.join(working_directory, 'MFA_output'))
    tmp_directory = abp(os.path.join(working_directory, 'MFA_temp'))
    dictionary_path = abp(dictionary_path)
    
    # assert working_directory is empty (don't want to delete any important files)
    from glob import glob
    if os.path.exists(working_directory) and glob(f'{working_directory}/*'):
        print(f'"{working_directory}" (working directory) is not Empty!')
        if quiet or input('Delete? (y/n)\n> ').lower() in ('y','yes'):
            import shutil
            shutil.rmtree(working_directory)
        else:
            raise Exception(f'"{working_directory}" is not Empty!')
    
    # link/rename audio paths and write quote-files. (MFA fails on filenames with spaces, though it seems to be fine in the folder paths)
    path_lookup = {path: os.path.join(working_directory, str(i)+os.path.splitext(path)[-1]) for i, (path, quote) in enumerate(path_quotes)}
    inv_path_lookup = {v:k for k,v in path_lookup.items()}
    for path, quote in path_quotes:
        new_path = path_lookup[path]
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        try:
            os.link(path, new_path) # hardlink
        except:
            os.symlink(path, new_path) # symlink
        with open(os.path.splitext(new_path)[0]+'.txt', 'w') as f:
            f.write(quote)
    
    # run the aligner
    wd_old = os.getcwd()
    os.chdir(binary_folder)
    if platform == "linux" or platform == "linux2":
        subprocess.call(f'"{binary_folder}/mfa_align"{" -q" if quiet else ""} -b {beam_width} -j {n_jobs} -v -d{" -t" if ignore_exceptions else ""} -t "{tmp_directory}" "{working_directory}" "{dictionary_path}" "{model_path}" "{output_directory}"', shell=True)
    elif platform == "win32":
        subprocess.call(f'"{binary_folder}/mfa_align.exe"{" -q" if quiet else ""} -b {beam_width} -j {n_jobs} -v -d{" -t" if ignore_exceptions else ""} -t "{tmp_directory}" "{working_directory}" "{dictionary_path}" "{model_path}" "{output_directory}"', shell=True)
    os.chdir(wd_old)
    
    # load the outputs and sort into dicts
    if len(open(os.path.join(output_directory, 'utterance_oovs.txt'), 'r').read().split("\n")) > 1:
        inv_basename_lookup = {os.path.splitext(os.path.split(k)[-1])[0]: v for k, v in inv_path_lookup.items()}
        print(' ---- MISSING VOCAB ---- ')
        print('\n'.join( [f'WORD: "{x.split(" ")[-1]}"\nPATH: "{inv_basename_lookup[x.split(" ")[0]]}"\n' for x in open(os.path.join(output_directory, 'utterance_oovs.txt'), 'r').read().split("\n") if len(x)] ))
        print(' ---- ############# ---- ')
    
    missing_vocab = [x for x in open(os.path.join(output_directory, 'oovs_found.txt'), 'r').read() if len(x)]
    
    # (optional) dump missing vocab to file
    if dump_missing_vocab:
        with open(dump_missing_vocab, 'a') as f:
                f.write('\n'.join([f'MISSING WORD: "{x.split(" ")[-1]}"\nPATH: "{inv_basename_lookup[x.split(" ")[0]]}"\n' for x in open(os.path.join(output_directory, 'utterance_oovs.txt'), 'r').read().split("\n") if len(x)]))
    
    # collect together all the new data
    mfa_data = []
    for i, ((path, new_path), (_, quote)) in enumerate(zip(list(path_lookup.items()), path_quotes)):
        textgrid_path = os.path.join(output_directory, os.path.split(working_directory)[-1], str(i)+'.TextGrid')
        if os.path.exists(textgrid_path):
            data = load_TextGrid(textgrid_path, quote=quote)
        else:
            data = None
        mfa_data.append(data)
        del data, textgrid_path
    
    # cleanup
    # unlink (DELETE!) the WORKING DIRECTORY files
    for new_path in path_lookup.values():
        os.unlink(new_path)
        new_txt = os.path.splitext(new_path)[0]+'.txt'
        os.unlink(new_txt)
        del new_txt
    
    import shutil
    shutil.rmtree(working_directory)
    
    return mfa_data, missing_vocab


if __name__ == "__main__":
    #force_align('../../_1_preprocess/tests/littlepip_sample', '../../dict/merged.dict.txt')
    
    # get paths
    # get quotes
    
    #force_align_path_quote_pairs('../../_1_preprocess/tests/littlepip_sample', '../../dict/merged.dict.txt')
    pass