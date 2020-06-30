import os
from glob import glob
import soundfile as sf
import librosa

def NancySplitRawIntoClips(directory, label_folder, Nancy_CorpusToArchive, in_ext=".wav", out_ext=".wav", SAMPLE_RATE=96000):
    """
    Take original 96Khz studio files and match them with the filenames used for the quote files.
    Writes audio segments into 'Sliced' folder inside the specified `directory`.
    PARAMS:
        directory: Where the input raw 96Khz audio files are located.
        label_folder: Where the prompts.data and timing information is located.
        Nancy_CorpusToArchive: Path to "NancyCorpusToArchiveMap.txt" where the output filenames are held as well as any exceptions.
    RETURNS:
        None
    """
    Nancy_ignore = {studio: output for output, studio, exception in [line.split("\t") for line in ((open(Nancy_CorpusToArchive, "r").read()).splitlines())] if exception}
    print(Nancy_ignore)
    Nancy_lookup = {studio: output for output, studio, exception in [line.split("\t") for line in ((open(Nancy_CorpusToArchive, "r").read()).splitlines())]}
    
    os.makedirs(os.path.join(directory,'Sliced'), exist_ok=True)
    available_labels = ["/".join(x.split("/")[-1:]) for x in glob(label_folder+"/*.txt")]
    available_audio = glob(directory+"/*"+in_ext)
    for audio_file in available_audio:
        print(audio_file)
        audio_filename = "/".join(audio_file.split("/")[-1:])
        audio_basename = audio_filename.replace(in_ext,"")
        audio_basename = audio_basename.replace("341_763","343_763") # exception, easier than rewriting entire label file
        ID_offset = int(audio_basename.split("_")[-2]) - 1
        ID_end = int(audio_basename.split("_")[-1]) - 1
        Prepend_ID = "_".join(audio_basename.split("_")[:-2]) # empty unless ARCTIC or LTI files
        if Prepend_ID: Prepend_ID += "_"
        if audio_filename.replace(in_ext,".txt") in available_labels:
            label_path = os.path.join(label_folder, audio_filename.replace(in_ext,".txt") ) # get label file
            beeps = []
            for line in ((open(label_path, "r").read()).splitlines()):
                beeps+=[line.split("\t")] # [beep_start, beep_stop, ID]
            print("beep count", len(beeps))
            print("ID_offset", ID_offset)
            print("ID_end", ID_end)
            print("end - offset", ID_end-ID_offset)
            assert (len(beeps)-1) == (ID_end-ID_offset), "Ensure each beep is labelled and matches the ArchiveMap"
            sound, local_SR = sf.read(audio_file)
            assert SAMPLE_RATE == local_SR, f"{audio_file} sample rate pf {local_SR} does not match expected sample rate of {SAMPLE_RATE}"
            #sound, _ = librosa.core.load(audio_file, sr=SAMPLE_RATE)
            for i in range(len(beeps)):
                clip_start = int(float(beeps[i][1])*SAMPLE_RATE) # end of previous beep
                clip_end = int(float(beeps[i+1][0])*SAMPLE_RATE) if i+1 < len(beeps) else len(sound) # start of next beep or end of file if no beeps left
                ID = Prepend_ID + str(ID_offset + int(beeps[i][2]))
                if ID in Nancy_ignore.keys():
                    continue
                print(ID,"-->",Nancy_lookup[ID])
                ID = Nancy_lookup[ID]
                clip_outpath = os.path.join(directory, 'Sliced', ID+out_ext)
                sound_clipped = sound[clip_start:clip_end]
                sf.write(clip_outpath, sound_clipped, SAMPLE_RATE)
                #librosa.output.write_wav(clip_outpath, sound_clipped, SAMPLE_RATE)
        else:
            print(audio_file, "doesn't have an available label")

def NancyWriteTranscripts(directory, prompts, ext=".wav"):
    Nancy_lookup = {filename: quote[1:-1].strip() for filename, quote in [line[2:-2].split(" ",1) for line in ((open(prompts, "r").read()).splitlines())]}
    for audio_fpath in glob(directory+"/**/*"+ext, recursive=True):
        audio_fpathname = "/".join(audio_fpath.split("/")[-1:])
        if audio_fpathname.replace(ext,"") in Nancy_lookup.keys():
            quote = Nancy_lookup[audio_fpathname.replace(ext,"")]
            quote = unidecode(quote)
            with open(audio_fpathname.replace(ext,".txt"), "w") as f:
                f.write(quote)
        else:
            print(f"{audio_fpath} Has no Quote.")