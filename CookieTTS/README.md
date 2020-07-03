
### \_0_download

`_0_download` contains scripts to download datasets.

---

### \_1_preprocess

`_1_preprocess` will extract/move files downloaded from `_0_download` (you can also move your own datasets into the `datasets` folder).

Files are;

 - Copied from downloads
 - Extracted
 - (Blizzard2011) Sliced into chunks
 - High-passed to remove frequencies under 40Hz
 - Low-passed to remove frequencies over 18000Hz
 - Resampled to target sample rate
 - Trimmed to remove silence
 - Volume normalized

then metadata is collected;

- audio paths
- transcripts
- speaker names
- speaker ids
- emotions (if in dataset)
- noise levels (if in dataset)
- source
- source type
- native sample rates
- aligned phonemes
- inferred emotion embeddings (from text with pretrained torchMoji)

This info is dumped as `filelist_train.txt` and `filelist_validation.txt` inside each dataset, as well as a `meta_dump.json` with **everything** in one file.

---

### \_2_ttm

`_2_ttm` is the Text-to-Mel folder. This contains Tacotron2 (and Flow-TTS later).

Separate usage instructions should be inside each folder. 

---

### \_3\_generate_postnets

`_3_generate_postnets` will generate Ground Truth aligned mel spectrograms from Tacotron2.

This can be used to train WaveGlow/WaveFlow, which is useful because Tacotron2 produces over-smoothed spectrograms.

Rather than making Tacotron2 perfect, it is easier to train WaveGlow/Waveflow to compensate for the inaccuracies.

---

### \_4_mtw

`_4_mtw` contains the Vocoders. Currently only WaveGlow/WaveFlow exists however more might be added in the future.

Again, each folder inside `_4_mtw` should have it's own readme with usage instructions.

---

### \_5_infer

`_5_infer` contains a Flask inference server, `.ipynb` notebook and `.py` script for using models trained in previous steps. 

---

### dict

`dict` contains pronunciation dictionary(s).

---

### scripts

`scripts` contains lots of stuff I use for testing. **I recommend ignoring this folder.**

---

### utils

`utils` contains code to be shared between other modules.