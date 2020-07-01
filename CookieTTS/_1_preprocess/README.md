### This is the preprocessing folder.

---

### Usage:

Run `python start_preprocess.py`

##

This will:

1. extract/copy files downloaded from `_0_download`
2. (optional) slice Blizzard2011 dataset into chunks
3. (optional) delete Noisy/Very Noisy clips from Clipper_MLP dataset 
4. High-passed to remove frequencies under 40Hz
5. Low-passed to remove frequencies over 18000Hz
6. Resample audio to target sample rate
7. Trim all audio files to remove silence
8. RMS Normalise volume to `0.08`
9. Collect metadata:
	1. audio paths
	2. transcripts
	3. speaker names
	4. speaker ids
	5. emotions (if in dataset)
	6. noise levels (if in dataset)
	7. source type
	8. native sample rates
	9. aligned phonemes
	10. inferred emotion embeddings (from text with pretrained torchMoji)
10. Write a `speaker_info.txt` and `emotion_info.txt` inside the `_1_preprocess` directory.
10. Dump dataset filelists as `filelist_train.txt` and `filelist_validation.txt` inside each dataset. (this contains **most** of the important info, but not all of it)
11. Dump **everything** as `meta_dump.txt` in datasets directory.

---

### Custom Datasets:

The preprocessing script is intended to be flexible enough for new datasets.

1. Make a folder inside the datasets folder. The name of this folder will be the name of the dataset.
	- e.g: go into datasets folder and create a folder called `Red vs Blue`. `Red vs Blue` is now the name of that dataset.
2. Place **a copy** of your custom datasets files inside the newly created folder.
	- e.g: inside the `Red vs Blue` folder, place a **copy** of any audio files and text files you have.
	- Any `.zip` and `.tar` files will be automatically extracted so those can be used to package your custom data.
	- This is tested on `.flac` and `.wav` files. Other audio file-types are unlikely to work.
	- If using stereo audio, note only the left channel will be included/used.

3. Run `python start_preprocess.py`
	- it will prompt you with basic questions, then process your dataset the same as the others.

4. Fix skipped clips because of missing vocab or typos.
	- Montreal Forced Aligner cannot align to words that do not have a pronunciation in the dictionary.
	- In the datasets folder will be a file called `missing_vocab.txt`.
		- This contains words without pronounciations and their relavent audio path.
		- If the word is a typo, fix it.
		- If there is no typo then the word likely does not exist in the pronunciation dictionary. Post the word and relevant audio file somewhere and they can be added to the dictionary by me or another person!
		- e.g: The word `Sangheili` is not in our dict, a pronunciation would need to be figured out and added to the dictionary in order to use a clip with that word in it.
5. Once you have your new dictionary file, delete the datasets folder and start from step 1.
	- If `missing_vocab.txt` is empty or close enough, you're good to start training models.