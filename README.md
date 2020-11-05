# This repo kinda works!

Check back in a week. Thanks.

Missing stuff;
- Normalize transcripts before running Montreal Forced Aligner

---

### Install/Setup:

##

Clone this repo: `https://github.com/CookiePPP/cookietts.git`

This will make a folder called `cookietts` where the command is run, and clone the repo into said folder.

##

Run `cd cookietts`

This will move you into the cloned repo.

##

Run `pip install -e .`

This will 'install' the package (without moving around any files).

##

Run `pip install -r requirements.txt`

This will install dependencies.

e.g: pytorch, numpy, scipy, tensorboard, librosa

##

Install [Nvidia/Apex](https://github.com/nvidia/apex#linux).

Nvidia/Apex contains the LAMB optimizer and faster running fused optimizers.

This also allows for fp16 (mixed-precision) training, which saves VRAM and runs faster on RTX cards.

(please nag me if you cannot install Apex. Pytorch added native fp16 (mixed-precision) support some time ago and I might be able to set that up as an alternative)

##

That should be the main stuff.

If something fails during preprocessing, check you have `ffmpeg` and `sox` installed as well.

If something else fails, you can create an 'issue' on the top of the github page. 

---

### Usage:

Head into the `CookieTTS` folder and read around.

If you want to train custom multispeaker models then folders **0**, **1** and **2** are of interest.

If you want to use an already trained model to generate new speech then folder **5** (`_5_infer`) is of interest.