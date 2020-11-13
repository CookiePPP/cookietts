This step will download the datasets for TTS.

---

Open `config.json` to configure the datasets that will be downloaded.

Once ready, run the command below which will download all datasets selected.

```
python start_downloads.py
```

---

Here is a summary of the datasets available for easy downloading and sorting.


| Dataset                | Speakers | Material Type      | Source                                     | Total Duration (Hours) | Sampling Rate | Gender(s) | Notes                                                                           | URL                                                                  |
|------------------------|----------|--------------------|--------------------------------------------|------------------------|---------------|-----------|---------------------------------------------------------------------------------|----------------------------------------------------------------------|
| LJSpeech               | 1        | Audiobook          | LibriVox project                           | 24                     | 22050         | Female    |                                                                                 | https://keithito.com/LJ-Speech-Dataset/                              |
| VCTK                   | 109~     | Newspaper Extracts | University of Edinburgh                    | 26                     | 96000         | Mixed     | Dataset is 44 Hours without trimming                                            | https://datashare.is.ed.ac.uk/handle/10283/2774?show=full            |
| Clipper/MLP            | 230~     | TV Show            | My Little Pony                             | 24 (80~)               | 48000 (Mixed) | Mixed     | Emotions are Labelled. Contains Labelled 'Clean','Noisy' and 'Very Noisy' data. | https://mega.nz/folder/L952DI4Q#nibaVrvxbwgCgXMlPHVnVw               |
| Blizzard2011/Nancy     | 1        | Audiobook          | Lessac Technologies Inc                    | 16                     | 96000         | Female    | Non-Commercial: Requires License with username and password                     | http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/ |
| Blizzard2013/Catherine | 1        | Audiobook          | Lessac Technologies Inc                    | 300                    | 44100         | Female    | Non-Commercial: Requires License with username and password                     | http://www.openslr.org/60/                                           |
| LibriTTS               | 2456     | Audiobook          | LibriVox project                           | 585~                   | 24000         | Mixed     |                                                                                 | https://desuarchive.org/mlp/thread/35074020/#35075476                |
| Littlepip              | 1        | Audiodrama         | Equestrian Broadcasting Company /Nowacking | <0.1                   | 48000/44100   | Female    | Emotions are Labelled. Contains Labelled 'Clean','Noisy' and 'Very Noisy' data. |                                                                      |
| Blaze the Cat          | 1        | Game               | Sonic the Hedgehog                         | 0.25                   | 48000         | Female    |                                                                                 |                                                                      |
| Persona 4 Golden       | 17       | Game               | Persona 4 Golden                           | TODO                   | 48000         | Mixed     |                                                                                 | https://desuarchive.org/mlp/thread/35205100/#35205486                |
| TF2                    | 12       | Game               | Valve/Team Fortress 2                      | TODO                   | 48000          | Mixed     | Announcer and Administrator voices contain extreme echo                         |                                                                      |
| Doofenshmirtz          | 1        | TV Show            | Phineas and Ferb                           | TODO                   | TODO          | Male      |                                                                                 |                                                                      |
| Caleb                  | 1        | Game               | Blood                                      | TODO                   | TODO          | Male      |                                                                                 |                                                                      |
| Wang                   | 1        | Game               | Shadow Warrior (1997)                      | TODO                   | TODO          | Male      |                                                                                 |                                                                      |
| Brock "Fucking" Samson | 1        | Game            | Poker Night At The Inventory                          | TODO                   | TODO          | Male      | MP3 files, possible quality issue.                                              | https://desuarchive.org/mlp/thread/35459053/#35465945                |
| Combine Soldiers       | 2        | Game               | Half Life 2                                | TODO                   | TODO          | Male      |                                                                                 | https://desuarchive.org/mlp/thread/35308325/#35324614                |
| Josh                   | 1        | TODO               | TODO                                       | TODO                   | TODO          | Male      |                                                                                 |                                                                      |
| Persona 5      | 23       | Game               | Persona 5                           | TODO                   | 22050         | Mixed     |                                                                                 | https://discordapp.com/channels/@me/761501978810777610/764349457138712587 https://discordapp.com/channels/@me/761501978810777610/764353982297800724              |
| Persona 5: Joker      | 1       | Game               | Persona 5: Dancing All Night                           | TODO                   | 48000         | Male     |                                                                                 | https://drive.google.com/file/d/1BgVxuEaUu0dby4ShNATZqAkb6fdBE80_/view?usp=sharing  
| Aperture Science Announcer      | 1       | Game               | Aperture Hand Labs                          | TODO                   | 48000         | Male     |                                                                                 | https://file.house/jC8K.7z
| Planetside 2      | 1       | Game               | Planetside 2                         | TODO                   | 48000         | Female     |                                                                                 | https://file.house/hcjg.7z
- Datasets with sample rates below the f_max of the TTM models can still be used when combined with masked mel bands.
- Datasets are volume normalized, however they will still have differences that make training more challenging for the models. **Quality over Quantity** in this case.
