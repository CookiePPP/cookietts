import json
import os
import sys
os.chdir(os.path.split(__file__)[0])
sys.path.append(os.path.abspath('..'))

from scripts import download_urls, download_clipper

# load config file
with open("config.json") as f:
    conf = json.loads(f.read())

# move into downloads folder
os.chdir(conf['downloads_folder'])
del conf['downloads_folder']

#######################
# do the downloading  #
#######################

def dl_dataset(conf, dataset, urls, filenames=None, force_dl=None):
    if type(urls) == type(' '):
        urls = [urls,]
    if 'filenames' in conf[dataset]:
        filenames = conf[dataset]['filenames']
    if force_dl is None and 'force_dl' in conf[dataset].keys():
        force_dl = conf[dataset]['force_dl']
    if conf[dataset]['download']:
        os.makedirs(dataset, exist_ok=True); os.chdir(dataset)
        if 'username' in conf[dataset].keys() or 'password' in conf[dataset].keys():
            download_urls.download(urls, dataset, filenames=filenames, force_dl=force_dl, username=conf[dataset]['username'], password=conf[dataset]['password'], auth_needed=True)
        else:
            download_urls.download(urls, dataset, filenames=filenames, force_dl=force_dl)
        os.chdir('..')

# Download Easy Datasets
simple_datasets = [
    'LJSpeech',
    'VCTK',
    'Blizzard2011',
    'Blizzard2013',
    'Littlepip',
    "Blaze the Cat",
    "Persona 4 Golden",
    "TF2",
    "TF2_announcer",
    "TF2_administrator",
    "Doofenshmirtz",
    "Caleb",
    "Wang",
    "Brock_Fucking_Samson",
    "Combine_soldiers",
    "Combine_overwatch",
    "Josh",
]

total_to_dl = len([conf[x]['download'] for x in conf.keys() if 'download' in conf[x].keys() and conf[x]['download']])
dls_position = 1

for dataset in [x for x in simple_datasets if conf[x]['download']]:
    print(f"On Dataset {dls_position}/{total_to_dl}")
    urls = conf[dataset]['url'] if ('url' in conf[dataset].keys()) else (conf[dataset]['urls'] if 'urls' in conf[dataset].keys() else None)
    if urls is None:
        print(f"{dataset} Dataset is missing URL(s).")
    dl_dataset(conf, dataset, urls)
    del urls
    dls_position+=1

# Clipper_MLP <- Master File Hosted by Clipper
print(f"On Dataset {dls_position}/{total_to_dl}")
dataset = 'Clipper_MLP'
os.makedirs(dataset, exist_ok=True); os.chdir(dataset)
if conf[dataset]['download']:
    download_clipper.download(conf[dataset])
os.chdir('..')
dls_position+=1

# LibriTTS
print(f"On Dataset {dls_position}/{total_to_dl}")
dataset = 'LibriTTS'
urls = []
if conf[dataset]['download_clean']:
    urls.extend(conf[dataset]['urls_clean'])
if conf[dataset]['download_other']:
    urls.extend(conf[dataset]['urls_other'])
if len(urls):
    dl_dataset(conf, dataset, urls)
del urls