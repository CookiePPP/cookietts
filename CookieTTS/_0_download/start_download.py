import json
import os
import sys
try:
    os.chdir(os.path.split(__file__)[0])
except:
    pass
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
    if 'filenames' in conf[dataset] and filenames is None:
        filenames = conf[dataset]['filenames']
    if force_dl is None and 'force_dl' in conf[dataset].keys():
        force_dl = conf[dataset]['force_dl']
    if not 'download' in conf[dataset].keys() or conf[dataset]['download']:
        os.makedirs(dataset, exist_ok=True); os.chdir(dataset)
        if 'username' in conf[dataset].keys() or 'password' in conf[dataset].keys():
            download_urls.download(urls, dataset, filenames=filenames, force_dl=force_dl, username=conf[dataset]['username'], password=conf[dataset]['password'], auth_needed=True)
        else:
            download_urls.download(urls, dataset, filenames=filenames, force_dl=force_dl)
        os.chdir('..')

# Download Normal Datasets
simple_datasets = [x for x in conf.keys() if x not in ('Clipper_MLP', 'LibriTTS')]

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

# download Unique datasets

# LibriTTS
print(f"On Dataset {dls_position}/{total_to_dl}")
dataset = 'LibriTTS'
urls = []
filenames =[]
if conf[dataset]['download_clean']:
    urls.extend(conf[dataset]['urls_clean'])
    filenames.extend(conf[dataset]['filenames_clean'])
if conf[dataset]['download_other']:
    urls.extend(conf[dataset]['urls_other'])
    filenames.extend(conf[dataset]['filenames_other'])
if len(urls):
    dl_dataset(conf, dataset, urls, filenames=filenames)
del urls, filenames

# Clipper_MLP <- Master File Hosted by Clipper
print(f"On Dataset {dls_position}/{total_to_dl}")
dataset = 'Clipper_MLP'
os.makedirs(dataset, exist_ok=True); os.chdir(dataset)
if conf[dataset]['download']:
    download_clipper.download(conf[dataset])
os.chdir('..')
dls_position+=1