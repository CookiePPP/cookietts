import urllib.request
from CookieTTS._0_download.scripts.download_mega import megadown


from pathlib import Path
def rmdir(directory, white_list=[], ignore_strs=[]):
    directory = Path(directory)
    doing_whitelist = len(white_list) > 0
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            if doing_whitelist and not any([x.lower() in str(item).lower() for x in white_list]):
                continue
            if any([x.lower() in str(item).lower() for x in ignore_strs]):
                continue
            item.unlink()
    if not len(directory.iterdir()):
        directory.rmdir()


def download(conf):
    # Download the master file
    print("Downloading Clippers Master Folder\nthis has many small files so patience is appreciated.")
    from time import sleep; sleep(2)
    import os
    
    # download files from MEGA
    megadown(conf['url'], filename='./')
    
    # cleanup
    if conf["delete_non_voice_data"]:
        rmdir('./', ignore_strs=['Sliced Dialogue'])
    
    if conf["delete_songs"]:
        rmdir('./Sliced Dialogue/Songs', ignore_strs=[])
    
    if  conf["delete_noisy"]:
        rmdir('./Sliced Dialogue', white_list=['_Noisy_'], ignore_strs=[])
    
    if conf["delete_very_noisy"]:
        rmdir('./Sliced Dialogue', white_list=['_Very Noisy_'], ignore_strs=[])
    
    white_list = []
    if len(conf["removable_folders"]):
        white_list.extend(conf["removable_folders"])
    if len(conf["removable_emotions"]):
        white_list.extend([f"_{x}" for x in conf["removable_folders"]])
    if len(white_list):
        rmdir('./', white_list=white_list, ignore_strs=[])
    
    print("Finished!")