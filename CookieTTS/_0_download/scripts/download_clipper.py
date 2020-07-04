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
    if not len(list(directory.iterdir())):
        directory.rmdir()


def download(conf):
    # Download the master file
    print("Downloading Clippers Master Folder\nthis takes multiple days to download due to mega.nz free bandwidth limits! An alternative download will be figured out later.")
    from time import sleep; sleep(2)
    import os
    
    # download files from MEGA
    megadown(conf['url'], filename='./')
    
    print("Finished!")