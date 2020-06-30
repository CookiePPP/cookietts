# megatools download urls
win64_url = "https://megatools.megous.com/builds/experimental/megatools-1.11.0-git-20200503-win64.zip"
win32_url = "https://megatools.megous.com/builds/experimental/megatools-1.11.0-git-20200503-win32.zip"
linux_url = "https://megatools.megous.com/builds/experimental/megatools-1.11.0-git-20200503-linux-x86_64.tar.gz"

# download megatools
from sys import platform
import os
import urllib.request
import subprocess
from CookieTTS.utils.dataset.extract_unknown import extract
from time import sleep
prev_wd = os.getcwd()
os.chdir(os.path.split(__file__)[0])

if not os.path.exists('megatools'):
    print('"megatools" not found. Downloading...')
    sleep(0.10)
    if platform == "linux" or platform == "linux2":
            dl_url = linux_url
    elif platform == "darwin":
        raise NotImplementedError('MacOS not supported.')
    elif platform == "win32":
            dl_url = win64_url
    dlname = dl_url.split("/")[-1]
    if not os.path.exists(dlname):
        urllib.request.urlretrieve(dl_url, dlname)
    assert os.path.exists(dlname), 'failed to download.'
    extract(dlname)
    sleep(0.10)
    os.unlink(dlname)
    print("Done!")

binary_folder = os.path.abspath(dlname)
os.chdir(prev_wd)

def megadown(download_link, filename='.'):
    """Use megatools binary executable to download files and folders from MEGA.nz ."""
    filename = ' --path "'+os.path.abspath(filename)+'"' if filename else ""
    wd_old = os.getcwd()
    os.chdir(binary_folder)
    if platform == "linux" or platform == "linux2":
        subprocess.call(f'./megatools dl{filename} --debug http {download_link}', shell=True)
    elif platform == "win32":
        subprocess.call(f'megatools.exe dl{filename} --debug http {download_link}', shell=True)
    os.chdir(wd_old)