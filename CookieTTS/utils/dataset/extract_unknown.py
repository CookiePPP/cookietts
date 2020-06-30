import tarfile
import os
from zipfile import ZipFile
import py7zr

def extract(path):
    if path.endswith(".zip"):
        with ZipFile(path, 'r') as zipObj:
           zipObj.extractall(os.path.split(path)[0])
    elif path.endswith(".tar.bz2"):
        tar = tarfile.open(path, "r:bz2")
        tar.extractall(os.path.split(path)[0])
        tar.close()
    elif path.endswith(".tar.gz"):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(os.path.split(path)[0])
        tar.close()
    elif path.endswith(".tar"):
        tar = tarfile.open(path, "r:")
        tar.extractall(os.path.split(path)[0])
        tar.close()
    elif path.endswith(".7z"):
        archive = py7zr.SevenZipFile(path, mode='r')
        archive.extractall(path=os.path.split(path)[0])
        archive.close()
    else:
        raise NotImplementedError(f"{path} extension not implemented.")