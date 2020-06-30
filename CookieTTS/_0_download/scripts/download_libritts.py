import urllib.request

def download(urls):
    for i, url in enumerate(urls):
        print(f"Downloading LibriTTS Dataset (File {i+1}/{len(urls)})...")
        urllib.request.urlretrieve(url, url.split("/")[-1])
    print("Finished!")