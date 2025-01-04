import os
import requests
from bs4 import BeautifulSoup

def download_file(url, download_dir='downloads'):
    """
    using requests to download file from url
    """
    os.makedirs(download_dir, exist_ok=True)

    local_filename = os.path.join(download_dir, url.split('/')[-1])
    print(f"开始下载 {url} ...")

    # stream=True: using requests to download large files or data streams
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"下载完成: {local_filename}\n")


def get_download_links(page_url):
    """
    get download links from the page
    """
    print(f"Getting download links from {page_url} ...")
    response = requests.get(page_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    download_suffixes = ('.zip', '.tar', '.tar.gz', '.tgz', '.7z')

    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.lower().endswith(download_suffixes):
            if href.startswith('/'):
                href = "https://lila.science" + href
            links.append(href)

    return links


def main():
    nacti_url = "https://lila.science/datasets/nacti"
    download_links = get_download_links(page_url=nacti_url)

    if not download_links:
        print("No download links found.")
        return

    for link in download_links:
        download_file(link, download_dir='downloads')


if __name__ == "__main__":
    main()
