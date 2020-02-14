from typing import List
import urllib.request

import requests
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector


def find_hlinks(endpoint: str) -> List[str]:
    resp = requests.get(endpoint)

    # Find appropriate encodings
    http_enc = resp.encoding if 'chatset' in resp.headers.get('content-type', '').lower() else None
    html_enc = EncodingDetector.from_declared_encoding(resp.content, is_html=True)
    enc = html_enc or http_enc

    soup = BeautifulSoup(resp.content, from_encoding=encoding)

    return [link['href'] for link in soup.find_all('a', href=True)]


def download_file(url: str, dst: str) -> None:
    urllib.request.urlretrieve(url, dst)
