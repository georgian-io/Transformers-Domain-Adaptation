import hashlib


def md5(fname: str, buffer_size: int = 4096) -> str:
    md5_hash = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(buffer_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()
