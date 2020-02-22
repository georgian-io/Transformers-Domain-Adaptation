"""Class definition for GeneralPath object."""
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import List, Union, Tuple, Optional

import boto3


class GeneralPath:
    """Contains READ ONLY functionalities."""
    def __init__(self, uri: str):
        if '://' not in uri:
            raise ValueError('Invalid uri specified')
        self.url = urlparse(uri)
        self.path = Path(self.url.path.strip('/'))
        self.client = None

    def __str__(self) -> str:
        return urlunparse((self.url.scheme, self.url.netloc,
                           self.path.as_posix(), self.url.params,
                           self.url.query, self.url.fragment))

    def __repr__(self) -> str:
        return f"GeneralPath({str(self)})"

    def __copy__(self) -> 'GeneralPath':
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}) -> 'GeneralPath':
        return deepcopy(self)

    def __div__(self, other: str) -> 'GeneralPath':
        if not isinstance(other, str):
            raise ValueError(f'GeneralPath can only be append with str type')
        ret = self.copy()
        ret.path /= other
        return ret

    def _init_client(self) -> 'GeneralPath':
        if 's3' in self.url.scheme:
            self.client = S3Client(self.url.netloc)
        else:
            raise NotImplementedError()
        return self

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def suffix(self) -> str:
        return self.path.suffix

    def parts(self) -> Tuple[str]:
        return tuple([self.url.scheme] + self.path.parts)

    def as_uri(self) -> str:
        return str(self)

    def is_dir(self) -> bool:
        if self.client is None:
            self._init_client()
        result = self.client.search(str(self.path))
        if len(result) == 1 and result[0].endswith('/'):
            return True
        else:
            return False

    def is_file(self) -> bool:
        if self.client is None:
            self._init_client()
        result = self.client.search(str(self.path))
        if len(result) == 1 and '.' in self.path.name:
            return True
        else:
            return False

    def exists(self) -> bool:
        return self.is_file() or self.is_dir()

    def rglob(self, pattern: str = '') -> List['GeneralPath']:
        splits = pattern.split('*')
        if len(splits) > 2:
            raise ValueError('Only one wildcard character * allowed')

        if self.client is None:
            self._init_client()
        result = [GeneralPath(x) for x in self.client.search(splits[0])]

        if len(splits) == 2:
            result = [x for x in result if x.path.as_posix().endswith(splits[1])]

        return result

    def glob(self, pattern: str = '') -> List['GeneralPath']:
        ret = []
        base = pattern.split('*')[0]  # Bug when pattern is not a complete folder name
        for hit in self.rglob(pattern=pattern):
            if '/' not in str(hit.path)[(len(str(self.path)) + len(base) + 1):]:  # Plus one is to account for possible '/'
                ret.append(hit)
        return ret

    def download(self, dst: str) -> None:
        if self.is_file():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            return self.client.download(str(self.path), dst)
        elif self.is_dir():
            raise NotImplementedError()
        else:
            raise FileNotFoundError('File does not exist')




class S3Client:
    def __init__(self, bucket: str) -> "S3Client":
        self.bucket = bucket.strip('/')
        self.client = boto3.client('s3')

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    def search(self, prefix: Optional[str] = None) -> List[str]:
        hits = []

        search_params = {'Bucket': self.bucket}
        if prefix not in (None, ''):
            search_params['Prefix'] = prefix

        response = self.client.list_objects_v2(**search_params)
        while True:
            if 'Contents' not in response:
                return []
            keys = [x['Key'] for x in response['Contents']]

            # Sort in increasing length
            keys = sorted(keys, key=len)

            # Only check if folders (ending with '/') is redundant
            ret = []
            for i, key in enumerate(keys):
                if not key.endswith('/'):
                    ret.append(key)
                    continue

                for j in range(i + 1, len(keys)):
                    if keys[i] in keys[j]:
                        break
                else:
                    ret.append(key)
            hits += ret

            # Continue search if 'NextContinuationToken' exists
            if 'NextContinuationToken' in response:
                search_params['ContinuationToken'] = response['NextContinuationToken']
                response = self.client.list_objects_v2(**search_params)
            else:
                break

        # Convert `hits` into S3 urls
        hits = [f's3://{self.bucket}/{x}' for x in hits]
        return hits

    def download(self, key: str, dst: str) -> None:
        self.client.download_file(Bucket=self.bucket, Key=key, Filename=dst)
