"""Script to download Pubmed's 2019 baseline files."""
import os
import argparse
import logging
from pathlib import Path
import urllib.request

from tqdm import tqdm

from web import download_file
from utils import parallelize
from hash import md5


N = 1015
FTP_ROOT = 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline'


class FileCorruptedError(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to download Pubmed's 2019 baseline files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dst', type=Path,
                        default=Path('data/biology/corpus/pubmed/'),
                        help='Directory to save all results.')
    parser.add_argument('-N', type=int, default=N,
                        help='Number of file shards to download (out of 1015).')
    parser.add_argument('--ignore-downloaded', action='store_true',
                        help='Default behaviour is to skip download if file '
                             'exists. If provided, ignore existing file and '
                             'download it again.')
    parser.add_argument('-w', type=int, default=10,
                        help='Number of concurrent downloads to do at once.')
    return parser.parse_args()


def generate_filename(n: int) -> str:
    return f'pubmed20n{n:04}.xml.gz'


def is_file_completely_downloaded(file_path: str) -> bool:
    """Check if file is completely downloaded.

    In case it is corrupted by an abrupt termination midway during downloading.

    Arguments:
        filename {str} -- Path to downloaded file
    """
    file_path = Path(file_path)
    file_hash = md5(file_path)
    with urllib.request.urlopen(os.path.join(FTP_ROOT, f'{file_path.name}.md5')) as f:
        checksum = f.read().strip().split()[1].decode()

    if file_hash != checksum:
        msg = f'Checksum of {file_path.name} does not match. Re-downloading...'
        logging.warning(msg)
    return file_hash == checksum


def _download_parallel(i: int, base: Path, ignore_downloaded: bool) -> None:
    """Unit function to parallelize."""
    filename = generate_filename(i)

    # Skip if file is already present
    if (not ignore_downloaded
        and (base / filename).is_file()
        and is_file_completely_downloaded(base / filename)):
        return

    download_file(os.path.join(FTP_ROOT, filename), dst=(base / filename))


def main(args):
    parallelize(_download_parallel, range(1, args.N + 1),
                n_workers=args.w,
                base=args.dst,
                ignore_downloaded=args.ignore_downloaded,
                desc='Downloading pubmed')


if __name__ == '__main__':
    args = parse_args()
    args.dst.mkdir(exist_ok=True, parents=True)
    main(args)
