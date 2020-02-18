"""Script to unzip all tar.gz files for US Court Jurisdictions."""
import argparse
import shutil
from pathlib import Path

from utils import parallelize


ALL_DIR = Path('data/law/corpus/us_court_jurisdictions_opinions/all')


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to unzip all tar.gz files for US Court Jurisdictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--src', type=Path,
                        default=ALL_DIR,
                        help='Directory containing zips.')
    parser.add_argument('--dst', type=Path,
                        default=ALL_DIR.parent,
                        help='Directory to write results to.')
    return parser.parse_args()


def _load_in_parallel(zip_file: Path,
                      dst: Path,
                      delete_zip_file: bool = False) -> None:
    """Unit function to load zip files.

    The resulting unzipped files will be placed in a folder named with the
    stem of the zip_file. For example, if the file is "apple.tar.gz", the
    unzipped files will be placed at "dst/apple".

    Arguments:
        zip_file {Path} -- tar.gz file to unzip
        dst {Path} -- Directory to store zipped files

    Keyword Arguments:
        delete_zip_file {bool} -- If True, delete zip file to save on space.
                                  (Corpus is about 36GB) (default: {False})
    """
    extract_path = dst / zip_file.name.split('.tar.gz')[0]

    # Extract file to target location
    extract_path.mkdir(exist_ok=True, parents=True)
    shutil.unpack_archive(zip_file, extract_path)

    # Delete file to save space
    if delete_zip_file:
        zip_file.unlink()


def main(args):
    parallelize(_load_in_parallel, list(args.src.glob('*.tar.gz')),
                dst=args.dst,
                delete_zip_file=True,
                desc='Unpacking ZIP files')


if __name__ == '__main__':
    args = parse_args()
    main(args)
