"""Script to unzip all tar.gz files for US Court Jurisdictions."""
import argparse
import logging
import shutil
from pathlib import Path

from src.utils.general_path import GeneralPath
from src.utils.multiproc import parallelize

logger = logging.getLogger(__name__)

CORPUS_URL = GeneralPath(
    "s3://nlp-domain-adaptation/domains/law/corpus/us_courts/all.tar"
)
WORK_DIR = Path("data/law/corpus/us_courts/")
ZIPPED_FOLDER = "zipped"
UNZIPPED_FOLDER = "unzipped"


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to download, and unzip all.tar and subsequent tar.gz files for US Court Jurisdictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--work-dir", type=Path, default=WORK_DIR, help="Working directory."
    )
    parser.add_argument(
        "--delete-loaded-zip",
        action="store_true",
        help="If provided, delete zip file after it has been "
        "unpacked to save on storage.",
    )
    return parser.parse_args()


def _load_in_parallel(zip_file: Path, dst: Path, delete_zip_file: bool = False) -> None:
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
    extract_path = dst / zip_file.name.split(".tar.gz")[0]

    # Extract file to target location
    extract_path.mkdir(exist_ok=True, parents=True)
    shutil.unpack_archive(str(zip_file), str(extract_path))

    # Delete file to save space
    if delete_zip_file:
        zip_file.unlink()


def main(args):
    logging.basicConfig(level=logging.INFO)

    # Download corpus from S3
    if not (args.work_dir / "all.tar").exists():
        logger.INFO("Corpus does not exist. Downloading from S3...")
        CORPUS_URL.download(str(args.work_dir))

    # Create directory to store individual tar.gz files from unzipping all.tar
    logger.INFO("Unzipping corpus...")
    (args.work_dir / ZIPPED_FOLDER).mkdir(exists_ok=True, parents=True)
    shutil.unpack_archive(
        str(args.work_dir / "all.tar"), str(args.work_dir / ZIPPED_FOLDER)
    )

    parallelize(
        _load_in_parallel,
        list((args.work_dir / ZIPPED_FOLDER).glob("*.tar.gz")),
        dst=args.work_dir / UNZIPPED_FOLDER,
        delete_zip_file=args.delete_loaded_zip,
        desc="Unpacking secondary ZIP files",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
