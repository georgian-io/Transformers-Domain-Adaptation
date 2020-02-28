"""Script to convert EURLEX57k dataset to a parquet file."""
import json
import shutil
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.multiproc import parallelize
from src.utils.general_path import GeneralPath


logger = logging.getLogger(__name__)

EURLEX_S3 = GeneralPath('s3://nlp-domain-adaptation/domains/law/tasks/eurlex57k/eurlex57k.zip')
OUTPUT_FILENAME = 'eurlex57k.parquet.gzip'


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract text data from pubmed baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--work-dir', type=Path,
                        default='data/law/tasks/eurlex57k',
                        help='Working directory for EURLEX57k dataset.')
    return parser.parse_args()


def extract(filepath: Path) -> pd.Series:
    with open(filepath, 'r') as f:
        series = pd.Series(json.load(f))

    # Include additional 'dataset' field for retraceability
    series['dataset'] = filepath.parent.name
    return series


def main(args):
    args.work_dir.mkdir(exist_ok=True, parents=True)

    # Download and extract zipped corpus if not available locally
    EURLEX_ZIP = args.work_dir / 'eurlex57k.zip'
    if not (EURLEX_ZIP).exists():
        logger.info(f'File not found. Downloading from S3 to {EURLEX_ZIP}')
        EURLEX_S3.download(EURLEX_ZIP)

        logger.info(f'Unzipping corpus')
        shutil.unpack_archive(str(EURLEX_ZIP), str(args.work_dir))

    # Process filepaths in parallel
    filepaths = list(args.work_dir.rglob('*.json'))
    logger.info(f'Processing {len(filepaths)} JSON files in {args.work_dir}')
    result = parallelize(extract, filepaths,
                         desc='Extracting EURLEX data')

    # Create dataframe
    df = (
        pd.concat(result, axis=1).T
        .pipe(lambda df: df.assign(type=df['type'].astype('category'),
                                   dataset=df['dataset'].astype('category')))
    )

    logger.info(f'Writing result to {args.work_dir / "eurlex57k.parquet"}')
    df.to_parquet(args.work_dir / 'eurlex57k.parquet.gzip', compression='gzip')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
