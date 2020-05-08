"""Script to generate training, dev, and test sets for Eurlex57k dataset."""
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.etl.law.tasks.eurlex57k.eurlex57k_dataset import Eurlex57kDataset


logger = logging.getLogger(__name__)


def parse_args(args: Optional[str] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        "Generate training, dev, and test sets for Eurlex57k dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--eurlex-src', type=Path,
                        help='Eurlex57k Parquet GZIP file',
                        default='data/law/tasks/eurlex57k/eurlex57k.parquet.gzip')
    parser.add_argument('--dst', type=Path,
                        help='Output directory',
                        default='data/law/tasks/eurlex57k')
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    """Main function."""
    logging.info(f'Loading Eurlex57k dataset from {args.eurlex_src}')
    df = pd.read_parquet(args.eurlex_src)
    df['main_body'] = df['main_body'].apply(lambda x: '\n'.join(x))

    for mode in ('train', 'dev', 'test'):
        logging.info(f'Generating texts for {mode} set')
        subset = df[df['dataset'] == mode]
        text_cols = subset[Eurlex57kDataset.TEXT_COLS]
        texts = Eurlex57kDataset.preprocess_texts(text_cols)

        output_text_file = (args.dst / mode).with_suffix('.txt')
        output_text_file.write_text('\n'.join(texts), encoding='utf-8')
        logging.info(f'Saved text for {mode} set to {output_text_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
