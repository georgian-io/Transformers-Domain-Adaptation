"""Script to generate training, dev, and test sets for Eurlex57k dataset."""
import logging
import argparse
import itertools as it
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

    # Write text data for each dataset
    for mode in ('train', 'dev', 'test'):
        logger.info(f'Generating texts for {mode} set')
        subset = df[df['dataset'] == mode]
        text_cols = subset[Eurlex57kDataset.TEXT_COLS]
        texts = Eurlex57kDataset.preprocess_texts(text_cols)

        output_text_file = args.dst / f'{mode}.txt'
        output_text_file.write_text('\n'.join(texts), encoding='utf-8')
        logger.info(f'Saved text for {mode} set to {output_text_file}')

    # Remove zero-shot labels from "concepts" field
    logger.info('Removing zero-shot labels')
    concept_sets = (
        df.groupby('dataset')['concepts']
        .apply(lambda series: set(it.chain.from_iterable(series.tolist())))
    )
    dev, test, train = concept_sets
    zero_shot_labels = (dev - train) | (test - train)
    df['concepts'] = df['concepts'].apply(lambda concepts: set(concepts) - zero_shot_labels)

    for mode in ('train', 'dev', 'test'):
        logger.info(f'Generating labels for {mode} set')
        labels_subset = (
            df[df['dataset'] == mode]['concepts']
            .apply(lambda x: ' '.join(x))
        )

        output_label_file = args.dst / f'{mode}_labels.txt'
        output_label_file.write_text('\n'.join(labels_subset.tolist()))
        logger.info(f'Saved labels for {mode} set to {output_label_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
