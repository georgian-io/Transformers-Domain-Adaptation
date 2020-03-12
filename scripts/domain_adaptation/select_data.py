"""Script to select subset of corpus for downstream domain adaptation.

Shuffling is not done here as that is handled by the domain pre-training script.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        "Script to select subset of corpus for downstream domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--corpus', type=Path, required=True, help='Corpus')
    parser.add_argument('--dst', type=Path, required=True,
                        help='Directory to save corpus subset')
    parser.add_argument('--filename', type=str, default=None,
                        help='Filename for corpus subset')

    # Args for "random" mode
    subparsers = parser.add_subparsers(help='Method to select subset of data',
                                       dest='mode')
    subparser = subparsers.add_parser('random', help='Randomly select data')
    subparser.add_argument('-p', '--percentage', type=float, default=1.,
                           help='Percentage of data to select')
    subparser.add_argument('-s', '--seed', type=int, default=None,
                           help='Random seed for reproducability')

    return parser.parse_args()


def main(args):
    logger.info('Reading corpus')
    with open(args.corpus) as f:
        n_lines = sum(1 for _ in tqdm(f, desc='Reading', leave=False))

    lines_index = pd.Series(np.arange(n_lines))

    # Get a random subset of lines
    logger.info(f'Randomly sampling {args.percentage} of corpus with '
                f'a seed of {args.seed}')
    np.random.seed(args.seed)
    sample_index = np.random.choice([0, 1],
                                    size=(n_lines,),
                                    p=[1 - args.percentage, args.percentage])

    # Parse filename if not provided
    if args.filename is None:
        args.filename = args.corpus.stem
        args.filename += f'_{args.mode}'
        args.filename += f'_{int(100 * args.percentage)}pct'
        if args.seed is not None:
            args.filename += f'_seed{args.seed}'
        args.filename += args.corpus.suffix

    # Save corpus
    logger.info(f'Saving subset corpus to {args.dst / args.filename}')
    args.dst.mkdir(exist_ok=True, parents=True)
    with open(args.corpus) as reader:
        with open(args.dst / args.filename, 'w+') as writer:
            # Read and sample
            lines = (line for line, should_sample in zip(reader, sample_index)
                        if should_sample)

            # Write
            lines = tqdm(lines, desc='Writing',
                         leave=False, total=sample_index.sum())
            list(writer.write(line) for line in lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
