"""Script to select subset of corpus for downstream domain adaptation."""
import argparse
import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        """Script to select subset of corpus for downstream domain adaptation.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--corpus', type=Path, required=True, help='Corpus')
    parser.add_argument('--dst', type=Path, required=True, help='Directory to save corpus subset')
    # parser.add_argument('-m', '--mode', choices=('random',), required=True,
    #                     help='Method to select subset of data')
    parser.add_argument('--filename', type=str, default='corpus_subset.txt',
                        help='Filename for corpus subset')

    # Args for "random" mode
    subparsers = parser.add_subparsers(help='Method to select subset of data', dest='mode')
    subparser = subparsers.add_parser('random', help='Randomly select data')
    subparser.add_argument('-p', '--percentage', type=float, default=1.,
                           help='Percentage of data to select')
    subparser.add_argument('-s', '--seed', type=int, default=None,
                           help='Random seed for reproducability')

    return parser.parse_args()


def main(args):
    logger.info('Reading corpus')
    lines = pd.Series(args.corpus
                      .read_text(encoding="utf-8")
                      .splitlines())

    # Get a random subset of lines
    logger.info(f'Randomly sampling {args.percentage} of corpus')
    lines = lines.sample(frac=args.percentage, random_state=args.seed)

    # Save corpus
    logger.info(f'Saving subset corpus to {args.dst / args.filename}')
    args.dst.mkdir(exist_ok=True, parents=True)
    (args.dst / args.filename).write_text('\n'.join(lines.tolist()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
