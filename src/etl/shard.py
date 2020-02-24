"""Script to shard large text file.

Borrowed from https://gist.github.com/Guitaricet/937264d4958cb62084008cc5d6da0685
"""
import shutil
import logging
import argparse
from pathlib import Path

from tqdm import tqdm


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        'General script to shard a text file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', type=Path, required=True,
                        help='Input text file to be sharded')
    parser.add_argument('--dst', type=Path, required=True,
                        help='Folder to store sharded files')
    parser.add_argument('-s', '--shard-size', default=int(1e6), type=int,
                        help='Number of lines per shard')
    return parser.parse_args()


def read_in_chunks(file_object, n_lines=1024) -> str:
    while True:
        ret = []
        for _ in range(n_lines):
            data = file_object.readline()
            if not data:
                break
            ret.append(data)

        if not len(ret):
            break
        yield ''.join(ret)


def main(args):
    logger.info(f'Writing shards to {args.dst}...')
    if args.dst.exists():
        logger.info(f'Removed existing folder at {args.dst}')
        shutil.rmtree(str(args.dst), ignore_errors=True)
    args.dst.mkdir(parents=True)

    with open(args.src) as f:
        for i, chunk in tqdm(enumerate(read_in_chunks(f,
                                                      n_lines=args.shard_size)),
                             desc='Sharding'):
            shard = (
                (args.dst / f'{args.src.stem}_{i:05d}')
                .with_suffix('.txt')
            )
            tqdm.write(f'Created shard {shard}')
            shard.write_text(chunk)

    logger.info(f'Successfully sharded {args.src} into {i} shards at {args.dst}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
