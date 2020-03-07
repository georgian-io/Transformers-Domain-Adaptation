"""Tokenize corpus for quick lookup during domain pre-training."""
import logging
import argparse
import itertools as it
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer

from src.utils.iter import batch


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        "Tokenize corpus using a pre-defined vocabulary using "
        "BERT's WordPiece tokenization algorithm and converts it to h5 for "
        "quick look-up during domain pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src',
                        required=True,
                        type=lambda x: [Path(p) for p in x.split(',')],
                        help='Path to corpus text file. Alternatively, provide '
                             'a folder and all text files within that folder '
                             'will be searched. Also possible to specify '
                             'multiple comma-separated paths/folders')
    parser.add_argument('--dst', required=True, type=Path,
                        help='Path to save h5 file')
    parser.add_argument('--vocab', type=Path, required=True,
                        help='Vocabulary file to tokenize corpus.')
    parser.add_argument('--filename', type=str, default='corpus.h5',
                        help='Parquet file name')
    parser.add_argument('-b', '--block-size', type=str, default=510,
                        help='Block size for domain pre-training')
    parser.add_argument('-c', '--chunk-size',
                        type=int, default=1000,
                        help='Number of lines in each batch of tokenization')
    args = parser.parse_args()

    # Find all txt files specified in `args.src`
    args.src = list(it.chain.from_iterable(p.glob('*.txt')
                                           if p.is_dir()
                                           else [p]
                                           for p in args.src))
    return args


def read_text_with_logging(p: Path) -> List[str]:
    logger.info(f'Reading text from {p}')
    return p.read_text(encoding="utf-8").splitlines()


def main(args) -> None:
    tokenizer = BertWordPieceTokenizer(str(args.vocab))
    cls_token, sep_token = tokenizer.encode('').ids

    lines = it.chain.from_iterable(read_text_with_logging(p) for p in args.src)
    chunks = batch(lines, args.chunk_size)
    tokenized = it.chain.from_iterable(
        encodings.ids[1:-1]
        for chunk in chunks
        for encodings in tokenizer.encode_batch(list(chunk))
    )
    blocks = ([cls_token] + list(block) + [sep_token] for block in batch(tokenized, args.block_size))
    (
        pd.Series(list(tqdm(blocks, desc='Tokenizing texts')), name='token_blocks')
        .to_frame()
        .to_hdf(args.dst / args.filename, key=args.filename.split('.')[0])
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Check if any text files are very large
    for p in args.src:
        if p.stat().st_size > 1e9:
            logger.warning(f'Size of {p}. This may impact lead to CPU-memory '
                           'issues. Consider sharding file.')

    # Create directory for output corpus
    args.dst.mkdir(exist_ok=True, parents=True)

    main(args)
