"""Script to select subset of corpus for downstream domain adaptation.

Shuffling is not done here as that is handled by the domain pre-training script.
"""
import sys
import argparse
import logging
import itertools as it
from pathlib import Path
from types import SimpleNamespace
from typing import List, Iterable, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer

from src.utils.iter import batch
sys.path.append('learn-to-select-data')
import similarity
from constants import SIMILARITY_FUNCTIONS


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
    subparser.add_argument('-p', '--pct', type=float, required=True,
                           help='Percentage of data to select w.r.t corpus size')
    subparser.add_argument('-s', '--seed', type=int, default=None,
                           help='Random seed for reproducability')

    # Args for "similarity" mode
    subparser = subparsers.add_parser(
        'similar', help='Select data based on token similarity'
    )
    subparser.add_argument('--fine-tune-text', type=Path, required=True,
                           help='Path to fine tuning (training) text. '
                                'Similarity of individual documents in corpus '
                                'will be compard against this text.' )
    subparser.add_argument('--sim-func', choices=SIMILARITY_FUNCTIONS,
                           default='jensen-shannon',
                           help='Similarity function to use')
    subparser.add_argument('-v', '--vocab-file', type=Path, required=True,
                           help='BERT vocabulary file')

    group = subparser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pct', type=float,
                       help='Percentage of data to select w.r.t. corpus size')
    group.add_argument('-n', '--n-docs', type=int,
                       help='Number of documents to select')
    group.add_argument('-t', '--threshold', type=float,
                       help='Select documents with similarities above '
                            '(or below if --invert is supplied)')

    subparser.add_argument('-i', '--invert', action='store_true',
                           help='If provided, pick most dissimilar documents '
                                'instead')
    subparser.add_argument('--no-lowercase',
                           action='store_false', dest='lowercase',
                           help='If provided, will not perform lowercasing '
                                'during tokenization')
    subparser.add_argument('-c', '--chunk-size', type=int, default=2**13,
                           help='Tokenization chunk size')

    args = parser.parse_args()

    if args.pct is not None and not 0 < args.pct <= 1:
        raise ValueError(f'Invalid percentage value of {args.pct} provided')

    return args


def parse_filename(args) -> str:
    filename = args.corpus.stem
    if args.mode == 'random':
        filename += f'_{args.mode}'
        filename += f'_{int(100 * args.pct)}pct'
        if args.seed is not None:
            filename += f'_seed{args.seed}'
    elif args.mode == 'similar':
        filename += '_similar' if not args.invert else '_dissimilar'
        filename += f'_{args.sim_func}'
        filename += f'_{args.fine_tune_text.stem}'
        if args.pct is not None:
            filename += f'_{args.pct}pct'
        elif args.n_docs is not None:
            filename += f'_{args.n_docs}docs'
        elif args.threshold is not None:
            filename += f'_{args.threshold}threshold'
    else:
        raise NotImplementedError
    filename += args.corpus.suffix
    return filename


def get_file_obj(filepath: Union[str, Path]) -> Iterable[str]:
    logger.info(f'Reading {filepath}')
    with open(filepath) as f:
        n_lines = sum(1 for _ in f)
    return tqdm(open(filepath), desc='Reading', leave=False, total=n_lines)


def copy_selected_docs(index: np.array, args) -> None:
    # Save corpus
    logger.info(f'Saving subset corpus to {args.dst / args.filename}')
    args.dst.mkdir(exist_ok=True, parents=True)
    with open(args.corpus) as reader:
        with open(args.dst / args.filename, 'w+') as writer:
            # Read and sample
            lines = (line for line, should_sample in zip(reader, index)
                          if should_sample)

            # Write
            lines = tqdm(lines, desc='Writing',
                         leave=False, total=index.sum())
            list(writer.write(line) for line in lines)


def docs_to_term_dist(docs: Iterable[str],
                      vocab_file: Path,
                      lowercase: bool,
                      chunk_size: int,
                     ) -> Iterable[np.array]:
    tokenizer = BertWordPieceTokenizer(str(vocab_file), lowercase=lowercase)

    # Create a duck-typed Vocabulary object to work on `similarity.get_term_dist`
    vocab = vocab_file.read_text().splitlines()
    vocab_obj = SimpleNamespace()
    vocab_obj.size = len(vocab)
    vocab_obj.word2id = {word: i for i, word in enumerate(vocab)}

    # Tokenize each document
    tokenized: Iterable[List[str]] = (
        enc.tokens[1:-1]
        for b in batch(docs, chunk_size)
        for enc in tokenizer.encode_batch(list(b))
    )

    # Convert each token into a term distribution
    term_dists: Iterable[np.array] = (
        similarity.get_term_dist([x], vocab=vocab_obj, lowercase=lowercase)
        for x in tokenized
    )
    return term_dists


def select_random(args) -> np.array:
    f = get_file_obj(args.corpus)
    n_lines = sum(1 for _ in f)
    f.close()

    # Get a random subset of lines
    logger.info(f'Randomly sampling {args.pct} of corpus with '
                f'a seed of {args.seed}')
    np.random.seed(args.seed)
    selection_index = (
        np.random.choice([0, 1], size=(n_lines,),
                         p=[1 - args.pct, args.pct])
        .astype(bool)
    )
    return selection_index


def select_similar(args) -> np.array:
    # Docs are '\n'-delimited text

    # Create a partial-ed function for conciseness
    to_term_dist = (
        lambda texts: docs_to_term_dist(texts,
                                        vocab_file=args.vocab_file,
                                        lowercase=args.lowercase,
                                        chunk_size=args.chunk_size)
    )

    # Get term distribution for fine-tune dataset
    # Chain all FT docs into one huge doc to obtain a
    # proper normalized term distribution
    f = get_file_obj(args.fine_tune_text)
    ft_text = [' '.join(line.strip() for line in f)]
    ft_term_dist = next(to_term_dist(ft_text))
    f.close()

    # Get term distribution for each doc in the corpus
    corpus_f = get_file_obj(args.corpus)
    corpus_term_dists = to_term_dist(corpus_f)

    # Calculate similarity for each doc in corpus
    similarities = pd.Series(
        similarity.similarity_name2value(args.sim_func,
                                         ft_term_dist, doc_term_dist)
        for doc_term_dist in tqdm(corpus_term_dists,
                                  desc=f'Computing {args.sim_func} similarities')
    )
    corpus_f.close()

    # Create the selection index
    selection_index = np.zeros((len(similarities)), dtype=bool)
    if args.threshold is None:
        if args.pct is not None:
            n_docs = int(len(similarities) * args.pct)
        else:
            n_docs = args.n_docs
        doc_indices = (
            similarities
            .sort_values(ascending=args.invert)
            .index[:n_docs]
        )
    else:
        # Select documents with similarities above `args.threshold`
        # If args.invert is provided, then select those below `args.threshold`
        predicate = (
            (similarities >= args.threshold)
            if not args.invert else
            (similarities <= args.threshold)
        )
        doc_indices = similarities[predicate].index

    for doc_index in doc_indices:
        selection_index[doc_index] = True
    return selection_index


def main(args):
    # Parse filename if not provided
    if args.filename is None:
        args.filename = parse_filename(args)

    if args.mode == 'random':
        selection_index = select_random(args)
    elif args.mode == 'similar':
        selection_index = select_similar(args)
    else:
        raise NotImplementedError

    # Create subset of corpus and writes it to args.dst
    copy_selected_docs(selection_index, args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
