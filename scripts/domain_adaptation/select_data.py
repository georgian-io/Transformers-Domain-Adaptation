"""Script to select subset of corpus for downstream domain adaptation.

Shuffling is not done here as that is handled by the domain pre-training script.
"""
import sys
import argparse
import logging
import itertools as it
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from typing import List, Iterable, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from sklearn.preprocessing import MinMaxScaler

from src.utils.iter import batch
sys.path.append('learn-to-select-data')
import similarity
import features as diversity
from constants import SIMILARITY_FUNCTIONS, DIVERSITY_FEATURES


DIVERSITY_FUNCTIONS = [f for f in DIVERSITY_FEATURES if f != 'quadratic_entropy']
logger = logging.getLogger(__name__)


def parse_args(raw_args: Optional[List[str]] = None):
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        "Script to select subset of corpus for downstream domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--corpus', type=Path, required=True, help='Corpus')
    parser.add_argument('--dst', type=Path, required=True,
                        help='Directory to save corpus subset')
    parser.add_argument('--filename', type=str, default=None,
                        help='Filename for corpus subset')
    parser.add_argument('--ignore-cache', action='store_true',
                        help='If True, do not use cache '
                             '(applies to metric-based techniques).')

    # Args for "random" mode
    subparsers = parser.add_subparsers(help='Method to select subset of data',
                                       dest='mode')
    subparser = subparsers.add_parser('random', help='Randomly select data')
    subparser.add_argument('-p', '--pct', type=float, required=True,
                           help='Percentage of data to select w.r.t corpus size')
    subparser.add_argument('-s', '--seed', type=int, default=None,
                           help='Random seed for reproducability')

    # Create metric parser which holds shared args for all child metric subparsers
    metric_parser = argparse.ArgumentParser(add_help=False)

    group = metric_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pct', type=float,
                       help='Percentage of data to select w.r.t. corpus size')
    group.add_argument('-n', '--n-docs', type=int,
                       help='Number of documents to select')
    group.add_argument('-t', '--threshold', type=float,
                       help='Select documents with scores above '
                            '(or below if --invert is supplied)')

    metric_parser.add_argument('-i', '--invert', action='store_true',
                               help='If provided, pick documents with lowest '
                                    'scores instead')
    metric_parser.add_argument('--no-lowercase',
                               action='store_false', dest='lowercase',
                               help='If provided, will not perform lowercasing '
                                    'during tokenization')
    metric_parser.add_argument('-v', '--vocab-file', type=Path,
                               default='bert-base-uncased-vocab.txt',
                               help='Vocabulary for tokenization')
    metric_parser.add_argument('-c', '--chunk-size', type=int, default=2**13,
                               help='Tokenization chunk size')

    # Args for "similarity" mode
    subparser = subparsers.add_parser(
        'similar', parents=[metric_parser],
        help='Select data based on token similarity'
    )
    subparser.add_argument('--fine-tune-text', type=Path, required=True,
                           help='Path to fine tuning (training) text. '
                                'Similarity of individual documents in corpus '
                                'will be compard against this text.' )
    subparser.add_argument('--sim-func', choices=SIMILARITY_FUNCTIONS,
                           default='jensen-shannon',
                           help='Similarity function to use')

    # Args for "diverse" mode
    subparser = subparsers.add_parser(
        'diverse', parents=[metric_parser],
        help='Select data based on token diversity'
    )
    subparser.add_argument('--div-func', choices=DIVERSITY_FUNCTIONS,
                           default='entropy',
                           help='Diversity function to use')

    # Args for "similar+diverse" mode
    subparser = subparsers.add_parser(
        'similar+diverse', parents=[metric_parser],
        help='Select data based on token diversity'
    )
    subparser.add_argument('--fine-tune-text', type=Path, required=True,
                           help='Path to fine tuning (training) text. '
                                'Similarity of individual documents in corpus '
                                'will be compard against this text.' )
    subparser.add_argument('--sim-func', choices=SIMILARITY_FUNCTIONS,
                           default='jensen-shannon',
                           help='Similarity function to use')
    subparser.add_argument('--div-func', choices=DIVERSITY_FUNCTIONS,
                           default='entropy',
                           help='Diversity function to use')
    subparser.add_argument('--fuse-by', choices=('linear_combination', 'union'),
                           default='linear_combination',
                           help='Method by which to combine similarity and '
                                'diversity metrics. "linear_combination" '
                                'combines both metrics using linear combination '
                                'of weights supplied in `--sim-div-weights`. '
                                'When using "union", the final corpus subset is '
                                'a union of the most/least similar set of docs '
                                'with the most/least diverse set of docs. '
                                'As there may be overlap, the final corpus '
                                'subset would likely differ from the specified '
                                'amount of documents.')
    subparser.add_argument('-w', '--sim-div-weights', default=[1, 1],
                           type=lambda x: [float(w) for w in x.split(',')],
                           help='Weights for similarity and diversity metrics. '
                                'Applies only if --fuse-by="linear_combination". '
                                'Provide as a comma-separated string. '
                                'For example, to compute 2 * sim + 0.5 * div, '
                                'specify --sim-div-weights=2,0.5')

    args = parser.parse_args(args=raw_args)

    # Check validity of corpus
    if not args.corpus.exists():
        raise FileNotFoundError(f'The corpus {args.corpus} does not exist')
    elif args.corpus.stat().st_size == 0:
        raise ValueError(f'The corpus {args.corpus} is empty.')

    # Check validity of args.pct, if specified
    if args.pct is not None and not 0 < args.pct <= 1:
        raise ValueError(f'Invalid percentage value of {args.pct} provided')

    return args


def parse_filename(args: argparse.Namespace) -> str:
    """Parse filename based on data selection mode."""
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
    elif args.mode == 'diverse':
        filename += '_most_diverse' if not args.invert else '_least_diverse'
        filename += f'_{args.div_func}'
    elif args.mode == 'similar+diverse':
        filename += '_most_sim_div' if not args.invert else '_least_sim_div'
        sim_weight, div_weight = [str(w).replace('.', ',')
                                  for w in args.sim_div_weights]
        filename += f'_{sim_weight}_{args.sim_func}_{div_weight}_{args.div_func}'
        filename += f'_{args.fine_tune_text.stem}'
        filename += f'_{args.fuse_by}'
    else:
        raise NotImplementedError

    # Append subset size parameter for non-random data selection method
    if args.mode != 'random':
        if args.pct is not None:
            filename += f'_{int(100 * args.pct)}pct'
        elif args.n_docs is not None:
            filename += f'_{args.n_docs}docs'
        elif args.threshold is not None:
            filename += f'_{args.threshold}threshold'
    filename += args.corpus.suffix
    return filename


def get_file_obj(filepath: Union[str, Path]):
    """Return a file object for streaming."""
    logger.info(f'Reading {filepath}')

    # Get the total number of lines for processing time estimates
    with open(filepath) as f:
        n_lines = sum(1 for _ in f)

    return tqdm(open(filepath), desc='Reading', leave=False, total=n_lines)


def copy_selected_docs(index: np.ndarray, args: argparse.Namespace) -> None:
    """Create a subset corpus by copying selected documents."""
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


def create_vocab(vocab_file: Path) -> SimpleNamespace:
    """Create a duck-type Vocabulary object.

    The Vocabulary object is a user-defined object from the
    `learn-to-select-data` repo. It is used in `similarity.get_term_dist`.

    Arguments:
        vocab_file {Path} -- Path to vocabulary file

    Returns:
        SimpleNamespace -- A duck-typed Vocabulary object
    """
    # Create a duck-typed Vocabulary object to work on `similarity.get_term_dist`.
    vocab = vocab_file.read_text().splitlines()
    vocab_obj = SimpleNamespace()
    vocab_obj.size = len(vocab)
    vocab_obj.word2id = {word: i for i, word in enumerate(vocab)}
    return vocab_obj


def docs_to_tokens(docs: Iterable[str],
                   vocab_file: Path,
                   lowercase: bool = True,
                   chunk_size: int = 2**13,
                  ) -> Iterable[List[str]]:
    """Tokenize documents.

    Arguments:
        docs {Iterable[str]} -- Documents
        vocab_file {Path} -- Path to vocabulary file
        lowercase {bool} -- If True, performs lowercasing
        chunk_size {int} -- Tokenization batch size

    Returns:
        Iterable[List[str]] -- A tokenized corpus
    """
    special_tokens = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')
    tokenizer = BertWordPieceTokenizer(str(vocab_file), lowercase=lowercase)

    # Tokenize each document
    tokenized: Iterable[List[str]] = (
        enc.tokens[1:-1]
        for b in batch(docs, chunk_size)
        for enc in tokenizer.encode_batch(list(b))
    )

    # Filter out possible unknown tokens
    tokenized = ([x for x in tokens if x not in special_tokens] for tokens in tokenized)
    return tokenized


def docs_to_term_dist(docs: Iterable[str],
                      vocab_file: Path,
                      lowercase: bool = True,
                      chunk_size: int = 2**13,
                      level: str = 'corpus'
                     ) -> Union[np.ndarray, Iterable[np.ndarray]]:
    """Convert documents into term (token) distributions.

    This is done by first tokenizing documents and then converting them into
    distributions based on vocab available in `vocab_file`.

    Arguments:
        docs {Iterable[str]} -- Documents
        vocab_file {Path} -- Path to vocabulary file
        lowercase {bool} -- If True, perform lowercasing
        chunk_size {int} -- Tokenization batch size

    Keyword Arguments:
        level {str} -- Level at which to form term distribution.
                       Valid values are {"corpus", "doc"}. If "corpus", create
                       a corpus-level term distribution. If "doc", create
                       a document-level term distribution (default: {'corpus'})

    Raises:
        ValueError: If an invalid value for `level` is provided

    Returns:
        Union[np.ndarray, Iterable[np.ndarray]] -- The term distribution(s)
    """
    tokenized = docs_to_tokens(docs=docs,
                               vocab_file=vocab_file,
                               lowercase=lowercase,
                               chunk_size=chunk_size)

    # Create a duck-typed Vocabulary object to work on `similarity.get_term_dist`
    vocab_obj = create_vocab(vocab_file)

    if level == 'corpus':
        # Convert tokenized corpus into a corpus-level term distribution
        term_dist: np.ndarray = (
            similarity.get_term_dist(tokenized, vocab=vocab_obj, lowercase=lowercase)
        )
    elif level == 'doc':
        # Convert tokenized docs into doc-level term distributions
        term_dist: Iterable[np.ndarray] = (  # type: ignore
            similarity.get_term_dist([x], vocab=vocab_obj, lowercase=lowercase)
            for x in tokenized
        )
    else:
        raise ValueError
    return term_dist


def calculate_similarity(args: argparse.Namespace) -> pd.Series:
    """Compute similarities of documents on a fine-tune corpus."""
    # Create a partial-ed function for conciseness
    to_term_dist = partial(docs_to_term_dist,
                           vocab_file=args.vocab_file,
                           lowercase=args.lowercase,
                           chunk_size=args.chunk_size)

    # Get term distribution for fine-tune dataset
    # Chain all FT docs into one huge doc to obtain a
    # proper normalized term distribution
    f = get_file_obj(args.fine_tune_text)
    ft_text = [' '.join(line.strip() for line in f)]
    ft_term_dist = to_term_dist(ft_text, level="corpus")
    f.close()

    # Get term distribution for each doc in the corpus
    corpus_f = get_file_obj(args.corpus)
    corpus_term_dists = to_term_dist(corpus_f, level="doc")

    # Calculate similarity for each doc in corpus
    similarities = pd.Series(
        similarity.similarity_name2value(args.sim_func,
                                         ft_term_dist, doc_term_dist)
        for doc_term_dist in tqdm(corpus_term_dists,
                                  desc=f'Computing {args.sim_func} similarities')
    )
    corpus_f.close()

    return similarities


def calculate_diversity(args: argparse.Namespace) -> pd.Series:
    """Compute intra-document diversity."""
    # Get term distribution for each doc in the corpus
    corpus_f = get_file_obj(args.corpus)
    corpus_f1, corpus_f2 = it.tee(corpus_f)

    # Tokenize the corpus
    corpus = docs_to_tokens(corpus_f1,
                            vocab_file=args.vocab_file,
                            lowercase=args.lowercase,
                            chunk_size=args.chunk_size)

    # Get a document-level term distribution
    doc_term_dist = docs_to_term_dist(corpus_f2,
                                      vocab_file=args.vocab_file,
                                      lowercase=args.lowercase,
                                      chunk_size=args.chunk_size,
                                      level='corpus')

    # Calculate diversity for each doc in the corpus
    word2id = create_vocab(args.vocab_file).word2id
    diversity_scores = pd.Series(
        diversity.diversity_feature_name2value(args.div_func, example=doc,
                                               train_term_dist=doc_term_dist,
                                               word2id=word2id, word2vec='')
        for doc in corpus
    )
    corpus_f.close()

    return diversity_scores


def _rank_metric_and_select(scores: pd.Series,
                            args: argparse.Namespace) -> np.ndarray:
    """
    Rank metrics and select top (or bottom) values.

    Called by metric-based selection methods.
    """
    # Create the selection index
    selection_index = np.zeros((len(scores)), dtype=bool)
    if args.threshold is None:
        if args.pct is not None:
            n_docs = int(len(scores) * args.pct)
        else:
            n_docs = args.n_docs
        doc_indices = (
            scores
            .sort_values(ascending=args.invert)
            .index[:n_docs]
        )
    else:
        # Select documents with scores above `args.threshold`
        # If args.invert is provided, then select those below `args.threshold`
        predicate = (
            (scores >= args.threshold)
            if not args.invert else
            (scores <= args.threshold)
        )
        doc_indices = scores[predicate].index

    for doc_index in doc_indices:
        selection_index[doc_index] = True
    return selection_index


def select_random(args: argparse.Namespace) -> np.ndarray:
    """Randomly select documents."""
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


def select_similar(args: argparse.Namespace) -> np.ndarray:
    """Select documents that are most / least similar to a fine-tuning corpus."""
    cache_path = (
        args.dst / 'cache' / f'similar_{args.corpus.stem}_{args.sim_func}_'
                             f'{args.fine_tune_text.stem}.pkl'
    )
    if not args.ignore_cache and cache_path.exists():
        logger.info(f'Using cache found at {cache_path}')
        similarities = pd.read_pickle(cache_path)
    else:
        similarities = calculate_similarity(args)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        similarities.to_pickle(cache_path)
        logger.info(f'Cached similarity scores at {cache_path}')
    return _rank_metric_and_select(similarities, args)


def select_diverse(args: argparse.Namespace) -> np.ndarray:
    """Select documents that are most / least diverse."""
    cache_path = args.dst / 'cache' / f'diverse_{args.corpus.stem}.pkl'

    if not args.ignore_cache and cache_path.exists():
        logger.info(f'Using cache found at {cache_path}')
        diversity_scores = pd.read_pickle(cache_path)
    else:
        diversity_scores = calculate_diversity(args)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        diversity_scores.to_pickle(cache_path)
        logger.info(f'Cached diversity scores at {cache_path}')
    return _rank_metric_and_select(diversity_scores, args)


def select_similar_and_diverse(args: argparse.Namespace) -> np.ndarray:
    """Select documents that are most / least (similar + diverse)."""
    # Parse cache file paths
    cache_dir = args.dst / 'cache'
    sim_cache = (
        cache_dir / f'similar_{args.corpus.stem}_{args.sim_func}_'
                    f'{args.fine_tune_text.stem}.pkl'
    )
    div_cache = cache_dir / f'diverse_{args.corpus.stem}.pkl'

    if not args.ignore_cache and sim_cache.exists():
        logger.info(f'Using similarity scores cache found at {sim_cache}')
        similarities = pd.read_pickle(sim_cache)
    else:
        similarities = calculate_similarity(args)
        sim_cache.parent.mkdir(exist_ok=True, parents=True)
        similarities.to_pickle(sim_cache)
        logger.info(f'Cached similarity scores at {sim_cache}')

    if not args.ignore_cache and div_cache.exists():
        logger.info(f'Using diversity scores cache found at {div_cache}')
        diversity_scores = pd.read_pickle(div_cache)
    else:
        diversity_scores = calculate_diversity(args)
        div_cache.parent.mkdir(exist_ok=True, parents=True)
        diversity_scores.to_pickle(div_cache)
        logger.info(f'Cached diversity scores at {div_cache}')


    # Calculate composite metric
    if args.fuse_by == 'linear_combination':
        # Ensure metrics are on the same scale before combining them
        similarities = (
            MinMaxScaler().fit_transform(similarities.values.reshape(-1, 1))
        )
        diversity_scores = (
            MinMaxScaler().fit_transform(diversity_scores.values.reshape(-1, 1))
        )

        # Calculate composite metric using linear combination
        sim_weight, div_weight = args.sim_div_weights
        composite_metric = sim_weight * similarities + div_weight * diversity_scores
        composite_metric = pd.Series(composite_metric.ravel())

        return _rank_metric_and_select(composite_metric, args)
    else:
        sim_selection_index = _rank_metric_and_select(similarities, args)
        div_selection_index = _rank_metric_and_select(diversity_scores, args)

        # Get composite selection index using set union
        composite_selection_index = sim_selection_index | div_selection_index
        return composite_selection_index


def main(args: argparse.Namespace):
    """Execute script."""
    # Parse filename if not provided
    if args.filename is None:
        args.filename = parse_filename(args)

    if args.mode == 'random':
        selection_index = select_random(args)
    elif args.mode == 'similar':
        selection_index = select_similar(args)
    elif args.mode == 'diverse':
        selection_index = select_diverse(args)
    elif args.mode == 'similar+diverse':
        selection_index = select_similar_and_diverse(args)
    else:
        raise NotImplementedError

    # Create subset of corpus and writes it to args.dst
    copy_selected_docs(selection_index, args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
