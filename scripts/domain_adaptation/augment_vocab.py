"""BERT vocabulary update functionality."""
import logging
import argparse
import itertools as it
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Iterable, Union, Optional

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import BertWordPieceTokenizer

from src.utils.iter import batch


logger = logging.getLogger(__name__)

VOCAB_CACHE_PREFIX = 'temp-in-domain'


def parse_args():
    parser = argparse.ArgumentParser(
        "Augment BERT's vocabulary with relevant in-domain tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--bert-vocab', type=str, required=True,
                        help='Path to original BERT vocabulary text file.')
    parser.add_argument('--corpus', required=True, type=Path,
                        help='Path to in-domain corpus text file. '
                             'Can be a text file, or a folder of text files.')
    parser.add_argument('--dst', type=str, required=True,
                        help='Directory to output the augmented '
                             'vocabulary text file.')
    parser.add_argument('--no-lowercase',
                        action='store_false', dest='lowercase',
                        help='If provided, will not perform lowercasing of '
                             'corpus.')
    parser.add_argument('--vocab-size', type=int, default=30519,
                        help='Vocabulary size of newly trained '
                             'WordPieceTokenizer')
    parser.add_argument('--rank-by', choices=('count', 'tfidf'),
                        default='count', help='Ranking heuristic')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='If provided, train tokenizer vocabulary from '
                             'scratch.')
    parser.add_argument('-t', '--tokenization-batch-size',
                        type=int, default=2**11,
                        help='Number of lines to tokenize each time. '
                             'Larger values lead to time savings at the '
                             'expense of larger memory requirements.')

    args = parser.parse_args()

    if not args.corpus.exists():
        raise ValueError(f'Specified corpus value of {args.corpus} '
                          'does not exist.')
    if args.corpus.is_dir():
        args.corpus = [str(x) for x in args.corpus.rglob('*.txt')]
    else:  # When it is a file
        args.corpus = [str(args.corpus)]

    return args


def train_tokenizer(corpus: Union[str, List[str]],
                    vocab_size: int = 30519,
                    overwrite: bool = True,
                    lowercase: bool = True,
                    save_vocab: bool = False,
                    dst: Optional[str] = None,
                    in_domain_vocab: str = VOCAB_CACHE_PREFIX,
                   ) -> BertWordPieceTokenizer:
    """Train a WordPiece tokenizer from scratch.

    Arguments:
        corpus {Union[str, List[str]]} -- In-domain corpus / corpora

    Keyword Arguments:
        vocab_size {int} -- Size of trained vocabulary (default: 30519)
        lowercase {bool} -- If True, perform lowercasing (default: True)
        save_vocab {bool} -- If True, save vocab to `in_domain_vocab`
                             (default: Fakse)
        in_domain_vocab {str} -- Path to save trained tokenizer vocabulary
                                 (default: {'in-domain-vocab.txt'})

    Returns:
        A BertWordPieceTokenizer trained on in-domain corpora.
    """
    if not isinstance(corpus, list):
        corpus = [corpus]

    # Load cached vocab if possible
    if not overwrite:
        cached_vocab = Path(dst) / (VOCAB_CACHE_PREFIX + '-vocab.txt')

        if cached_vocab.exists():
            logger.info(f'Loading cached vocabulary at {cached_vocab}')
            return BertWordPieceTokenizer(str(cached_vocab))
        else:
            logger.info(f'Cached vocabulary not found at {cached_vocab}')

    # Train tokenizer
    logger.info('Training new WordPiece tokenizer on in-domain corpora')
    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)
    tokenizer.train(corpus, vocab_size=vocab_size)

    if save_vocab:
        tokenizer.save('.' if dst is None else dst, in_domain_vocab)
        logger.info('Saved in-domain vocabulary to '
                    f'{Path(dst) / (in_domain_vocab + "-vocab.txt")}')
    return tokenizer


def tokenize(texts: List[str],
             tokenizer: BertWordPieceTokenizer,
             flat_map: bool = False,
            ) -> Union[List[str],
                       List[List[str]]]:
    """Tokenize texts using BERT WordPiece tokenizer implemented in Rust.

    Arguments:
        texts {List[str]} -- Text data to tokenize
        tokenizer {BertWordPieceTokenizer}
            -- A BertWordPieceTokenizer from the `tokenizers` library
        flat_map {bool} -- If True, flat maps results into a List[str],
                           instead of List[List[str]].

    Returns:
        A tokenized string or a list of tokenized string.
    """
    # Instantiate the tokenizer
    if not hasattr(tokenizer, 'encode_batch'):
        raise AttributeError(f'Provided `tokenizer` is not from `tokenizers` '
                             'library.')

    if flat_map:
        tokenized = [t for enc in tokenizer.encode_batch(texts)
                       for t in enc.tokens]
    else:
        tokenized = [enc.tokens for enc in tokenizer.encode_batch(texts)]
    return tokenized


def fused_tokenize_and_rank(corpora: List[str],
                            tokenizer: BertWordPieceTokenizer,
                            batch_size: int
                           ) -> List[str]:
    def read_text_with_logging(corpus: str) -> List[str]:
        logger.info(f'Loading text from {corpus}')
        return Path(corpus).read_text(encoding="utf-8").splitlines()

    lines: Iterable[str] = it.chain.from_iterable(read_text_with_logging(c)
                                                  for c in corpora)
    batches: Iterable[Tuple[str]] = batch(lines, batch_size)
    token_ids: Iterable[int] = it.chain.from_iterable(
        tokens.tokens[1:-1]
        for batch in batches
        for tokens in tokenizer.encode_batch(list(batch))
    )
    counts = Counter(tqdm(token_ids, desc='Counting tokens'))
    ranked_tokens = [token for token, _ in counts.most_common()]
    return ranked_tokens


def rank_tokens(tokenized_docs: List[List[str]],
                mode: str = 'count'
               ) -> List[str]:
    """Rank in-domain tokens.

    This ranking is used to decide which tokens are used to replcae the
    [USUSED*] tokens in BERT's vocabulary.

    Ranking heuristic:
        "count" -- Rank tokens by freq. of occurence in desc. order
        "tfidf" -- Rank tokens by TFIDF in desc. order

    Arguments:
        tokens {List[str]} -- The tokenized corpus. Inner list represents
                              a tokenized document in the corpus.

    Keyword Arguments:
        mode {str} -- Ranking heuristic. Choose between {'count' and 'tfidf'}
                      (default: {'count'})

    Returns:
        List[str] -- A ranked list of tokens
    """
    MODES = ('count', 'tfidf')
    if mode not in MODES:
        raise ValueError(f'Invalid mode {mode} provided. '
                         f'Expecting value from {MODES}.')

    logger.info(f'Ranking tokens by {mode}')
    if mode == 'count':
        return _rank_tokens_by_count(tokenized_docs)
    else:
        return _rank_tokens_by_tfidf(tokenized_docs)


def _rank_tokens_by_count(tokenized_docs: List[List[str]]) -> List[str]:
    tokens = [t for tokens in tokenized_docs for t in tokens]
    tokens = pd.Series(tokens)
    ranked = tokens.value_counts().index.tolist()
    return ranked


def _rank_tokens_by_tfidf(tokenized_docs: List[List[str]]) -> List[str]:
    # Convert the list of tokens in each doc into a space-delimited string
    documents = pd.Series(tokenized_docs).apply(lambda x: ' '.join(x))

    # Fit a TfidfVectorizer
    tfidf = TfidfVectorizer(lowercase=False, max_features=5000,
                            use_idf=True, smooth_idf=True,
                            token_pattern='\S+')
    tfidf.fit(documents)

    # Get TFIDFs for corpora
    corpus_str = ' '.join([t for tokens in tokenized_docs for t in tokens])
    tfidfs = np.array(tfidf.transform([corpus_str]).todense()).squeeze()
    ranked = (
        pd.Series(tfidfs, index=sorted(tfidf.vocabulary_.keys(),
                                       key=lambda x: x.__getitem__))
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return ranked


def _get_stop_words() -> List[str]:
    """Load stop words from NLTK.

    Will attempt to download stopwords and reload up to 5 times.
    """
    num_retries = 5
    for _ in range(num_retries):
        try:
            stopwords = nltk.corpus.stopwords.words('english')
        except LookupError:  # Stopwords folder not yet downloaded
            nltk.download('stopwords')
            continue
        break
    else:
        raise ValueError(f'{num_retries} attempts at loading stopwords failed.')
    return stopwords


def create_updated_vocab_txt(top_terms: List[str],
                             ori_vocab_path: str,
                             updated_vocab_path: str,
                            ) -> None:
    """Update BERT tokenizer vocabulary with relevant in-domain tokens.

    This is done by replacing '[unused*]' tokens in BERT's vocabulary with
    in-domain terms that do not already exist in the existing vocabulary.

    Results are saved in a txt file.

    Arguments:
        top_terms {List[str]} -- Ranked in-domain terms in descending order

    Keyword Arguments:
        ori_vocab_path {str} -- Path to existing vocabulary txt file
        updated_vocab_path {str} -- Path to save updated vocabulary txt file
    """
    logger.info('Updating vocabulary')
    # Get stop words
    stopwords = _get_stop_words() + ["[CLS]", "[SEP]"]

    # Get original vocab
    with open(ori_vocab_path) as f:
        vocab = [x.strip() for x in f.readlines()]

    # Filter out tokens that are not stop words or part of existing vocab
    top_terms = [
        x
        for x in top_terms
        if (x not in stopwords and x not in vocab)
    ]

    # Create top term generator
    unused_tokens = [x for x in vocab if '[unused' in x]
    mapping = dict(zip(unused_tokens, top_terms))

    # Update original vocab with the next top term is the token is '[unused*]'
    for i, ori_term in enumerate(vocab):
        if ori_term in mapping:
            vocab[i] = mapping[ori_term]

    # Saves vocab
    with open(updated_vocab_path, 'w+') as f:
        f.write('\n'.join(vocab))
    logger.info(f'Updated vocabulary saved at {updated_vocab_path}')


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create directory
    Path(args.dst).mkdir(exist_ok=True, parents=True)

    # Train and save in-domain corpora as text file
    tokenizer = train_tokenizer(args.corpus,
                                vocab_size=args.vocab_size,
                                overwrite=args.overwrite_cache,
                                lowercase=args.lowercase,
                                dst=args.dst,
                                save_vocab=True)

    if args.rank_by == 'count':
        logger.info('Using fused tokenization and ranking for better performance')
        ranked_tokens = fused_tokenize_and_rank(
            args.corpus,
            tokenizer=tokenizer,
            batch_size=args.tokenization_batch_size
        )
    else:
        # Load corpus / corpora
        tokenized_corpus = []
        for c in args.corpus:
            logger.info(f'Tokenizing {c} with in-domain tokenizer')
            with open(c) as f:
                tokenized_corpus += tokenize(f.readlines(),
                                            tokenizer=tokenizer)

        # Rank tokens
        ranked_tokens = rank_tokens(tokenized_corpus, mode=args.rank_by)

    # Save augmented vocabulary to text file
    updated_vocab_path = (Path(args.dst) / 'vocab.txt').as_posix()
    create_updated_vocab_txt(ranked_tokens,
                             ori_vocab_path=args.bert_vocab,
                             updated_vocab_path=updated_vocab_path)
