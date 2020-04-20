"""Module containing featurization functions."""
import sys
import itertools as it
from pathlib import Path
from types import SimpleNamespace
from typing import List, Iterable, Union, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import BertWordPieceTokenizer

sys.path.append('learn-to-select-data')
from similarity import get_term_dist
from src.utils.iter import batch

from .io import get_docs


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

    Keyword Arguments:
        lowercase {bool} -- If True, performs lowercasing (default: {True})
        chunk_size {int} -- Tokenization batch size (default: {2**13})

    Returns:
        Iterable[List[str]] -- A tokenized corpus
    """
    special_tokens = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')
    tokenizer = BertWordPieceTokenizer(str(vocab_file), lowercase=lowercase)

    # Tokenize each document, filtering out special tokens
    return (
        [token for token in enc.tokens[1:-1] if token not in special_tokens]
        for b in batch(docs, chunk_size)
        for enc in tokenizer.encode_batch(list(b))
    )


def tokens_to_term_dist(tokenized: Iterable[List[str]],
                        vocab_file: Path,
                        lowercase: bool = True,
                        level: str = 'corpus',
                       ) -> Union[np.ndarray, Iterable[np.ndarray]]:
    """Convert tokens into term distributions.

    The tokens are converted into a distribution based on vocab available
    in `vocab_file`.

    Arguments:
        tokenized {Iterable[List[str]]} -- Tokenized documents
        vocab_file {Path} -- Path to vocabulary file

    Keyword Arguments:
        lowercase {bool} -- If True, perform lowercasing (default: {True})
        level {str} -- Level at which to form term distribution.
                       Valid values are {"corpus", "doc"}. If "corpus", create
                       a corpus-level term distribution. If "doc", create
                       document-level term distributions (default: {'corpus'})

    Raises:
        ValueError: If an invalid value for `level` is provided

    Returns:
        Union[np.ndarray, Iterable[np.ndarray]] -- The term distribution(s)
    """
    # Create a duck-typed Vocabulary object to work on `similarity.get_term_dist`
    vocab_obj = create_vocab(vocab_file)

    # Convert documents to doc/corpus-level term distributions
    if level == 'corpus':
        term_dist: np.ndarray = get_term_dist(tokenized,
                                              vocab=vocab_obj,
                                              lowercase=lowercase)
    elif level == 'doc':
        term_dist: Iterable[np.ndarray] = (  # type: ignore
            get_term_dist([x], vocab=vocab_obj, lowercase=lowercase)
            for x in tokenized
        )
    else:
        raise ValueError
    return term_dist


def tokens_to_tfidf(tokenized: Iterable[Iterable[str]],
                    vectorizer: TfidfVectorizer,
                    level: str,
                   ) -> np.ndarray:
    """Convert tokens to a TF-IDF vector(s).

    Arguments:
        tokenized {Iterable[Iterable[str]]} -- Tokenized documents
        vectorizer {TfidfVectorizer} -- A fitted sklearn TF-IDF Vectorizer
        level {str} -- Level at which to form TF-IDF vector.
                       Valid values are {"corpus", "doc"}. If "corpus", create
                       a corpus-level TF-IDF vector. If "doc", create
                       document-level TF-IDF vectors (default: {'corpus'})

    Returns:
        np.ndarray
            -- A TF-IDF vector. Given N documents and a TF-IDF vocabulary of
               length V, the output has a shape of (N,) if level=='corpus'
               otherwise (N, V).

    """
    if level == 'corpus':
        tokenized = it.chain.from_iterable(tokenized)
        return vectorizer.transform([tokenized]).toarray().squeeze()
    return vectorizer.transform(tokenized).toarray()


def get_fitted_tfidf_vectorizer(file: Path,
                                norm: Optional[str] = 'l1',
                               ) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer.

    Arguments:
        file {Path} -- File containing documents to fit vectorizer

    Keyword Arguments:
        norm {Optional[str]} -- If not None, normalize TF-IDF vectors either by
                                L1 or L2 norm (default: {'l1'})

    Returns:
        TfidfVectorizer -- A fitted TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(lowercase=False, token_patern=None,
                                 norm=norm, tokenizer=lambda x: x)
    docs = docs_to_tokens(get_docs(file), 
    vectorizer.fit(tokenized)
    return vectorizer
