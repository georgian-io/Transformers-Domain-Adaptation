"""BERT Tokenizer update functionality."""
import logging
import argparse
from typing import List, Union

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import BertWordPieceTokenizer


logger = logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        "Augment BERT's vocabulary with relevant in-domain tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--bert-vocab', type=str, required=True,
                        help='Path to original BERT vocabulary text file.')
    parser.add_argument('--corpus', required=True,
                        type=lambda paths: paths.split(','),
                        help='Path to in-domain corpus text file.')
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
    return parser.parse_args()


def train_tokenizer(corpus: Union[str, List[str]],
                    vocab_size: int = 30519,
                    lowercase: bool = True,
                    save_vocab: bool = False,
                    in_domain_vocab: str = 'in-domain-vocab.txt',
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

    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)

    logging.info('Training new WordPiece tokenizer on in-domain corpora...')
    tokenizer.train(corpus, vocab_size=vocab_size)

    if save_vocab:
        tokenizer.save('.', in_domain_vocab.rsplit('-', 1)[0])
        logging.info(f'Saved trained tokenizer to {in_domain_vocab}')
    return tokenizer


def tokenize(texts: List[str],
             tokenizer: Union[str, BertWordPieceTokenizer],
             lowercase: bool = True,
             flat_map: bool = False,
            ) -> Union[List[str],
                       List[List[str]]]:
    """Tokenize texts using BERT WordPiece tokenizer implemented in Rust.

    Arguments:
        texts {List[str]} -- Text data to tokenize
        tokenizer {Union[str, BertWordPieceTokenizer]}
            -- Path to tokenizer vocabulary.
               Alternatively, supply a BertWordPieceTokenizer from the
               `tokenizers` library
        lowercase {bool} -- If True, perform lowercasing.
        flat_map {bool} -- If True, flat maps results into a List[str],
                           instead of List[List[str]].

    Returns:
        A tokenized string or a list of tokenized string.
    """
    # Instantiate the tokenizer
    if isinstance(tokenizer, str):
        tokenizer = BertWordPieceTokenizer(str(tokenizer), lowercase=lowercase)
    elif not hasattr(tokenizer, 'encode_batch'):
        raise AttributeError(f'Provided `tokenizer` is not from `tokenizers` '
                             'library.')

    logging.info('Tokenizing texts...')
    if flat_map:
        tokenized = [t for enc in tokenizer.encode_batch(texts)
                       for t in enc.tokens]
    else:
        tokenized = [enc.tokens for enc in tokenizer.encode_batch(texts)]
    return tokenized


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

    logging.info(f'Ranking tokens by {mode}...')
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
    logging.info('Updating vocabulary...')
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
    assert len(unused_tokens) <= len(top_terms)  # TODO Handle the inverse situation
    mapping = dict(zip(unused_tokens, top_terms))

    # Update original vocab with the next top term is the token is '[unused*]'
    for i, ori_term in enumerate(vocab):
        if ori_term in mapping:
            vocab[i] = mapping[ori_term]

    # Saves vocab
    with open(updated_vocab_path, 'w+') as f:
        f.write('\n'.join(vocab))
    logging.info(f'Updated vocabulary saved at {updated_vocab_path}')


if __name__ == '__main__':
    args = parse_args()

    # Train and save in-domain corpora as text file
    logging.info('\n')
    tokenizer = train_tokenizer(args.corpus, lowercase=args.lowercase)

    # Load corpus / corpora
    corpus = []
    for c in args.corpus:
        with open(c) as f:
            corpus += [x.strip() for x in f.readlines()]

    # Tokenize corpus
    tokenized_corpus = tokenize(corpus,
                                tokenizer=tokenizer,
                                lowercase=args.lowercase)

    # Rank tokens
    ranked_tokens = rank_tokens(tokenized_corpus, mode=args.rank_by)

    # Save augmented vocabulary to text file
    create_updated_vocab_txt(ranked_tokens,
                             ori_vocab_path=args.bert_vocab,
                             updated_vocab_path=args.dst)
