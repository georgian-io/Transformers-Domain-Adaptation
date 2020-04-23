"""Module to test vectorized implementation of diversity functions."""
from types import SimpleNamespace

import numpy as np

import pytest

import sys
sys.path.append('learn-to-select-data')
import similarity
import features
from constants import DIVERSITY_FEATURES


CORPUS = ('apples are sweet', 'this is a similar document')
VOCABULARY = ('a', 'apples', 'are',
              'document', 'fine-tune', 'is',
              'similar', 'sweet', 'this')


@pytest.fixture(scope='session')
def vocab_obj():
    """Create a duck-type Vocabulary object to `get_term_dist`."""
    vocab = SimpleNamespace()
    vocab.size = len(VOCABULARY)
    vocab.word2id = dict(map(reversed, enumerate(VOCABULARY)))
    return vocab


@pytest.fixture(scope='session')
def tokenized_corpus():
    """Tokenize the corpus."""
    return [x.split(' ') for x in CORPUS]


@pytest.fixture(scope='session')
def corpus_dist(tokenized_corpus, vocab_obj):
    """Return a distribution for each document in the corpus."""
    return similarity.get_term_dist(tokenized_corpus, vocab_obj)


@pytest.fixture(scope='session')
def word_vectors(vocab_obj):
    """Create random word vectors for each term in the vocabulary."""
    vectors = np.random.normal(size=(len(VOCABULARY), 10))
    return dict(zip(vocab_obj.word2id, vectors))


@pytest.mark.parametrize('div_func', DIVERSITY_FEATURES)
def test_diversity_fast_correctness(div_func, tokenized_corpus, corpus_dist,
                                    vocab_obj, word_vectors):
    """Ensure that the fast implementation has similar results as the original."""
    res1 = [
        features.diversity_feature_name2value(
            div_func, doc, corpus_dist, vocab_obj.word2id, word_vectors)
            for doc in tokenized_corpus
    ]
    res2 = [
        features.diversity_feature_name2value_fast(
            div_func, doc, corpus_dist, vocab_obj.word2id, word_vectors)
            for doc in tokenized_corpus
    ]
    assert np.allclose(res1, res2)
