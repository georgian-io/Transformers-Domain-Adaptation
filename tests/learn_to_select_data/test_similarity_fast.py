"""Module to test vectorized implementation of similarity functions."""
from types import SimpleNamespace

import numpy as np

import pytest

import sys
sys.path.append('learn-to-select-data')
import similarity
import similarity_fast
from constants import SIMILARITY_FUNCTIONS


FINE_TUNE_TEXT = ('this is a fine-tune document')
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
def corpus_dists(vocab_obj):
    """Return a distribution for each document in the corpus."""
    corpus = [x.split(' ') for x in CORPUS]
    return [similarity.get_term_dist([doc], vocab_obj) for doc in corpus]


@pytest.fixture(scope='session')
def fine_tune_text_dist(vocab_obj):
    """Return a fine tune text distribution."""
    fine_tune_text = FINE_TUNE_TEXT.split(' ')
    return similarity.get_term_dist([fine_tune_text], vocab_obj)


@pytest.mark.parametrize('sim_func', SIMILARITY_FUNCTIONS)
def test_similarity_fast_correctness(sim_func, fine_tune_text_dist, corpus_dists):
    """Ensure that the fast implementation has similar results as the original."""
    res1 = [similarity.similarity_name2value(sim_func, fine_tune_text_dist, dist)
            for dist in corpus_dists]
    res2 = similarity.similarity_name2value_fast(
        sim_func, fine_tune_text_dist.reshape(1, -1), corpus_dists
    )
    assert np.allclose(res1, res2)
