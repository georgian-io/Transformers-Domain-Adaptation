"""Test data selection module."""
import shlex
from collections import Iterable

import pytest
import numpy as np

from scripts.domain_adaptation import select_data

CORPUS = [
    "apple apple apple",
    "apple banana apple",
    "apple banana document",
]
VOCABULARY = ['a', 'apple', 'are', 'banana', 'document', '##n', 'is',
              'it', '##s', 'this', 'these', '[UNK]']


@pytest.fixture(scope='session')
def vocab_file(tmp_path_factory):
    """Create a vocabulary file shared by all test functions."""
    vocab = tmp_path_factory.mktemp('shared') / 'vocab.txt'
    vocab.write_text('\n'.join(VOCABULARY))
    return vocab


@pytest.fixture(scope='session')
def corpus_file(tmp_path_factory):
    """Create a corpus file shared by all test functions."""
    corpus = tmp_path_factory.mktemp('shared') / 'corpus.txt'
    corpus.write_text('\n'.join(CORPUS))
    return corpus


def test_corpus_does_not_exist(tmp_path):
    """Raise FileNotFoundError is corpus does not exist."""
    corpus = tmp_path / 'nonexistent.txt'
    args = shlex.split(f'--corpus {corpus} --dst {tmp_path} random -p 0.02')
    with pytest.raises(FileNotFoundError):
        select_data.parse_args(args)


def test_empty_corpus(tmp_path):
    """Raise ValueError when corpus exists but is empty."""
    corpus = tmp_path / 'nonexistent.txt'
    corpus.write_text('')
    args = shlex.split(f'--corpus {corpus} --dst {tmp_path} random -p 0.02')
    with pytest.raises(ValueError):
        select_data.parse_args(args)


def test_invalid_pct(tmp_path):
    """Raise ValueError when an invalid percentage is provided."""
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('This is a test corpus')
    args = shlex.split(f'--corpus {corpus} --dst {tmp_path} random -p 2')
    with pytest.raises(ValueError):
        select_data.parse_args(args)


def test_create_vocab(vocab_file):
    """Test whether the output `create_vocab` has the correct attributes."""
    vocab_obj = select_data.create_vocab(vocab_file)

    assert hasattr(vocab_obj, 'size')
    assert vocab_obj.size == len(VOCABULARY)

    assert hasattr(vocab_obj, 'word2id')
    assert vocab_obj.word2id == dict(map(reversed, enumerate(VOCABULARY)))


def test_docs_to_tokens_bert_uncased_type(vocab_file):
    """Test whether the output of `docs_to_tokens` has the correct types."""
    docs = ['This is a document']
    tokenized = select_data.docs_to_tokens(docs, vocab_file)
    assert isinstance(tokenized, Iterable)
    tokenized_doc = next(tokenized)
    assert isinstance(tokenized_doc, list)
    assert all(isinstance(x, str) for x in tokenized_doc)


def test_docs_to_tokens_bert_uncased_correctness(vocab_file):
    """Test the correctness of the output of `docs_to_tokens`."""
    docs = ['These are bananas', 'It is an apple']
    tokenized = select_data.docs_to_tokens(docs, vocab_file)
    tokenized = list(tokenized)

    assert tokenized == [['these', 'are', 'banana', '##s'],
                         ['it', 'is', 'a', '##n', 'apple']]


def test_docs_to_tokens_bert_uncased_no_special_tokens(vocab_file):
    """Test that `docs_to_tokens`'s output does not contain special tokens."""
    docs = ['[CLS] [PAD] [UNK] [SEP] [MASK]']
    tokenized = select_data.docs_to_tokens(docs, vocab_file)
    tokenized = list(tokenized)
    assert tokenized == [[]]


def test_docs_to_term_dist_level_doc_invalid_level(vocab_file):
    """Raise ValueError when an invalid level is provided."""
    docs = ['This is a document']
    with pytest.raises(ValueError):
        select_data.docs_to_term_dist(docs, vocab_file,
                                      level='invalid-level-#12313')


def test_docs_to_term_dist_level_corpus_type(vocab_file):
    """Ensure that output type is correct when level == "corpus"."""
    docs = ['This is a document']
    corpus_dist = select_data.docs_to_term_dist(docs, vocab_file, level='corpus')
    assert isinstance(corpus_dist, np.ndarray)
    assert corpus_dist.shape == (len(VOCABULARY),)


def test_docs_to_term_dist_level_corpus_correctness(vocab_file):
    """Ensure that output is correct when level == "corpus"."""
    docs = ['this is an apple', 'this is a banana this is a document']
    term_dist = select_data.docs_to_term_dist(docs, vocab_file, level='corpus')
    answers = np.array([3, 1, 0, 1, 1, 1, 3, 0, 0, 3, 0, 0]) / 13
    assert np.allclose(term_dist, answers)


def test_docs_to_term_dist_level_doc_type(vocab_file):
    """Ensure that output type is correct when level == "doc"."""
    docs = ['This is a document']
    term_dists = select_data.docs_to_term_dist(docs, vocab_file, level='doc')
    assert isinstance(term_dists, Iterable)

    term_dist = next(term_dists)
    assert isinstance(term_dist, np.ndarray)
    assert term_dist.shape == (len(VOCABULARY),)


def test_docs_to_term_dist_level_doc_correctness(vocab_file):
    """Ensure that output is correct when level == "doc"."""
    docs = ['this is an apple', 'this is a banana this is a document']
    term_dists = select_data.docs_to_term_dist(docs, vocab_file, level='doc')
    term_dists = list(term_dists)

    answers = [
        [0.2, 0.2, 0, 0, 0, 0.2, 0.2, 0, 0, 0.2, 0, 0],
        [0.25, 0, 0, 0.125, 0.125, 0, 0.25, 0, 0, 0.25, 0, 0],
    ]
    assert np.allclose(term_dists, answers)


def test_integration_select_random_pct_not_modified(tmp_path, corpus_file):
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} random -p 0.4')
    args = select_data.parse_args(args)
    select_data.main(args)

    corpus_subset = next(tmp_path.glob('*.txt'))
    docs_subset = corpus_subset.read_text().splitlines()
    import pdb; pdb.set_trace()
    assert any([doc in CORPUS for doc in docs_subset])


def test_integration_select_random_pct_correct_number(tmp_path, corpus_file):
    pass

def test_integration_select_similar_jensen_shannon():
    pass

def test_integration_select_similar_jensen_shannon_inverse():
    pass

def test_integration_select_diverse_entropy():
    pass

def test_integration_select_diverse_entropy_inverse():
    pass

# TODO include unit tests before integration tests