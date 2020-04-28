"""Unit tests for data selection module."""
import shlex
from typing import List
from pathlib import Path
from collections import Iterable

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.domain_adaptation import select_data


CORPUS = (
    'apple apple apple apple apple',
    'apple banana apple apple apple',
    'apple banana carrot apple apple',
    'apple banana carrot document apple',
    'apple banana carrot document emu',
    'this is an apple',
    'These are documents and this is a banana'
)
VOCABULARY = ('a', 'apple', 'are', 'banana', 'carrot', 'document',
              'emu', 'is', 'this', 'these', '##d', '##n', '##s', '[UNK]')


@pytest.fixture(scope='session')
def corpus_file(tmp_path_factory) -> Path:
    """Create a corpus file shared by all test functions."""
    corpus = tmp_path_factory.mktemp('shared') / 'corpus.txt'
    corpus.write_text('\n'.join(CORPUS))
    return corpus


@pytest.fixture(scope='session')
def documents(corpus_file) -> List[str]:
    """Return a list of documents to be shared by all test functions."""
    return corpus_file.read_text().splitlines()


@pytest.fixture(scope='session')
def vocab_file(tmp_path_factory) -> Path:
    """Create a vocab file shared by all test functions."""
    vocab = tmp_path_factory.mktemp('shared') / 'vocab.txt'
    vocab.write_text('\n'.join(VOCABULARY))
    return vocab


def test_corpus_does_not_exist(tmp_path):
    """Ensure that error is raised if specified corpus does not exist."""
    corpus = tmp_path / 'corpus.txt'
    args = shlex.split(f'--corpus {corpus} --dst {tmp_path} random -p 0.02')
    with pytest.raises(FileNotFoundError):
        select_data.parse_args(args)


def test_corpus_is_empty(tmp_path):
    """Ensure that error is raised if specified corpus is empty."""
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('')
    args = shlex.split(f'--corpus {corpus} --dst {tmp_path} random -p 0.02')
    with pytest.raises(ValueError):
        select_data.parse_args(args)


def test_invalid_pct(tmp_path):
    """Ensure that error is raised if specified pct is invalid."""
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('this is a document')
    args = shlex.split(f'--corpus {corpus} --dst {tmp_path} random -p 2')
    with pytest.raises(ValueError):
        select_data.parse_args(args)


def test_create_vocab(vocab_file):
    """Ensure that vocab object has appropriate and correct attributes."""
    vocab_obj = select_data.create_vocab(vocab_file)
    assert hasattr(vocab_obj, 'size')
    assert hasattr(vocab_obj, 'word2id')
    assert vocab_obj.size == len(VOCABULARY)
    assert vocab_obj.word2id == dict(map(reversed, enumerate(VOCABULARY)))


def test_docs_to_tokens_type(documents, vocab_file):
    """Test the output types of `docs_to_tokens`."""
    tokenized = select_data.docs_to_tokens(documents, vocab_file)
    assert isinstance(tokenized, Iterable)

    tokenized_doc = next(tokenized)
    assert isinstance(tokenized_doc, list)
    assert isinstance(tokenized_doc[0], str)


def test_docs_to_tokens_correctness(vocab_file):
    """Test the output correctness of `docs_to_tokens`."""
    docs = ['This is an apple']
    tokenized = select_data.docs_to_tokens(docs, vocab_file)
    tokenized = list(tokenized)
    assert tokenized == [['this', 'is', 'a', '##n', 'apple']]


def test_docs_to_tokens_no_special_tokens(vocab_file):
    """Ensure that `docs_to_tokens`'s output does not contain BERT special tokens."""
    docs = ['[CLS] [SEP] [UNK] [MASK] [PAD]']
    tokenized = select_data.docs_to_tokens(docs, vocab_file)
    tokenized = list(tokenized)
    assert tokenized == [[]]


def test_docs_to_term_dist_invalid_level(documents, vocab_file):
    """Ensure that error is raised if an invalid level arg is specified."""
    with pytest.raises(ValueError):
        select_data.docs_to_term_dist(documents, vocab_file, level='invalid-level-123')


def test_docs_to_term_dist_level_corpus_type(documents, vocab_file):
    """Test the output types of `docs_to_term_dist`'s output when level=corpus."""
    term_dist = select_data.docs_to_term_dist(documents, vocab_file,
                                              level='corpus')
    assert isinstance(term_dist, np.ndarray)
    assert term_dist.shape == (len(VOCABULARY),)


def test_docs_to_term_dist_level_corpus_correctness(documents, vocab_file):
    """Test the output correctness of `docs_to_term_dist` when level=corpus."""
    term_dist = select_data.docs_to_term_dist(documents, vocab_file, level='corpus')
    answer = np.array([3, 16, 1, 5, 3, 3, 1, 2, 2, 1, 1, 2, 1, 0]) / 41
    assert np.allclose(term_dist, answer)


def test_docs_to_term_dist_level_doc_type(documents, vocab_file):
    """Test the output types of `docs_to_term_dist`'s output when level=doc."""
    term_dists = select_data.docs_to_term_dist(documents, vocab_file, level='doc')
    assert isinstance(term_dists, Iterable)

    for term_dist in term_dists:
        assert isinstance(term_dist, np.ndarray)
        assert term_dist.shape == (len(VOCABULARY),)


def test_docs_to_term_dist_level_doc_correctness(documents, vocab_file):
    """Test the output correctness of `docs_to_term_dist` when level=corpus."""
    term_dists = select_data.docs_to_term_dist(documents, vocab_file, level='doc')
    term_dists = np.array(list(term_dists))
    answers = np.array([
        [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
        [2, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
    ])
    answers = answers / answers.sum(axis=1, keepdims=True)
    assert np.allclose(term_dists, answers)


def test_docs_to_tfidf_invalid_level(documents, vocab_file):
    """Ensure that error is raised if an invalid level arg is specified."""
    with pytest.raises(ValueError):
        select_data.docs_to_tfidf(documents, vocab_file, level='invalid-level-123')



# def test_docs_to_tfidf_level_corpus_correctness(documents, vocab_file):
#     """Test the output correctness of `docs_to_tfidf` when level=corpus."""
#     term_dist, vectorizer = select_data.docs_to_tfidf(documents, vocab_file, level='corpus')
#     import pdb; pdb.set_trace()
#     answer = np.array([3, 16, 1, 5, 3, 3, 1, 2, 2, 1, 1, 2, 1, 0]) / (1 + np.log(2))
#     assert np.allclose(term_dist, answer)


def test_docs_to_tfidf_level_doc_type(documents, vocab_file):
    """Test the output types of `docs_to_tfidf`'s output when level='doc'."""
    term_dist, vectorizer = select_data.docs_to_tfidf(documents, vocab_file,
                                                      level='doc')
    assert isinstance(term_dist, np.ndarray)
    assert term_dist.shape == (len(documents), len(VOCABULARY) - 1)  # Without '[UNK]'

    assert isinstance(vectorizer, TfidfVectorizer)
    assert hasattr(vectorizer, 'vocabulary_')
    assert hasattr(vectorizer, 'idf_')


# def test_docs_to_tfidf_level_doc_correctness(documents, vocab_file):
#     """Test the output correctness of `docs_to_tfidf` when level=corpus."""
#     term_dists, _ = select_data.docs_to_tfidf(documents, vocab_file, level='doc')
#     answers = np.array([
#         [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
#         [2, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
#     ])
#     answers = answers / answers.sum(axis=1, keepdims=True)
#     assert np.allclose(term_dists, answers)
