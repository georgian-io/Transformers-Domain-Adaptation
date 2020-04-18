"""Integration tests for data selection module."""
import shlex
import itertools as it
from pathlib import Path

import pytest
import pandas as pd

from scripts.domain_adaptation import select_data

from .test_unit_select_data import corpus_file, vocab_file, CORPUS, VOCABULARY


FINE_TUNE_TEXT = ('emu carrot', 'this is an apple')


@pytest.fixture(scope='session')
def fine_tune_corpus_file(tmp_path_factory) -> Path:
    """Create a fine-tune corpus file shared by all test functions."""
    fine_tune_corpus = tmp_path_factory.mktemp('shared') / 'fine-tune-text.txt'
    fine_tune_corpus.write_text('\n'.join(FINE_TUNE_TEXT))
    return fine_tune_corpus


def test_select_random_not_modified(tmp_path, corpus_file):
    """Ensure that individual docs in subset corpus is not modified."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} random -p 0.3')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    assert all(doc in CORPUS for doc in docs_subset)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_similar_not_modified(tmp_path, corpus_file, vocab_file,
                                     fine_tune_corpus_file, invert):
    """Ensure that individual docs in subset corpus is not modified."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar --fine-tune-text {fine_tune_corpus_file} '
                       f'-v {vocab_file} -p 0.3 {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    assert all(doc in CORPUS for doc in docs_subset)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_similar_correct_num_docs(tmp_path, corpus_file, vocab_file,
                                         fine_tune_corpus_file, invert):
    """Ensure that the correct number of docs are selected."""
    PCT = 0.3
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar --fine-tune-text {fine_tune_corpus_file} '
                       f'-v {vocab_file} -p {PCT} {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = list(tmp_path.rglob('*.txt'))
    assert len(output_file) == 1

    output_file = output_file[0]
    docs_subset = output_file.read_text().splitlines()
    assert len(docs_subset) == int(len(CORPUS) * PCT)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_similar_correct_subset(tmp_path, corpus_file, vocab_file,
                                       fine_tune_corpus_file, invert):
    """Test that the corpus subset from `select_similar` is correct."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar --fine-tune-text {fine_tune_corpus_file} '
                       f'-v {vocab_file} -p 0.3 {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    if invert == '':
        assert tuple(docs_subset) == (CORPUS[4], CORPUS[5])
    else:
        assert tuple(docs_subset) == (CORPUS[0], CORPUS[1])


def test_select_similar_cache_correctness(tmp_path, corpus_file, vocab_file,
                                          fine_tune_corpus_file):
    """Test that the calculated similarities are cached properly."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar --fine-tune-text {fine_tune_corpus_file} '
                       f'-v {vocab_file} -p 0.3')
    args = select_data.parse_args(args)
    select_data.main(args)

    cache_path = next((tmp_path / 'cache').rglob('*.pkl'))
    assert isinstance(cache_path, Path)

    cache = pd.read_pickle(cache_path)
    similarities = select_data.calculate_similarity(args)
    assert isinstance(cache, pd.Series)
    assert (cache.values == similarities.values).all()


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_diverse_not_modified(tmp_path, corpus_file, vocab_file, invert):
    """Ensure that individual docs in subset corpus is not modified."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'diverse -v {vocab_file} -p 0.3 {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    assert all(doc in CORPUS for doc in docs_subset)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_diverse_correct_num_docs(tmp_path, corpus_file,
                                         vocab_file, invert):
    """Ensure that the correct number of docs are selected."""
    PCT = 0.3
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'diverse -v {vocab_file} -p {PCT} {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = list(tmp_path.rglob('*.txt'))
    assert len(output_file) == 1

    output_file = output_file[0]
    docs_subset = output_file.read_text().splitlines()
    assert len(docs_subset) == int(len(CORPUS) * PCT)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_diverse_correct_subset(tmp_path, corpus_file,
                                       vocab_file, invert):
    """Test that the corpus subset from `select_diverse` is correct."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'diverse -v {vocab_file} -p 0.5 {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    if invert == '':
        assert tuple(docs_subset) == (CORPUS[4], CORPUS[5], CORPUS[6])
    else:
        assert tuple(docs_subset) == (CORPUS[0], CORPUS[1], CORPUS[2])


def test_select_diverse_cache_correctness(tmp_path, corpus_file, vocab_file,
                                          fine_tune_corpus_file):
    """Test that the calculated similarities are cached properly."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'diverse -v {vocab_file} -p 0.5')
    args = select_data.parse_args(args)
    select_data.main(args)

    cache_path = next((tmp_path / 'cache').rglob('*.pkl'))
    assert isinstance(cache_path, Path)

    cache = pd.read_pickle(cache_path)
    diversity_scores = select_data.calculate_diversity(args)
    assert isinstance(cache, pd.Series)
    assert (cache.values == diversity_scores.values).all()


@pytest.mark.parametrize('invert,fuse_by', it.product(('', '-i'), ('linear_combination', 'union')))
def test_select_similar_diverse_not_modified(tmp_path, corpus_file, vocab_file,
                                             fine_tune_corpus_file, invert,
                                             fuse_by):
    """Ensure that individual docs in subset corpus is not modified."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar+diverse --fine-tune-text {fine_tune_corpus_file} '
                       f'-v {vocab_file} -p 0.3 {invert} --fuse-by {fuse_by}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    assert all(doc in CORPUS for doc in docs_subset)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_similar_diverse_correct_num_docs(tmp_path, corpus_file,
                                                 vocab_file, invert,
                                                 fine_tune_corpus_file):
    """Ensure that the correct number of docs are selected."""
    PCT = 0.3
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar+diverse --fine-tune-text {fine_tune_corpus_file} '
                       f'-v {vocab_file} -p {PCT} {invert}')
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = list(tmp_path.rglob('*.txt'))
    assert len(output_file) == 1

    output_file = output_file[0]
    docs_subset = output_file.read_text().splitlines()
    assert len(docs_subset) == int(len(CORPUS) * PCT)


@pytest.mark.parametrize('invert', ['', '-i'])
def test_select_similar_diverse_correct_subset(tmp_path, corpus_file, vocab_file,
                                               fine_tune_corpus_file, invert):
    """Test that the corpus subset from `select_diverse` is correct."""
    args = shlex.split(f'--corpus {corpus_file} --dst {tmp_path} '
                       f'similar+diverse --fine-tune-text {fine_tune_corpus_file} '
                       f'--sim-div-weights 1,1 '
                       f'-v {vocab_file} -p 0.3 {invert}')
    # import pdb; pdb.set_trace()
    args = select_data.parse_args(args)
    select_data.main(args)

    output_file = next(tmp_path.rglob('*.txt'))
    docs_subset = output_file.read_text().splitlines()
    if invert == '':
        assert CORPUS[6] in docs_subset  # Not the best test
    else:
        assert tuple(docs_subset) == (CORPUS[0], CORPUS[1])
