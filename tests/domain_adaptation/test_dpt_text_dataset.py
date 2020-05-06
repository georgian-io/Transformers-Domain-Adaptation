"""Module to test functionality of TextDataset."""
from types import SimpleNamespace
from shutil import copyfile

import torch
import pytest
from tokenizers import BertWordPieceTokenizer

from scripts.domain_adaptation.domain_pre_train import TextDataset


TEXTS = ('These are texts from a document',
         'What happens when a sheep, drum and a snake fall down a cliff?',
         'ba-dum-tss!')
BLOCK_SIZE = 4
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102


@pytest.fixture(scope='session')
def text_file(tmp_path_factory):
    """Return a path to a text file."""
    text_filepath = tmp_path_factory.mktemp('tmp') / 'corpus.txt'
    text_filepath.write_text('\n'.join(TEXTS))
    return text_filepath


@pytest.fixture(scope='session')
def tokenizer():
    """Create a Bert Tokenizer implemented in Rust."""
    return BertWordPieceTokenizer('bert-base-uncased-vocab.txt')


@pytest.fixture(scope='session')
def text_dataset(text_file, tokenizer, tmp_path_factory):
    """Create a session-scoped TextDataset to be used by all test functions."""
    output_dir = tmp_path_factory.mktemp('tmp_output_dir', numbered=False)
    args = SimpleNamespace(output_dir=output_dir, cache_dtype=torch.short,
                           overwrite_cache=True, chunk_size=5)
    return TextDataset(tokenizer, args, file_paths=[text_file], block_size=4)


def test_load_from_multiple_txt_files(tmp_path, text_file, tokenizer):
    """Ensure that dataset is able to load from multiple text files."""
    # To simulate reading from multiple text files,
    # create a second text file (a copy of `text_file`)
    text_file_2 = tmp_path / 'another_corpus.txt'
    copyfile(text_file, text_file_2)

    file_paths = [text_file, text_file_2]
    args = SimpleNamespace(output_dir=tmp_path, cache_dtype=torch.short,
                           overwrite_cache=True, chunk_size=5)

    dataset = TextDataset(tokenizer, args,
                          file_paths=file_paths,
                          block_size=BLOCK_SIZE)
    assert dataset.examples.shape == (29, BLOCK_SIZE)


def test_output_dtype(text_dataset):
    """Ensure that TextDataset returns the correct output type when index.

    Output should be
    1. torch tensors
    2. of dtype `torch.short`
    3. with length `block_size`
    """
    for i in range(len(text_dataset)):
        tensor = text_dataset[i]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.short
        assert len(tensor) == BLOCK_SIZE


def test_output_starts_and_ends_with_bert_tokens(text_dataset):
    """Ensure that the torch tensors start and end the correct BERT token ids."""
    for i in range(len(text_dataset)):
        tensor = text_dataset[i]
        if i < len(text_dataset) - 1:
            assert tensor[0] == CLS_TOKEN_ID
            assert tensor[-1] == SEP_TOKEN_ID


def test_last_output_is_padded(text_dataset):
    """Ensure that the output is right-padded with zeros."""
    tensor = text_dataset[-1]
    first_zero_index = torch.where(tensor == 0)[0][0]

    # Check that everything before `first_zero_index` is non-zero
    # and everything after `first_zero_index` is zero
    assert all(tensor[:first_zero_index] > 0)
    assert all(tensor[first_zero_index:] == 0)

    # Ensure that first and last non-zero tokens are special BERT tokens
    assert tensor[0] == CLS_TOKEN_ID
    assert tensor[first_zero_index - 1] == SEP_TOKEN_ID


def test_that_caching_occurs(tmp_path_factory, text_dataset):
    """Test that results are cached."""
    cache_path = (
        tmp_path_factory.getbasetemp() / 'tmp_output_dir' / 'corpus_cache.pt'
    )
    assert cache_path.exists()


def test_cache_dtype(tmp_path_factory, text_dataset):
    """Test that cache has the correct dtypes.

    It should be a single torch tensor, of dtype `torch.short`
    with length `block_size`.
    """
    cache_path = (
        tmp_path_factory.getbasetemp() / 'tmp_output_dir' / 'corpus_cache.pt'
    )
    cache = torch.load(cache_path)
    assert isinstance(cache, torch.Tensor)
    assert cache.dtype == torch.short
    assert cache.shape == (15, BLOCK_SIZE)
