from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from transformers_domain_adaptation.type import Corpus
from transformers_domain_adaptation.vocab_augmentor import VocabAugmentor

TOKENIZERS_TO_TEST = ("bert-base-uncased", "bert-base-cased", "roberta-base")


@pytest.fixture(params=TOKENIZERS_TO_TEST)
def tokenizer(request) -> PreTrainedTokenizerFast:
    _tokenizer = AutoTokenizer.from_pretrained(request.param)
    return _tokenizer


@pytest.fixture
def named_tmpfile() -> IO[str]:
    with NamedTemporaryFile("w+") as tmpfile:
        yield tmpfile


@pytest.fixture
def augmentor(tokenizer) -> VocabAugmentor:
    return VocabAugmentor(
        tokenizer, cased=False, target_vocab_size=int(1.1 * len(tokenizer))
    )


@pytest.fixture(scope="session")
def training_corpus() -> Corpus:
    text = """
    It's Supercalifragilisticexpialidocious
    Even though the sound of it
    Is something quite atrocious
    If you say it loud enough
    You'll always sound precocious
    Supercalifragilisticexpialidocious
    Um-dittle-ittl-um-dittle-I
    Um-dittle-ittl-um-dittle-I
    Um-dittle-ittl-um-dittle-I
    Um-dittle-ittl-um-dittle-I
    """
    corpus = text.strip().split("\n")
    return corpus


##################################
##### Input Validation Tests #####
##################################
@pytest.mark.parametrize("vocab_size_multiplier", (0.1, 0.5, 0.9))
def test_VocabAugmentor_error_raised_when_target_vocab_size_is_less_than_tokenizer_vocab_size(
    vocab_size_multiplier,
):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    target_vocab_size = int(vocab_size_multiplier * len(tokenizer))
    with pytest.raises(ValueError):
        VocabAugmentor(tokenizer, cased=True, target_vocab_size=target_vocab_size)


################################
##### Private Methods Tests #####
################################
def test__remove_overlapping_tokens_correctness(augmentor: VocabAugmentor):
    c = Counter(["apple", "a_new_token", "day", "a_new_token"])
    output = augmentor._remove_overlapping_tokens(c)
    assert set(output) == {"a_new_token"}


########################################
##### `._get_training_files` Tests #####
########################################
def test__get_training_files_raise_error_on_nonexistent_file(named_tmpfile: IO[str]):
    with pytest.raises(FileNotFoundError):
        VocabAugmentor._get_training_files("nonexistent_file.txt", named_tmpfile)


@pytest.mark.parametrize("input_corpus_as_str", (True, False))
def test__get_training_files_correctness_single_file(
    tmp_path, input_corpus_as_str, named_tmpfile: IO[str]
):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("")  # Creates the file

    train_files = VocabAugmentor._get_training_files(
        corpus if input_corpus_as_str else Path(corpus), named_tmpfile
    )

    assert len(train_files) == 1
    assert isinstance(train_files[0], str)


@pytest.mark.parametrize("input_corpus_as_str", (True, False))
def test__get_training_files_correctness_single_directory(
    tmp_path, input_corpus_as_str, named_tmpfile: IO[str]
):
    n_files = 3
    corpus_dir = tmp_path
    # Create multiple text files
    for i in range(3):
        (corpus_dir / f"corpus{i}.txt").write_text("")

    train_files = VocabAugmentor._get_training_files(
        corpus_dir if input_corpus_as_str else Path(corpus_dir), named_tmpfile
    )

    assert len(train_files) == n_files
    assert all(isinstance(file, str) for file in train_files)


def test__get_training_files_return_tmpfile_when_corpus_is_of_type_Corpus(
    named_tmpfile: IO[str],
):
    corpus: Corpus = ["This is a document.", "The document following the first."]
    train_files = VocabAugmentor._get_training_files(corpus, named_tmpfile)
    assert len(train_files) == 1
    assert train_files[0] == named_tmpfile.name


def test__get_training_files_tmpfile_returned_properly_saves_text(
    named_tmpfile: IO[str],
):
    corpus: Corpus = ["This is a document.", "The document following the first."]
    train_files = VocabAugmentor._get_training_files(corpus, named_tmpfile)
    assert Path(train_files[0]).read_text() == "".join(corpus)


###################################
##### `.get_new_tokens` Tests #####
###################################
def test_get_new_tokens_learns_lowercased_tokens_when_cased_arg_is_True(
    tokenizer: PreTrainedTokenizerFast, training_corpus: Corpus
):
    augmentor = VocabAugmentor(
        tokenizer, cased=True, target_vocab_size=len(tokenizer) + 3
    )
    new_tokens = augmentor.get_new_tokens(training_corpus)
    assert any(c.isupper() for c in "".join(new_tokens))


@pytest.mark.parametrize("n_extra_tokens", (3, 5, 10))
def test_get_new_tokens_return_correct_number_of_new_tokens(
    tokenizer: PreTrainedTokenizerFast, training_corpus: Corpus, n_extra_tokens
):
    augmentor = VocabAugmentor(
        tokenizer, cased=False, target_vocab_size=len(tokenizer) + n_extra_tokens
    )
    new_tokens = augmentor.get_new_tokens(training_corpus)
    assert len(new_tokens) <= n_extra_tokens


def test_get_new_tokens_does_not_return_existing_tokens(augmentor: VocabAugmentor):
    training_corpus = ["An apple a day keeps the doctors away"]
    new_tokens = augmentor.get_new_tokens(training_corpus)
    assert set(new_tokens) < set(training_corpus[0].split())
