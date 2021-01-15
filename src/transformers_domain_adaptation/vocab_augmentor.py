"""Class definition for VocabAugmentor."""
from pathlib import Path
from collections import Counter
from types import MappingProxyType
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import IO, List, Type, Union, Counter as CounterType

from sklearn.base import BaseEstimator
from tokenizers import Tokenizer, trainers
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import BPE, Unigram, WordPiece

from transformers_domain_adaptation.type import Corpus, Token


class VocabAugmentor(BaseEstimator):
    """Find new tokens to add to a :obj:`tokenizer`'s vocabulary.

    A new vocabulary is learnt from the training corpus
    using the same tokenization model (WordPiece, BPE, Unigram).
    The most common tokens of this new vocabulary that do not exist
    in the existing vocabulary are selected.
    """

    supported_trainers = MappingProxyType(
        {
            BPE: trainers.BpeTrainer,
            WordPiece: trainers.WordPieceTrainer,
            Unigram: trainers.UnigramTrainer,
        }
    )

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, cased: bool, target_vocab_size: int
    ):
        """
        Args:
            tokenizer: A Rust-based 🤗 Tokenizer
            cased: If False, ignore uppercases in corpus
            target_vocab_size: Size of augmented vocabulary

        Raises:
            ValueError: If :obj:`target_vocab_size` is larger or equal to the existing vocabulary of :obj:`tokenizer`
            RuntimeError: If :obj:`tokenizer` uses an unsupported tokenization model
        """
        if target_vocab_size <= tokenizer.vocab_size:
            raise ValueError(
                f"Ensure that `target_vocab_size` is larger than tokenizer's vocab size."
            )
        self.tokenizer = tokenizer
        self.cased = cased
        self.target_vocab_size = target_vocab_size
        self.model_cls: Type[
            BaseTokenizer
        ] = tokenizer.backend_tokenizer.model.__class__

        # Instantiate rust tokenizer
        rust_tokenizer = Tokenizer(self.model_cls())
        if not cased:
            rust_tokenizer.normalizer = Lowercase()
        rust_tokenizer.pre_tokenizer = Whitespace()
        self.rust_tokenizer = rust_tokenizer

        # Instantiate the appropriate Trainer based on `self.model` (i.e. BPE, WordPiece, etc)
        trainer_cls = self.supported_trainers.get(self.model_cls, None)
        if trainer_cls is None:
            raise RuntimeError(f"{self.model_cls} is not supported")
        self.trainer = trainer_cls(
            vocab_size=self.target_vocab_size,
            special_tokens=list(self.tokenizer.special_tokens_map.values()),
        )

    def get_new_tokens(
        self,
        training_corpus: Union[Corpus, Path, str],
    ) -> List[Token]:
        """Obtain new tokens found in :obj:`training_corpus`.

        New tokens contains the most common tokens that do not exist in the :obj:`tokenizer`'s vocabulary.

        Args:
            training_corpus: The training corpus
        """
        # Training has to be wrapped with the `tmpfile` context
        with NamedTemporaryFile("w+") as tmpfile:  # If we need to save Corpus type
            # Train new tokenizer on `ft_corpus`
            train_files = self._get_training_files(training_corpus, _tmpfile=tmpfile)
            self.rust_tokenizer.train(self.trainer, train_files)

            # Include unknown token to vocab
            with TemporaryDirectory() as tmpdir:
                files = self.rust_tokenizer.model.save(tmpdir)
                self.rust_tokenizer.model = self.model_cls.from_file(
                    *files, unk_token="[UNK]"
                )

            # Find most common tokens in vocab
            token_counts = self._count_tokens(train_files)

        # Remove overlapping tokens from original tokenizer
        token_counts = self._remove_overlapping_tokens(token_counts)
        new_tokens = [
            token
            for token, _ in token_counts.most_common(
                self.target_vocab_size - self.tokenizer.vocab_size
            )
        ]
        return new_tokens

    @staticmethod
    def _get_training_files(
        corpus: Union[Corpus, Path, str], _tmpfile: IO[str]
    ) -> List[str]:
        """Return files for training.

        If `corpus is a sequence of documents, it will be written to a temporary file,
        and that temporary file's name will be returned.

        If `corpus` is a Path or str, it will return the path, or paths if `corpus` is a directory.

        Args:
            corpus: Text data or path to training corpus
            _tmpfile: Temporary file object. Used when `corpus` is not a path

        Raises:
            FileNotFoundError: If `corpus` is a str or Path and it does not exist on the filesystem.
        """
        if isinstance(corpus, str) or isinstance(corpus, Path):
            corpus = Path(corpus)

            if not corpus.exists():
                raise FileNotFoundError(
                    f"Training corpus {corpus.as_posix()} does not exist."
                )

            files = list(corpus.rglob("*.*")) if corpus.is_dir() else [corpus]
            files = [f.as_posix() for f in files]
            return files

        else:  # Corpus type
            for doc in corpus:
                _tmpfile.write(doc)
            _tmpfile.seek(0)
            return [_tmpfile.name]

    def _count_tokens(self, files: List[str]) -> CounterType[str]:
        """Count number of tokens in a list of files."""
        token_counts: CounterType[str] = Counter()
        for file in files:
            with open(file) as f:
                token_counts += Counter(
                    token
                    for enc in self.rust_tokenizer.encode_batch(f.readlines())
                    for token in enc.tokens
                )
        return token_counts

    def _remove_overlapping_tokens(
        self, token_counts: CounterType[str]
    ) -> CounterType[str]:
        """Remove tokens from `token_counts` that exist in the current tokenizer's vocab."""
        _token_counts = token_counts.copy()

        for vocab_term in self.tokenizer.get_vocab().keys():
            if vocab_term in _token_counts:
                del _token_counts[vocab_term]
        return _token_counts
