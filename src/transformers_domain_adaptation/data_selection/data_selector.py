from collections import Counter
from typing import List, Optional, Sequence, Union, Counter as CounterType

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.preprocessing import RobustScaler
from transformers import PreTrainedTokenizerFast
from sklearn.base import BaseEstimator, TransformerMixin

from transformers_domain_adaptation.type import Corpus, Token
from transformers_domain_adaptation.data_selection.metrics import (
    SIMILARITY_FEATURES,
    DIVERSITY_FEATURES,
    similarity_func_factory,
    diversity_func_factory,
)


class DataSelector(BaseEstimator, TransformerMixin):
    """Select subset of data that is likely to be beneficial for domain pre-training.

    This class is sklearn-compatible and implements the sklearn Transformers interface.
    """

    def __init__(
        self,
        keep: Union[int, float],
        tokenizer: PreTrainedTokenizerFast,
        similarity_metrics: Optional[Sequence[str]] = None,
        diversity_metrics: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            keep: Quantity of documents from corpus to keep.
                  To specify number of documents, use :obj:`int`.
                  To specify percentage of documents in corpus, use :obj:`float`.
            tokenizer: A Rust-based 🤗 Tokenizer
            similarity_metrics: An optional list of similarity metrics
            diversity_metrics: An optional list of diversity metrics

        Note:
            For a list of similarity and diversity metrics, refer to :ref:`data-selection-metrics`

        Note:
            At least one similarity/diversity metric must be provided.
        """
        if isinstance(keep, int) and keep <= 0:
            raise ValueError(f"Int value for `keep` must be strictly positive.")
        if isinstance(keep, float) and not 0 <= keep <= 1:
            raise ValueError(
                f"Float value for `keep` must be between 0 and 1 (inclusive)."
            )
        if similarity_metrics is not None:
            _invalid_sim_metrics = set(similarity_metrics) - SIMILARITY_FEATURES
            if _invalid_sim_metrics:
                raise ValueError(
                    f"Invalid similarity metric(s) {_invalid_sim_metrics} found"
                )
        if diversity_metrics is not None:
            _invalid_div_metrics = set(diversity_metrics) - DIVERSITY_FEATURES
            if _invalid_div_metrics:
                raise ValueError(
                    f"Invalid diversity metric(s) {_invalid_div_metrics} found"
                )
        if similarity_metrics is None and diversity_metrics is None:
            raise ValueError(
                f"No metrics provided. Please provide at least one similarity or diversity metric."
            )

        self.keep = keep
        self.tokenizer = tokenizer
        self.similarity_metrics = similarity_metrics
        self.diversity_metrics = diversity_metrics

    def to_term_dist(self, text: str) -> np.ndarray:
        if not len(text.strip()):
            raise ValueError(f"A non-empty string must be provided.")

        tokenized: List[Token] = self.tokenizer.tokenize(text)
        term_counts = Counter(tokenized)

        vocab = self.tokenizer.vocab

        # Create a term distribution
        term_dist: np.ndarray = np.zeros(len(vocab))
        for term, count in term_counts.items():
            term_dist[vocab[term]] = count
        term_dist /= term_dist.sum()

        return term_dist

    def to_term_dist_batch(self, texts: Sequence[str]) -> np.ndarray:
        # * Assumption: Token ID 0 is a special token id and never appears in tokenization with `add_special_tokens=False`

        # Tokenize all documents using Rust tokenizer
        counters: List[CounterType[int]] = [
            Counter(enc.ids)
            for enc in self.tokenizer.backend_tokenizer.encode_batch(
                texts, add_special_tokens=False
            )
        ]

        rows = np.array(
            [val for i, counter in enumerate(counters) for val in [i] * len(counter)]
        )
        cols = np.array([key for counter in counters for key in counter.keys()])
        data = np.array([value for counter in counters for value in counter.values()])
        term_counts = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(counters), len(self.tokenizer)),
            dtype=np.uint16 if len(self.tokenizer) < 2 ** 16 else np.uint32,
        )

        term_dist = term_counts / term_counts.sum(axis=1)
        return np.array(term_dist)

    def fit(self, ft_corpus: Corpus):
        """Compute corpus-level term distribution of :obj:`ft_corpus`.

        A new fitted attribute ``.ft_term_dist_`` of shape (:math:`V`,) is created,
        where :math:`V` is the size of the :obj:`tokenizer` vocabulary.

        Args:
            ft_corpus: The fine-tuning corpus. Not to be confused with
                       the domain pre-training corpus (which is used in :meth:`transform`)

        Note:
            The :obj:`ft_corpus` is treated as a single "document", which will be compared
            against other documents in the in-domain corpus in :meth:`transform`
        """
        self.ft_term_dist_ = self.to_term_dist(" ".join(ft_corpus))
        return self

    def transform(self, docs: Corpus) -> Corpus:
        """Create a relevant subset of documents from the training corpus based on the provided data selection metrics.

        Args:
            docs: The training corpus

        Returns:
            A subset of relevant :obj:`docs` for domain pre-training
        """
        scores = self.compute_metrics(docs)
        composite_scores = scores["composite"].sort_values(ascending=False)

        n_select = (
            self.keep if isinstance(self.keep, int) else int(self.keep * len(docs))
        )
        selection_index = composite_scores.index[:n_select]
        subset_corpus = pd.Series(docs)[selection_index]

        return subset_corpus.tolist()

    def compute_metrics(self, docs: Corpus) -> pd.DataFrame:
        scores = pd.concat(
            [
                self.compute_similarities(docs),
                self.compute_diversities(docs),
            ],
            axis=1,
        )

        # Ensure metrics are normalized, before combining them into a composite score
        scores = pd.DataFrame(
            RobustScaler().fit_transform(scores), columns=scores.columns
        )
        scores["composite"] = scores.sum(axis=1)
        return scores

    def compute_similarities(self, docs: Corpus) -> pd.DataFrame:
        similarities = pd.DataFrame()  # of shape (n_docs, n_metrics)
        if (
            self.similarity_metrics is None
        ):  # Short-circuit function to avoid unnecessary computations
            return similarities

        term_dists = self.to_term_dist_batch(docs)

        pbar = tqdm(
            self.similarity_metrics,
            desc="computing similarity",
            unit="metric",
            dynamic_ncols=True,
        )
        for metric in pbar:
            sim_func = similarity_func_factory(metric)
            similarities[metric] = sim_func(
                term_dists, self.ft_term_dist_.reshape(1, -1)
            )

        return similarities

    def compute_diversities(self, docs: Corpus) -> pd.DataFrame:
        diversities = pd.DataFrame()  # of shape (n_docs, n_metrics)
        if self.diversity_metrics is None:
            return diversities

        tokenized_docs: List[List[Token]] = [
            enc.tokens for enc in self.tokenizer.backend_tokenizer.encode_batch(docs)
        ]

        pbar = tqdm(
            self.diversity_metrics,
            desc="computing diversity",
            unit="metric",
            dynamic_ncols=True,
        )
        for metric in pbar:
            div_func = diversity_func_factory(
                metric,
                train_term_dist=self.ft_term_dist_,
                vocab2id=self.tokenizer.vocab,
            )
            diversities[metric] = pd.Series(
                (div_func(tokenized_doc) for tokenized_doc in tokenized_docs)
            )

        return diversities
