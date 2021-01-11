from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from nlp_domain_adaptation.type import Corpus


class RandomDataSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        select: Union[int, float],
        random_state: Optional[int] = None,
    ):
        if isinstance(select, int) and select <= 0:
            raise ValueError(f"Int value for `select` must be strictly positive.")
        if isinstance(select, float) and not 0 <= select <= 1:
            raise ValueError(
                f"Float value for `select` must be between 0 and 1 (inclusive)."
            )
        if random_state is not None:
            np.random.seed(random_state)
        self.select = select

    def fit(self, docs: Corpus):
        return self

    def transform(self, docs: Corpus) -> Corpus:
        p = self.select if isinstance(self.select, float) else self.select / len(docs)
        selection_index = np.random.choice([0, 1], size=len(docs), p=[1 - p, p]).astype(
            bool
        )
        selected_docs = np.array(docs)[selection_index].tolist()
        return selected_docs
