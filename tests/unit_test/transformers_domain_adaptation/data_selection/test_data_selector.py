import string

import pytest
import numpy as np
from hypothesis import given, strategies as st
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from transformers_domain_adaptation.type import Corpus
from transformers_domain_adaptation.data_selection.data_selector import DataSelector


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def data_selector(tokenizer) -> DataSelector:
    return DataSelector(
        keep=2,
        tokenizer=tokenizer,
        similarity_metrics=[
            "jensen-shannon",
            "renyi",
            "cosine",
            "euclidean",
            "variational",
            "bhattacharyya",
        ],
        diversity_metrics=[
            "num_token_types",
            "type_token_ratio",
            "entropy",
            "simpsons_index",
            "renyi_entropy",
        ],
    )


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    _corpus = (
        "apple apple apple apple apple",
        "apple banana apple apple apple",
        "apple banana carrot apple apple",
        "apple banana carrot document apple",
        "apple banana carrot document emu",
        "this is an apple",
        "These are documents and this is a banana",
    )
    return _corpus


##################################
##### Input validation tests #####
##################################
@given(keep=st.integers(max_value=0))
def test_DataSelector_raise_error_with_zero_or_negative_select_int(keep, tokenizer):
    with pytest.raises(ValueError):
        DataSelector(
            keep=keep, tokenizer=tokenizer, similarity_metrics=["euclidean"]
        )


@given(
    keep=st.one_of(
        st.floats(None, 0, exclude_max=True), st.floats(1, None, exclude_min=True)
    )
)
def test_DataSelector_raise_error_with_invalid_select_float(keep, tokenizer):
    with pytest.raises(ValueError):
        DataSelector(
            keep=keep, tokenizer=tokenizer, similarity_metrics=["euclidean"]
        )


def test_DataSelector_raise_error_with_invalid_similarity_metric(tokenizer):
    with pytest.raises(ValueError):
        DataSelector(
            keep=2, tokenizer=tokenizer, similarity_metrics=["invalid_metric"]
        )


def test_DataSelector_raise_error_with_invalid_diversity_metric(tokenizer):
    with pytest.raises(ValueError):
        DataSelector(
            keep=2, tokenizer=tokenizer, diversity_metrics=["invalid_metric"]
        )


def test_DataSelector_raise_error_when_both_similarity_and_diversity_metrics_are_not_specified(
    tokenizer,
):
    with pytest.raises(ValueError):
        DataSelector(keep=0.5, tokenizer=tokenizer)


#################################
##### `.to_term_dist` tests #####
#################################
@pytest.mark.parametrize("text", ["", " "])
def test_to_term_dist_raise_error_with_empty_str(data_selector: DataSelector, text):
    with pytest.raises(ValueError):
        data_selector.to_term_dist(text)


# @given(
#     text=st.text(alphabet=string.printable, min_size=1, max_size=100).filter(
#         lambda x: len(x.strip()) > 1
#     )
# )
@pytest.mark.parametrize("text", ["a very short sentence", "knock knock tell me a joke"])
def test_to_term_dist_return_a_valid_proba_dist(data_selector: DataSelector, text):
    term_dist = data_selector.to_term_dist(text)
    assert np.isclose(term_dist.sum(), 1.0)
    assert (term_dist >= 0).all()


@pytest.mark.parametrize("text", ("apples are red", "there is a bird on the tree"))
def test_to_term_dist_correctness(data_selector: DataSelector, text):
    term_dist = data_selector.to_term_dist(text)
    assert len(term_dist.nonzero()[0]) == len(text.split(" "))


##############################
##### `.compute_*` tests #####
##############################
def test_compute_similarities_return_dataframe_of_correct_shape(
    data_selector: DataSelector, corpus: Corpus
):
    data_selector.fit(corpus)
    scores = data_selector.compute_similarities(corpus)
    assert scores.shape[1] == len(data_selector.similarity_metrics)


def test_compute_diversity_return_dataframe_of_correct_shape(
    data_selector: DataSelector, corpus: Corpus
):
    data_selector.fit(corpus)
    scores = data_selector.compute_diversities(corpus)
    assert scores.shape[1] == len(data_selector.diversity_metrics)


def test_compute_metrics_adds_composite_score_column(
    data_selector: DataSelector, corpus: Corpus
):
    expected_features = (
        len(data_selector.similarity_metrics)
        + len(data_selector.diversity_metrics)
        + 1  # composite score
    )

    data_selector.fit(corpus)
    scores = data_selector.compute_metrics(corpus)
    assert scores.shape[1] == expected_features
    assert "composite" in scores


##############################
##### DataSelector tests #####
##############################
@pytest.mark.parametrize("keep", (2, 5))
def test_DataSelector_selects_correct_num_of_docs_with_int_select_arg(
    corpus: Corpus, tokenizer: PreTrainedTokenizerFast, keep: int
):
    data_selector = DataSelector(
        keep=keep,
        tokenizer=tokenizer,
        similarity_metrics=[
            "jensen-shannon",
            "renyi",
            "cosine",
            "euclidean",
            "variational",
            "bhattacharyya",
        ],
        diversity_metrics=[
            "num_token_types",
            "type_token_ratio",
            "entropy",
            "simpsons_index",
            "renyi_entropy",
        ],
    )
    selected_corpus = data_selector.fit_transform(corpus)
    assert len(selected_corpus) == keep


@pytest.mark.parametrize(
    "keep,correct_n_docs", ((0.01, 0), (0.2, 1), (0.8, 5), (1.0, 7))
)
def test_DataSelector_selects_correct_num_of_docs_with_float_select_arg(
    corpus: Corpus,
    tokenizer: PreTrainedTokenizerFast,
    keep: float,
    correct_n_docs: int,
):
    data_selector = DataSelector(
        keep=keep,
        tokenizer=tokenizer,
        similarity_metrics=[
            "jensen-shannon",
            "renyi",
            "cosine",
            "euclidean",
            "variational",
            "bhattacharyya",
        ],
        diversity_metrics=[
            "num_token_types",
            "type_token_ratio",
            "entropy",
            "simpsons_index",
            "renyi_entropy",
        ],
    )
    selected_corpus = data_selector.fit_transform(corpus)
    assert len(selected_corpus) == correct_n_docs
