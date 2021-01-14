#
# TODO: Cite code
from typing import Callable, Dict
from typing_extensions import Literal

import numpy as np
import scipy.stats
import scipy.spatial.distance


def jensen_shannon_similarity(repr1: np.ndarray, repr2: np.ndarray) -> np.ndarray:
    """Calculate similairty based on Jensen-Shannon divergence.

    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    """
    if len(repr1) == 1:
        repr1 = np.repeat(repr1, len(repr2), axis=0)
    elif len(repr2) == 1:
        repr2 = np.repeat(repr2, len(repr1), axis=0)
    avg_repr = 0.5 * (repr1 + repr2)
    sim = np.array(
        [
            1 - 0.5 * (scipy.stats.entropy(p, avg) + scipy.stats.entropy(q, avg))
            for p, q, avg in zip(repr1, repr2, avg_repr)
        ]
    )

    # the similarity is -inf if no term in the document is in the vocabulary
    sim = np.where(np.isinf(sim), 0, sim)
    return sim


def renyi_similarity(
    repr1: np.ndarray, repr2: np.ndarray, alpha: float = 0.99
) -> np.ndarray:
    """Calculate similarity based on Rényi divergence.

    https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R.C3.A9nyi_divergence
    """
    log_sum = (np.power(repr1, alpha) / np.power(repr2, alpha - 1)).sum(axis=-1)
    renyi_divergence = 1 / (alpha - 1) * np.log(log_sum)
    return -renyi_divergence


def cosine_similarity(repr1: np.ndarray, repr2: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)."""
    if len(repr1) == 1:
        repr1 = np.repeat(repr1, len(repr2), axis=0)
    elif len(repr2) == 1:
        repr2 = np.repeat(repr2, len(repr1), axis=0)

    assert not (np.isnan(repr2).any() or np.isinf(repr2).any())
    assert not (np.isnan(repr1).any() or np.isinf(repr1).any())
    sim = np.array(
        [1 - scipy.spatial.distance.cosine(p, q) for p, q in zip(repr1, repr2)]
    )

    # the similarity is nan if no term in the document is in the vocabulary
    sim = np.where(np.isnan(sim), 0, sim)
    return sim


def euclidean_similarity(repr1: np.ndarray, repr2: np.ndarray) -> np.ndarray:
    """Calculate similarity based on Euclidean distance.

    https://en.wikipedia.org/wiki/Euclidean_distance
    """
    euclidean_distance = np.sqrt(((repr1 - repr2) ** 2).sum(axis=-1))
    return -euclidean_distance


def variational_similarity(repr1: np.ndarray, repr2: np.ndarray) -> np.ndarray:
    """Calculate similarity based on L1 / Manhattan distance.

    https://en.wikipedia.org/wiki/Taxicab_geometry
    """
    manhattan_distance = np.abs(repr1 - repr2).sum(axis=-1)
    return -manhattan_distance


def bhattacharyya_similarity(repr1: np.ndarray, repr2: np.ndarray) -> np.ndarray:
    """Calculate similarity based on Bhattacharyya distance.

    https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    distance = -np.log(np.sqrt(repr1 * repr2).sum(axis=-1))
    assert not np.isnan(distance).any(), "Error: Similarity is nan."

    # the distance is -inf if no term in the review is in the vocabulary
    distance = np.where(np.isinf(distance), 0, distance)
    return -distance


############################
##### Function factory #####
############################
SimilarityMetric = Literal[
    "jensen-shannon",
    "renyi",
    "cosine",
    "euclidean",
    "variational",
    "bhattacharyya",
]
SimilarityFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
SIMILARITY_FEATURES = {
    "jensen-shannon",
    "renyi",
    "cosine",
    "euclidean",
    "variational",
    "bhattacharyya",
}


def similarity_func_factory(metric: SimilarityMetric) -> SimilarityFunc:
    """Return the corresponding similarity function based on the provided metric.

    Args:
        metric (str): Similarity metric

    Raises:
        ValueError: If `metric` does not exist in SIMILARITY_FEATURES
    """
    if metric not in SIMILARITY_FEATURES:
        raise ValueError(f'"{metric}" is not a valid similarity metric.')

    mapping: Dict[SimilarityMetric, SimilarityFunc] = {
        "jensen-shannon": jensen_shannon_similarity,
        "renyi": renyi_similarity,
        "cosine": cosine_similarity,
        "euclidean": euclidean_similarity,
        "variational": variational_similarity,
        "bhattacharyya": bhattacharyya_similarity,
    }

    similarity_function = mapping[metric]
    return similarity_function
