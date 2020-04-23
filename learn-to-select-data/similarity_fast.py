import numpy as np
import scipy.stats
import scipy.spatial.distance


# SIMILARITY MEASURES

def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    if len(repr1) == 1:
        repr1 = np.repeat(repr1, len(repr2), axis=0)
    elif len(repr2) == 1:
        repr2 = np.repeat(repr2, len(repr1), axis=0)
    avg_repr = 0.5 * (repr1 + repr2)
    sim = np.array([1 - 0.5 * (scipy.stats.entropy(p, avg) + scipy.stats.entropy(q, avg))
                    for p, q, avg in zip(repr1, repr2, avg_repr)])

    # the similarity is -inf if no term in the document is in the vocabulary
    sim = np.where(np.isinf(sim), 0, sim)
    return sim


def renyi_divergence(repr1, repr2, alpha=0.99):
    """Calculates Renyi divergence (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R.C3.A9nyi_divergence)."""
    log_sum = (np.power(repr1, alpha) / np.power(repr2, alpha - 1)).sum(axis=-1)
    sim = 1 / (alpha - 1) * np.log(log_sum)

    # the similarity is -inf if no term in the document is in the vocabulary
    sim = np.where(np.isinf(sim), 0, sim)
    return sim


def cosine_similarity(repr1, repr2):
    """Calculates cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)."""
    if len(repr1) == 1:
        repr1 = np.repeat(repr1, len(repr2), axis=0)
    elif len(repr2) == 1:
        repr2 = np.repeat(repr2, len(repr1), axis=0)

    assert not (np.isnan(repr2).any() or np.isinf(repr2).any())
    assert not (np.isnan(repr1).any() or np.isinf(repr1).any())
    sim = np.array([1 - scipy.spatial.distance.cosine(p, q) for p, q in zip(repr1, repr2)])

    # the similarity is nan if no term in the document is in the vocabulary
    sim = np.where(np.isnan(sim), 0, sim)
    return sim


def euclidean_distance(repr1, repr2):
    """Calculates Euclidean distance (https://en.wikipedia.org/wiki/Euclidean_distance)."""
    return np.sqrt(((repr1 - repr2)**2).sum(axis=-1))


def variational_distance(repr1, repr2):
    """Also known as L1 or Manhattan distance (https://en.wikipedia.org/wiki/Taxicab_geometry)."""
    return np.abs(repr1 - repr2).sum(axis=-1)


def bhattacharyya_distance(repr1, repr2):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""
    sim = - np.log(np.sqrt(repr1 * repr2).sum(axis=-1))
    assert not np.isnan(sim).any(), 'Error: Similarity is nan.'

    # the similarity is -inf if no term in the review is in the vocabulary
    sim = np.where(np.isinf(sim), 0, sim)
    return sim