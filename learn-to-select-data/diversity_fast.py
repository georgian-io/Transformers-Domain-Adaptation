"""Module that implements vectorized versions of diversity functions."""
import numpy as np


def number_of_word_types(example):
    """Counts the number of word types of the example."""
    return len(set(example))


def type_token_ratio(example):
    """Calculates the type-token ratio of the example."""
    if not len(example):
        return 1
    return number_of_word_types(example) / len(example)


def entropy(example, train_term_dist, word2id):
    """Calculates Entropy (https://en.wikipedia.org/wiki/Entropy_(information_theory))."""
    summed = 0
    for word in set(example):
        if word in word2id:
            p_word = train_term_dist[word2id[word]]
            summed += p_word * np.log(p_word)
    return - summed


def simpsons_index(example, train_term_dist, word2id):
    """Calculates Simpson's Index (https://en.wikipedia.org/wiki/Diversity_index#Simpson_index)."""
    if not len(example):
        return 0
    example = {word for word in example if word in word2id}
    word_ids = [word2id[word] for word in example]
    score = (train_term_dist[word_ids]**2).sum()
    return score



def quadratic_entropy(example, train_term_dist, word2id, word2vec):
    """Calculates Quadratic Entropy."""
    assert word2vec is not None, ('Error: Word vector representations have to '
                                  'be available for quadratic entropy.')

    if not len(example):
        return 0

    # Only retain words that exist in both `word2id` and `word2vec` since
    # the product will be 0 otherwise
    example = {word for word in example
                    if (word in word2id and word in word2vec)}

    # Get probabiltiies
    word_ids = [word2id[word] for word in example]
    p = train_term_dist[word_ids]  # (N,)
    p_mat = p.reshape(-1, 1) * p.reshape(1, -1)  # (N, N)

    # Get embeddings
    vec = np.array([word2vec[word] for word in example])  # (N, D)

    def cosine_similarity_vect(vec: np.ndarray) -> np.ndarray:
        """Calculate cosine_similarity (vectorized).

        Arguments:
            vec {np.ndarray} -- Embeddings (N x D)

        Returns:
            np.ndarray -- Pairwise cosine similarities (N x N)
        """
        N, D = vec.shape
        uv = (vec.reshape(N, 1, D) * vec.reshape(1, N, D)).mean(axis=-1)
        uu = (vec**2).mean(axis=-1)
        vv = (vec**2).mean(axis=-1)
        dist = uv / np.sqrt(uu.reshape(N, 1) * vv.reshape(1, N))
        return dist

    summed = (cosine_similarity_vect(vec) * p_mat).sum()
    return summed


def renyi_entropy(example, domain_term_dist, word2id):
    """Calculates Rényi Entropy (https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy)."""
    example = {word for word in example if word in word2id}
    word_ids = [word2id[word] for word in example]

    alpha = 0.99
    summed = (domain_term_dist[word_ids]**alpha).sum()
    if summed == 0:
        # 0 if none of the words appear in the dictionary;
        # set to a small constant == low prob instead
        summed = 0.0001
    score = 1 / (1 - alpha) * np.log(summed)
    return score