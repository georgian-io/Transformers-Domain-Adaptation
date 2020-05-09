"""Metric definitions."""
import numpy as np


def r_precision_at_k(pred: np.ndarray, label: np.ndarray, k: int = 5) -> float:
    """Calculate R-Precision@K as defined in the Eurlex57K paper.

    Reference: https://arxiv.org/abs/1905.10892

    Arguments:
        pred {np.ndarray} -- Multi-hot encoded predictions of shape (N x C)
        label {np.ndarray} -- Multi-hot encoded predictions of shape (N x C)
    -- where C is the number of classes

    Keyword Arguments:
        k {int} -- Number of positive class predictions per example (default: {5})

    Raises:
        ValueError: If any prediction doesn't have the same number of positive
                    class predictions as `k`
    """
    if not ((pred != 0).sum(axis=1) == k).all():
        raise ValueError('Ensure that number of predictions for all examples '
                         'are the same as `k`.')

    # `& label.astype(bool)` is a Boolean mask to enforce
    # exclusive matching of non-zero labels
    num = (pred == label) & label.astype(bool)
    denom = np.minimum(label.sum(axis=1, keepdims=True), k)

    return (num / denom).sum(axis=1).mean(axis=0)
