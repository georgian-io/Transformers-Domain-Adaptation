"""
This module contains similarity- and diversity-based data selection metrics.

These metrics were proposed by Ruder and Plank
in their `paper <https://arxiv.org/pdf/1707.05246.pdf>`_.
"""

from .similarity import SIMILARITY_FEATURES, similarity_func_factory
from .diversity import DIVERSITY_FEATURES, diversity_func_factory
