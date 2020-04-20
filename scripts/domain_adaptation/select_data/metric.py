"""Module to calculate similarity or diversity of documents."""
import sys
from functools import partial

sys.path.append('learn-to-select-data')
import similarity
import features as diversity
from constants import SIMILARITY_FUNCTIONS, DIVERSITY_FEATURES

from .io import get_docs
from . import featurization


DIVERSITY_FUNCTIONS = [f for f in DIVERSITY_FEATURES if f != 'quadratic_entropy']


def calculate_similarity(args: argparse.Namespace):
    tokenize_fn = partial()
    # Featurize the Fine Tune corpus into a term distribution / TF-IDF vector
    if args.use_tfidf:
        tokenized = featurization.docs_to_tokens(docs=get_docs(args.corpus), vocab_file=args.vocab_file, lowercase=args.lowercase, chunk_size=args.chunk_size)
        vectorizer = featurization.get_fitted_tfidf_vectorizer(
            get_docs(args.corpus), norm=args.norm
        )

        ft_repr = (
            featurization.tokens_to_tfidf(tokenized=get_docs(args.fine_tune_text),
                                          vectorizer=vectorizer,
                                          level='corpus')
        )

        corpus_repr = (
            featurization.tokens_to_tfidf(tokenized)
        )



def calculate_diversity():
    pass