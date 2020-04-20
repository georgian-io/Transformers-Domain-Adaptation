"""Module to randomly select data."""
import logging
import argparse

import numpy as np


logger = logging.getLogger(__name__)


def select_random(args: argparse.Namespace) -> np.ndarray:
    """Randomly select documents.

    Arguments:
        args {argparse.Namespace} -- Args with the following attributes
            pct {float} -- The percentage of the corpus to randomly select
            seed {Optional[int]} -- If provided, fix random seed

    Returns:
        np.ndarray -- A boolean index indicating which document to select
    """
    with open(args.corpus) as f:
        n_lines = sum(1 for _ in f)

    # Get a random subset of lines
    logger.info(f'Randomly sampling {args.pct} of corpus with '
                f'a seed of {args.seed}')
    np.random.seed(args.seed)
    selection_index = (
        np.random.choice([0, 1], size=(n_lines,),
                         p=[1 - args.pct, args.pct])
        .astype(bool)
    )
    return selection_index
