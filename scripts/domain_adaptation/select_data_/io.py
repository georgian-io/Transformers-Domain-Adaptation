"""File I/O utility module."""
import logging
import argparse
from pathlib import Path
from typing import Union, Iterable

import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def get_docs(filepath: Union[str, Path]) -> Iterable[str]:
    """Return a stream of documents."""
    logger.info(f'Reading {filepath}')

    # Get the total number of lines for processing time estimates
    with open(filepath) as f:
        n_lines = sum(1 for _ in f)

    # Yield a stream of text
    with open(filepath) as f:
        yield from tqdm(f, desc='Reading', total=n_lines)


def create_selected_docs_subset(index: np.ndarray,
                                args: argparse.Namespace) -> None:
    """Create a subset corpus by copying selected documents.

    Arguments:
        index {np.ndarray} -- A boolean index indicating which document from the corpus to include in the subset
        args {argparse.Namespace} -- Args with the following attributes
            dst {Path} -- Path to output directory
            filename {Path} -- Docs subset filename
            corpus {Path} -- Path to corpus file
    """
    # Save corpus
    logger.info(f'Saving subset corpus to {args.dst / args.filename}')
    args.dst.mkdir(exist_ok=True, parents=True)
    with open(args.corpus) as reader:
        with open(args.dst / args.filename, 'w+') as writer:
            # Read and sample
            lines = (line for line, should_sample in zip(reader, index)
                          if should_sample)

            # Write
            lines = tqdm(lines, desc='Writing',
                         leave=False, total=index.sum())
            list(writer.write(line) for line in lines)
