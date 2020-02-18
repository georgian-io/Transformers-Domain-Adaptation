"""Extract text data from US Court Jurisdiction JSON files."""
import json
import logging
import argparse
from pathlib import Path
from typing import List, Union, Optional

import pandas as pd
from retrying import retry

from src.utils.multiproc import parallelize
from src.utils.shell import is_file_in_use
from src.utils.text import clean as clean_text


TEXT_FIELDS = ['plain_text']
CORPUS_PATH = Path('data/law/corpus/us_court_jurisdictions_opinions/acca')
OUTPUT_FILENAME = 'us_courts_corpus.txt'


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract text data from pubmed baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--src', type=Path,
                        default=CORPUS_PATH,
                        help='Directory containing court opinions.')
    parser.add_argument('--dst', type=Path,
                        default=CORPUS_PATH.parent / OUTPUT_FILENAME,
                        help='Path of file to write extracted texts to.')
    parser.add_argument('--text-fields', type=lambda x: x.split(','),
                        default=['plain_text'],
                        help='A comma-delimited string of text columns to use '
                             'in processing. Valid columns are '
                             f'{set(TEXT_FIELDS)}. '
                             'Concatenation occurs in the order the text '
                             'columns are specified')
    parser.add_argument('--concat-str', type=str, default=' ',
                        help='String to concatenate text columns of the same '
                             'article together.')
    args = parser.parse_args()

    if len(set(args.text_fields) - set(TEXT_FIELDS)):
        raise ValueError('Invalid text column values provided.')

    return args


@retry(retry_on_result=lambda x: x == True,
       wait_random_min=1000,
       wait_random_max=3000)
def append_write(text: Union[str, List[str]],
                 dst: Union[str, Path],
                 filename: Union[str, Path],
                ) -> Optional[bool]:
    # Retry append if the file is being used by another process
    if is_file_in_use(str(dst)):
        logging.warning(f'{dst} is in use. Re-attempting write of {filename}')
        return True

    if isinstance(text, list):
        text = '\n'.join(text)
    text += '\n'
    with open(dst, 'a+') as f:
        f.write(text)


def _extract_text(filename: str,
                  text_fields: List[str],
                  concat_str: str
                 ) -> str:
    with open(filename, 'r') as f:
        data = json.load(f)
    texts = concat_str.join([clean_text(data[field]) for field in text_fields])
    return texts


def extract_and_write(filename, args):
    """Extract and append text to the target file.

    Text has to be appended to be memory efficient — Expected file size ~20GB.
    """
    texts = _extract_text(filename,
                          text_fields=args.text_fields,
                          concat_str=args.concat_str)
    append_write(text=texts, dst=args.dst, filename=filename)


def main(args):
    # Create the blank target file for all processes to write to
    with open(args.dst, 'w+') as f:
        f.write('')

    # Extract text and append them to file
    result: List[List[str]] = (
        parallelize(extract_and_write,
                    [str(x) for x in Path(args.src).rglob('*.json')],
                    args=args,
                    n_workers=2,  # TEMP
                    desc='Extracting text')
    )


if __name__ == "__main__":
    logging.getLogger()
    args = parse_args()
    main(args)
