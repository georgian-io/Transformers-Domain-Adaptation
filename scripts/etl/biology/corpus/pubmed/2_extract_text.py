"""Extract text data from PubMed baselines."""
import logging
import argparse
from pathlib import Path
from typing import List, Union, Optional

import pandas as pd
from lxml import etree as et
from retrying import retry

from src.utils.multiproc import parallelize
from src.utils.shell import is_file_in_use



PUBMED_DIR = Path('data/biology/corpus/pubmed')
OUTPUT_FILENAME = 'pubmed_corpus.txt'
ARTICLE_XML_PATH = 'PubmedArticle/MedlineCitation/Article'
TAG_MAPPINGS = {
    'abstract': 'Abstract/AbstractText',
    'article_title': 'ArticleTitle',
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract text data from pubmed baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--src', type=Path,
                        default=PUBMED_DIR,
                        help='Directory containing zips.')
    parser.add_argument('--dst', type=Path,
                        default=PUBMED_DIR / OUTPUT_FILENAME,
                        help='Path of file to write results to.')
    parser.add_argument('--text-fields', type=lambda x: x.split(','),
                        default=['abstract'],
                        help='A comma-delimited string of text columns to use '
                             'in processing. Valid columns are '
                             f'{set(TAG_MAPPINGS.keys())}. '
                             'Concatenation occurs in the order the text '
                             'columns are specified')
    parser.add_argument('--concat-str', type=str, default=' ',
                        help='String to concatenate text columns of the same '
                             'article together.')
    args = parser.parse_args()

    if len(set(args.text_fields) - set(TAG_MAPPINGS.keys())):
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
                 ) -> List[str]:
    root = et.parse(filename)
    df = (
        pd.Series(root.findall(ARTICLE_XML_PATH), name='articles')
        .to_frame()
    )

    def find_tag(elem, tag):
        if (child := elem.find(TAG_MAPPINGS[tag])) is not None:
            return child
        else:
            return

    # Extract tags from the appropriate XML child element
    for field in text_fields:
        df[field] = (
            df['articles']
            .apply(lambda x: child.text
                             if (child := find_tag(x, field)) is not None
                             and child.text is not None
                             else '')
        )

    # Concatenate texts
    texts = (
        df
        .drop('articles', axis=1)
        .apply(lambda x: concat_str.join(x), axis=1)
    )

    # Filter rows that are empty
    texts = texts[~texts.apply(lambda x: x == '')]
    return texts.tolist()


def extract_and_write(filename, args):
    """Extract and append text to the target file.

    Text has to be appended to be memory efficient — Expected file size ~20GB.
    """
    texts = _extract_text(filename,
                          text_fields=args.text_fields,
                          concat_str=args.concat_str)
    append_write(text=texts, dst=args.dst, filename=str(filename))


def main(args):
    # Create the blank target file for all processes to write to
    with open(args.dst, 'w+') as f:
        f.write('')

    # Extract text and append them to file
    result: List[List[str]] = (
        parallelize(extract_and_write,
                    [str(x) for x in Path(args.src).glob('*.xml.gz')],
                    args=args,
                    desc='Extracting text')
    )


if __name__ == "__main__":
    logging.getLogger()
    args = parse_args()
    main(args)
