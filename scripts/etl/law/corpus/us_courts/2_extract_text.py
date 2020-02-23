"""Extract text data from US Court Jurisdictions JSON files."""
import json
import shutil
import logging
import argparse
from pathlib import Path

import apache_beam as beam
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


logger = logging.getLogger(__name__)

TEXT_FIELDS = ['plain_text']
CORPUS_PATH = Path('data/law/corpus/us_courts/all')
OUTPUT_FOLDER = CORPUS_PATH / 'corpus'
FILE_PREFIX = 'us_courts_corpus'


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract text data from US Court Jurisdictions JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', type=Path, default=CORPUS_PATH,
                        help='Directory containing JSON court opinions.')
    parser.add_argument('--dst', type=Path, default=OUTPUT_FOLDER,
                        help='Output folder to write corpus shards.')
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
    known_args, pipeline_args = parser.parse_known_args()

    if len(set(known_args.text_fields) - set(TEXT_FIELDS)):
        raise ValueError('Invalid text fields specified.')
    return known_args, pipeline_args


def main(known_args, pipeline_args, save_main_session=True):
    # Find all JSON files
    files = [str(x) for x in Path(known_args.src).rglob('*.json')]
    logger.info(f'Found {len(files)} JSON files to process.')

    # Create save_directory
    if known_args.dst.exists():
        logger.info(f'Found an existing destination folder. Deleting...')
        shutil.rmtree(known_args.dst, ignore_errors=True)
    known_args.dst.mkdir(parents=True)

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        lines = (
            p
            | 'InitPipelineWithFilePaths' >> beam.Create(files)
            | 'CreateJSON'
                >> beam.Map(lambda x: json.loads(Path(x).read_text()))
            | 'ExtractTextFromFields'
                >> beam.Map(lambda x: [v for k, v in x.items()
                                         if k in known_args.text_fields])
            | 'ConcatTextFields'
                >> beam.Map(lambda x: known_args.concat_str.join(x))
            | 'WriteToText' >> WriteToText(str(known_args.dst / FILE_PREFIX))
        )
    logging.info('Processing complete! '
                 f'The results are saved in {known_args.dst}.')


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    known_args, pipeline_args = parse_args()
    main(known_args, pipeline_args)
