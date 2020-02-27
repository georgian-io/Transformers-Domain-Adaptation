"""Extract text data from US Court Jurisdictions JSON files."""
import json
import shutil
import psutil
import logging
import argparse
from pathlib import Path
from functools import partial

from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t


logger = logging.getLogger(__name__)

TEXT_FIELDS = ['html_with_citations']
CORPUS_PATH = Path('data/law/corpus/us_courts/unzipped')
OUTPUT_FOLDER = CORPUS_PATH.parent / 'corpus'


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
                        default=TEXT_FIELDS,
                        help='A comma-delimited string of text columns to use '
                             'in processing. Valid columns are '
                             f'{set(TEXT_FIELDS)}. '
                             'Concatenation occurs in the order the text '
                             'columns are specified')
    parser.add_argument('--concat-str', type=str, default=' ',
                        help='String to concatenate text columns of the same '
                             'article together.')
    parser.add_argument('--spark-driver-mem', type=int, default=None,
                        help='Memory (GBs) to allocate to the Spark driver')
    args = parser.parse_args()

    if len(set(args.text_fields) - set(TEXT_FIELDS)):
        raise ValueError('Invalid text fields specified.')
    return args


def parse_html(x, concat_str: str = ' '):
    return BeautifulSoup(x, 'lxml').get_text(concat_str)


def main(args):
    # Create a directory to save outputs
    if args.dst.exists():
        logger.info(f'Found an existing destination folder. Deleting...')
        shutil.rmtree(args.dst, ignore_errors=True)

    # Create Spark Session
    if args.spark_driver_mem is not None:
        driver_mem = f'{args.spark_driver_mem}g'
    else:
        driver_mem = '{0}g'.format(int(psutil.virtual_memory().total // 1e9))
    spark = (
        SparkSession
        .builder
        .appName(__name__)
        .config('spark.driver.memory', driver_mem)
        .getOrCreate()
    )

    # Build schema
    schema = t.StructType([
        t.StructField('absolute_url', t.StringType()),
        t.StructField('author', t.StringType()),
        t.StructField('author_str', t.StringType()),
        t.StructField('cluster', t.StringType()),
        t.StructField('date_created', t.DateType()),
        t.StructField('date_modified', t.DateType()),
        t.StructField('download_url', t.StringType()),
        t.StructField('extracted_by_ocr', t.BooleanType()),
        t.StructField('html', t.StringType()),
        t.StructField('html_columbia', t.StringType()),
        t.StructField('html_lawbox', t.StringType()),
        t.StructField('html_with_citations', t.StringType()),
        t.StructField('id', t.LongType()),
        t.StructField('joined_by', t.ArrayType(t.StringType())),
        t.StructField('local_path', t.StringType()),
        t.StructField('opinions_cited', t.ArrayType(t.StringType())),
        t.StructField('page_count', t.IntegerType()),
        t.StructField('per_curiam', t.BooleanType()),
        t.StructField('plain_text', t.StringType()),
        t.StructField('resource_uri', t.StringType()),
        t.StructField('sha1', t.StringType()),
        t.StructField('type', t.StringType()),
    ])

    # Find all json files
    json_df = spark.read.json(str(args.src), schema=schema, multiLine=True)
    logger.info(f'Processing {json_df.count()} JSON files...')


    # Create UDF to parse html markups
    global parse_html
    parse_html = partial(parse_html, concat_str=args.concat_str)
    parse_html_udf = f.udf(parse_html, f.StringType())

    # Concatenating columns
    logger.info(f"Extracting text columns ({', '.join(args.text_fields)}) and "
                f"concatenating with '{args.concat_str}'...")
    CAT_COL = 'concat'
    # First concat text columns in a row (i.e. JSON file)
    # Then, concat with all rows together
    texts_df = (
        json_df
        .select(*args.text_fields)
        .withColumn(CAT_COL, f.concat_ws(' ', *args.text_fields))
        .withColumn(CAT_COL, parse_html_udf(CAT_COL))
        .agg(f.collect_list(CAT_COL).alias(CAT_COL))
        .withColumn(CAT_COL, f.concat_ws(' ', CAT_COL))
    )

    # Write file
    logger.info(f'Writing text to {args.dst}')
    texts_df.write.text(str(args.dst))
    logger.info(f'Text successfully saved to {args.dst}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
