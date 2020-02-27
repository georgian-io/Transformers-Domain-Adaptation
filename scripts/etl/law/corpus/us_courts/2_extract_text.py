"""Extract text data from US Court Jurisdictions JSON files."""
import json
import shutil
import psutil
import logging
import argparse
from pathlib import Path
from functools import partial

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t

from src.etl.cat_text import cat


logger = logging.getLogger(__name__)

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
    parser.add_argument('--spark-driver-mem', type=int, default=None,
                        help='Memory (GBs) to allocate to the Spark driver')
    args = parser.parse_args()

    return args


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

    # Concatenating columns
    CAT_COL = 'concat'
    texts_df = (
        json_df
        .withColumnRenamed('plain_text', CAT_COL)
        .select(CAT_COL)
    )

    texts_df = texts_df.filter(~((texts_df[CAT_COL] == "")
                                 | texts_df[CAT_COL].isNull()
                                 | f.isnan(texts_df[CAT_COL])))
    logger.info(f'{texts_df.count()} JSON files remain after filtering for null values.')

    # Write file into one merged text file
    merged_file = (args.dst.parent / args.dst.stem).with_suffix('.txt')
    logger.info(f'Writing text to {merged_file}')

    texts_df.write.text(str(args.dst))
    cat(args.dst, merged_file)
    shutil.rmtree(args.dst, ignore_errors=True)

    logger.info(f'Text successfully written to {merged_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
