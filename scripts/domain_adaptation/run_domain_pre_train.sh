#!/bin/sh
TRAIN_FILE=$1
EVAL_FILE=$2
TOKENIZER_VOCAB=$3

# --block_size and --overwrite* are TEMP until a workaround if found
python -m tutorials.dpt.run_language_modelling \
    --output_dir=output \
    --model_type=bert \
    --tokenizer_name=$TOKENIZER_VOCAB \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --overwrite_cache \
    --overwrite_output_dir \
    --block_size=512 \
    --mlm
