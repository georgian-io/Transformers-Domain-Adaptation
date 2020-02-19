#!/bin/sh

OUTPUT_DIR=$1
MAX_LENGTH=128
BATCH_SIZE=8
NUM_EPOCHS=1
SAVE_STEPS=750

python run_ner.py --data_dir ./ \
    --model_type bert \
    --labels ./labels.txt \
    --model_name_or_path ../../output \
    --output_dir $OUTPUT_DIR \
    --max_seq_length $MAX_LENGTH \
    --num_train_epochs $NUM_EPOCHS \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --do_train \
    --do_eval \
    --do_predict \
    --overwrite_cache \
    --overwrite_output_dir