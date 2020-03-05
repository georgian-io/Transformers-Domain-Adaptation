#!/bin/zsh

DATASET="linnaeus"
OUTPUT_FOLDER="results/biology/$DATASET"
FINE_TUNE_DATA_DIR="data/biology/tasks/$DATASET"
LABELS="$FINE_TUNE_DATA_DIR/labels.txt"
NUM_EPOCHS_NER=3
MAX_LENGTH=128
SUFFIX=".tsv"

OVERWRITE="TRUE"
FP16="FALSE"

# Overwrite cache and output dir if $OVERWRITE is set
OVERWRITE_ARGS=()
if ! [ -z $OVERWRITE ]; then
    OVERWRITE_ARGS+=("--overwrite_cache")
    OVERWRITE_ARGS+=("--overwrite_output_dir")
fi

FP16_ARGS=""
if [ $FP16 = "TRUE" ]; then
    FP16_ARGS="--fp16"
fi

# Create labels if they do not exist
if ! [ -e $LABELS ]; then
    python -m scripts.etl.biology.tasks.extract_ner_labels $FINE_TUNE_DATA_DIR
    if [ $? -ne 0 ]; then
        echo "Label generation failed"
        exit 0
    fi
fi

# Run NER
python -m scripts.domain_adaptation.fine_tune_ner \
    --data_dir $FINE_TUNE_DATA_DIR \
    --labels $LABELS \
    --suffix $SUFFIX \
    --output_dir $OUTPUT_FOLDER \
    --model_type bert \
    --model_name_or_path "bert-base-uncased" \
    --do_lower_case \
    --max_seq_length $MAX_LENGTH \
    --do_train \
    --num_train_epochs $NUM_EPOCHS_NER \
    --do_eval \
    --eval_all_checkpoints \
    --do_predict \
    $FP16_ARGS \
    ${OVERWRITE_ARGS[@]}