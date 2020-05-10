#!/bin/bash

MODE=${1:-"dpt"}
VALID_MODES=("dpt" "ft")
if [ -z $MODE ]; then echo "Mode required as first arg."; exit 1; fi
if ! [ $MODE = "dpt" ] && ! [ $MODE = "ft" ]; then
    echo "Invalid `mode` provided."
    exit 1
fi

BUCKET="s3://nlp-domain-adaptation"

if [ $(basename $(pwd)) != "NLP-Domain-Adaptation" ]; then
    echo "This script is intended to run in the NLP-Domain-Adaptation folder."
    echo "Move to the correct folder before running this again."
    exit 1
fi

# Copy cached folders
if ! [ -e results ]; then mkdir results; fi

FINE_TUNE_DATASET="eurlex57k"
EXP_NAME="2pct_random_seed42"
CORPUS="us_courts_corpus_random_2pct_seed42.txt"

EXP_DIR="data_select/$FINE_TUNE_DATASET/$EXP_NAME"
CORPUS_PATH="law/corpus/subsets/random/$CORPUS"

# Copy corpus and fine-tuning datasets from S3
DOMAINS=("law")
SUBDIRECTORIES=("corpus" "tasks")
for domain in $DOMAINS; do
    # Load corpus
    aws s3 cp "$BUCKET/domains/$domain/corpus/" "data/$domain/corpus" \
        --recursive --exclude "*" --include "*.txt" --exclude "*corpus*" --exclude "*/*"

    # Load task dataset
    aws s3 cp "$BUCKET/domains/$domain/tasks/" "data/$domain/tasks" \
        --recursive --exclude "*" --include "*.t*"
done

function get_latest_checkpoint() {
    latest_checkpoint=$(
        aws s3 ls $1 \
            | grep "checkpoint-" \
            | grep -v "end" \
            | awk -F "(PRE checkpoint-|/)" '{print $2}' \
            | sort -n \
            | tail -1
    )
    echo $latest_checkpoint
}

aws s3 cp "$BUCKET/domains/$CORPUS_PATH" "data/$CORPUS_PATH"

if [ $MODE = "dpt" ]; then
    aws s3 cp "$BUCKET/runs/$EXP_DIR" \
        "results/$EXP_DIR" \
        --recursive --exclude="*checkpoint*.pt"

    # Copy state dicts for latest checkpoint, if available
    latest_checkpoint="$(get_latest_checkpoint $BUCKET/runs/$EXP_DIR/domain-pre-trained/)"
    if ! [ -z $latest_checkpoint ]; then
        aws s3 sync "$BUCKET/runs/$EXP_DIR/domain-pre-trained/checkpoint-$latest_checkpoint" \
            "results/$EXP_DIR/domain-pre-trained/checkpoint-$latest_checkpoint"
    fi
else
    aws s3 cp "$BUCKET/runs/$EXP_DIR" \
        "results/$EXP_DIR" \
        --recursive --exclude="*checkpoint-*"
fi


# Update write permissions so other scripts can write into these folders
sudo chmod -R 777 data
sudo chmod -R 777 results
