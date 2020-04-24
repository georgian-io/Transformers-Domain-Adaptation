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

FINE_TUNE_DATASET="linnaeus"
PCT=2
MOD="similar"
EXP_NAME="pubmed_${PCT}pct_${MOD}_js_pubmed_vocab"

DATA_DIR="data/biology/corpus/subsets/"
RESULTS_DIR="results/$FINE_TUNE_DATASET/$EXP_NAME/domain-pre-trained"
CORPUS="pubmed_corpus_${MOD}_jensen-shannon_linnaeus_train_2pct_pubmed_vocab.txt"
mkdir -p $DATA_DIR
mkdir -p $RESULTS_DIR

# Copy corpus and fine-tuning datasets from S3
DOMAINS=("biology")
SUBDIRECTORIES=("corpus" "tasks")
for domain in $DOMAINS; do
    # # Load corpus
    mkdir -p "data/$domain/corpus"
    aws s3 cp "$BUCKET/domains/$domain/corpus/" "data/$domain/corpus" \
      --recursive --exclude "*" --include "*.txt" --exclude "*/*"

    # Load task dataset
    if [ $domain = "biology" ]; then
        mkdir -p "data/$domain/tasks"
        aws s3 cp "$BUCKET/domains/$domain/tasks/" "data/$domain/tasks" --recursive
    fi
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

aws s3 cp "$BUCKET/domains/biology/corpus/subsets/$CORPUS" \
    "data/biology/corpus/subsets/$CORPUS"

if [ $MODE = "dpt" ]; then
    aws s3 cp "$BUCKET/runs/$FINE_TUNE_DATASET/$EXP_NAME" \
        "results/$FINE_TUNE_DATASET/$EXP_NAME" \
        --recursive --exclude="*.pt"

    # Copy state dicts for latest checkpoint, if available
    latest_checkpoint="$(get_latest_checkpoint $BUCKET/runs/$FINE_TUNE_DATASET/$EXP_NAME/domain-pre-trained/)"
    if ! [ -z $latest_checkpoint ]; then
        aws s3 sync "$BUCKET/runs/$FINE_TUNE_DATASET/$EXP_NAME/domain-pre-trained/checkpoint-$latest_checkpoint" \
            "results/$FINE_TUNE_DATASET/$EXP_NAME/domain-pre-trained/checkpoint-$latest_checkpoint"
    fi
else
    aws s3 cp "$BUCKET/runs/$FINE_TUNE_DATASET/$EXP_NAME" \
        "results/$FINE_TUNE_DATASET/$EXP_NAME" \
        --recursive --exclude="*checkpoint-*"
fi


# Update write permissions so other scripts can write into these folders
sudo chmod -R 777 data
sudo chmod -R 777 results
