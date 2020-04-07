#!/bin/bash
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
MOD="most"
EXP_NAME="pubmed_${PCT}pct_${MOD}_sim_div"

DATA_DIR="data/biology/corpus/subsets/"
RESULTS_DIR="results/$FINE_TUNE_DATASET/pubmed_${PCT}pct_${MOD}_diverse/domain-pre-trained"
CORPUS="pubmed_corpus_${MOD}_sim_div_1_0_jensen-shannon_1_0_entropy_linnaeus_train_2pct.txt"
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

aws s3 cp "$BUCKET/domains/biology/corpus/subsets/$CORPUS" \
    "data/biology/corpus/subsets/$CORPUS"

aws s3 cp "$BUCKET/runs/$FINE_TUNE_DATASET/$EXP_NAME" \
    "results/$FINE_TUNE_DATASET/$EXP_NAME" \
    --recursive

# Update write permissions so other scripts can write into these folders
sudo chmod -R 777 data
sudo chmod -R 777 results
