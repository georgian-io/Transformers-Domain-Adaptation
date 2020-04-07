#!/bin/bash
BUCKET="s3://nlp-domain-adaptation"

if [ $(basename $(pwd)) != "NLP-Domain-Adaptation" ]; then
    echo "This script is intended to run in the NLP-Domain-Adaptation folder."
    echo "Move to the correct folder before running this again."
    exit 1
fi

# Copy corpus and fine-tuning datasets from S3
DOMAINS=("biology" "law")
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

# Copy cached folders
if ! [ -e results ]; then mkdir results; fi

DPT_COMPLETIONS=( [10]=10000 [25]=30000 [50]=60000 [75]=95000 )
for DPT_COMPLETION in ${!DPT_COMPLETIONS[@]}; do
    CKPT_NUM=${DPT_COMPLETIONS[$DPT_COMPLETION]}
    DEST=results/linnaeus/pubmed_2pct_seed281_${DPT_COMPLETION}pct_dpt/domain-pre-trained
    mkdir -p $DEST
    aws s3 cp "$BUCKET/cache/pubmed_2pct_seed281/domain-pre-trained/checkpoint-$CKPT_NUM" $DEST --recursive --exclude "*pt"
done

# Update write permissions so other scripts can write into these folders
sudo chmod -R 777 data
sudo chmod -R 777 results
