#!/bin/zsh
FINE_TUNE_DATASET="BC2GM"
PCT=2
SEED=281
DPT_COMPLETION=$1
if [ -z $DPT_COMPLETION ]; then echo "Require DPT COMPLETION as first arg"; exit 1; fi
CORPUS="data/biology/corpus/shards"
FINE_TUNE_TEXT="data/biology/corpus/${FINE_TUNE_DATASET}_train.txt"
EVAL_CORPUS="data/biology/corpus/${FINE_TUNE_DATASET}_dev.txt"
TASK_DIR="data/biology/tasks/$FINE_TUNE_DATASET"
OUTPUT_DIR="results/$FINE_TUNE_DATASET/pubmed_${PCT}pct_seed${SEED}_${DPT_COMPLETION}pct_dpt"
MAX_STEPS="10000"

LABELS=$TASK_DIR/labels.txt


# NER fine tuning args
export MAX_LENGTH=128
export NUM_EPOCHS_NER=25

# Create labels if they do not exist
if ! [ -e $LABELS ]; then
    python -m scripts.etl.biology.tasks.extract_ner_labels $TASK_DIR
    if [ $? -ne 0 ]; then
        echo "Label generation failed"
        exit 0
    fi
fi

# Set up daemon to sync training results
function sync_results() {
    aws s3 sync $OUTPUT_DIR \
        "s3://nlp-domain-adaptation/runs/$FINE_TUNE_DATASET/$(basename $OUTPUT_DIR)" \
        --recursive
}
watch -n 1800 sync_results &
SYNC_PID=$!

# Run domain adaptation
./domain_adaptation_pipeline.sh \
    --corpus $CORPUS \
    -o $OUTPUT_DIR \
    --overwrite-output-dir \
    --fine-tune-data-dir $TASK_DIR \
    --max-steps $MAX_STEPS \
    --batch-size 8 \
    --save-steps 2500 \
    --skip-augment-vocab \
    --skip-domain-pre-train \
    -v
./scripts/sync_tb_logs.sh $OUTPUT_DIR

# Clean up sync daemon and run end-of-training sync
kill $SYNC_PID
sync_results
