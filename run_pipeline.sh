#!/bin/zsh
MODE=$1
VALID_MODES=("dpt" "ft")
if [ -z $MODE ]; then echo "Mode required as first arg."; exit 1; fi
if ! [ $MODE = "dpt" ] && ! [ $MODE = "ft" ]; then
    echo "Invalid `mode` provided."
    exit 1
fi

BUCKET="s3://nlp-domain-adaptation"
FINE_TUNE_DATASET="linnaeus"
PCT=2
MOD="most"
CORPUS="data/biology/corpus/subsets/pubmed_corpus_${MOD}_sim_div_1_jensen-shannon_1_entropy_linnaeus_train_union_2pct.txt"
FINE_TUNE_TEXT="data/biology/corpus/${FINE_TUNE_DATASET}_train.txt"
EVAL_CORPUS="data/biology/corpus/${FINE_TUNE_DATASET}_dev.txt"
TASK_DIR="data/biology/tasks/$FINE_TUNE_DATASET"
OUTPUT_DIR="results/$FINE_TUNE_DATASET/pubmed_${PCT}pct_${MOD}_sim_div_union"
MAX_STEPS="128194"
CONTINUE="TRUE"

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

# Periodically sync training artifacts to S3
watch -n 300 \
    aws s3 sync $OUTPUT_DIR \
        $BUCKET/runs/${OUTPUT_DIR/results\/} \
        --exclude "*fine-tuned/checkpoint*" \
        > /dev/null 2>&1 &
DAEMON_PID=$!

# Continue domain pre-training from a checkpoint if possible
if [ $CONTINUE = "TRUE" ] \
   && [ -e $OUTPUT_DIR/domain-pre-trained ] \
   && ! [ $(ls $OUTPUT_DIR/domain-pre-trained | grep "checkpoint" | wc -l) = 0 ]; then
    CONTINUE_ARG="--should-continue"
fi

# Run domain adaptation
if [ $MODE = "dpt" ]; then  # Domain pre-training
    ./domain_adaptation_pipeline.sh \
    --corpus $CORPUS \
    --eval-corpus $EVAL_CORPUS \
    -o $OUTPUT_DIR \
    --overwrite-output-dir \
    --fine-tune-data-dir $TASK_DIR \
    --max-steps $MAX_STEPS \
    --batch-size 8 \
    --save-steps 2500 \
    --skip-augment-vocab \
    --skip-fine-tune \
    --distributed-train \
    -v $CONTINUE_ARG
else  # Fine tuning
    ./domain_adaptation_pipeline.sh \
    --corpus $CORPUS \
    --eval-corpus $EVAL_CORPUS \
    -o $OUTPUT_DIR \
    --overwrite-output-dir \
    --fine-tune-data-dir $TASK_DIR \
    --max-steps $MAX_STEPS \
    --batch-size 8 \
    --save-steps 2500 \
    --skip-augment-vocab \
    --skip-domain-pre-train \
    -v $CONTINUE_ARG
fi

# Run end-of-training sync
kill $DAEMON_PID # Kill syncing daemon to prevent race conditions
aws s3 sync $OUTPUT_DIR \
    $BUCKET/runs/${OUTPUT_DIR/results\/} \
    --exclude "*fine-tuned/checkpoint*"
