#!/bin/zsh
MODE=$1
VALID_MODES=("dpt" "ft")
if [ -z $MODE ]; then echo "Mode required as first arg."; exit 1; fi
if ! [ $MODE = "dpt" ] && ! [ $MODE = "ft" ]; then
    echo "Invalid `mode` provided."
    exit 1
fi

BUCKET="s3://nlp-domain-adaptation"
FINE_TUNE_DATASET="eurlex57k"
CORPUS="data/law/corpus/us_courts_corpus.txt"
FINE_TUNE_TEXT="data/law/corpus/${FINE_TUNE_DATASET}_train.txt"
EVAL_CORPUS="data/law/corpus/${FINE_TUNE_DATASET}_dev.txt"
TASK_DIR="data/law/tasks/$FINE_TUNE_DATASET"
OUTPUT_DIR="results/$FINE_TUNE_DATASET/bert/100pct"
CONTINUE="TRUE"

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
