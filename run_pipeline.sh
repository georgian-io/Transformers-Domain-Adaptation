#!/bin/zsh
CORPUS="data/biology/tasks/linnaeus/corpus/linnaeus_corpus.txt"
TASK_DIR="data/biology/tasks/linnaeus"
OUTPUT_DIR="results/bio_linnaeus/dpt_default"
CONTINUE_DPT="FALSE"
MAX_STEPS="10000"

LABELS=$TASK_DIR/labels.txt


# NER fine tuning args
export MAX_LENGTH=128
export NUM_EPOCHS_NER=3

# Create labels if they do not exist
if ! [ -e $LABELS ]; then
    python -m scripts.etl.biology.tasks.extract_ner_labels $TASK_DIR
    if [ $? -ne 0 ]; then
        echo "Label generation failed"
        exit 0
    fi
fi

SHOULD_CONTINUE=""
if [ $CONTINUE_DPT = "TRUE" ]; then
    SHOULD_CONTINUE="--should-continue"
fi

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
    $SHOULD_CONTINUE \
    -v
