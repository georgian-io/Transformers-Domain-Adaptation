#!/bin/zsh
# Set default arguments
POSITIONAL=()
BERT_VOCAB="bert-base-uncased-vocab.txt"
BATCH_SIZE=4
EPOCHS_DPT=1
MAX_STEPS=-1
LEARNING_RATE=5e-5
WARMUP_STEPS=0

FP16=""

SAVE_STEPS=2500
SAVE_TOTAL_LIMIT=50

OVERWRITE_CACHE=""
OVERWRITE_OUTPUT_DIR=""
SHOULD_CONTINUE=""

DISTRIBUTED_TRAIN="FALSE"

IFS="="  # Change the internal field separator
while [ $# -gt 0 ]; do
    ARG1="$1"
    EXTRA_SHIFT="TRUE"  # Performs double shifts of args
    if grep -q $IFS <<< $ARG1; then
        read -r ARG1 ARG2 <<< $ARG1
        EXTRA_SHIFT="FALSE"  # Perform only single shift of args
    elif [ $# -gt 1 ]; then
        ARG2="$2"
    fi

    case $ARG1 in
        -o|--output-dir)
        OUTPUT_DIR="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --corpus)
        CORPUS="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --eval-corpus)
        EVAL_CORPUS="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --fine-tune-text)
        FINE_TUNE_TEXT="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --bert-vocab)
        BERT_VOCAB="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --learning-rate)
        LEARNING_RATE="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        -b|--batch-size)
        BATCH_SIZE=$ARG2
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --epochs-dpt)
        EPOCHS_DPT=$ARG2
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --warmup-steps)
        WARMUP_STEPS=$ARG2
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --max-steps)
        MAX_STEPS=$ARG2
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --fine-tune-data-dir)
        FINE_TUNE_DATA_DIR="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --save-steps)
        SAVE_STEPS=$ARG2
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --save-total-limit)
        SAVE_TOTAL_LIMIT=$ARG2
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --fp16)
        FP16="--fp16"
        shift
        ;;
        --overwrite-cache)
        OVERWRITE_CACHE="--overwrite_cache"
        shift
        ;;
        --overwrite-output-dir)
        OVERWRITE_OUTPUT_DIR="--overwrite_output_dir"
        shift
        ;;
        --should-continue)
        SHOULD_CONTINUE="--should_continue"
        shift
        ;;
        --skip-augment-vocab)
        SKIP_AUGMENT_VOCAB=TRUE
        shift
        ;;
        --skip-domain-pre-train)
        SKIP_DOMAIN_PRE_TRAIN=TRUE
        shift
        ;;
        --skip-fine-tune)
        SKIP_FINE_TUNE=TRUE
        shift
        ;;
        --distributed-train)
        DISTRIBUTED_TRAIN=TRUE
        shift
        ;;
        -v|--verbose)
        VERBOSE=TRUE
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITIONAL[@]}"  # Restore positional parameters
IFS=" "  # Reset IFS

# TODO Error out on new args

# ----------- Args post-processing -----------
# Check that args are properly specified
if [ -z $OUTPUT_DIR ]; then
    echo "--output-dir must be specified."
    exit 1
fi
if [[ -z $CORPUS && \
      ! ( $SKIP_AUGMENT_VOCAB = "TRUE" || \
          $SKIP_DOMAIN_PRE_TRAIN = "TRUE" ) ]]; then
    echo "--corpus must be specified unless \
          --skip-augment-vocab and --skip-domain-pre-train \
          are provided."
    exit 1
fi
if [[ -z $FINE_TUNE_DATA_DIR && ! $SKIP_FINE_TUNE = "TRUE" ]]; then
    echo "--fine-tune-data-dir must be specified unless --skip-fine-tune is provided."
    exit 1
fi

# Directories
AUGMENTED_VOCAB_FOLDER="$OUTPUT_DIR/augmented-vocab"
DOMAIN_PRE_TRAIN_FOLDER="$OUTPUT_DIR/domain-pre-trained"
FINE_TUNE_FOLDER="$OUTPUT_DIR/fine-tuned"

# Create intermediary args for use in scripts
# 1. Expand $CORPUS to all child text files
# 2. Include additional corpora like an eval corpus or corpora from fine tuning
if [ -f $CORPUS ]; then
    CORPUS_ARGS=$CORPUS
elif [ -d $CORPUS ]; then
    for shard in $CORPUS/*.txt; do
        if [ -z $CORPUS_ARGS ]; then
            CORPUS_ARGS=$shard
        else
            CORPUS_ARGS="$CORPUS_ARGS,$shard"
        fi
    done
fi
if ! [ -z $FINE_TUNE_TEXT ]; then
    CORPUS_ARGS="$CORPUS_ARGS,$FINE_TUNE_TEXT"
fi

if ! [ -z $SHOULD_CONTINUE ]; then
    OVERWRITE_OUTPUT_DIR="--overwrite_output_dir"
fi

EVAL_CORPUS_ARGS=()
if ! [ -z $EVAL_CORPUS ]; then
    # read -r EVAL_CORPUS_ARGS <<< "--do_eval --eval_data_file $EVAL_CORPUS"  # TODO Figure out
    if ! [ -e $EVAL_CORPUS ]; then
        echo "$EVAL_CORPUS does not exist"
        exit 1
    fi
    EVAL_CORPUS_ARGS+=("--do_eval")
    EVAL_CORPUS_ARGS+=("--eval_data_file")
    EVAL_CORPUS_ARGS+=($EVAL_CORPUS)
fi

TOKENIZER_VOCAB=""
if [[ -z $SKIP_AUGMENT_VOCAB || ( -e "$AUGMENTED_VOCAB_FOLDER/vocab.txt" ) ]]; then
    TOKENIZER_VOCAB="$AUGMENTED_VOCAB_FOLDER/vocab.txt"
else
    AUGMENTED_VOCAB_FOLDER="$OUTPUT_DIR/vocab"
    TOKENIZER_VOCAB="$AUGMENTED_VOCAB_FOLDER/vocab.txt"
    mkdir -p $AUGMENTED_VOCAB_FOLDER
    cp "bert-base-uncased-vocab.txt" $TOKENIZER_VOCAB
fi


# If --verbose is provided, print all parameters
if ! [ -z $VERBOSE ]; then
    echo
    echo '----------- Parameters -----------'
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "CORPUS: $CORPUS"
    echo "EVAL_CORPUS: $EVAL_CORPUS"
    echo "FINE_TUNE_TEXT: $FINE_TUNE_TEXT"
    echo "CORPUS_ARGS: $CORPUS_ARGS"
    echo "BERT_VOCAB: $BERT_VOCAB"
    echo "LEARNING_RATE: $LEARNING_RATE"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "FP16: $FP16"
    echo "EPOCHS_DPT: $EPOCHS_DPT"
    echo "MAX_STEPS: $MAX_STEPS"
    echo "WARMUP_STEPS: $WARMUP_STEPS"
    echo "TOKENIZER_VOCAB: $TOKENIZER_VOCAB"
    echo "FINE_TUNE_DATA_DIR: $FINE_TUNE_DATA_DIR"
    echo "SAVE_STEPS: $SAVE_STEPS"
    echo "SAVE_TOTAL_LIMIT: $SAVE_TOTAL_LIMIT"
    echo "OVERWRITE_CACHE: $OVERWRITE_CACHE"
    echo "OVERWRITE_OUTPUT_DIR: $OVERWRITE_OUTPUT_DIR"
    echo "SKIP_AUGMENT_VOCAB: $SKIP_AUGMENT_VOCAB"
    echo "SKIP_DOMAIN_PRE_TRAIN: $SKIP_DOMAIN_PRE_TRAIN"
    echo "SKIP_FINE_TUNING: $SKIP_FINE_TUNE"
    echo "DISTRIBUTE_TRAIN: $DISTRIBUTE_TRAIN"
    echo "AUGMENTED_VOCAB_FOLDER: $AUGMENTED_VOCAB_FOLDER"
    echo "DOMAIN_PRE_TRAIN_FOLDER: $DOMAIN_PRE_TRAIN_FOLDER"
    echo "FINE_TUNE_FOLDER: $FINE_TUNE_FOLDER"
    echo
fi

# Augmenting tokenizer
if ! [ -z $SKIP_AUGMENT_VOCAB ]; then
    echo "Skipping vocabulary augmentation as specified by user."
else
    echo
    echo "**************************************************"
    echo "Augmenting vocabulary with in-domain terminologies"
    echo "**************************************************"
    python -m scripts.domain_adaptation.augment_vocab \
        --bert-vocab $BERT_VOCAB \
        --corpus $CORPUS_ARGS \
        --dst $AUGMENTED_VOCAB_FOLDER \
        $OVERWRITE_CACHE
    if [ $? -ne 0 ]; then
        echo "Vocabulary augmentation failed. Halting pipeline."
        exit 1
    fi
fi

# Domain pre-train model
if ! [ -z $SKIP_DOMAIN_PRE_TRAIN ]; then
    echo "Skipping domain pre-training as specified by user."
else
    echo
    echo "********************************************************"
    echo "Performing domain pre-training on BERT with $CORPUS"
    echo "********************************************************"
    if [ $DISTRIBUTED_TRAIN = "TRUE" ]; then
        N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        CMD="python -m torch.distributed.launch --nproc_per_node 8 ./scripts/domain_adaptation/domain_pre_train.py"
    else
        CMD="python -m scripts.domain_adaptation.domain_pre_train"
    fi
    ${(z)CMD} \
        --output_dir $DOMAIN_PRE_TRAIN_FOLDER \
        --model_type "bert" \
        --tokenizer_vocab $TOKENIZER_VOCAB \
        --model_name_or_path "bert-base-uncased" \
        --block_size 512 \
        --do_train \
        --num_train_epochs $EPOCHS_DPT \
        --warmup_steps $WARMUP_STEPS \
        --max_steps $MAX_STEPS \
        --train_data_file $CORPUS_ARGS \
        --learning_rate $LEARNING_RATE \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --mlm \
        --save_steps $SAVE_STEPS \
        --save_total_limit $SAVE_TOTAL_LIMIT \
        --evaluate_during_training \
        --eval_all_checkpoints \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        $FP16 \
        ${EVAL_CORPUS_ARGS[@]} \
        $SHOULD_CONTINUE \
        $OVERWRITE_CACHE \
        $OVERWRITE_OUTPUT_DIR
    if [ $? -ne 0 ]; then
        echo "Domain pre-training failed. Halting pipeline."
        exit 1
    fi
fi

# Fine tune model (currently only available for NER)
if ! [ -z $SKIP_FINE_TUNE ]; then
    echo "Skipping fine-tuning as specified by user."
else
    echo
    echo "************************************"
    echo "Fine-tuning domain pre-trained model"
    echo "************************************"
    python -m scripts.domain_adaptation.fine_tune_ner \
        --data_dir $FINE_TUNE_DATA_DIR \
        --labels "$FINE_TUNE_DATA_DIR/labels.txt" \
        --suffix ".tsv" \
        --output_dir $FINE_TUNE_FOLDER \
        --model_type "bert" \
        --model_name_or_path $DOMAIN_PRE_TRAIN_FOLDER \
        --do_lower_case \
        --max_seq_length $MAX_LENGTH \
        --do_train \
        --num_train_epochs $NUM_EPOCHS_NER \
        --do_eval \
        --eval_all_checkpoints \
        --evaluate_during_training \
        --do_predict \
        --save_steps 1000 \
        $FP16 \
        $OVERWRITE_OUTPUT_DIR
fi
