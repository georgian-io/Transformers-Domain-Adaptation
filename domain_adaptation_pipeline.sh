#!/bin/zsh
# Set default arguments
POSITIONAL=()
BERT_VOCAB="bert-base-uncased-vocab.txt"
BATCH_SIZE=4
EPOCHS_DPT=1

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
        --fine-tune-data-dir)
        FINE_TUNE_DATA_DIR="$ARG2"
        shift;
        if [ $EXTRA_SHIFT = "TRUE" ]; then shift; fi
        ;;
        --fp16)
        FP16=TRUE
        shift
        ;;
        --overwrite)
        OVERWRITE=TRUE
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


# Create intermediary args for use in scripts
CORPUS_ARGS="$CORPUS"
for VAR in $EVAL_CORPUS $FINE_TUNE_TEXT; do
    if ! [ -z $VAR ]; then
        CORPUS_ARGS="$VAR,$CORPUS"
    fi
done

OVERWRITE_ARGS=()
if ! [ -z $OVERWRITE ]; then
    OVERWRITE_ARGS+=("--overwrite_cache")
    OVERWRITE_ARGS+=("--overwrite_output_dir")
fi

DPT_EVAL_ARGS=()
if ! [ -z $EVAL_CORPUS ]; then
    read -ra DPT_EVAL_ARGS <<< "--do eval --eval_data_file $EVAL_CORPUS"
fi

FP16_ARGS=""
if ! [ -z $FP16 ]; then
    FP16_ARGS="--fp16"
fi

# Directories
DOMAIN_PRE_TRAIN_FOLDER="$OUTPUT_DIR/domain-pre-trained"
AUGMENTED_VOCAB_PATH="$OUTPUT_DIR/augmented-vocab/vocab.txt"
FINE_TUNE_FOLDER="$OUTPUT_DIR/fine-tuned"

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
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "FP16: $FP16"
    echo "EPOCHS_DPT: $EPOCHS_DPT"
    echo "FINE_TUNE_DATA_DIR: $FINE_TUNE_DATA_DIR"
    echo "POSITIONAL: $POSITIONAL"
    echo "OVERWRITE: $OVERWRITE"
    echo "OVERWRITE_ARGS: $OVERWRITE_ARGS"
    echo "SKIP_AUGMENT_VOCAB: $SKIP_AUGMENT_VOCAB"
    echo "SKIP_DOMAIN_PRE_TRAIN: $SKIP_DOMAIN_PRE_TRAIN"
    echo "SKIP_FINE_TUNING: $SKIP_FINE_TUNE"
    echo "AUGMENTED_VOCAB_PATH: $AUGMENTED_VOCAB_PATH"
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
        --dst $AUGMENTED_VOCAB_PATH
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
    python -m scripts.domain_adaptation.domain_pre_train \
        --output_dir $DOMAIN_PRE_TRAIN_FOLDER \
        --model_type bert \
        --tokenizer_vocab $AUGMENTED_VOCAB_PATH \
        --block_size 512 \
        --do_train \
        --num_train_epochs $EPOCHS_DPT \
        --train_data_file $CORPUS \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --mlm \
        $FP16_ARGS \
        ${DPT_EVAL_ARGS[@]} \
        ${OVERWRITE_ARGS[@]}
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
        --output_dir $FINE_TUNE_FOLDER \
        --model_type bert \
        --model_name_or_path $DOMAIN_PRE_TRAIN_FOLDER \
        --do_lower_case \
        --max_seq_length $MAX_LENGTH \
        --do_train \
        --num_train_epochs $NUM_EPOCHS_NER \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --do_eval \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --do_predict \
        $FP16_ARGS \
        ${OVERWRITE_ARGS[@]}
fi
