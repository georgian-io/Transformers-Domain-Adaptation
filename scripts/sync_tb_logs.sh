#!/bin/sh
EXPERIMENT_DIR=$1
EXPERIMENT=$(basename $EXPERIMENT_DIR)
DOMAIN=$(basename $(dirname $EXPERIMENT_DIR))

if [ -z $EXPERIMENT_DIR ]; then
    echo "Experiment directory required."
    exit 1
fi

BUCKET="s3://nlp-domain-adaptation"

aws s3 cp --recursive $EXPERIMENT_DIR/domain-pre-trained/runs $BUCKET/runs/$DOMAIN/$EXPERIMENT/domain-pre-trained
aws s3 cp --recursive $EXPERIMENT_DIR/fine-tuned/runs $BUCKET/runs/$DOMAIN/$EXPERIMENT/fine-tuned
