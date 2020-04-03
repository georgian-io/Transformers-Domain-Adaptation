#!/bin/zsh

TAG_NAME=$1

if [ -z $TAG_NAME ]; then
    echo "Full tag name expected as first argument"
    exit 1
fi

aws ec2 describe-instances \
    --filter "Name=tag:Name,Values=$TAG_NAME" \
    --query "Reservations[0].Instances[0].PublicDnsName" \
    | sed -e 's/^"//g' -e 's/"$//g'
