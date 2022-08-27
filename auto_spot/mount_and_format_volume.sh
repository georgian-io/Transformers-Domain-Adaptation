#!/bin/zsh

PROJ="autonlp"
BLOCK="/dev/xvdf"
ROOT="/$PROJ"

# Get device file type
file_type=$(sudo file -s $BLOCK | cut -d " " -f 2)
if ! [ $file_type = "data" ]; then
    echo "File type other than data found for $BLOCK"
fi

# Format device and create a new filesystem
sudo mkfs -t xfs $BLOCK

# Create directory for data
sudo mkdir $ROOT
sudo mount $BLOCK $ROOT
