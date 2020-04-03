#!/bin/bash

# EC2 Constants
PROJ="autonlp"
AWS_REGION=us-east-1
VOLUME_TAG_NAME="$PROJ-checkpoints"
SNAPSHOT_TAG_NAME="$VOLUME_TAG_NAME-snapshot"

# Device and directory constants
ROOT="/$PROJ"
BLOCK="/dev/nvme1n1"
USER="ubuntu"

# Repo constants
REPO="https://github.com/georgianpartners/NLP-Domain-Adaptation"
WORK_DIR="NLP-Domain-Adaptation"

# Clean up spot fleet requests
teardown() {
    SPOT_FLEET_REQUEST_ID=$(
        aws ec2 describe-spot-instance-requests \
            --region $AWS_REGION \
            --filter "Name=instance-id,Values=$INSTANCE_ID" \
            --query "SpotInstanceRequests[].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value[]" \
            --output text
    )
    aws ec2 cancel-spot-fleet-requests \
        --region $AWS_REGION \
        --spot-fleet-request-ids $SPOT_FLEET_REQUEST_ID \
        --terminate-instances
}


# Get instance ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_AZ=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)

# Get Volume Id and availability zone
VOLUME_ID=$(
    aws ec2 describe-volumes \
        --region $AWS_REGION \
        --filter "Name=tag:Name,Values=$VOLUME_TAG_NAME" \
        --query "Volumes[].VolumeId" \
        --output text)
if [ -z $VOLUME_ID ]; then
    echo "Volume $VOLUME_TAG_NAME not found"
    teardown
fi

VOLUME_AZ=$(
    aws ec2 describe-volumes \
        --region $AWS_REGION \
        --filter "Name=tag:Name,Values=$VOLUME_TAG_NAME" \
        --query "Volumes[].AvailabilityZone" \
        --output text)

if [ $VOLUME_AZ != $INSTANCE_AZ ]; then
    SNAPSHOT_ID=$(
        aws ec2 create-snapshot \
            --region $AWS_REGION \
            --volume-id $VOLUME_ID \
            --description "`date +"%D %T"`" \
            --tag-specifications \
                "ResourceType=snapshot,Tags=[{Key=Name,Value=$SNAPSHOT_TAG_NAME}]" \
            --query SnapshotId \
            --output text) \
    && aws ec2 wait --region \
        $AWS_REGION snapshot-completed \
        --snapshot-ids $SNAPSHOT_ID \
    && echo "Snapshot $SNAPSHOT_ID successfully created " \
    || echo ("Failed to create snapshot."; teardown)

    aws ec2 --region $AWS_REGION delete-volume --volume-id $VOLUME_ID

    VOLUME_ID=$(
        aws ec2 create-volume \
            --region $AWS_REGION \
            --availability-zone $INSTANCE_AZ \
            --snapshot-id $SNAPSHOT_ID \
            --volume-type gp2 \
            --tag-specification \
                "ResourceType=volume,Tags=[{Key=Name,Value=$VOLUME_TAG_NAME}]" \
            --query VolumeId \
            --output text) \
    && aws ec2 wait volume-available \
        --region $AWS_REGION \
        --volume-id $VOLUME_ID \
    && echo "Volume $VOLUME_ID successfully ported to $INSTANCE_AZ" \
    || echo ("Failed to port volume"; teardown)
fi

aws ec2 attach-volume \
    --region $AWS_REGION \
    --volume-id $VOLUME_ID \
    --instance-id $INSTANCE_ID \
    --device /dev/sdf
sleep 10

# Mount volume and change ownership
mkdir $ROOT
mount $BLOCK $ROOT
chown -R $USER: $ROOT
cd /home/$USER

# Load code for training
git clone $REPO
chown -R $USER: $WORK_DIR
cd $WORK_DIR

# Change to appropriate git branch to run experiments
git checkout use-best-checkpoints

# Download data
./sync_s3_data.sh &

# Install dependencies
sleep 120  # Wait until apt lock is released
apt install zsh htop -y &> install.log
sudo -H -u $USER zsh -c "source activate pytorch_p36; pip install -U pip; pip install -r requirements.txt" >> install.log 2>&1
# sudo -H -u $USER zsh -c "curl https://pyenv.run | zsh; pyenv install $PYTHON_VERSION; pyenv virtualenv $PYTHON_VERSION autonlp"

# Initiate training
sudo -H -u $USER zsh -c "./scripts/train.sh"

# Sync data to S3 and terminate spot fleet
teardown
