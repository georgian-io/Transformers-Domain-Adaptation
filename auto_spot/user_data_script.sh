#!/bin/bash

# EC2 Constants
PROJ="autonlp"
AWS_REGION=us-east-1
VOLUME_TAG_NAME="$PROJ-checkpoints"
SNAPSHOT_TAG_NAME="$VOLUME_TAG_NAME-snapshot"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)


# Repo constants
USER="ubuntu"
REPO="https://github.com/georgianpartners/NLP-Domain-Adaptation"
WORK_DIR="/home/$USER/NLP-Domain-Adaptation"
LOG_DIR="$WORK_DIR/logs"
GIT_BRANCH="git-branch-temp"  # Value to be replaced by sed in `submit_spot_request`

# Clean up spot fleet requests
function teardown() {
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

# Load git repo
cd $(dirname $WORK_DIR)
git clone $REPO
chown -R $USER: $WORK_DIR
cd $WORK_DIR
mkdir -p $LOG_DIR

# Change to appropriate git branch to run experiments
git checkout $GIT_BRANCH

# Give script permissions to write files and folder
sudo chmod -R 777 $WORK_DIR

# Download data
./scripts/sync_s3_data.sh &> $LOG_DIR/setup.log 2>&1 &

# Install dependencies
sleep 120  # Wait until apt lock is released
apt install zsh htop -y >> $LOG_DIR/setup.log 2>&1
sudo -H -u $USER zsh -c "source /home/ubuntu/anaconda3/bin/activate pytorch_p36; pip install -U pip jupyterlab; pip install -r requirements.txt" >> $LOG_DIR/setup.log 2>&1
# sudo -H -u $USER zsh -c "curl https://pyenv.run | zsh; pyenv install $PYTHON_VERSION; pyenv virtualenv $PYTHON_VERSION autonlp"

# Initiate training
echo "Starting training job" &> $LOG_DIR/training.log
sudo -H -u $USER zsh -c "source /home/ubuntu/anaconda3/bin/activate pytorch_p36; ./run_pipeline.sh" >> $LOG_DIR/training.log 2>&1
echo "Training job complete" >> $LOG_DIR/training.log 2>&1

echo "Syncing log to S3" >> $LOG_DIR/setup.log 2>&1
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
echo "Instance type is $INSTANCE_TYPE" >> $LOG_DIR/setup.log 2>&1
aws s3 cp $LOG_DIR s3://nlp-domain-adaptation/log --recursive

# Sync data to S3 and terminate spot fleet
teardown
