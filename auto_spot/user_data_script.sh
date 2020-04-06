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

# Load code for training
cd $(dirname $WORK_DIR)
git clone $REPO
chown -R $USER: $WORK_DIR
cd $WORK_DIR

# Change to appropriate git branch to run experiments
git checkout $GIT_BRANCH

# Download data
./scripts/sync_s3_data.sh &

# Install dependencies
sleep 120  # Wait until apt lock is released
apt install zsh htop -y &> install.log
sudo -H -u $USER zsh -c "source /home/ubuntu/anaconda3/bin/activate pytorch_p36; pip install -U pip jupyterlab; pip install -r requirements.txt" >> install.log 2>&1
# sudo -H -u $USER zsh -c "curl https://pyenv.run | zsh; pyenv install $PYTHON_VERSION; pyenv virtualenv $PYTHON_VERSION autonlp"

# Initiate training
sudo -H -u $USER zsh -c "./scripts/train.sh"

# Sync data to S3 and terminate spot fleet
# teardown
