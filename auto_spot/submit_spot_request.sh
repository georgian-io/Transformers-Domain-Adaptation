#!/bin/zsh

MODE=$1
VALID_MODES=("dpt" "ft")
if [ -z $MODE ]; then echo "Mode required as first arg."; exit 1; fi
if ! [ $MODE = "dpt" ] && ! [ $MODE = "ft" ]; then
    echo "Invalid `mode` provided."
    exit 1
fi

INSTANCE_TYPE=$([ $MODE = "dpt" ] && echo "p3.16xlarge" || echo "p3.2xlarge")
SPOT_FLEET_CONFIG="auto_spot/spot_fleet_config.json"
USER_DATA_SCRIPT="auto_spot/user_data_script.sh"

GIT_BRANCH=$2
if [ -z $GIT_BRANCH ]; then GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD); fi

# Check if remote git branch exists to run code on VM
function remote_branch_exists() {
    git branch --remote | grep "/$1$" | wc -l | xargs
}
if ! [ $(remote_branch_exists $GIT_BRANCH) = 1 ]; then
    echo "\"$GIT_BRANCH\" branch does not exist at origin."
    exit 1
else
    echo "Using git branch \"$GIT_BRANCH\" when initiating spot instance"
fi

# Chain of
# 1. Update user script into spot_fleet_config.json
# 2. Submit spot fleet request
# 3. Reset values in folder

sed -i '' "s/git-branch-temp/$GIT_BRANCH/g" $USER_DATA_SCRIPT
USER_DATA_B64=$(base64 $USER_DATA_SCRIPT -b0) \
    && sed -i '' "s|base64_encoded_bash_script|$USER_DATA_B64|g" $SPOT_FLEET_CONFIG \
    && sed -i '' "s/instance_type/$INSTANCE_TYPE/g" $SPOT_FLEET_CONFIG \
    && aws ec2 request-spot-fleet \
        --spot-fleet-request-config file://$SPOT_FLEET_CONFIG \

# Undo sed substitutions
sed -i '' "s/$GIT_BRANCH/git-branch-temp/g" $USER_DATA_SCRIPT
git checkout @ -- $SPOT_FLEET_CONFIG
