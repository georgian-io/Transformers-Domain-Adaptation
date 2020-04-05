#!/bin/zsh

SPOT_FLEET_CONFIG="auto_spot/spot_fleet_config.json"
USER_DATA_SCRIPT="auto_spot/user_data_script.sh"

GIT_BRANCH=$1
if [ -z $GIT_BRANCH ]; then
    echo "Valid git branch expected as the first argument"
    exit 1
elif ! [ $(git branch --remote | grep $GIT_BRANCH | wc -l | xargs) = 1 ]; then
    echo "Git branch provided does not exist at origin."
    exit 1
fi

# Chain of
# 1. Update user script into spot_fleet_config.json
# 2. Submit spot fleet request
# 3. Reset values in folder

sed -i '' "s/git-branch-temp/$GIT_BRANCH/g" $USER_DATA_SCRIPT
USER_DATA_B64=$(base64 $USER_DATA_SCRIPT -b0) \
    && sed -i '' "s|base64_encoded_bash_script|$USER_DATA_B64|g" $SPOT_FLEET_CONFIG \
    && aws ec2 request-spot-fleet \
        --spot-fleet-request-config file://$SPOT_FLEET_CONFIG \

# Undo sed substitutions
sed -i '' "s/$GIT_BRANCH/git-branch-temp/g" $USER_DATA_SCRIPT
git checkout @ -- $SPOT_FLEET_CONFIG
