#!/bin/zsh

# Ensure that instance used for setup is terminated
# (so that volume can be deleted if in different AZ as spot instance when
# running user_data_script
EBS_ATTACHED=$(
    aws ec2 describe-volumes \
    --filter "Name=tag:Name,Values=autonlp-checkpoints" \
    --query "Volumes[0].Attachments" \
    --output text
)
if ! [ -z $EBS_ATTACHED ]; then
    echo "EBS volume may still be connected to the setup machine. \
    Please terminate the instance or detach the volume to ensure that \
    user_data_script.sh works."
    exit 1
fi

# Chain of
# 1. Update user script into spot_fleet_config.json
# 2. Submit spot fleet request
# 3. Reset values in folder
USER_DATA=$(base64 auto_spot/user_data_script.sh -b0) \
    && sed -i '' \
        "s|base64_encoded_bash_script|$USER_DATA|g"  \
        auto_spot/spot_fleet_config.json \
    && aws ec2 request-spot-fleet \
        --spot-fleet-request-config file://auto_spot/spot_fleet_config.json \
    && git checkout @ -- auto_spot/spot_fleet_config.json
