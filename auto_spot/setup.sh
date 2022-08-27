#!/bin/zsh

PROJ="autonlp"
AWS_REGION="us-east-1"
AZ="${AWS_REGION}b"
IMAGE_ID="ami-0dbb717f493016a1a"

# Create a regular instance to prepare EBS volume
instance_id=$(
    aws ec2 run-instances \
        --image-id $IMAGE_ID \
        --security-group-ids sg-758c4705 \
        --count 1 \
        --instance-type m4.xlarge \
        --key-name chris-ssh \
        --placement "AvailabilityZone=$AZ" \
        --tag-specification \
            "ResourceType=instance,Tags=[{Key=Name,Value=$PROJ-vol-setup}]" \
        --query "Instances[0].InstanceId" \
        --output text
    ) \
    && echo "Instance created with ID $instance_id" \
    || (echo "Instance creation failed"; exit 1)
# Possible TODO include subnet ID ^

# Create volume and obtain volume-id
volume_id=$(
    aws ec2 create-volume \
        --size 1000 \
        --region $AWS_REGION \
        --availability-zone $AZ \
        --volume-type gp2 \
        --tag-specification \
            "ResourceType=volume,Tags=[{Key=Name,Value=$PROJ-checkpoints}]" \
        --query "VolumeId" \
        --output text
    ) \
    && echo "Volume created with ID $volume_id" \
    || (echo "Volume creation failed"; exit 1)

# Pausing for a while to ensure instance is already running
echo "Waiting for instance $instance_id to be running"
aws ec2 wait instance-running --instance-ids $instance_id

# Attach volume to instance
(aws ec2 attach-volume \
    --volume-id $volume_id \
    --instance-id $instance_id \
    --device /dev/sdf) \
&& echo "Volume $volume_id successfully attached to instance $instance_id" \
|| (echo "Failed to attach volume $volume_id to instance $instance_id"; exit 1)
