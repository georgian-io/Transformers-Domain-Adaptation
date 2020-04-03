#!/bin/zsh

PROJ="autonlp"
EC2_ROLE=$PROJ
EC2_POLICY="ec2-permissions-$PROJ"
EC2_INSTANCE_PROFILE="$PROJ-profile"

SPOT_FLEET_ROLE="$PROJ-spot-fleet-role"

# Create role and policies for EC2 orchestration
aws iam create-role \
    --role-name $EC2_ROLE \
    --assume-role-policy-document \
        '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --description "Role for a program to orchestrate EC2 spot instances"

aws iam create-policy \
    --policy-name $EC2_POLICY \
    --policy-document file://auto_spot/${PROJ}-permissions.json \
    --description "Permissions for EC2 spot instance orchestration"

aws iam attach-role-policy \
    --policy-arn arn:aws:iam::823217009914:policy/$EC2_POLICY \
    --role-name $PROJ

# Create instance profile and attach role to it
aws iam create-instance-profile \
    --instance-profile-name $EC2_INSTANCE_PROFILE \
    && aws iam add-role-to-instance-profile \
        --instance-profile-name $EC2_INSTANCE_PROFILE \
        --role-name $EC2_ROLE

# Create role for requesting spot fleet
aws iam create-role \
    --role-name $SPOT_FLEET_ROLE \
    --assume-role-policy-document \
        '{"Version":"2012-10-17","Statement":[{"Sid":"","Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --description "Role to request spot fleets"

aws iam attach-role-policy \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole \
    --role-name $SPOT_FLEET_ROLE
