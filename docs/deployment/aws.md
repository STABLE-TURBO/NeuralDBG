# AWS Deployment Guide

Deploy Neural DSL on Amazon Web Services using ECS, EKS, and other AWS services.

## Table of Contents

- [Architecture Options](#architecture-options)
- [ECS Deployment](#ecs-deployment)
- [EKS Deployment](#eks-deployment)
- [RDS and ElastiCache](#rds-and-elasticache)
- [Load Balancing](#load-balancing)
- [Storage](#storage)
- [Monitoring](#monitoring)
- [Cost Optimization](#cost-optimization)

## Architecture Options

### Option 1: ECS Fargate (Serverless)
- **Best for**: Simple deployments, variable traffic
- **Pros**: No server management, auto-scaling, pay-per-use
- **Cons**: Higher per-container cost, limited customization

### Option 2: ECS EC2
- **Best for**: Predictable workloads, cost optimization
- **Pros**: Lower cost, more control, GPU support
- **Cons**: Manual capacity management

### Option 3: EKS (Kubernetes)
- **Best for**: Complex microservices, multi-cloud
- **Pros**: Kubernetes ecosystem, portability
- **Cons**: Higher operational complexity

## ECS Deployment

### Prerequisites

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure

# Install ECS CLI
sudo curl -Lo /usr/local/bin/ecs-cli https://amazon-ecs-cli.s3.amazonaws.com/ecs-cli-linux-amd64-latest
sudo chmod +x /usr/local/bin/ecs-cli
```

### Create Infrastructure

```bash
# Create VPC
aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=neural-vpc}]'

# Create subnets
aws ec2 create-subnet \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=neural-subnet-1}]'

aws ec2 create-subnet \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.2.0/24 \
  --availability-zone us-east-1b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=neural-subnet-2}]'

# Create security group
aws ec2 create-security-group \
  --group-name neural-sg \
  --description "Security group for Neural DSL" \
  --vpc-id vpc-xxxxx

# Allow inbound traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 80 \
  --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0
```

### Create ECS Cluster

```bash
# Create cluster
aws ecs create-cluster --cluster-name neural-prod

# Or use CloudFormation
cat > ecs-cluster.yaml <<'EOF'
AWSTemplateFormatVersion: '2010-09-09'
Description: Neural DSL ECS Cluster

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>

Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: neural-prod
      ClusterSettings:
        - Name: containerInsights
          Value: enabled
  
  ECSTaskExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Statement:
        - Effect: Allow
          Principal:
            Service: ecs-tasks.amazonaws.com
          Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
        - arn:aws:iam::aws:policy/SecretsManagerReadWrite
  
  ECSTaskRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Statement:
        - Effect: Allow
          Principal:
            Service: ecs-tasks.amazonaws.com
          Action: sts:AssumeRole
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Statement:
            - Effect: Allow
              Action:
                - s3:GetObject
                - s3:PutObject
                - s3:ListBucket
              Resource:
                - arn:aws:s3:::neural-models/*
                - arn:aws:s3:::neural-models

Outputs:
  ClusterName:
    Value: !Ref ECSCluster
  TaskExecutionRoleArn:
    Value: !GetAtt ECSTaskExecutionRole.Arn
  TaskRoleArn:
    Value: !GetAtt ECSTaskRole.Arn
EOF

aws cloudformation create-stack \
  --stack-name neural-ecs \
  --template-body file://ecs-cluster.yaml \
  --parameters ParameterKey=VpcId,ParameterValue=vpc-xxxxx \
               ParameterKey=SubnetIds,ParameterValue="subnet-xxxxx,subnet-yyyyy" \
  --capabilities CAPABILITY_IAM
```

### Push Image to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name neural-dsl

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t neural-dsl:latest .
docker tag neural-dsl:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/neural-dsl:latest

# Push image
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/neural-dsl:latest
```

### Create Task Definition

```json
{
  "family": "neural-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "taskRoleArn": "arn:aws:iam::123456789:role/NeuralTaskRole",
  "executionRoleArn": "arn:aws:iam::123456789:role/NeuralExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/neural-dsl:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "API_PORT",
          "value": "8000"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://neural:password@neural-db.xxxxx.us-east-1.rds.amazonaws.com:5432/neural_api"
        },
        {
          "name": "REDIS_HOST",
          "value": "neural-redis.xxxxx.cache.amazonaws.com"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:neural/production:secret_key::"
        },
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:neural/production:db_password::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/neural-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Create Service

```bash
# Create service
aws ecs create-service \
  --cluster neural-prod \
  --service-name neural-api \
  --task-definition neural-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx,subnet-yyyyy],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789:targetgroup/neural-api/xxxxx,containerName=api,containerPort=8000" \
  --health-check-grace-period-seconds 60

# Enable auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/neural-prod/neural-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 3 \
  --max-capacity 20

# CPU-based scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/neural-prod/neural-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name neural-api-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

**scaling-policy.json:**
```json
{
  "TargetValue": 70.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  },
  "ScaleInCooldown": 300,
  "ScaleOutCooldown": 60
}
```

## EKS Deployment

### Create EKS Cluster

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name neural-prod \
  --region us-east-1 \
  --nodegroup-name neural-nodes \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed \
  --with-oidc \
  --ssh-access \
  --ssh-public-key my-key

# Or use config file
cat > eks-cluster.yaml <<'EOF'
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: neural-prod
  region: us-east-1
  version: "1.28"

vpc:
  cidr: 10.0.0.0/16

managedNodeGroups:
  - name: neural-general
    instanceType: t3.xlarge
    minSize: 3
    maxSize: 10
    desiredCapacity: 3
    volumeSize: 100
    ssh:
      allow: true
      publicKeyName: my-key
    labels:
      role: general
    tags:
      nodegroup-role: general
  
  - name: neural-workers
    instanceType: c5.2xlarge
    minSize: 2
    maxSize: 20
    desiredCapacity: 2
    volumeSize: 100
    labels:
      role: worker
    tags:
      nodegroup-role: worker

iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: neural-api
        namespace: neural-prod
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/AmazonS3FullAccess
        - arn:aws:iam::aws:policy/SecretsManagerReadWrite

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
EOF

eksctl create cluster -f eks-cluster.yaml
```

### Deploy to EKS

```bash
# Update kubeconfig
aws eks update-kubeconfig --name neural-prod --region us-east-1

# Create namespace
kubectl create namespace neural-prod

# Install ALB Ingress Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=neural-prod \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Deploy Neural DSL (see kubernetes.md for manifests)
kubectl apply -f k8s/
```

## RDS and ElastiCache

### RDS PostgreSQL

```bash
# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name neural-db-subnet \
  --db-subnet-group-description "Neural DSL DB subnet group" \
  --subnet-ids subnet-xxxxx subnet-yyyyy

# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier neural-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.3 \
  --master-username neural \
  --master-user-password $(openssl rand -hex 32) \
  --allocated-storage 100 \
  --storage-type gp3 \
  --storage-encrypted \
  --vpc-security-group-ids sg-xxxxx \
  --db-subnet-group-name neural-db-subnet \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "mon:04:00-mon:05:00" \
  --multi-az \
  --publicly-accessible false \
  --enable-cloudwatch-logs-exports '["postgresql"]'

# Create read replica
aws rds create-db-instance-read-replica \
  --db-instance-identifier neural-db-replica \
  --source-db-instance-identifier neural-db \
  --db-instance-class db.t3.medium \
  --publicly-accessible false
```

### ElastiCache Redis

```bash
# Create subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name neural-cache-subnet \
  --cache-subnet-group-description "Neural DSL cache subnet group" \
  --subnet-ids subnet-xxxxx subnet-yyyyy

# Create Redis cluster
aws elasticache create-replication-group \
  --replication-group-id neural-redis \
  --replication-group-description "Neural DSL Redis cluster" \
  --engine redis \
  --engine-version 7.0 \
  --cache-node-type cache.t3.medium \
  --num-cache-clusters 3 \
  --automatic-failover-enabled \
  --multi-az-enabled \
  --cache-subnet-group-name neural-cache-subnet \
  --security-group-ids sg-xxxxx \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --auth-token $(openssl rand -hex 16) \
  --snapshot-retention-limit 5 \
  --snapshot-window "03:00-05:00"
```

## Load Balancing

### Application Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name neural-alb \
  --subnets subnet-xxxxx subnet-yyyyy \
  --security-groups sg-xxxxx \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4

# Create target group
aws elbv2 create-target-group \
  --name neural-api-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxxxx \
  --target-type ip \
  --health-check-protocol HTTP \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3

# Create listener
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:123456789:loadbalancer/app/neural-alb/xxxxx \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:us-east-1:123456789:certificate/xxxxx \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:us-east-1:123456789:targetgroup/neural-api-tg/xxxxx

# Create HTTP to HTTPS redirect
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:123456789:loadbalancer/app/neural-alb/xxxxx \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=redirect,RedirectConfig="{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}"
```

## Storage

### S3 Bucket Setup

```bash
# Create bucket
aws s3api create-bucket \
  --bucket neural-models-prod \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket neural-models-prod \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket neural-models-prod \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'

# Set lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket neural-models-prod \
  --lifecycle-configuration file://lifecycle-policy.json
```

**lifecycle-policy.json:**
```json
{
  "Rules": [
    {
      "Id": "archive-old-models",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "models/"
      },
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 180,
          "StorageClass": "GLACIER"
        }
      ]
    },
    {
      "Id": "delete-temp-files",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "temp/"
      },
      "Expiration": {
        "Days": 7
      }
    }
  ]
}
```

### EFS for Shared Storage

```bash
# Create EFS file system
aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --encrypted \
  --tags Key=Name,Value=neural-efs

# Create mount targets
aws efs create-mount-target \
  --file-system-id fs-xxxxx \
  --subnet-id subnet-xxxxx \
  --security-groups sg-xxxxx

aws efs create-mount-target \
  --file-system-id fs-xxxxx \
  --subnet-id subnet-yyyyy \
  --security-groups sg-xxxxx
```

## Monitoring

### CloudWatch Setup

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/neural-api

# Set retention
aws logs put-retention-policy \
  --log-group-name /ecs/neural-api \
  --retention-in-days 30

# Create metric filter
aws logs put-metric-filter \
  --log-group-name /ecs/neural-api \
  --filter-name error-count \
  --filter-pattern "[time, request_id, level = ERROR*, ...]" \
  --metric-transformations \
      metricName=ErrorCount,metricNamespace=Neural,metricValue=1

# Create alarm
aws cloudwatch put-metric-alarm \
  --alarm-name neural-api-errors \
  --alarm-description "Alert on API errors" \
  --metric-name ErrorCount \
  --namespace Neural \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:123456789:neural-alerts
```

### Container Insights

```bash
# Enable Container Insights
aws ecs put-account-setting \
  --name containerInsights \
  --value enabled

# Update cluster
aws ecs update-cluster-settings \
  --cluster neural-prod \
  --settings name=containerInsights,value=enabled
```

## Cost Optimization

### Use Spot Instances

```bash
# Create capacity provider for Spot
aws ecs create-capacity-provider \
  --name neural-spot-provider \
  --auto-scaling-group-provider file://spot-provider.json
```

**spot-provider.json:**
```json
{
  "autoScalingGroupArn": "arn:aws:autoscaling:us-east-1:123456789:autoScalingGroup:xxxxx:autoScalingGroupName/neural-spot-asg",
  "managedScaling": {
    "status": "ENABLED",
    "targetCapacity": 100,
    "minimumScalingStepSize": 1,
    "maximumScalingStepSize": 10
  },
  "managedTerminationProtection": "ENABLED"
}
```

### Reserved Instances

```bash
# Purchase RDS Reserved Instance
aws rds purchase-reserved-db-instances-offering \
  --reserved-db-instances-offering-id xxxxx \
  --reserved-db-instance-id neural-db-ri \
  --db-instance-count 1

# Purchase ElastiCache Reserved Nodes
aws elasticache purchase-reserved-cache-nodes-offering \
  --reserved-cache-nodes-offering-id xxxxx \
  --reserved-cache-node-id neural-cache-ri \
  --cache-node-count 3
```

### S3 Intelligent-Tiering

```bash
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket neural-models-prod \
  --id neural-intelligent-tiering \
  --intelligent-tiering-configuration file://intelligent-tiering.json
```

**intelligent-tiering.json:**
```json
{
  "Id": "neural-intelligent-tiering",
  "Status": "Enabled",
  "Tierings": [
    {
      "Days": 90,
      "AccessTier": "ARCHIVE_ACCESS"
    },
    {
      "Days": 180,
      "AccessTier": "DEEP_ARCHIVE_ACCESS"
    }
  ]
}
```

## Complete Terraform Example

See [terraform-aws](../examples/terraform/aws/) for complete Infrastructure as Code examples.
