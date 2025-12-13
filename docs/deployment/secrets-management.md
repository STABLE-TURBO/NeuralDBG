# Secrets Management

Comprehensive guide to managing secrets in production deployments using various secrets management solutions.

## Table of Contents

- [Overview](#overview)
- [HashiCorp Vault](#hashicorp-vault)
- [AWS Secrets Manager](#aws-secrets-manager)
- [Google Secret Manager](#google-secret-manager)
- [Azure Key Vault](#azure-key-vault)
- [Kubernetes Secrets](#kubernetes-secrets)
- [Best Practices](#best-practices)

## Overview

**Never store secrets in:**
- Source code
- Docker images
- Environment files committed to git
- Configuration management systems (without encryption)
- Container logs

**Store secrets in:**
- Dedicated secrets management systems
- Encrypted key-value stores
- Hardware security modules (HSM)
- Cloud provider secrets services

## HashiCorp Vault

### Setup Vault Server

```bash
# Install Vault
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Create configuration
sudo mkdir -p /etc/vault
sudo tee /etc/vault/config.hcl > /dev/null <<EOF
storage "file" {
  path = "/var/lib/vault/data"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  tls_cert_file = "/etc/vault/ssl/cert.pem"
  tls_key_file = "/etc/vault/ssl/key.pem"
}

ui = true
api_addr = "https://vault.example.com:8200"
cluster_addr = "https://vault.example.com:8201"
EOF

# Create systemd service
sudo tee /etc/systemd/system/vault.service > /dev/null <<EOF
[Unit]
Description=HashiCorp Vault
Documentation=https://www.vaultproject.io/docs/
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/local/bin/vault server -config=/etc/vault/config.hcl
ExecReload=/bin/kill --signal HUP \$MAINPID
KillMode=process
Restart=on-failure
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# Start Vault
sudo systemctl enable vault
sudo systemctl start vault

# Initialize Vault
vault operator init
# Save unseal keys and root token securely!

# Unseal Vault
vault operator unseal <unseal-key-1>
vault operator unseal <unseal-key-2>
vault operator unseal <unseal-key-3>

# Login
vault login <root-token>
```

### Configure Vault for Neural DSL

```bash
# Enable KV secrets engine
vault secrets enable -path=neural kv-v2

# Store secrets
vault kv put neural/production \
  secret_key="$(openssl rand -hex 32)" \
  db_password="$(openssl rand -hex 32)" \
  redis_password="$(openssl rand -hex 16)" \
  jwt_secret="$(openssl rand -hex 32)"

# Read secrets
vault kv get neural/production

# Create policy
vault policy write neural-policy - <<EOF
path "neural/data/production/*" {
  capabilities = ["read", "list"]
}
EOF

# Create AppRole for Neural DSL
vault auth enable approle
vault write auth/approle/role/neural-role \
  token_policies="neural-policy" \
  token_ttl=1h \
  token_max_ttl=4h

# Get role ID and secret ID
vault read auth/approle/role/neural-role/role-id
vault write -f auth/approle/role/neural-role/secret-id
```

### Integrate with Neural DSL

```python
# neural/secrets/vault.py
import hvac
import os
from typing import Dict, Any

class VaultSecrets:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv("VAULT_ADDR", "https://vault.example.com:8200"),
            verify=os.getenv("VAULT_CACERT", True)
        )
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Vault using AppRole."""
        role_id = os.getenv("VAULT_ROLE_ID")
        secret_id = os.getenv("VAULT_SECRET_ID")
        
        if role_id and secret_id:
            self.client.auth.approle.login(
                role_id=role_id,
                secret_id=secret_id
            )
        elif token := os.getenv("VAULT_TOKEN"):
            self.client.token = token
        else:
            raise ValueError("No Vault authentication method configured")
    
    def get_secret(self, path: str, key: str = None) -> Any:
        """Get secret from Vault."""
        secret = self.client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point="neural"
        )
        
        data = secret["data"]["data"]
        return data.get(key) if key else data
    
    def get_all_secrets(self, path: str = "production") -> Dict[str, str]:
        """Get all secrets for environment."""
        return self.get_secret(path)

# Usage in application
vault = VaultSecrets()
secrets = vault.get_all_secrets("production")

SECRET_KEY = secrets["secret_key"]
DB_PASSWORD = secrets["db_password"]
REDIS_PASSWORD = secrets["redis_password"]
```

### Docker Compose with Vault

```yaml
# docker-compose.vault.yml
version: '3.8'

services:
  vault:
    image: vault:1.15.0
    cap_add:
      - IPC_LOCK
    environment:
      - VAULT_ADDR=https://0.0.0.0:8200
    volumes:
      - ./vault/config:/vault/config
      - vault-data:/vault/data
    ports:
      - "8200:8200"
    command: server

  api:
    image: neural-dsl:latest
    environment:
      - VAULT_ADDR=https://vault:8200
      - VAULT_ROLE_ID=${VAULT_ROLE_ID}
      - VAULT_SECRET_ID=${VAULT_SECRET_ID}
      - VAULT_SKIP_VERIFY=true  # Only for development!
    depends_on:
      - vault

volumes:
  vault-data:
```

### Kubernetes with Vault

```yaml
# vault-injector.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: neural-api
  namespace: neural-prod
---
apiVersion: v1
kind: Pod
metadata:
  name: neural-api
  namespace: neural-prod
  annotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "neural-role"
    vault.hashicorp.com/agent-inject-secret-config: "neural/data/production"
    vault.hashicorp.com/agent-inject-template-config: |
      {{- with secret "neural/data/production" -}}
      export SECRET_KEY="{{ .Data.data.secret_key }}"
      export DB_PASSWORD="{{ .Data.data.db_password }}"
      export REDIS_PASSWORD="{{ .Data.data.redis_password }}"
      {{- end }}
spec:
  serviceAccountName: neural-api
  containers:
  - name: api
    image: neural-dsl:latest
    command: ["/bin/sh", "-c"]
    args:
    - source /vault/secrets/config && python -m neural.api.main
```

## AWS Secrets Manager

### Setup

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Create secret
aws secretsmanager create-secret \
  --name neural/production \
  --description "Neural DSL production secrets" \
  --secret-string '{
    "secret_key":"'$(openssl rand -hex 32)'",
    "db_password":"'$(openssl rand -hex 32)'",
    "redis_password":"'$(openssl rand -hex 16)'",
    "jwt_secret":"'$(openssl rand -hex 32)'"
  }'

# Read secret
aws secretsmanager get-secret-value \
  --secret-id neural/production \
  --query SecretString \
  --output text

# Update secret
aws secretsmanager update-secret \
  --secret-id neural/production \
  --secret-string '{
    "secret_key":"new-secret-key"
  }'

# Enable automatic rotation
aws secretsmanager rotate-secret \
  --secret-id neural/production \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789:function:neural-rotation \
  --rotation-rules AutomaticallyAfterDays=30
```

### Integrate with Neural DSL

```python
# neural/secrets/aws.py
import boto3
import json
from typing import Dict

class AWSSecretsManager:
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client(
            'secretsmanager',
            region_name=region_name
        )
    
    def get_secret(self, secret_name: str) -> Dict[str, str]:
        """Get secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except Exception as e:
            raise ValueError(f"Error retrieving secret: {e}")
    
    def get_secret_value(self, secret_name: str, key: str) -> str:
        """Get specific key from secret."""
        secrets = self.get_secret(secret_name)
        return secrets.get(key)

# Usage
secrets_manager = AWSSecretsManager()
secrets = secrets_manager.get_secret("neural/production")

SECRET_KEY = secrets["secret_key"]
DB_PASSWORD = secrets["db_password"]
REDIS_PASSWORD = secrets["redis_password"]
```

### IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:123456789:secret:neural/production-*"
    }
  ]
}
```

### ECS Task Definition

```json
{
  "family": "neural-api",
  "taskRoleArn": "arn:aws:iam::123456789:role/NeuralTaskRole",
  "executionRoleArn": "arn:aws:iam::123456789:role/NeuralExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "neural-dsl:latest",
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:neural/production:secret_key::"
        },
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:neural/production:db_password::"
        }
      ]
    }
  ]
}
```

## Google Secret Manager

### Setup

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Authenticate
gcloud auth login
gcloud config set project neural-project

# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com

# Create secrets
echo -n "$(openssl rand -hex 32)" | gcloud secrets create secret-key --data-file=-
echo -n "$(openssl rand -hex 32)" | gcloud secrets create db-password --data-file=-
echo -n "$(openssl rand -hex 16)" | gcloud secrets create redis-password --data-file=-

# Access secret
gcloud secrets versions access latest --secret="secret-key"

# Grant access to service account
gcloud secrets add-iam-policy-binding secret-key \
  --member="serviceAccount:neural-sa@neural-project.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Integrate with Neural DSL

```python
# neural/secrets/gcp.py
from google.cloud import secretmanager
from typing import Dict

class GCPSecretManager:
    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
    
    def get_secret(self, secret_id: str, version_id: str = "latest") -> str:
        """Get secret from Google Secret Manager."""
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version_id}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')
    
    def get_all_secrets(self) -> Dict[str, str]:
        """Get all Neural DSL secrets."""
        return {
            "secret_key": self.get_secret("secret-key"),
            "db_password": self.get_secret("db-password"),
            "redis_password": self.get_secret("redis-password"),
            "jwt_secret": self.get_secret("jwt-secret"),
        }

# Usage
secrets_manager = GCPSecretManager("neural-project")
secrets = secrets_manager.get_all_secrets()

SECRET_KEY = secrets["secret_key"]
DB_PASSWORD = secrets["db_password"]
```

### Cloud Run Configuration

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: neural-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      serviceAccountName: neural-sa@neural-project.iam.gserviceaccount.com
      containers:
      - image: gcr.io/neural-project/neural-dsl:latest
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: secret-key
              key: latest
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-password
              key: latest
```

## Azure Key Vault

### Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create Key Vault
az keyvault create \
  --name neural-vault \
  --resource-group neural-rg \
  --location eastus

# Store secrets
az keyvault secret set \
  --vault-name neural-vault \
  --name secret-key \
  --value "$(openssl rand -hex 32)"

az keyvault secret set \
  --vault-name neural-vault \
  --name db-password \
  --value "$(openssl rand -hex 32)"

# Get secret
az keyvault secret show \
  --vault-name neural-vault \
  --name secret-key \
  --query value -o tsv

# Grant access to managed identity
az keyvault set-policy \
  --name neural-vault \
  --object-id <managed-identity-id> \
  --secret-permissions get list
```

### Integrate with Neural DSL

```python
# neural/secrets/azure.py
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from typing import Dict

class AzureKeyVault:
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault."""
        secret = self.client.get_secret(secret_name)
        return secret.value
    
    def get_all_secrets(self) -> Dict[str, str]:
        """Get all Neural DSL secrets."""
        return {
            "secret_key": self.get_secret("secret-key"),
            "db_password": self.get_secret("db-password"),
            "redis_password": self.get_secret("redis-password"),
            "jwt_secret": self.get_secret("jwt-secret"),
        }

# Usage
vault = AzureKeyVault("https://neural-vault.vault.azure.net/")
secrets = vault.get_all_secrets()

SECRET_KEY = secrets["secret_key"]
DB_PASSWORD = secrets["db_password"]
```

### AKS Configuration

```yaml
# aks-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: neural-api
  labels:
    azure.workload.identity/use: "true"
spec:
  serviceAccountName: neural-sa
  containers:
  - name: api
    image: neural.azurecr.io/neural-dsl:latest
    env:
    - name: AZURE_CLIENT_ID
      value: <managed-identity-client-id>
    - name: KEY_VAULT_URL
      value: https://neural-vault.vault.azure.net/
    volumeMounts:
    - name: secrets-store
      mountPath: "/mnt/secrets"
      readOnly: true
  volumes:
  - name: secrets-store
    csi:
      driver: secrets-store.csi.k8s.io
      readOnly: true
      volumeAttributes:
        secretProviderClass: "azure-neural-secrets"
---
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: azure-neural-secrets
spec:
  provider: azure
  parameters:
    usePodIdentity: "false"
    useVMManagedIdentity: "true"
    userAssignedIdentityID: <managed-identity-client-id>
    keyvaultName: neural-vault
    objects: |
      array:
        - |
          objectName: secret-key
          objectType: secret
        - |
          objectName: db-password
          objectType: secret
```

## Kubernetes Secrets

### Create Secrets

```bash
# From literal values
kubectl create secret generic neural-secrets \
  --namespace=neural-prod \
  --from-literal=secret-key=$(openssl rand -hex 32) \
  --from-literal=db-password=$(openssl rand -hex 32) \
  --from-literal=redis-password=$(openssl rand -hex 16)

# From file
kubectl create secret generic neural-secrets \
  --namespace=neural-prod \
  --from-file=secret-key=./secret-key.txt \
  --from-file=db-password=./db-password.txt

# From env file
kubectl create secret generic neural-secrets \
  --namespace=neural-prod \
  --from-env-file=.env.production
```

### Use Secrets in Pods

```yaml
# pod-with-secrets.yaml
apiVersion: v1
kind: Pod
metadata:
  name: neural-api
spec:
  containers:
  - name: api
    image: neural-dsl:latest
    env:
    # Mount specific secret as environment variable
    - name: SECRET_KEY
      valueFrom:
        secretKeyRef:
          name: neural-secrets
          key: secret-key
    # Mount all secrets as environment variables
    envFrom:
    - secretRef:
        name: neural-secrets
    # Mount secrets as files
    volumeMounts:
    - name: secrets-volume
      mountPath: /etc/secrets
      readOnly: true
  volumes:
  - name: secrets-volume
    secret:
      secretName: neural-secrets
```

### Encrypt Secrets at Rest

```bash
# Generate encryption key
head -c 32 /dev/urandom | base64

# Create encryption config
cat > /etc/kubernetes/encryption-config.yaml <<EOF
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: <base64-encoded-key>
    - identity: {}
EOF

# Configure API server
# Add to /etc/kubernetes/manifests/kube-apiserver.yaml:
# - --encryption-provider-config=/etc/kubernetes/encryption-config.yaml
```

### Sealed Secrets

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Install kubeseal CLI
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-0.24.0-linux-amd64.tar.gz
tar xfz kubeseal-0.24.0-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal

# Create sealed secret
kubectl create secret generic neural-secrets \
  --dry-run=client \
  --from-literal=secret-key=$(openssl rand -hex 32) \
  -o yaml | \
  kubeseal -o yaml > sealed-secret.yaml

# Apply sealed secret (safe to commit to git)
kubectl apply -f sealed-secret.yaml
```

## Best Practices

### Rotation Strategy

```python
# neural/secrets/rotation.py
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SecretRotation:
    def __init__(self, secrets_manager):
        self.secrets_manager = secrets_manager
        self.rotation_interval = timedelta(days=90)
    
    def should_rotate(self, secret_name: str) -> bool:
        """Check if secret should be rotated."""
        metadata = self.secrets_manager.get_secret_metadata(secret_name)
        last_rotated = metadata.get("last_rotated")
        
        if not last_rotated:
            return True
        
        last_rotated_date = datetime.fromisoformat(last_rotated)
        return datetime.utcnow() - last_rotated_date > self.rotation_interval
    
    def rotate_secret(self, secret_name: str):
        """Rotate secret."""
        logger.info(f"Rotating secret: {secret_name}")
        
        # Generate new secret
        new_value = self.generate_secret_value()
        
        # Update in secrets manager
        self.secrets_manager.update_secret(secret_name, new_value)
        
        # Update applications (gradual rollout)
        self.update_applications(secret_name)
        
        logger.info(f"Secret rotated successfully: {secret_name}")
```

### Access Control

```python
# neural/secrets/access_control.py
from enum import Enum
from typing import List

class SecretPermission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"

class SecretAccessControl:
    def __init__(self):
        self.acl = {}
    
    def grant_access(
        self,
        secret_path: str,
        user: str,
        permissions: List[SecretPermission]
    ):
        """Grant user access to secret."""
        if secret_path not in self.acl:
            self.acl[secret_path] = {}
        
        self.acl[secret_path][user] = permissions
    
    def has_permission(
        self,
        secret_path: str,
        user: str,
        permission: SecretPermission
    ) -> bool:
        """Check if user has permission."""
        if secret_path not in self.acl:
            return False
        
        user_permissions = self.acl[secret_path].get(user, [])
        return permission in user_permissions
```

### Audit Logging

```python
# neural/secrets/audit.py
import logging
from datetime import datetime

audit_logger = logging.getLogger("secrets_audit")

def log_secret_access(user: str, secret_name: str, action: str):
    """Log secret access for audit."""
    audit_logger.info({
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "secret": secret_name,
        "action": action,
    })
```

### Environment-Specific Secrets

```
secrets/
├── development/
│   ├── secret-key
│   ├── db-password
│   └── api-keys
├── staging/
│   ├── secret-key
│   ├── db-password
│   └── api-keys
└── production/
    ├── secret-key
    ├── db-password
    └── api-keys
```

### Secrets Checklist

- [ ] Never commit secrets to version control
- [ ] Use different secrets for each environment
- [ ] Rotate secrets regularly (every 90 days)
- [ ] Enable encryption at rest and in transit
- [ ] Implement least-privilege access
- [ ] Audit all secret access
- [ ] Monitor for unauthorized access
- [ ] Have incident response plan
- [ ] Document secret recovery procedures
- [ ] Test secret rotation process
- [ ] Use strong random generation
- [ ] Implement secrets expiration
- [ ] Enable MFA for secrets management access
- [ ] Backup secrets securely
- [ ] Review access permissions quarterly
