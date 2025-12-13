# Azure Deployment Guide

Deploy Neural DSL on Microsoft Azure using AKS, Container Instances, and other Azure services.

## Table of Contents

- [Architecture Options](#architecture-options)
- [AKS Deployment](#aks-deployment)
- [Container Instances](#container-instances)
- [Azure Database and Cache](#azure-database-and-cache)
- [Load Balancing](#load-balancing)
- [Storage](#storage)
- [Monitoring](#monitoring)
- [Cost Optimization](#cost-optimization)

## Architecture Options

### Option 1: Azure Container Instances (Serverless)
- **Best for**: Simple workloads, dev/test
- **Pros**: Fast deployment, pay-per-second billing
- **Cons**: No auto-scaling, limited networking

### Option 2: AKS (Azure Kubernetes Service)
- **Best for**: Production workloads, microservices
- **Pros**: Full Kubernetes features, auto-scaling, Azure integration
- **Cons**: More complex, higher minimum cost

### Option 3: Azure Container Apps
- **Best for**: Event-driven apps, serverless containers
- **Pros**: Simplified Kubernetes, auto-scaling, KEDA support
- **Cons**: Less control than AKS

## AKS Deployment

### Create AKS Cluster

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login
az account set --subscription "Neural DSL Subscription"

# Create resource group
az group create --name neural-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group neural-rg \
  --name neural-aks \
  --location eastus \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --node-osdisk-size 100 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 20 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --network-plugin azure \
  --network-policy azure \
  --load-balancer-sku standard \
  --enable-rbac \
  --enable-azure-rbac \
  --attach-acr neuralacr

# Or use ARM template
az deployment group create \
  --resource-group neural-rg \
  --template-file aks-cluster.json \
  --parameters aks-parameters.json
```

**aks-cluster.json:**
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "clusterName": {
      "type": "string",
      "defaultValue": "neural-aks"
    },
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    },
    "nodeCount": {
      "type": "int",
      "defaultValue": 3
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerService/managedClusters",
      "apiVersion": "2023-01-01",
      "name": "[parameters('clusterName')]",
      "location": "[parameters('location')]",
      "properties": {
        "kubernetesVersion": "1.27.0",
        "dnsPrefix": "[concat(parameters('clusterName'), '-dns')]",
        "agentPoolProfiles": [
          {
            "name": "agentpool",
            "count": "[parameters('nodeCount')]",
            "vmSize": "Standard_D4s_v3",
            "osType": "Linux",
            "mode": "System",
            "enableAutoScaling": true,
            "minCount": 3,
            "maxCount": 20,
            "osDiskSizeGB": 100
          }
        ],
        "networkProfile": {
          "networkPlugin": "azure",
          "networkPolicy": "azure",
          "loadBalancerSku": "standard",
          "serviceCidr": "10.0.0.0/16",
          "dnsServiceIP": "10.0.0.10"
        },
        "addonProfiles": {
          "omsagent": {
            "enabled": true
          },
          "azurepolicy": {
            "enabled": true
          }
        }
      },
      "identity": {
        "type": "SystemAssigned"
      }
    }
  ]
}
```

### Get AKS Credentials

```bash
# Get credentials
az aks get-credentials --resource-group neural-rg --name neural-aks

# Verify connection
kubectl get nodes

# Install kubectl (if needed)
az aks install-cli
```

### Push to Azure Container Registry

```bash
# Create ACR
az acr create \
  --resource-group neural-rg \
  --name neuralacr \
  --sku Premium \
  --admin-enabled false

# Login to ACR
az acr login --name neuralacr

# Build and push
docker build -t neuralacr.azurecr.io/neural-dsl:latest .
docker push neuralacr.azurecr.io/neural-dsl:latest

# Or use ACR Tasks to build
az acr build \
  --registry neuralacr \
  --image neural-dsl:latest \
  --file Dockerfile .
```

### Deploy to AKS

```bash
# Create namespace
kubectl create namespace neural-prod

# Create service principal for AKS to pull from ACR
az aks update \
  --resource-group neural-rg \
  --name neural-aks \
  --attach-acr neuralacr

# Deploy application (see kubernetes.md for manifests)
kubectl apply -f k8s/ -n neural-prod
```

## Container Instances

### Deploy with Azure CLI

```bash
# Create container group
az container create \
  --resource-group neural-rg \
  --name neural-api \
  --image neuralacr.azurecr.io/neural-dsl:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server neuralacr.azurecr.io \
  --registry-username $(az acr credential show --name neuralacr --query username -o tsv) \
  --registry-password $(az acr credential show --name neuralacr --query passwords[0].value -o tsv) \
  --dns-name-label neural-api \
  --ports 8000 \
  --environment-variables \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
  --secure-environment-variables \
    SECRET_KEY=$(az keyvault secret show --vault-name neural-vault --name secret-key --query value -o tsv) \
    DB_PASSWORD=$(az keyvault secret show --vault-name neural-vault --name db-password --query value -o tsv) \
  --restart-policy OnFailure
```

### Deploy with YAML

```yaml
# container-group.yaml
apiVersion: '2021-09-01'
location: eastus
name: neural-api
properties:
  containers:
  - name: api
    properties:
      image: neuralacr.azurecr.io/neural-dsl:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: API_HOST
        value: '0.0.0.0'
      - name: API_PORT
        value: '8000'
      - name: SECRET_KEY
        secureValue: '<from-keyvault>'
  imageRegistryCredentials:
  - server: neuralacr.azurecr.io
    username: <acr-username>
    password: <acr-password>
  osType: Linux
  ipAddress:
    type: Public
    dnsNameLabel: neural-api
    ports:
    - protocol: TCP
      port: 8000
  restartPolicy: OnFailure
tags: null
type: Microsoft.ContainerInstance/containerGroups
```

```bash
az container create --resource-group neural-rg --file container-group.yaml
```

## Azure Database and Cache

### Azure Database for PostgreSQL

```bash
# Create PostgreSQL Flexible Server
az postgres flexible-server create \
  --resource-group neural-rg \
  --name neural-db \
  --location eastus \
  --admin-user neural \
  --admin-password $(openssl rand -hex 32) \
  --sku-name Standard_D4s_v3 \
  --tier GeneralPurpose \
  --version 15 \
  --storage-size 128 \
  --storage-auto-grow Enabled \
  --backup-retention 7 \
  --geo-redundant-backup Enabled \
  --high-availability Enabled \
  --zone 1 \
  --standby-zone 2

# Create database
az postgres flexible-server db create \
  --resource-group neural-rg \
  --server-name neural-db \
  --database-name neural_api

# Configure firewall (allow Azure services)
az postgres flexible-server firewall-rule create \
  --resource-group neural-rg \
  --name neural-db \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Enable private endpoint
az network vnet subnet create \
  --resource-group neural-rg \
  --vnet-name neural-vnet \
  --name db-subnet \
  --address-prefixes 10.0.2.0/24

az postgres flexible-server update \
  --resource-group neural-rg \
  --name neural-db \
  --subnet /subscriptions/<sub-id>/resourceGroups/neural-rg/providers/Microsoft.Network/virtualNetworks/neural-vnet/subnets/db-subnet \
  --private-dns-zone /subscriptions/<sub-id>/resourceGroups/neural-rg/providers/Microsoft.Network/privateDnsZones/neural-db.postgres.database.azure.com
```

### Azure Cache for Redis

```bash
# Create Redis cache
az redis create \
  --resource-group neural-rg \
  --name neural-cache \
  --location eastus \
  --sku Premium \
  --vm-size P1 \
  --redis-version 6 \
  --enable-non-ssl-port false \
  --minimum-tls-version 1.2 \
  --zones 1 2 3 \
  --replicas-per-primary 2

# Configure persistence
az redis patch-schedule create \
  --resource-group neural-rg \
  --name neural-cache \
  --schedule-entries '[{"dayOfWeek":"Sunday","startHourUtc":3,"maintenanceWindow":"PT5H"}]'

# Get connection string
az redis show \
  --resource-group neural-rg \
  --name neural-cache \
  --query [hostName,sslPort,accessKeys.primaryKey] \
  --output tsv
```

## Load Balancing

### Application Gateway

```bash
# Create public IP
az network public-ip create \
  --resource-group neural-rg \
  --name neural-appgw-ip \
  --allocation-method Static \
  --sku Standard

# Create Application Gateway
az network application-gateway create \
  --resource-group neural-rg \
  --name neural-appgw \
  --location eastus \
  --vnet-name neural-vnet \
  --subnet appgw-subnet \
  --capacity 2 \
  --sku Standard_v2 \
  --public-ip-address neural-appgw-ip \
  --frontend-port 443 \
  --http-settings-port 8000 \
  --http-settings-protocol Http \
  --priority 100

# Add SSL certificate
az network application-gateway ssl-cert create \
  --resource-group neural-rg \
  --gateway-name neural-appgw \
  --name neural-ssl \
  --cert-file neural.pfx \
  --cert-password <password>

# Configure health probe
az network application-gateway probe create \
  --resource-group neural-rg \
  --gateway-name neural-appgw \
  --name neural-health \
  --protocol Http \
  --path /health \
  --interval 30 \
  --timeout 30 \
  --threshold 3

# Add backend pool
az network application-gateway address-pool create \
  --resource-group neural-rg \
  --gateway-name neural-appgw \
  --name neural-backend \
  --servers <aks-node-ips>
```

### AKS Ingress with App Gateway

```bash
# Enable App Gateway Ingress Controller addon
az aks enable-addons \
  --resource-group neural-rg \
  --name neural-aks \
  --addons ingress-appgw \
  --appgw-name neural-appgw
```

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-ingress
  namespace: neural-prod
  annotations:
    kubernetes.io/ingress.class: azure/application-gateway
    appgw.ingress.kubernetes.io/ssl-redirect: "true"
    appgw.ingress.kubernetes.io/request-timeout: "300"
spec:
  tls:
  - hosts:
    - neural.example.com
    secretName: neural-tls
  rules:
  - host: neural.example.com
    http:
      paths:
      - path: /api/*
        pathType: Prefix
        backend:
          service:
            name: neural-api
            port:
              number: 8000
```

## Storage

### Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --resource-group neural-rg \
  --name neuralstorage \
  --location eastus \
  --sku Standard_GRS \
  --kind StorageV2 \
  --access-tier Hot \
  --https-only true \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false

# Create container
az storage container create \
  --account-name neuralstorage \
  --name models \
  --auth-mode login

# Set lifecycle management
cat > lifecycle-policy.json <<'EOF'
{
  "rules": [
    {
      "enabled": true,
      "name": "archive-old-models",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 90
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 180
            }
          }
        },
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["models/"]
        }
      }
    }
  ]
}
EOF

az storage account management-policy create \
  --account-name neuralstorage \
  --policy @lifecycle-policy.json
```

### Azure Files (Shared Storage)

```bash
# Create file share
az storage share create \
  --account-name neuralstorage \
  --name neural-shared \
  --quota 1024

# Mount in AKS
kubectl create secret generic azure-storage-secret \
  --from-literal=azurestorageaccountname=neuralstorage \
  --from-literal=azurestorageaccountkey=$(az storage account keys list --account-name neuralstorage --query [0].value -o tsv) \
  -n neural-prod
```

```yaml
# azure-files-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neural-shared
  namespace: neural-prod
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: azurefile
  resources:
    requests:
      storage: 1Ti
```

## Monitoring

### Azure Monitor

```bash
# Enable Container Insights
az aks enable-addons \
  --resource-group neural-rg \
  --name neural-aks \
  --addons monitoring \
  --workspace-resource-id /subscriptions/<sub-id>/resourceGroups/neural-rg/providers/Microsoft.OperationalInsights/workspaces/neural-workspace

# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group neural-rg \
  --workspace-name neural-workspace \
  --location eastus \
  --retention-time 30

# Create action group
az monitor action-group create \
  --resource-group neural-rg \
  --name neural-alerts \
  --short-name neural \
  --email-receiver name=admin email=admin@example.com

# Create metric alert
az monitor metrics alert create \
  --resource-group neural-rg \
  --name neural-high-cpu \
  --description "Alert when CPU > 80%" \
  --scopes /subscriptions/<sub-id>/resourceGroups/neural-rg/providers/Microsoft.ContainerService/managedClusters/neural-aks \
  --condition "avg Percentage CPU > 80" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action neural-alerts
```

### Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
  --app neural-api \
  --location eastus \
  --resource-group neural-rg \
  --application-type web

# Get instrumentation key
az monitor app-insights component show \
  --app neural-api \
  --resource-group neural-rg \
  --query instrumentationKey -o tsv
```

```python
# neural/monitoring/azure_insights.py
from applicationinsights import TelemetryClient
import os

def setup_app_insights():
    """Setup Application Insights."""
    instrumentation_key = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY")
    tc = TelemetryClient(instrumentation_key)
    return tc

# Usage
tc = setup_app_insights()

@app.post("/api/compile")
async def compile_model(model: Model):
    tc.track_event("model_compilation_started")
    try:
        result = compile_model_service(model)
        tc.track_event("model_compilation_success")
        return result
    except Exception as e:
        tc.track_exception()
        raise
```

## Cost Optimization

### Azure Reserved Instances

```bash
# Purchase AKS node reservation
az reservations reservation-order purchase \
  --reservation-order-id <order-id> \
  --sku Standard_D4s_v3 \
  --location eastus \
  --quantity 3 \
  --term P1Y \
  --billing-scope /subscriptions/<sub-id>
```

### Spot Instances

```bash
# Add spot node pool
az aks nodepool add \
  --resource-group neural-rg \
  --cluster-name neural-aks \
  --name spotpool \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --node-count 3 \
  --min-count 0 \
  --max-count 20 \
  --enable-cluster-autoscaler \
  --node-vm-size Standard_D4s_v3
```

### Storage Tiering

```bash
# Configure blob lifecycle
az storage account management-policy create \
  --account-name neuralstorage \
  --policy @lifecycle-policy.json
```

### Autoscaling

```bash
# Configure cluster autoscaler
az aks nodepool update \
  --resource-group neural-rg \
  --cluster-name neural-aks \
  --name agentpool \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 20
```

## Complete Terraform Example

See [terraform-azure](../examples/terraform/azure/) for complete Infrastructure as Code examples.

## Best Practices

1. **Use Managed Identities** - Don't use service principals with passwords
2. **Enable Azure Policy** - Enforce governance and compliance
3. **Use Private Endpoints** - Reduce public exposure
4. **Enable RBAC** - Fine-grained access control
5. **Use Azure Key Vault** - Centralized secrets management
6. **Enable diagnostic logs** - Track all resource changes
7. **Use tags** - For cost tracking and organization
8. **Implement least privilege** - Minimal permissions
9. **Use availability zones** - For high availability
10. **Regular security assessments** - Use Azure Defender
