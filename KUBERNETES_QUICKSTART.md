# Kubernetes Quick Start Guide

Deploy Neural DSL to Kubernetes in minutes.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- At least 3 nodes with 4GB RAM each
- 50GB available storage

## Quick Start with kubectl

### 1. Update Secrets

```bash
# Edit k8s/secret.yaml and replace ALL default passwords
# Use secure random values:
python -c "import secrets; print(secrets.token_urlsafe(32))"

nano k8s/secret.yaml
```

### 2. Deploy

```bash
# Make script executable (Linux/Mac)
chmod +x scripts/deploy-k8s.sh

# Run deployment
./scripts/deploy-k8s.sh

# Or manually:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/nocode-deployment.yaml
kubectl apply -f k8s/aquarium-deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
```

### 3. Verify

```bash
# Check all resources
kubectl get all -n neural-dsl

# Check pod status
kubectl get pods -n neural-dsl

# Wait for pods to be ready
kubectl wait --for=condition=ready pod --all -n neural-dsl --timeout=300s
```

### 4. Access Services

```bash
# Port forward for local access
kubectl port-forward svc/neural-api 8000:8000 -n neural-dsl
kubectl port-forward svc/neural-dashboard 8050:8050 -n neural-dsl

# Access at:
# API: http://localhost:8000
# Dashboard: http://localhost:8050
```

## Quick Start with Helm

### 1. Configure Values

```bash
# Copy and edit values
cp helm/neural-dsl/values.yaml my-values.yaml
nano my-values.yaml
```

Update at minimum:
```yaml
secrets:
  secretKey: "your-random-secret-key"

redis:
  auth:
    password: "your-redis-password"

postgresql:
  auth:
    password: "your-postgres-password"

ingress:
  hosts:
    - host: your-domain.com
```

### 2. Install

```bash
# Install with Helm
helm install neural-dsl ./helm/neural-dsl \
  -n neural-dsl \
  --create-namespace \
  --values my-values.yaml

# Or use script
chmod +x scripts/deploy-helm.sh
RELEASE_NAME=neural-dsl NAMESPACE=neural-dsl VALUES_FILE=my-values.yaml ./scripts/deploy-helm.sh
```

### 3. Verify

```bash
# Check release
helm status neural-dsl -n neural-dsl

# Check pods
kubectl get pods -n neural-dsl
```

## Common Operations

### View Logs

```bash
# API logs
kubectl logs -f deployment/neural-api -n neural-dsl

# Worker logs
kubectl logs -f deployment/neural-worker -n neural-dsl

# All API pods
kubectl logs -f -l app=neural-api -n neural-dsl
```

### Scale Services

```bash
# Scale API
kubectl scale deployment neural-api --replicas=5 -n neural-dsl

# Scale workers
kubectl scale deployment neural-worker --replicas=10 -n neural-dsl
```

### Update Configuration

```bash
# Edit ConfigMap
kubectl edit configmap neural-dsl-config -n neural-dsl

# Restart pods to pick up changes
kubectl rollout restart deployment/neural-api -n neural-dsl
```

### Execute Commands

```bash
# Shell in API pod
kubectl exec -it deployment/neural-api -n neural-dsl -- bash

# Run Python
kubectl exec -it deployment/neural-api -n neural-dsl -- python

# Check database
kubectl exec -it deployment/postgres -n neural-dsl -- psql -U neural -d neural_db
```

## Troubleshooting

### Pods Not Starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n neural-dsl

# Check events
kubectl get events -n neural-dsl --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n neural-dsl
```

### PVC Issues

```bash
# Check PVCs
kubectl get pvc -n neural-dsl

# Describe PVC
kubectl describe pvc <pvc-name> -n neural-dsl

# Check storage class
kubectl get storageclass
```

### Ingress Issues

```bash
# Check ingress
kubectl get ingress -n neural-dsl
kubectl describe ingress neural-dsl-ingress -n neural-dsl

# Check ingress controller
kubectl get pods -n ingress-nginx
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n neural-dsl
kubectl describe svc neural-api -n neural-dsl

# Check endpoints
kubectl get endpoints -n neural-dsl
```

## Resource Monitoring

```bash
# Pod resource usage
kubectl top pods -n neural-dsl

# Node resource usage
kubectl top nodes

# HPA status
kubectl get hpa -n neural-dsl
```

## Cleanup

### Delete Namespace (removes everything)

```bash
kubectl delete namespace neural-dsl
```

### Helm Uninstall

```bash
helm uninstall neural-dsl -n neural-dsl
kubectl delete namespace neural-dsl
```

### Delete Specific Resources

```bash
kubectl delete -f k8s/api-deployment.yaml
kubectl delete -f k8s/worker-deployment.yaml
```

## Production Configuration

### High Availability Setup

```yaml
# my-values.yaml
api:
  replicaCount: 5
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - neural-api
        topologyKey: kubernetes.io/hostname

worker:
  replicaCount: 10
  autoscaling:
    enabled: true
    minReplicas: 5
    maxReplicas: 50
```

### Resource Limits

```yaml
api:
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 4000m
      memory: 8Gi

worker:
  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 8000m
      memory: 16Gi
```

### Persistent Storage

```yaml
persistence:
  enabled: true
  size: 100Gi
  storageClass: fast-ssd
  accessMode: ReadWriteMany

postgresql:
  persistence:
    size: 50Gi
    storageClass: fast-ssd

redis:
  persistence:
    size: 20Gi
    storageClass: fast-ssd
```

## Monitoring Setup

### Prometheus

```bash
# Install Prometheus operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

### Grafana

```bash
# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring

# Default credentials: admin/prom-operator
```

## Next Steps

- Configure [Ingress with TLS](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- Set up [cert-manager](https://cert-manager.io/) for automatic SSL
- Configure [Prometheus monitoring](https://prometheus.io/)
- Set up [log aggregation with ELK](https://www.elastic.co/elk-stack)
- Implement [disaster recovery](https://kubernetes.io/docs/concepts/cluster-administration/disaster-recovery/)

## Support

- Full Documentation: [DEPLOYMENT.md](DEPLOYMENT.md)
- Kubernetes Docs: https://kubernetes.io/docs/
- Helm Docs: https://helm.sh/docs/
- Issues: https://github.com/Lemniscate-world/Neural/issues
