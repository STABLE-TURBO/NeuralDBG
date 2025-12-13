# Kubernetes Manifests for Neural DSL

This directory contains Kubernetes manifests for deploying Neural DSL.

## Files

- `namespace.yaml` - Creates the neural-dsl namespace
- `configmap.yaml` - Configuration for all services
- `secret.yaml` - Secrets (passwords, keys) - UPDATE BEFORE DEPLOYING
- `redis-deployment.yaml` - Redis cache and Celery broker
- `postgres-deployment.yaml` - PostgreSQL database
- `api-deployment.yaml` - Neural DSL API service
- `worker-deployment.yaml` - Celery workers for async tasks
- `dashboard-deployment.yaml` - NeuralDbg dashboard
- `nocode-deployment.yaml` - No-code interface
- `aquarium-deployment.yaml` - Aquarium IDE backend
- `ingress.yaml` - Ingress for external access
- `hpa.yaml` - Horizontal Pod Autoscalers

## Quick Start

1. **Update secrets:**

Edit `secret.yaml` and replace all default passwords and keys with secure values.

2. **Deploy:**

```bash
# Apply all manifests in order
kubectl apply -f namespace.yaml
kubectl apply -f secret.yaml
kubectl apply -f configmap.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f api-deployment.yaml
kubectl apply -f worker-deployment.yaml
kubectl apply -f dashboard-deployment.yaml
kubectl apply -f nocode-deployment.yaml
kubectl apply -f aquarium-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
```

Or use the deployment script:

```bash
../scripts/deploy-k8s.sh
```

3. **Verify:**

```bash
kubectl get all -n neural-dsl
kubectl get pods -n neural-dsl
kubectl logs -f deployment/neural-api -n neural-dsl
```

## Customization

### Resource Limits

Edit resource requests/limits in deployment files:

```yaml
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi
```

### Replicas

Change replica counts:

```yaml
spec:
  replicas: 3  # Change this value
```

### Storage

Adjust PVC sizes:

```yaml
spec:
  resources:
    requests:
      storage: 50Gi  # Change this value
```

### Ingress

Update hosts in `ingress.yaml`:

```yaml
spec:
  rules:
  - host: your-domain.com  # Change this
```

## Monitoring

View logs:

```bash
kubectl logs -f deployment/neural-api -n neural-dsl
kubectl logs -f deployment/neural-worker -n neural-dsl
```

Check resources:

```bash
kubectl top pods -n neural-dsl
kubectl top nodes
```

## Scaling

Manual scaling:

```bash
kubectl scale deployment neural-api --replicas=5 -n neural-dsl
kubectl scale deployment neural-worker --replicas=10 -n neural-dsl
```

Autoscaling is configured in `hpa.yaml`.

## Cleanup

Remove all resources:

```bash
kubectl delete namespace neural-dsl
```

Or remove specific resources:

```bash
kubectl delete -f .
```
