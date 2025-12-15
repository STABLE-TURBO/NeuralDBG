# Kubernetes Manifests for Neural DSL

This directory contains Kubernetes manifests for deploying Neural DSL services.

## Files

- `namespace.yaml` - Creates the neural-dsl namespace
- `configmap.yaml` - Configuration for all services
- `dashboard-deployment.yaml` - NeuralDbg dashboard
- `nocode-deployment.yaml` - No-code interface
- `aquarium-deployment.yaml` - Aquarium IDE backend
- `ingress.yaml` - Ingress for external access
- `hpa.yaml` - Horizontal Pod Autoscalers

## Quick Start

1. **Deploy:**

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f dashboard-deployment.yaml
kubectl apply -f nocode-deployment.yaml
kubectl apply -f aquarium-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
```

2. **Verify:**

```bash
kubectl get all -n neural-dsl
kubectl get pods -n neural-dsl
kubectl logs -f deployment/neural-dashboard -n neural-dsl
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

Adjust PVC sizes if needed:

```yaml
spec:
  resources:
    requests:
      storage: 10Gi  # Change this value
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
kubectl logs -f deployment/neural-dashboard -n neural-dsl
kubectl logs -f deployment/neural-nocode -n neural-dsl
kubectl logs -f deployment/neural-aquarium -n neural-dsl
```

Check resources:

```bash
kubectl top pods -n neural-dsl
kubectl top nodes
```

## Scaling

Manual scaling:

```bash
kubectl scale deployment neural-dashboard --replicas=3 -n neural-dsl
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
