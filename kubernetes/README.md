# Neural DSL Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Neural DSL services.

## Quick Start

1. **Create namespace:**
   ```bash
   kubectl create namespace neural
   kubectl config set-context --current --namespace=neural
   ```

2. **Update secrets:**
   
   Edit `neural-secrets.yaml` and replace placeholder values:
   ```bash
   # Generate a secure secret key
   python -c "import secrets; print(secrets.token_hex(32))"
   
   # Update the secret-key in neural-secrets.yaml
   ```

3. **Deploy infrastructure:**
   ```bash
   # Deploy secrets
   kubectl apply -f neural-secrets.yaml
   
   # Deploy Redis
   kubectl apply -f neural-redis-deployment.yaml
   ```

4. **Deploy services:**
   ```bash
   kubectl apply -f neural-api-deployment.yaml
   kubectl apply -f neural-dashboard-deployment.yaml
   kubectl apply -f neural-aquarium-deployment.yaml
   ```

5. **Verify deployment:**
   ```bash
   kubectl get pods
   kubectl get services
   ```

## Services

### API Service
- **Port:** 8000
- **Replicas:** 3
- **Endpoints:**
  - `/health` - Health check
  - `/health/live` - Liveness probe
  - `/health/ready` - Readiness probe
  - `/docs` - API documentation

### Dashboard Service
- **Port:** 8050
- **Replicas:** 2
- **Purpose:** NeuralDbg real-time debugging interface

### Aquarium Service
- **Port:** 8051
- **Replicas:** 2
- **Purpose:** Visual IDE for Neural DSL

### Redis Service
- **Port:** 6379
- **Replicas:** 1
- **Purpose:** Cache and message broker
- **Storage:** Requires PersistentVolume for persistence

## Health Checks

All services include:

- **Liveness Probe:** Ensures pod is alive (restarts if fails)
- **Readiness Probe:** Ensures pod is ready to receive traffic

### Testing Health Checks

```bash
# Port-forward to test locally
kubectl port-forward svc/neural-api 8000:8000

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/detailed
```

## Resource Requirements

### API Service
- Requests: 256Mi memory, 250m CPU
- Limits: 512Mi memory, 500m CPU

### Dashboard Service
- Requests: 512Mi memory, 500m CPU
- Limits: 1Gi memory, 1000m CPU

### Aquarium Service
- Requests: 512Mi memory, 500m CPU
- Limits: 1Gi memory, 1000m CPU

### Marketplace Service
- Requests: 256Mi memory, 250m CPU
- Limits: 512Mi memory, 500m CPU

### Redis Service
- Requests: 128Mi memory, 100m CPU
- Limits: 256Mi memory, 200m CPU

## Storage

### Redis PVC
- Name: `redis-data-pvc`
- Size: 5Gi
- Access Mode: ReadWriteOnce

## Secrets

The `neural-secrets` Secret contains:

- `secret-key`: Secret key for encryption (minimum 32 characters)
- `database-url`: PostgreSQL connection string
- `redis-password`: Redis password (optional)

**Important:** Update these values before deployment!

## Ingress (Optional)

Create an Ingress to expose services externally:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.neural.example.com
    - dashboard.neural.example.com
    - aquarium.neural.example.com
    secretName: neural-tls
  rules:
  - host: api.neural.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neural-api
            port:
              number: 8000
  - host: dashboard.neural.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neural-dashboard
            port:
              number: 8050
  - host: aquarium.neural.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neural-aquarium
            port:
              number: 8051
```

## Monitoring

### Check Pod Status

```bash
kubectl get pods -w
```

### View Logs

```bash
# API logs
kubectl logs -f deployment/neural-api

# Dashboard logs
kubectl logs -f deployment/neural-dashboard

# All services
kubectl logs -f -l app=neural
```

### Describe Pod

```bash
kubectl describe pod <pod-name>
```

### Events

```bash
kubectl get events --sort-by='.lastTimestamp'
```

## Scaling

### Manual Scaling

```bash
# Scale API service
kubectl scale deployment neural-api --replicas=5

# Scale dashboard
kubectl scale deployment neural-dashboard --replicas=3
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Updates and Rollbacks

### Rolling Update

```bash
# Update image
kubectl set image deployment/neural-api api=neural-dsl/api:v1.1.0

# Check rollout status
kubectl rollout status deployment/neural-api
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/neural-api

# Rollback to specific revision
kubectl rollout undo deployment/neural-api --to-revision=2
```

### Rollout History

```bash
kubectl rollout history deployment/neural-api
```

## Troubleshooting

### Pod Not Starting

1. Check pod status:
   ```bash
   kubectl describe pod <pod-name>
   ```

2. Check logs:
   ```bash
   kubectl logs <pod-name>
   ```

3. Check events:
   ```bash
   kubectl get events --field-selector involvedObject.name=<pod-name>
   ```

### Health Check Failing

1. Check probe configuration:
   ```bash
   kubectl describe pod <pod-name>
   ```

2. Test endpoint manually:
   ```bash
   kubectl port-forward <pod-name> 8000:8000
   curl http://localhost:8000/health/ready
   ```

3. Check logs for errors:
   ```bash
   kubectl logs <pod-name>
   ```

### Storage Issues

1. Check PVC status:
   ```bash
   kubectl get pvc
   ```

2. Describe PVC:
   ```bash
   kubectl describe pvc <pvc-name>
   ```

3. Check storage class:
   ```bash
   kubectl get storageclass
   ```

## Cleanup

Remove all Neural DSL resources:

```bash
# Delete deployments
kubectl delete -f neural-api-deployment.yaml
kubectl delete -f neural-dashboard-deployment.yaml
kubectl delete -f neural-aquarium-deployment.yaml
kubectl delete -f neural-redis-deployment.yaml

# Delete secrets
kubectl delete -f neural-secrets.yaml

# Delete PVCs (warning: this deletes data!)
kubectl delete pvc redis-data-pvc

# Delete namespace
kubectl delete namespace neural
```

## Production Checklist

- [ ] Update all secrets with secure values
- [ ] Configure persistent storage with appropriate storage class
- [ ] Set up database (PostgreSQL) with backups
- [ ] Configure Ingress with TLS certificates
- [ ] Set up monitoring and logging (Prometheus, Grafana, ELK)
- [ ] Configure resource limits based on load testing
- [ ] Set up Horizontal Pod Autoscaling
- [ ] Configure network policies
- [ ] Enable pod security policies
- [ ] Set up disaster recovery plan
- [ ] Configure backup strategy for PVCs
- [ ] Test failover scenarios
- [ ] Document incident response procedures

## See Also

- [Configuration Validation Guide](../docs/CONFIGURATION_VALIDATION.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)
- [API Documentation](../docs/API.md)
- [Security Best Practices](../SECURITY.md)
