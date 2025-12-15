# Neural DSL Deployment Guide

This guide covers deploying Neural DSL services using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Environment Configuration](#environment-configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### For Docker/Docker Compose
- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM
- 10GB free disk space

## Docker Deployment

### Building Individual Services

Build specific service images:

```bash
# Dashboard
docker build -f deployment/docker/Dockerfile.dashboard -t neural-dsl/dashboard:latest .

# No-Code Interface
docker build -f deployment/docker/Dockerfile.nocode -t neural-dsl/nocode:latest .

# Aquarium IDE
docker build -f deployment/docker/Dockerfile.aquarium -t neural-dsl/aquarium:latest .
```

### Running Individual Containers

```bash
# Run Dashboard
docker run -d \
  --name neural-dashboard \
  -p 8050:8050 \
  neural-dsl/dashboard:latest

# Run No-Code Interface
docker run -d \
  --name neural-nocode \
  -p 8051:8051 \
  neural-dsl/nocode:latest

# Run Aquarium IDE
docker run -d \
  --name neural-aquarium \
  -p 8052:8052 \
  neural-dsl/aquarium:latest
```

## Docker Compose Deployment

### Development Environment

For local development with hot-reload:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

Services will be available at:
- Dashboard (NeuralDbg): http://localhost:8050
- No-Code Interface: http://localhost:8051
- Aquarium IDE: http://localhost:8052
- Nginx (reverse proxy): http://localhost

### Production Environment

For production deployment:

```bash
# Start services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Updating Services

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# Or for specific service
docker-compose build dashboard
docker-compose up -d dashboard
```

## Kubernetes Deployment

### Manual Deployment with kubectl

1. **Create namespace and secrets:**

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Update secrets in deployment/kubernetes/secret.yaml with secure values
# Then apply
kubectl apply -f deployment/kubernetes/secret.yaml

# Apply ConfigMap
kubectl apply -f deployment/kubernetes/configmap.yaml
```

2. **Deploy infrastructure services:**

```bash
# Redis
kubectl apply -f deployment/kubernetes/redis-deployment.yaml

# PostgreSQL
kubectl apply -f deployment/kubernetes/postgres-deployment.yaml

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=redis -n neural-dsl --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres -n neural-dsl --timeout=300s
```

3. **Deploy application services:**

```bash
# Dashboard
kubectl apply -f deployment/kubernetes/dashboard-deployment.yaml

# No-Code
kubectl apply -f deployment/kubernetes/nocode-deployment.yaml

# Aquarium IDE
kubectl apply -f deployment/kubernetes/aquarium-deployment.yaml
```

4. **Configure Ingress:**

```bash
# Update hosts in deployment/kubernetes/ingress.yaml
# Then apply
kubectl apply -f deployment/kubernetes/ingress.yaml
```

5. **Enable autoscaling:**

```bash
kubectl apply -f deployment/kubernetes/hpa.yaml
```

### Verify Deployment

```bash
# Check all resources
kubectl get all -n neural-dsl

# Check pod status
kubectl get pods -n neural-dsl

# Check logs
kubectl logs -f deployment/neural-dashboard -n neural-dsl

# Check service endpoints
kubectl get svc -n neural-dsl

# Describe pod for troubleshooting
kubectl describe pod <pod-name> -n neural-dsl
```

## Helm Deployment

Helm provides the easiest way to deploy Neural DSL in production.

### Installing with Helm

1. **Add repository (if published):**

```bash
helm repo add neural-dsl https://charts.neural-dsl.com
helm repo update
```

2. **Install from local chart:**

```bash
# Install with default values
helm install neural-dsl ./deployment/helm/neural-dsl -n neural-dsl --create-namespace

# Or with custom values
helm install neural-dsl ./deployment/helm/neural-dsl -n neural-dsl \
  --create-namespace \
  --values custom-values.yaml
```

3. **Install with custom configuration:**

```bash
# Create custom values file
cat > custom-values.yaml <<EOF
dashboard:
  replicaCount: 3
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi

ingress:
  enabled: true
  hosts:
    - host: neural.example.com
      paths:
        - path: /
          pathType: Prefix
          service: dashboard
          port: 8050

secrets:
  secretKey: "your-production-secret-key"
EOF

# Install
helm install neural-dsl ./deployment/helm/neural-dsl \
  -n neural-dsl \
  --create-namespace \
  --values custom-values.yaml
```

### Managing Helm Releases

```bash
# List releases
helm list -n neural-dsl

# Get release status
helm status neural-dsl -n neural-dsl

# Upgrade release
helm upgrade neural-dsl ./deployment/helm/neural-dsl -n neural-dsl

# Rollback release
helm rollback neural-dsl -n neural-dsl

# Uninstall release
helm uninstall neural-dsl -n neural-dsl
```

### Helm Values Configuration

Key configuration options in `values.yaml`:

```yaml
# Scale Dashboard replicas
dashboard:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10

# Configure resources
dashboard:
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi

# Configure storage
persistence:
  enabled: true
  size: 50Gi
  storageClass: "fast-ssd"

# Configure ingress
ingress:
  enabled: true
  className: nginx
  hosts:
    - host: neural-dsl.example.com
  tls:
    - secretName: neural-dsl-tls
      hosts:
        - neural-dsl.example.com
```

## Environment Configuration

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Debug mode | false |
| `DASHBOARD_PORT` | Dashboard port | 8050 |
| `NOCODE_PORT` | No-Code interface port | 8051 |
| `AQUARIUM_PORT` | Aquarium IDE port | 8052 |
| `UPDATE_INTERVAL` | Dashboard update interval (ms) | 5000 |

## Monitoring

### Health Checks

Check service status:

```bash
# Docker Compose
docker-compose ps

# View logs
docker-compose logs -f dashboard
docker-compose logs -f nocode
docker-compose logs -f aquarium
```

### Resource Monitoring

```bash
# Docker stats
docker stats
```

## Troubleshooting

### Common Issues

#### 1. Services not starting

```bash
# Check logs
docker-compose logs dashboard
docker-compose logs nocode
docker-compose logs aquarium

# Check service status
docker-compose ps
```

#### 2. Port conflicts

```bash
# Change ports in docker-compose.yml or use environment variables
export DASHBOARD_PORT=8060
export NOCODE_PORT=8061
docker-compose up -d
```

#### 3. Out of memory errors

```bash
# Increase resource limits in docker-compose.yml
services:
  dashboard:
    deploy:
      resources:
        limits:
          memory: 2G
```

#### 4. Permission errors

```bash
# Check volume permissions
docker-compose exec dashboard ls -la /app/data

# Fix permissions
docker-compose exec --user root dashboard chown -R neural:neural /app/data
```

### Volume Backup

```bash
# Docker volumes
docker run --rm -v nocode-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/nocode-data-backup.tar.gz /data
```

## Production Checklist

- [ ] Configure SSL/TLS certificates
- [ ] Set up persistent volumes for data
- [ ] Configure backup strategy
- [ ] Set resource limits and requests
- [ ] Set up log aggregation
- [ ] Document custom configurations

## Support

For issues and questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
