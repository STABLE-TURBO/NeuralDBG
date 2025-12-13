# Neural DSL Deployment Guide

This guide covers deploying Neural DSL services using Docker, Docker Compose, and Kubernetes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Docker Compose Deployment](#docker-compose-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Helm Deployment](#helm-deployment)
- [Environment Configuration](#environment-configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### For Docker/Docker Compose
- Docker 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 20GB free disk space

### For Kubernetes
- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (for Helm deployments)
- At least 16GB RAM across nodes
- 100GB free disk space

## Docker Deployment

### Building Individual Services

Build specific service images:

```bash
# API Service
docker build -f dockerfiles/Dockerfile.api -t neural-dsl/api:latest .

# Worker Service
docker build -f dockerfiles/Dockerfile.worker -t neural-dsl/worker:latest .

# Dashboard
docker build -f dockerfiles/Dockerfile.dashboard -t neural-dsl/dashboard:latest .

# No-Code Interface
docker build -f dockerfiles/Dockerfile.nocode -t neural-dsl/nocode:latest .

# Aquarium IDE
docker build -f dockerfiles/Dockerfile.aquarium -t neural-dsl/aquarium:latest .
```

### Running Individual Containers

```bash
# Run API (requires Redis)
docker run -d \
  --name neural-api \
  -p 8000:8000 \
  -e SECRET_KEY=your-secret-key \
  -e REDIS_HOST=redis \
  neural-dsl/api:latest

# Run Dashboard
docker run -d \
  --name neural-dashboard \
  -p 8050:8050 \
  neural-dsl/dashboard:latest
```

## Docker Compose Deployment

### Development Environment

For local development with hot-reload:

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your settings
nano .env

# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

Services will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8050
- No-Code: http://localhost:8051
- Aquarium IDE: http://localhost:8052
- Flower (Celery monitoring): http://localhost:5555
- Nginx (reverse proxy): http://localhost

### Production Environment

For production deployment:

```bash
# Copy and configure environment
cp .env.example .env
nano .env  # Set secure passwords and keys

# Start services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=5

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
docker-compose build api
docker-compose up -d api
```

## Kubernetes Deployment

### Manual Deployment with kubectl

1. **Create namespace and secrets:**

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Update secrets in k8s/secret.yaml with secure values
# Then apply
kubectl apply -f k8s/secret.yaml

# Apply ConfigMap
kubectl apply -f k8s/configmap.yaml
```

2. **Deploy infrastructure services:**

```bash
# Redis
kubectl apply -f k8s/redis-deployment.yaml

# PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=redis -n neural-dsl --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres -n neural-dsl --timeout=300s
```

3. **Deploy application services:**

```bash
# API
kubectl apply -f k8s/api-deployment.yaml

# Worker
kubectl apply -f k8s/worker-deployment.yaml

# Dashboard
kubectl apply -f k8s/dashboard-deployment.yaml

# No-Code
kubectl apply -f k8s/nocode-deployment.yaml

# Aquarium IDE
kubectl apply -f k8s/aquarium-deployment.yaml
```

4. **Configure Ingress:**

```bash
# Update hosts in k8s/ingress.yaml
# Then apply
kubectl apply -f k8s/ingress.yaml
```

5. **Enable autoscaling:**

```bash
kubectl apply -f k8s/hpa.yaml
```

### Verify Deployment

```bash
# Check all resources
kubectl get all -n neural-dsl

# Check pod status
kubectl get pods -n neural-dsl

# Check logs
kubectl logs -f deployment/neural-api -n neural-dsl

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
helm install neural-dsl ./helm/neural-dsl -n neural-dsl --create-namespace

# Or with custom values
helm install neural-dsl ./helm/neural-dsl -n neural-dsl \
  --create-namespace \
  --values custom-values.yaml
```

3. **Install with custom configuration:**

```bash
# Create custom values file
cat > custom-values.yaml <<EOF
api:
  replicaCount: 5
  resources:
    requests:
      cpu: 1000m
      memory: 1Gi
    limits:
      cpu: 4000m
      memory: 4Gi

worker:
  replicaCount: 10
  autoscaling:
    maxReplicas: 50

ingress:
  enabled: true
  hosts:
    - host: neural.example.com
      paths:
        - path: /
          pathType: Prefix
          service: api
          port: 8000

secrets:
  secretKey: "your-production-secret-key"

redis:
  auth:
    password: "secure-redis-password"

postgresql:
  auth:
    password: "secure-postgres-password"
EOF

# Install
helm install neural-dsl ./helm/neural-dsl \
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
helm upgrade neural-dsl ./helm/neural-dsl -n neural-dsl

# Rollback release
helm rollback neural-dsl -n neural-dsl

# Uninstall release
helm uninstall neural-dsl -n neural-dsl
```

### Helm Values Configuration

Key configuration options in `values.yaml`:

```yaml
# Scale API replicas
api:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10

# Scale workers
worker:
  replicaCount: 5
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20

# Configure resources
api:
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

### Required Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SECRET_KEY` | Secret key for API authentication | - | Yes |
| `REDIS_PASSWORD` | Redis password | changeme | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | changeme | Yes |
| `DATABASE_URL` | PostgreSQL connection URL | - | Yes |
| `CELERY_BROKER_URL` | Celery broker URL | - | Yes |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Debug mode | false |
| `API_WORKERS` | Number of API workers | 4 |
| `CELERY_CONCURRENCY` | Celery worker concurrency | 4 |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | true |
| `RATE_LIMIT_REQUESTS` | Requests per period | 100 |
| `RATE_LIMIT_PERIOD` | Rate limit period (seconds) | 60 |
| `CORS_ORIGINS` | Allowed CORS origins | [] |

### Generating Secure Secrets

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate passwords
openssl rand -base64 32

# Or use Kubernetes secrets
kubectl create secret generic neural-dsl-secret \
  --from-literal=SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  -n neural-dsl
```

## Monitoring and Observability

### Health Checks

All services expose health check endpoints:

```bash
# API health
curl http://localhost:8000/health

# Dashboard health
curl http://localhost:8050/

# Check in Kubernetes
kubectl get pods -n neural-dsl
kubectl describe pod <pod-name> -n neural-dsl
```

### Logs

```bash
# Docker Compose
docker-compose logs -f api
docker-compose logs -f worker

# Kubernetes
kubectl logs -f deployment/neural-api -n neural-dsl
kubectl logs -f deployment/neural-worker -n neural-dsl

# Stream logs from all pods
kubectl logs -f -l app=neural-api -n neural-dsl
```

### Metrics

Access Celery Flower for worker metrics:
- Docker Compose: http://localhost:5555
- Kubernetes: Configure ingress for Flower service

### Resource Monitoring

```bash
# Docker stats
docker stats

# Kubernetes resource usage
kubectl top pods -n neural-dsl
kubectl top nodes
```

## Troubleshooting

### Common Issues

#### 1. Services not starting

```bash
# Check logs
docker-compose logs api
kubectl logs deployment/neural-api -n neural-dsl

# Check service dependencies
docker-compose ps
kubectl get pods -n neural-dsl
```

#### 2. Database connection errors

```bash
# Verify PostgreSQL is running
docker-compose ps postgres
kubectl get pods -l app=postgres -n neural-dsl

# Check connection
docker-compose exec api python -c "from neural.api.database import engine; print(engine)"
```

#### 3. Redis connection errors

```bash
# Test Redis connection
docker-compose exec redis redis-cli ping
kubectl exec -it deployment/redis -n neural-dsl -- redis-cli ping

# Check Redis logs
docker-compose logs redis
kubectl logs deployment/redis -n neural-dsl
```

#### 4. Worker not processing tasks

```bash
# Check worker logs
docker-compose logs worker
kubectl logs deployment/neural-worker -n neural-dsl

# Check Celery status via Flower
# Access http://localhost:5555

# Manually test worker
docker-compose exec worker celery -A neural.api.celery_app inspect active
```

#### 5. Out of memory errors

```bash
# Increase resource limits in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G

# Or in Kubernetes deployment
resources:
  limits:
    memory: 4Gi
```

#### 6. Permission errors

```bash
# Check volume permissions
docker-compose exec api ls -la /app/data

# Fix permissions
docker-compose exec --user root api chown -R neural:neural /app/data
```

### Performance Tuning

#### Scale Workers

```bash
# Docker Compose
docker-compose up -d --scale worker=10

# Kubernetes
kubectl scale deployment neural-worker --replicas=10 -n neural-dsl

# Or with HPA (automatic)
kubectl autoscale deployment neural-worker \
  --cpu-percent=75 \
  --min=5 \
  --max=20 \
  -n neural-dsl
```

#### Optimize Resource Allocation

```yaml
# Adjust in docker-compose.yml or k8s manifests
resources:
  requests:
    cpu: 1000m    # 1 CPU core
    memory: 2Gi   # 2GB RAM
  limits:
    cpu: 4000m    # 4 CPU cores
    memory: 8Gi   # 8GB RAM
```

### Backup and Recovery

#### Database Backup

```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U neural neural_db > backup.sql

# Kubernetes
kubectl exec deployment/postgres -n neural-dsl -- \
  pg_dump -U neural neural_db > backup.sql

# Restore
docker-compose exec -T postgres psql -U neural neural_db < backup.sql
```

#### Volume Backup

```bash
# Docker volumes
docker run --rm -v neural-api-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/api-data-backup.tar.gz /data

# Kubernetes PVC
kubectl exec deployment/neural-api -n neural-dsl -- \
  tar czf - /app/data | cat > api-data-backup.tar.gz
```

## Production Checklist

- [ ] Change all default passwords and secret keys
- [ ] Configure SSL/TLS certificates
- [ ] Set up persistent volumes for data
- [ ] Configure backup strategy
- [ ] Enable monitoring and alerting
- [ ] Set resource limits and requests
- [ ] Configure autoscaling
- [ ] Set up log aggregation
- [ ] Configure network policies
- [ ] Enable RBAC in Kubernetes
- [ ] Test disaster recovery procedures
- [ ] Document custom configurations
- [ ] Set up CI/CD pipelines
- [ ] Configure rate limiting
- [ ] Enable CORS properly

## Support

For issues and questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: See project README.md
- Email: Lemniscate_zero@proton.me
