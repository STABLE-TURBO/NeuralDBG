# Neural DSL Deployment

Consolidated deployment configurations for Neural DSL across multiple environments and platforms.

## Directory Structure

```
deployment/
├── docker/              # Docker configurations
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── Dockerfile.dashboard
│   ├── Dockerfile.nocode
│   ├── Dockerfile.aquarium
│   └── README.md
├── kubernetes/          # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── redis-deployment.yaml
│   ├── postgres-deployment.yaml
│   ├── api-deployment.yaml
│   ├── worker-deployment.yaml
│   ├── dashboard-deployment.yaml
│   ├── nocode-deployment.yaml
│   ├── aquarium-deployment.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   └── README.md
├── helm/                # Helm charts
│   └── neural-dsl/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── templates/
│       └── README.md
└── README.md           # This file
```

## Quick Start

### Docker Compose
```bash
# From repo root
docker-compose up -d
```

See [docker/README.md](docker/README.md) for building individual images.

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Or use Helm
helm install neural-dsl deployment/helm/neural-dsl/
```

See [kubernetes/README.md](kubernetes/README.md) for detailed K8s deployment guide.

### Helm
```bash
# Install with Helm
helm install neural-dsl deployment/helm/neural-dsl/ -n neural-dsl --create-namespace
```

See [helm/neural-dsl/README.md](helm/neural-dsl/README.md) for Helm configuration options.

## Deployment Options

### 1. Local Development
Use Docker Compose for local development:
```bash
docker-compose -f docker-compose.dev.yml up
```

### 2. Production (Docker)
Use production Docker Compose config:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Kubernetes
Deploy to any Kubernetes cluster:
```bash
cd deployment/kubernetes
./deploy.sh  # or kubectl apply -f .
```

### 4. Helm (Templated)
For flexible, templated deployments:
```bash
helm install neural-dsl deployment/helm/neural-dsl/ \
  --set api.replicaCount=5 \
  --set secrets.secretKey="your-secret"
```

## Configuration

### Docker
- Multi-stage builds for minimal image size
- Non-root user execution
- Health checks included
- See `deployment/docker/README.md`

### Kubernetes
- Namespace isolation
- ConfigMaps for configuration
- Secrets for sensitive data
- HPA for autoscaling
- Ingress for external access
- See `deployment/kubernetes/README.md`

### Helm
- Values-based configuration
- Templated manifests
- Easy upgrades and rollbacks
- See `deployment/helm/neural-dsl/README.md`

## Security Checklist

Before deploying to production:

- [ ] Update all default passwords in secrets
- [ ] Configure TLS/SSL certificates
- [ ] Set resource limits appropriately
- [ ] Enable network policies
- [ ] Configure backup strategy
- [ ] Set up monitoring and logging
- [ ] Review security context settings
- [ ] Scan container images for vulnerabilities

## Monitoring

All deployments include:
- Health check endpoints
- Prometheus-compatible metrics (when configured)
- Logging to stdout/stderr

## Support

For deployment issues:
- Check individual README files in subdirectories
- Review [DEPLOYMENT.md](../DEPLOYMENT.md) in repo root
- Open an issue: https://github.com/Lemniscate-world/Neural/issues

## Migration from Old Structure

This directory consolidates the previous structure:
- `dockerfiles/` → `deployment/docker/`
- `k8s/` → `deployment/kubernetes/`
- `helm/` → `deployment/helm/`
- `kubernetes/` → merged into `deployment/kubernetes/`
