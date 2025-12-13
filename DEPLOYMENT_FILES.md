# Neural DSL Deployment Files Index

Complete index of all deployment-related files in the Neural DSL project.

## Docker Images

Multi-stage Dockerfiles optimized for production:

```
dockerfiles/
├── Dockerfile.api          # Neural DSL REST API service
├── Dockerfile.worker       # Celery workers with ML backends
├── Dockerfile.aquarium     # Aquarium IDE backend
├── Dockerfile.dashboard    # NeuralDbg debugging dashboard
└── Dockerfile.nocode       # No-code model builder interface
```

## Docker Compose

### Production
- `docker-compose.yml` - Production deployment with all services
  - PostgreSQL database
  - Redis cache/broker
  - API with uvicorn
  - Celery workers
  - Dashboard, No-Code, Aquarium IDE
  - Nginx reverse proxy
  - Flower monitoring

### Development
- `docker-compose.dev.yml` - Development environment
  - Hot-reload enabled
  - Debug mode
  - Direct port access
  - Volume mounts for code

## Kubernetes Manifests

Raw Kubernetes YAML manifests:

```
k8s/
├── namespace.yaml              # neural-dsl namespace
├── configmap.yaml              # Environment configuration
├── secret.yaml                 # Passwords, keys (UPDATE BEFORE DEPLOY!)
├── redis-deployment.yaml       # Redis StatefulSet + Service + PVC
├── postgres-deployment.yaml    # PostgreSQL StatefulSet + Service + PVC
├── api-deployment.yaml         # API Deployment + Service + PVC
├── worker-deployment.yaml      # Worker Deployment (scalable)
├── dashboard-deployment.yaml   # Dashboard Deployment + Service
├── nocode-deployment.yaml      # No-Code Deployment + Service + PVC
├── aquarium-deployment.yaml    # Aquarium Deployment + Service + PVC
├── ingress.yaml                # Ingress controller configuration
├── hpa.yaml                    # Horizontal Pod Autoscalers
└── README.md                   # Kubernetes deployment guide
```

## Helm Chart

Production-ready Helm chart with full customization:

```
helm/neural-dsl/
├── Chart.yaml                  # Chart metadata (v0.3.0)
├── values.yaml                 # Default configuration values
├── README.md                   # Helm chart documentation
└── templates/
    ├── _helpers.tpl            # Template helper functions
    ├── namespace.yaml          # Namespace template
    ├── configmap.yaml          # ConfigMap template
    ├── secret.yaml             # Secret template
    ├── api-deployment.yaml     # API resources template
    ├── ingress.yaml            # Ingress template
    └── hpa.yaml                # HPA template
```

## Deployment Scripts

Automation scripts for building and deploying:

```
scripts/
├── build-images.sh             # Build all Docker images (Linux/Mac)
├── build-images.bat            # Build all Docker images (Windows)
├── push-images.sh              # Push images to registry (Linux/Mac)
├── push-images.bat             # Push images to registry (Windows)
├── deploy-k8s.sh               # Deploy to Kubernetes with kubectl
└── deploy-helm.sh              # Deploy to Kubernetes with Helm
```

Make scripts executable on Linux/Mac:
```bash
chmod +x scripts/*.sh
```

## Configuration Files

### Environment Configuration
- `.env.example` - Environment variable template
  - All configurable parameters documented
  - Secure defaults
  - Production and development settings

### Nginx Configuration
- `nginx.conf` - Reverse proxy configuration
  - Routes for all services
  - WebSocket support for dashboards
  - Subdomain configuration
  - Health check endpoint
  - SSL/TLS ready

### Docker Build Configuration
- `.dockerignore` - Build context exclusions
  - Optimizes image build time
  - Reduces image size
  - Excludes development files

## Documentation

### Main Guides
- `DEPLOYMENT.md` - **Complete deployment guide**
  - Prerequisites
  - Docker deployment
  - Docker Compose deployment
  - Kubernetes deployment
  - Helm deployment
  - Configuration
  - Monitoring
  - Troubleshooting
  - Production checklist

### Quick Start Guides
- `DOCKER_QUICKSTART.md` - Docker quick start (5 minutes)
  - Minimal setup
  - Common commands
  - Troubleshooting

- `KUBERNETES_QUICKSTART.md` - Kubernetes quick start
  - kubectl deployment
  - Helm deployment
  - Common operations

### Summary Documents
- `DEPLOYMENT_SUMMARY.md` - Architecture overview
  - Service architecture diagram
  - Key features
  - Resource requirements
  - Deployment options

- `DEPLOYMENT_FILES.md` - This file
  - Complete file index
  - Quick reference

## Service Ports

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| API | 8000 | HTTP | REST API endpoint |
| Dashboard | 8050 | HTTP/WS | NeuralDbg dashboard |
| No-Code | 8051 | HTTP/WS | Visual model builder |
| Aquarium IDE | 8052 | HTTP/WS | Integrated IDE |
| Flower | 5555 | HTTP | Celery monitoring |
| Redis | 6379 | TCP | Cache/message broker |
| PostgreSQL | 5432 | TCP | Database |
| Nginx | 80/443 | HTTP/HTTPS | Reverse proxy |

## Quick Reference Commands

### Docker Compose

```bash
# Start (development)
docker-compose -f docker-compose.dev.yml up -d

# Start (production)
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=5

# Rebuild
docker-compose build --no-cache
```

### Kubernetes with kubectl

```bash
# Deploy all
./scripts/deploy-k8s.sh

# Or manually
kubectl apply -f k8s/

# Check status
kubectl get all -n neural-dsl

# View logs
kubectl logs -f deployment/neural-api -n neural-dsl

# Scale
kubectl scale deployment neural-api --replicas=5 -n neural-dsl

# Delete
kubectl delete namespace neural-dsl
```

### Kubernetes with Helm

```bash
# Install
helm install neural-dsl ./helm/neural-dsl -n neural-dsl --create-namespace

# Install with custom values
helm install neural-dsl ./helm/neural-dsl \
  -n neural-dsl \
  --create-namespace \
  --values custom-values.yaml

# Upgrade
helm upgrade neural-dsl ./helm/neural-dsl -n neural-dsl

# Status
helm status neural-dsl -n neural-dsl

# Uninstall
helm uninstall neural-dsl -n neural-dsl
```

### Building Images

```bash
# Build all images (Linux/Mac)
./scripts/build-images.sh

# Build all images (Windows)
.\scripts\build-images.bat

# Build with custom registry
REGISTRY=myregistry.io REPO=neural TAG=v0.3.0 ./scripts/build-images.sh

# Push images
./scripts/push-images.sh
```

## Environment Variables

### Critical (Must be changed in production)
```bash
SECRET_KEY=<generate-random-key>
REDIS_PASSWORD=<secure-password>
POSTGRES_PASSWORD=<secure-password>
FLOWER_PASSWORD=<secure-password>
```

### Configuration
```bash
DEBUG=false
API_WORKERS=4
CELERY_CONCURRENCY=4
RATE_LIMIT_ENABLED=true
CORS_ORIGINS=["https://your-domain.com"]
```

See `.env.example` for complete list.

## File Sizes & Build Times

Approximate Docker image sizes:
- API: ~500MB
- Worker: ~800MB (includes ML frameworks)
- Dashboard: ~400MB
- No-Code: ~450MB
- Aquarium: ~400MB

Build times (on modern hardware):
- All images: ~10-15 minutes
- Single service: ~2-4 minutes

## Security Checklist

Before production deployment:

- [ ] Update `k8s/secret.yaml` with secure random values
- [ ] Change `SECRET_KEY` in `.env`
- [ ] Change all passwords in `.env`
- [ ] Configure SSL/TLS certificates
- [ ] Set `DEBUG=false`
- [ ] Configure proper `CORS_ORIGINS`
- [ ] Enable network policies (Kubernetes)
- [ ] Set up RBAC (Kubernetes)
- [ ] Configure resource limits
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy
- [ ] Test disaster recovery

## Resource Requirements

### Minimum (Development)
- 4GB RAM
- 2 CPU cores
- 20GB disk space
- Docker Desktop or minikube

### Recommended (Production)
- 16GB RAM (per node)
- 8 CPU cores (per node)
- 100GB disk space
- 3+ Kubernetes nodes
- Load balancer
- Persistent volume provisioner

### Kubernetes Resource Allocation

**API (per pod):**
- Requests: 500m CPU, 512Mi RAM
- Limits: 2 CPU, 2Gi RAM

**Worker (per pod):**
- Requests: 1 CPU, 2Gi RAM
- Limits: 4 CPU, 8Gi RAM

**Dashboard (per pod):**
- Requests: 250m CPU, 256Mi RAM
- Limits: 1 CPU, 1Gi RAM

## Storage Requirements

| Component | Size | Type | Notes |
|-----------|------|------|-------|
| API Data | 50Gi | ReadWriteMany | Models, experiments |
| No-Code | 20Gi | ReadWriteMany | User models |
| Aquarium | 10Gi | ReadWriteMany | IDE workspaces |
| PostgreSQL | 20Gi | ReadWriteOnce | Database |
| Redis | 10Gi | ReadWriteOnce | Cache persistence |

## Deployment Strategies

### Development
- Use `docker-compose.dev.yml`
- Single instance of each service
- Volume mounts for hot-reload
- Debug mode enabled

### Staging
- Use Kubernetes with limited replicas
- Test autoscaling
- Test backups and recovery
- Performance testing

### Production
- Use Kubernetes with Helm
- Multiple replicas for HA
- Autoscaling enabled
- Monitoring and alerting
- Regular backups
- SSL/TLS certificates
- CDN for static assets

## Monitoring Integration

Ready for integration with:
- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboards
- **ELK Stack** - Log aggregation
- **Jaeger** - Distributed tracing
- **Flower** - Celery task monitoring (included)

## Support & Links

- **Main Repository**: https://github.com/Lemniscate-world/Neural
- **Issues**: https://github.com/Lemniscate-world/Neural/issues
- **Documentation**: See README.md and docs/
- **Email**: Lemniscate_zero@proton.me

## Version History

- **v0.3.0** (Current)
  - Initial comprehensive deployment configuration
  - Docker, Docker Compose, Kubernetes, Helm
  - Complete documentation
  - Production-ready configurations
  - Security hardening

## Next Steps

After deployment:
1. Verify all services are running
2. Configure DNS for ingress
3. Set up SSL certificates
4. Configure monitoring
5. Test disaster recovery
6. Document custom configurations
7. Set up CI/CD pipeline

---

**Last Updated**: December 2024
**Deployment Version**: 0.3.0
