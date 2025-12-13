# Neural DSL Dockerfiles

Multi-stage Dockerfiles for all Neural DSL services.

## Overview

Each Dockerfile uses multi-stage builds to:
- Minimize final image size
- Separate build and runtime dependencies
- Optimize layer caching
- Improve security (non-root user)

## Available Dockerfiles

### Dockerfile.api
**Neural DSL REST API Service**

- Base: `python:3.10-slim`
- Includes: FastAPI, Uvicorn, Celery client
- Dependencies: Core + API packages
- Port: 8000
- Health check: `/health` endpoint

```bash
# Build
docker build -f dockerfiles/Dockerfile.api -t neural-dsl/api:latest .

# Run
docker run -p 8000:8000 neural-dsl/api:latest
```

### Dockerfile.worker
**Celery Workers for Async Tasks**

- Base: `python:3.10-slim`
- Includes: Celery, ML backends (TF, PyTorch)
- Dependencies: Core + API + Backends
- No exposed ports (worker process)

```bash
# Build
docker build -f dockerfiles/Dockerfile.worker -t neural-dsl/worker:latest .

# Run
docker run neural-dsl/worker:latest
```

### Dockerfile.dashboard
**NeuralDbg Debugging Dashboard**

- Base: `python:3.10-slim`
- Includes: Dash, Plotly, Flask
- Dependencies: Core + Dashboard + Visualization
- Port: 8050

```bash
# Build
docker build -f dockerfiles/Dockerfile.dashboard -t neural-dsl/dashboard:latest .

# Run
docker run -p 8050:8050 neural-dsl/dashboard:latest
```

### Dockerfile.nocode
**No-Code Model Builder Interface**

- Base: `python:3.10-slim`
- Includes: Dash, Visualization, Backends
- Dependencies: Core + Dashboard + Visualization + Backends
- Port: 8051

```bash
# Build
docker build -f dockerfiles/Dockerfile.nocode -t neural-dsl/nocode:latest .

# Run
docker run -p 8051:8051 neural-dsl/nocode:latest
```

### Dockerfile.aquarium
**Aquarium IDE Backend**

- Base: `python:3.10-slim`
- Includes: Dash, IDE components
- Dependencies: Core + Dashboard + Visualization
- Port: 8052

```bash
# Build
docker build -f dockerfiles/Dockerfile.aquarium -t neural-dsl/aquarium:latest .

# Run
docker run -p 8052:8052 neural-dsl/aquarium:latest
```

## Build All Images

Use the provided scripts:

### Linux/Mac
```bash
chmod +x ../scripts/build-images.sh
../scripts/build-images.sh
```

### Windows
```cmd
..\scripts\build-images.bat
```

### Custom Registry/Tag
```bash
REGISTRY=myregistry.io REPO=neural TAG=v0.3.0 ../scripts/build-images.sh
```

## Multi-Stage Build Structure

All Dockerfiles follow this pattern:

```dockerfile
# Stage 1: Builder
FROM python:3.10-slim as builder
# Install build dependencies (gcc, g++, git)
# Copy requirements and source
# Build Python packages in user space
# Install all dependencies

# Stage 2: Runtime
FROM python:3.10-slim
# Install only runtime dependencies
# Copy built packages from builder
# Copy application code
# Create non-root user
# Set up volumes and permissions
# Define health checks
# Set entrypoint
```

## Image Sizes

Approximate compressed sizes:

| Image | Size | Layers |
|-------|------|--------|
| api | ~500MB | 12 |
| worker | ~800MB | 12 |
| dashboard | ~400MB | 11 |
| nocode | ~450MB | 11 |
| aquarium | ~400MB | 11 |

## Security Features

All images include:
- Non-root user execution
- Minimal base image (slim)
- No unnecessary packages
- Health checks
- Read-only root filesystem capable
- Security context compatible

## Build Optimization

### Layer Caching
Dependencies are installed before copying application code to maximize cache hits:

```dockerfile
# Copy requirements first
COPY requirements.txt setup.py ./
RUN pip install -r requirements.txt

# Copy code last (changes frequently)
COPY neural/ ./neural/
```

### Build Arguments
Customize builds with arguments:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg PIP_INDEX_URL=https://my-pypi.org/simple \
  -f dockerfiles/Dockerfile.api \
  -t neural-dsl/api:latest \
  .
```

## Troubleshooting

### Build Fails on Dependencies
```bash
# Clear build cache
docker builder prune -a

# Build with no cache
docker build --no-cache -f dockerfiles/Dockerfile.api -t neural-dsl/api:latest .
```

### Image Too Large
```bash
# Check layers
docker history neural-dsl/api:latest

# Analyze size
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest neural-dsl/api:latest
```

### Permission Issues
All containers run as non-root user `neural` (UID 1000). Ensure volumes have correct permissions:

```bash
# Fix volume permissions
docker run --rm -v myvolume:/data alpine chown -R 1000:1000 /data
```

## Development vs Production

### Development
- Use volume mounts for hot-reload
- Enable debug mode
- Use `docker-compose.dev.yml`

### Production
- Build optimized images
- No volume mounts for code
- Disable debug mode
- Use multi-stage builds
- Scan for vulnerabilities

## Vulnerability Scanning

Scan images before deploying:

```bash
# Using Trivy
trivy image neural-dsl/api:latest

# Using Docker Scout
docker scout cves neural-dsl/api:latest

# Using Snyk
snyk container test neural-dsl/api:latest
```

## Registry Publishing

### Docker Hub
```bash
docker login
docker tag neural-dsl/api:latest username/neural-dsl-api:latest
docker push username/neural-dsl-api:latest
```

### Private Registry
```bash
docker login myregistry.io
docker tag neural-dsl/api:latest myregistry.io/neural-dsl/api:latest
docker push myregistry.io/neural-dsl/api:latest
```

### GitHub Container Registry
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker tag neural-dsl/api:latest ghcr.io/username/neural-dsl-api:latest
docker push ghcr.io/username/neural-dsl-api:latest
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Build and push Docker images
  run: |
    docker build -f dockerfiles/Dockerfile.api -t ${{ secrets.REGISTRY }}/neural-dsl-api:${{ github.sha }} .
    docker push ${{ secrets.REGISTRY }}/neural-dsl-api:${{ github.sha }}
```

### GitLab CI
```yaml
build-api:
  script:
    - docker build -f dockerfiles/Dockerfile.api -t $CI_REGISTRY_IMAGE/api:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE/api:$CI_COMMIT_SHA
```

## Related Documentation

- [Docker Compose Configuration](../docker-compose.yml)
- [Kubernetes Deployments](../k8s/)
- [Deployment Guide](../DEPLOYMENT.md)
- [Quick Start](../DOCKER_QUICKSTART.md)

## Support

For issues with Docker builds:
- Check [DEPLOYMENT.md](../DEPLOYMENT.md) troubleshooting section
- Open an issue: https://github.com/Lemniscate-world/Neural/issues
- Email: Lemniscate_zero@proton.me
