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
docker build -f dockerfiles/Dockerfile.dashboard -t neural-dsl/dashboard:latest .

# No-Code Interface
docker build -f dockerfiles/Dockerfile.nocode -t neural-dsl/nocode:latest .

# Aquarium IDE
docker build -f dockerfiles/Dockerfile.aquarium -t neural-dsl/aquarium:latest .
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
- Documentation: See project README.md
- Email: Lemniscate_zero@proton.me
