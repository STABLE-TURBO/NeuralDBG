# Single-Server Deployment

Deploy all Neural DSL services on a single server for development, staging, or low-traffic production environments.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Configuration](#configuration)
- [Service Management](#service-management)
- [Monitoring](#monitoring)
- [Backup & Recovery](#backup--recovery)
- [Scaling Considerations](#scaling-considerations)

## Overview

Single-server deployment runs all Neural DSL services on one machine:

```
┌─────────────────────────────────────────┐
│         Single Server (8GB+ RAM)        │
├─────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────────────┐  │
│  │   Nginx    │  │  API (Port 8000) │  │
│  │ (Port 80)  │  └──────────────────┘  │
│  └────────────┘  ┌──────────────────┐  │
│                  │ Dashboard (8050) │  │
│  ┌────────────┐  └──────────────────┘  │
│  │ PostgreSQL │  ┌──────────────────┐  │
│  │ (Port 5432)│  │ No-Code (8051)   │  │
│  └────────────┘  └──────────────────┘  │
│  ┌────────────┐  ┌──────────────────┐  │
│  │   Redis    │  │ Celery Workers   │  │
│  │ (Port 6379)│  └──────────────────┘  │
│  └────────────┘  ┌──────────────────┐  │
│                  │ Flower (5555)    │  │
│                  └──────────────────┘  │
└─────────────────────────────────────────┘
```

### When to Use

**Good for:**
- Development and staging environments
- Proof of concepts and demos
- Low to medium traffic (<100 req/s)
- Budget-constrained deployments
- Simple operational requirements

**Not recommended for:**
- High-traffic production (>500 req/s)
- Mission-critical applications requiring 99.9%+ uptime
- Applications requiring horizontal scaling
- Distributed training workloads

## Prerequisites

### Server Requirements

**Minimum Specifications:**
- **CPU:** 4 cores
- **RAM:** 8 GB
- **Disk:** 50 GB SSD
- **OS:** Ubuntu 22.04 LTS, Debian 11, or RHEL 8+

**Recommended Specifications:**
- **CPU:** 8 cores
- **RAM:** 16 GB
- **Disk:** 100 GB SSD
- **OS:** Ubuntu 22.04 LTS

### Software Requirements

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### Network Requirements

- Domain name (e.g., `neural.example.com`)
- DNS A record pointing to server IP
- Firewall ports open:
  - 80 (HTTP)
  - 443 (HTTPS)
  - 22 (SSH - for management only)

## Deployment Steps

### 1. Clone Repository

```bash
# Clone Neural DSL
git clone https://github.com/your-org/neural-dsl.git
cd neural-dsl
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required Variables:**

```bash
# .env
DEBUG=false
SECRET_KEY=$(openssl rand -hex 32)

# Database
POSTGRES_DB=neural_api
POSTGRES_USER=neural
POSTGRES_PASSWORD=$(openssl rand -hex 32)
DB_PASSWORD=${POSTGRES_PASSWORD}

# Redis
REDIS_PASSWORD=$(openssl rand -hex 16)

# Celery
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Storage
STORAGE_PATH=/app/data/storage
EXPERIMENTS_PATH=/app/data/experiments
MODELS_PATH=/app/data/models

# Security
AUTH_ENABLED=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### 3. SSL/TLS Setup

#### Option A: Let's Encrypt (Recommended)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot certonly --standalone -d neural.example.com

# Certificates will be at:
# /etc/letsencrypt/live/neural.example.com/fullchain.pem
# /etc/letsencrypt/live/neural.example.com/privkey.pem
```

#### Option B: Self-Signed (Development Only)

```bash
# Generate self-signed certificate
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem \
  -subj "/CN=neural.example.com"
```

### 4. Configure Nginx

Create `nginx.prod.conf`:

```nginx
events {
    worker_connections 2048;
}

http {
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_status 429;

    # Upstream services
    upstream neural_api {
        server api:8000;
        keepalive 32;
    }

    upstream neural_dashboard {
        server dashboard:8050;
        keepalive 16;
    }

    upstream neural_nocode {
        server nocode:8051;
        keepalive 16;
    }

    upstream flower {
        server flower:5555;
        keepalive 8;
    }

    # HTTP -> HTTPS redirect
    server {
        listen 80;
        server_name neural.example.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name neural.example.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;

        # File upload limits
        client_max_body_size 100M;
        client_body_timeout 300s;

        # API routes
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://neural_api/;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # Dashboard
        location /dashboard/ {
            rewrite ^/dashboard/(.*)$ /$1 break;
            proxy_pass http://neural_dashboard;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # No-Code GUI
        location /nocode/ {
            rewrite ^/nocode/(.*)$ /$1 break;
            proxy_pass http://neural_nocode;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Flower monitoring
        location /flower/ {
            auth_basic "Restricted Access";
            auth_basic_user_file /etc/nginx/.htpasswd;

            rewrite ^/flower/(.*)$ /$1 break;
            proxy_pass http://flower;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

### 5. Update Docker Compose

Create `docker-compose.prod.single.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: neural-postgres
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
    networks:
      - neural-net

  redis:
    image: redis:7-alpine
    container_name: neural-redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always
    networks:
      - neural-net

  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: neural-api
    environment:
      - DEBUG=false
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=${CELERY_BROKER_URL}
      - CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND}
      - API_HOST=${API_HOST}
      - API_PORT=${API_PORT}
      - API_WORKERS=${API_WORKERS}
      - STORAGE_PATH=${STORAGE_PATH}
      - EXPERIMENTS_PATH=${EXPERIMENTS_PATH}
      - MODELS_PATH=${MODELS_PATH}
    volumes:
      - api-data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always
    networks:
      - neural-net

  worker:
    build:
      context: .
      dockerfile: Dockerfile
      target: worker
    container_name: neural-worker
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=${CELERY_BROKER_URL}
      - CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND}
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - STORAGE_PATH=${STORAGE_PATH}
      - EXPERIMENTS_PATH=${EXPERIMENTS_PATH}
      - MODELS_PATH=${MODELS_PATH}
    volumes:
      - api-data:/app/data
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: always
    networks:
      - neural-net

  flower:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: neural-flower
    command: celery -A neural.api.celery_app flower --port=5555
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=${CELERY_BROKER_URL}
      - CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND}
    depends_on:
      - redis
      - worker
    restart: always
    networks:
      - neural-net

  nginx:
    image: nginx:alpine
    container_name: neural-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - /etc/letsencrypt:/etc/nginx/ssl:ro
      - ./nginx-htpasswd:/etc/nginx/.htpasswd:ro
    depends_on:
      - api
    restart: always
    networks:
      - neural-net

volumes:
  postgres-data:
  redis-data:
  api-data:

networks:
  neural-net:
    driver: bridge
```

### 6. Deploy Services

```bash
# Create password file for Flower
sudo apt install apache2-utils -y
htpasswd -c nginx-htpasswd admin

# Build and start services
docker-compose -f docker-compose.yml -f docker-compose.prod.single.yml up -d

# Check logs
docker-compose logs -f

# Verify all services are running
docker-compose ps
```

### 7. Initialize Database

```bash
# Run database migrations
docker-compose exec api python -m neural.api.database migrate

# Create admin user (if applicable)
docker-compose exec api python -m neural.api.cli create-admin
```

### 8. Verify Deployment

```bash
# Test API health
curl https://neural.example.com/health

# Test API endpoint
curl -H "X-API-Key: your-api-key" https://neural.example.com/api/v1/health

# Check Flower (requires auth)
# Open browser: https://neural.example.com/flower/

# Check Dashboard
# Open browser: https://neural.example.com/dashboard/
```

## Configuration

### Resource Limits

Add resource limits to `docker-compose.prod.single.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  worker:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '2.0'
          memory: 2G

  postgres:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  redis:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### PostgreSQL Tuning

Create `postgres.conf`:

```ini
# Connection settings
max_connections = 100
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 20MB
min_wal_size = 1GB
max_wal_size = 4GB
max_worker_processes = 4
max_parallel_workers_per_gather = 2
max_parallel_workers = 4
max_parallel_maintenance_workers = 2
```

Mount in PostgreSQL container:

```yaml
services:
  postgres:
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./postgres.conf:/etc/postgresql/postgresql.conf:ro
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
```

## Service Management

### Start/Stop Services

```bash
# Start all services
docker-compose -f docker-compose.yml -f docker-compose.prod.single.yml up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart api

# View logs
docker-compose logs -f api

# View logs for all services
docker-compose logs -f
```

### Scaling Workers

```bash
# Scale Celery workers
docker-compose up -d --scale worker=4

# Or in docker-compose file:
services:
  worker:
    deploy:
      replicas: 4
```

### Updates and Rollbacks

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart services
docker-compose -f docker-compose.yml -f docker-compose.prod.single.yml build
docker-compose -f docker-compose.yml -f docker-compose.prod.single.yml up -d

# Rollback if needed
git checkout <previous-commit>
docker-compose -f docker-compose.yml -f docker-compose.prod.single.yml up -d
```

## Monitoring

### System Monitoring

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Monitor resources
htop

# Monitor disk I/O
sudo iotop

# Monitor network
sudo nethogs
```

### Docker Monitoring

```bash
# Container stats
docker stats

# Detailed container info
docker-compose ps
docker-compose top

# Disk usage
docker system df
```

### Application Logs

```bash
# API logs
docker-compose logs -f --tail=100 api

# Worker logs
docker-compose logs -f --tail=100 worker

# All logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs --no-color > neural-logs.txt
```

### Prometheus Metrics

Add Prometheus and Grafana to `docker-compose.prod.single.yml`:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: neural-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    restart: always
    networks:
      - neural-net

  grafana:
    image: grafana/grafana:latest
    container_name: neural-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
    restart: always
    networks:
      - neural-net

volumes:
  prometheus-data:
  grafana-data:
```

## Backup & Recovery

### Database Backups

```bash
# Create backup script
cat > backup-db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

docker-compose exec -T postgres pg_dump -U neural neural_api | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
EOF

chmod +x backup-db.sh

# Add to crontab for daily backups at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup-db.sh") | crontab -
```

### Volume Backups

```bash
# Backup all Docker volumes
docker run --rm \
  -v neural_postgres-data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine \
  tar czf /backup/postgres-$(date +%Y%m%d).tar.gz -C /source .

docker run --rm \
  -v neural_api-data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine \
  tar czf /backup/api-data-$(date +%Y%m%d).tar.gz -C /source .
```

### Restore from Backup

```bash
# Restore database
gunzip < backup_20240101_020000.sql.gz | \
  docker-compose exec -T postgres psql -U neural neural_api

# Restore volume
docker run --rm \
  -v neural_postgres-data:/target \
  -v $(pwd)/backups:/backup \
  alpine \
  sh -c "cd /target && tar xzf /backup/postgres-20240101.tar.gz"
```

## Scaling Considerations

### When to Upgrade from Single-Server

Consider microservices or Kubernetes when you experience:

1. **Traffic exceeds capacity** (>500 req/s sustained)
2. **Frequent resource contention** (CPU/memory maxed out)
3. **Need for high availability** (>99.9% uptime SLA)
4. **Independent service scaling** (need more workers than API instances)
5. **Regional distribution** (multi-region deployment)

### Vertical Scaling (Interim Solution)

Before moving to microservices, try:

1. **Upgrade server specs** (more CPU, RAM, faster disks)
2. **Optimize database queries** (indexes, query optimization)
3. **Add caching layer** (Redis caching for frequent queries)
4. **Enable CDN** (for static assets)
5. **Optimize worker concurrency** (tune Celery workers)

### Migration Path

1. **Single-Server** → **Multi-Server** → **Kubernetes**
2. Start with managed database (RDS, Cloud SQL)
3. Separate API and workers to different servers
4. Add load balancer for multiple API instances
5. Eventually move to container orchestration

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

### Quick Fixes

**Services won't start:**
```bash
docker-compose logs
docker-compose down -v  # Remove volumes
docker-compose up -d
```

**Database connection issues:**
```bash
docker-compose exec postgres psql -U neural -d neural_api
# Check connectivity
```

**High memory usage:**
```bash
docker stats
# Adjust resource limits in docker-compose.yml
```

**SSL certificate renewal:**
```bash
sudo certbot renew
docker-compose restart nginx
```
