# Microservices Architecture Deployment

Deploy Neural DSL with distributed services for high availability and independent scaling.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Service Separation](#service-separation)
- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Load Balancing](#load-balancing)
- [Service Discovery](#service-discovery)
- [Inter-Service Communication](#inter-service-communication)
- [Scaling Strategy](#scaling-strategy)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                            │
│                   (Nginx / HAProxy)                          │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┬──────────────┬──────────────┐
    │                     │              │              │
┌───▼────┐         ┌──────▼─────┐  ┌────▼──────┐ ┌────▼──────┐
│ API #1 │         │  API #2    │  │  API #3   │ │Dashboard  │
│ (8000) │         │  (8000)    │  │  (8000)   │ │  (8050)   │
└───┬────┘         └──────┬─────┘  └────┬──────┘ └────┬──────┘
    │                     │              │             │
    └─────────────────────┴──────────────┴─────────────┘
                          │
              ┌───────────┴────────────┐
              │                        │
    ┌─────────▼──────┐      ┌─────────▼──────┐
    │ PostgreSQL     │      │  Redis Cluster │
    │ (Primary +     │      │  (3 nodes)     │
    │  Replicas)     │      │                │
    └────────────────┘      └────────────────┘
              │
    ┌─────────┴────────────────────┐
    │     Celery Workers           │
    │  ┌───────┐  ┌───────┐       │
    │  │Worker1│  │Worker2│  ...  │
    │  └───────┘  └───────┘       │
    └──────────────────────────────┘
```

## Service Separation

### API Tier (Stateless)

**Multiple instances for:**
- High availability
- Horizontal scaling
- Rolling updates without downtime
- Geographic distribution

**Resource allocation per instance:**
- CPU: 2 cores
- RAM: 2 GB
- Instances: Start with 3, scale based on load

### Worker Tier (Compute)

**Separate worker pools for:**
- Training tasks (GPU-enabled)
- Compilation tasks (CPU-intensive)
- Export tasks (CPU/Memory-intensive)
- General background tasks

**Resource allocation:**
- Training workers: 4 cores, 8 GB RAM, 1 GPU
- Compilation workers: 2 cores, 4 GB RAM
- Export workers: 2 cores, 4 GB RAM
- General workers: 1 core, 2 GB RAM

### Data Tier (Stateful)

**PostgreSQL cluster:**
- Primary (read/write): 4 cores, 8 GB RAM
- Replicas (read-only): 2 cores, 4 GB RAM
- Automatic failover with Patroni/Stolon

**Redis cluster:**
- 3 nodes for high availability
- Redis Sentinel for automatic failover
- 2 GB RAM per node

## Prerequisites

### Infrastructure

- **Minimum 5 servers:**
  - 3x API/Worker servers (4 cores, 8 GB RAM each)
  - 1x Database server (4 cores, 16 GB RAM)
  - 1x Redis cluster (3 nodes, 2 GB RAM each)

- **Load balancer:**
  - Dedicated server or cloud LB (AWS ALB, GCP LB)
  - SSL/TLS termination
  - Health checks configured

- **Shared storage:**
  - NFS or object storage (S3, GCS, Azure Blob)
  - For model files and experiment artifacts

### Network

- Private network for inter-service communication
- Firewall rules configured
- Service mesh (optional): Istio, Linkerd, Consul

## Deployment Steps

### 1. Prepare Infrastructure

```bash
# On each server, install Docker
curl -fsSL https://get.docker.com | sh

# Configure Docker daemon for production
sudo tee /etc/docker/daemon.json <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "userland-proxy": false,
  "live-restore": true
}
EOF

sudo systemctl restart docker
```

### 2. Set Up PostgreSQL Cluster

Using Docker with Patroni for HA:

```yaml
# docker-compose.postgres.yml
version: '3.8'

services:
  etcd1:
    image: quay.io/coreos/etcd:v3.5.0
    command: >
      etcd --name etcd1
      --initial-advertise-peer-urls http://etcd1:2380
      --listen-peer-urls http://0.0.0.0:2380
      --listen-client-urls http://0.0.0.0:2379
      --advertise-client-urls http://etcd1:2379
      --initial-cluster etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      --initial-cluster-state new
      --initial-cluster-token etcd-cluster
    networks:
      - dbnet

  patroni1:
    image: patroni/patroni:latest
    hostname: patroni1
    environment:
      PATRONI_NAME: patroni1
      PATRONI_RESTAPI_LISTEN: '0.0.0.0:8008'
      PATRONI_RESTAPI_CONNECT_ADDRESS: patroni1:8008
      PATRONI_POSTGRESQL_LISTEN: '0.0.0.0:5432'
      PATRONI_POSTGRESQL_CONNECT_ADDRESS: patroni1:5432
      PATRONI_POSTGRESQL_DATA_DIR: /var/lib/postgresql/data
      PATRONI_ETCD3_HOSTS: etcd1:2379,etcd2:2379,etcd3:2379
      PATRONI_SCOPE: neural_cluster
      PATRONI_POSTGRESQL_PGPASS: /tmp/pgpass
      POSTGRES_USER: neural
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: neural_api
    volumes:
      - patroni1-data:/var/lib/postgresql/data
    networks:
      - dbnet

  haproxy:
    image: haproxy:2.8-alpine
    ports:
      - "5432:5432"
      - "7000:7000"
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    depends_on:
      - patroni1
      - patroni2
      - patroni3
    networks:
      - dbnet

networks:
  dbnet:
    driver: overlay

volumes:
  patroni1-data:
  patroni2-data:
  patroni3-data:
```

HAProxy config for PostgreSQL:

```
# haproxy.cfg
global
    maxconn 100

defaults
    log global
    mode tcp
    retries 2
    timeout client 30m
    timeout connect 4s
    timeout server 30m
    timeout check 5s

listen stats
    mode http
    bind *:7000
    stats enable
    stats uri /

listen primary
    bind *:5432
    option httpchk OPTIONS /master
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server patroni1 patroni1:5432 maxconn 100 check port 8008
    server patroni2 patroni2:5432 maxconn 100 check port 8008
    server patroni3 patroni3:5432 maxconn 100 check port 8008

listen replicas
    bind *:5433
    option httpchk OPTIONS /replica
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server patroni1 patroni1:5432 maxconn 100 check port 8008
    server patroni2 patroni2:5432 maxconn 100 check port 8008
    server patroni3 patroni3:5432 maxconn 100 check port 8008
```

### 3. Set Up Redis Cluster

```yaml
# docker-compose.redis.yml
version: '3.8'

services:
  redis-1:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
      - "16379:16379"
    volumes:
      - redis-1-data:/data
    networks:
      - redis-net

  redis-2:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --requirepass ${REDIS_PASSWORD}
    ports:
      - "6380:6379"
      - "16380:16379"
    volumes:
      - redis-2-data:/data
    networks:
      - redis-net

  redis-3:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes --requirepass ${REDIS_PASSWORD}
    ports:
      - "6381:6379"
      - "16381:16379"
    volumes:
      - redis-3-data:/data
    networks:
      - redis-net

  redis-sentinel-1:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf
    depends_on:
      - redis-1
      - redis-2
      - redis-3
    networks:
      - redis-net

networks:
  redis-net:
    driver: overlay

volumes:
  redis-1-data:
  redis-2-data:
  redis-3-data:
```

Redis Sentinel config:

```
# sentinel.conf
port 26379
sentinel monitor neural-redis redis-1 6379 2
sentinel auth-pass neural-redis ${REDIS_PASSWORD}
sentinel down-after-milliseconds neural-redis 5000
sentinel parallel-syncs neural-redis 1
sentinel failover-timeout neural-redis 10000
```

### 4. Deploy API Instances

```yaml
# docker-compose.api.yml
version: '3.8'

services:
  api:
    image: neural-dsl:latest
    environment:
      - DEBUG=false
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=postgresql://neural:${DB_PASSWORD}@haproxy:5432/neural_api
      - DATABASE_READ_URL=postgresql://neural:${DB_PASSWORD}@haproxy:5433/neural_api
      - REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
      - REDIS_SENTINEL_MASTER=neural-redis
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=sentinel://sentinel-1:26379;sentinel://sentinel-2:26379;sentinel://sentinel-3:26379
      - CELERY_RESULT_BACKEND=sentinel://sentinel-1:26379;sentinel://sentinel-2:26379;sentinel://sentinel-3:26379
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - API_WORKERS=4
      - S3_ENABLED=true
      - S3_BUCKET=${S3_BUCKET}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - neural-net
```

### 5. Deploy Worker Pools

```yaml
# docker-compose.workers.yml
version: '3.8'

services:
  worker-training:
    image: neural-dsl:latest
    command: celery -A neural.api.celery_app worker -Q training -c 2 --max-tasks-per-child=10
    environment:
      - REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=sentinel://sentinel-1:26379
      - CELERY_RESULT_BACKEND=sentinel://sentinel-1:26379
      - DATABASE_URL=postgresql://neural:${DB_PASSWORD}@haproxy:5432/neural_api
      - S3_ENABLED=true
      - S3_BUCKET=${S3_BUCKET}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    networks:
      - neural-net

  worker-compilation:
    image: neural-dsl:latest
    command: celery -A neural.api.celery_app worker -Q compilation -c 4 --max-tasks-per-child=50
    environment:
      - REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=sentinel://sentinel-1:26379
      - CELERY_RESULT_BACKEND=sentinel://sentinel-1:26379
      - DATABASE_URL=postgresql://neural:${DB_PASSWORD}@haproxy:5432/neural_api
      - S3_ENABLED=true
      - S3_BUCKET=${S3_BUCKET}
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    networks:
      - neural-net

  worker-export:
    image: neural-dsl:latest
    command: celery -A neural.api.celery_app worker -Q export -c 2 --max-tasks-per-child=20
    environment:
      - REDIS_SENTINEL_HOSTS=sentinel-1:26379,sentinel-2:26379,sentinel-3:26379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - CELERY_BROKER_URL=sentinel://sentinel-1:26379
      - CELERY_RESULT_BACKEND=sentinel://sentinel-1:26379
      - DATABASE_URL=postgresql://neural:${DB_PASSWORD}@haproxy:5432/neural_api
      - S3_ENABLED=true
      - S3_BUCKET=${S3_BUCKET}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    networks:
      - neural-net

networks:
  neural-net:
    driver: overlay
```

### 6. Configure Load Balancer

```nginx
# nginx-lb.conf
upstream neural_api {
    least_conn;
    server api-1:8000 max_fails=3 fail_timeout=30s;
    server api-2:8000 max_fails=3 fail_timeout=30s;
    server api-3:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name api.neural.example.com;

    ssl_certificate /etc/ssl/certs/neural.crt;
    ssl_certificate_key /etc/ssl/private/neural.key;

    location / {
        proxy_pass http://neural_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Health check
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
    }

    location /health {
        access_log off;
        proxy_pass http://neural_api/health;
    }
}
```

## Load Balancing

### Strategy Selection

**Round Robin** (default):
```nginx
upstream neural_api {
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

**Least Connections** (recommended for varying request times):
```nginx
upstream neural_api {
    least_conn;
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

**IP Hash** (session affinity):
```nginx
upstream neural_api {
    ip_hash;
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

**Weighted** (different server capacities):
```nginx
upstream neural_api {
    server api-1:8000 weight=3;
    server api-2:8000 weight=2;
    server api-3:8000 weight=1;
}
```

### Health Checks

```nginx
upstream neural_api {
    server api-1:8000 max_fails=3 fail_timeout=30s;
    server api-2:8000 max_fails=3 fail_timeout=30s;
    server api-3:8000 max_fails=3 fail_timeout=30s;
}
```

## Service Discovery

### Using Consul

```yaml
# docker-compose.consul.yml
version: '3.8'

services:
  consul:
    image: consul:latest
    command: agent -server -bootstrap-expect=3 -ui -client=0.0.0.0
    ports:
      - "8500:8500"
      - "8600:8600/udp"
    networks:
      - neural-net

  registrator:
    image: gliderlabs/registrator:latest
    command: -internal consul://consul:8500
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock
    depends_on:
      - consul
    networks:
      - neural-net
```

### Dynamic Configuration

```bash
# Register service
curl -X PUT http://consul:8500/v1/agent/service/register \
  -d '{
    "id": "api-1",
    "name": "neural-api",
    "address": "10.0.1.10",
    "port": 8000,
    "check": {
      "http": "http://10.0.1.10:8000/health",
      "interval": "10s",
      "timeout": "5s"
    }
  }'
```

## Inter-Service Communication

### Service Mesh with Istio

```yaml
# istio-config.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: neural-api
spec:
  hosts:
  - neural-api
  http:
  - route:
    - destination:
        host: neural-api
        subset: v1
      weight: 90
    - destination:
        host: neural-api
        subset: v2
      weight: 10
    timeout: 300s
    retries:
      attempts: 3
      perTryTimeout: 60s
```

### Circuit Breaker

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: neural-api
spec:
  host: neural-api
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

## Scaling Strategy

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-api
  minReplicas: 3
  maxReplicas: 20
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

### Custom Metrics Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-worker
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-worker
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: celery_queue_length
        selector:
          matchLabels:
            queue_name: training
      target:
        type: AverageValue
        averageValue: "10"
```

## Monitoring

See [Monitoring & Observability](monitoring.md) for comprehensive monitoring setup.

### Key Metrics

- **API latency** (p50, p95, p99)
- **Request rate** (requests/second)
- **Error rate** (4xx, 5xx errors)
- **Worker queue length**
- **Database connections**
- **Cache hit rate**
- **Resource utilization** (CPU, memory, disk)

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues.

### Service-Specific Issues

**API instances not receiving traffic:**
```bash
# Check load balancer
curl http://lb:8080/health

# Check service registration
curl http://consul:8500/v1/health/service/neural-api

# Check network connectivity
docker exec api-1 ping api-2
```

**Database connection pool exhausted:**
```bash
# Monitor connections
docker exec postgres psql -U neural -c "SELECT count(*) FROM pg_stat_activity;"

# Increase pool size
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

**Redis failover issues:**
```bash
# Check Redis Sentinel
docker exec redis-sentinel-1 redis-cli -p 26379 SENTINEL masters

# Force failover
docker exec redis-sentinel-1 redis-cli -p 26379 SENTINEL failover neural-redis
```
