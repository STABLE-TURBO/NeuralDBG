# Neural DSL Helm Chart

Helm chart for deploying Neural DSL on Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.0+
- PV provisioner support in the underlying infrastructure (for persistence)

## Installing the Chart

### From local directory

```bash
helm install neural-dsl . -n neural-dsl --create-namespace
```

### With custom values

```bash
helm install neural-dsl . -n neural-dsl \
  --create-namespace \
  --values custom-values.yaml
```

### Using specific values

```bash
helm install neural-dsl . -n neural-dsl \
  --create-namespace \
  --set api.replicaCount=5 \
  --set worker.replicaCount=10 \
  --set secrets.secretKey="your-secret-key"
```

## Uninstalling the Chart

```bash
helm uninstall neural-dsl -n neural-dsl
```

## Configuration

The following table lists the configurable parameters of the Neural DSL chart and their default values.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Global Docker image registry | `""` |
| `global.imagePullSecrets` | Global Docker registry secret names | `[]` |
| `global.storageClass` | Global storage class | `""` |

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `namespace` | Kubernetes namespace | `neural-dsl` |
| `image.registry` | Image registry | `docker.io` |
| `image.repository` | Image repository | `neural-dsl` |
| `image.tag` | Image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |

### API Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.enabled` | Enable API service | `true` |
| `api.replicaCount` | Number of API replicas | `3` |
| `api.service.type` | Service type | `ClusterIP` |
| `api.service.port` | Service port | `8000` |
| `api.resources.requests.cpu` | CPU request | `500m` |
| `api.resources.requests.memory` | Memory request | `512Mi` |
| `api.resources.limits.cpu` | CPU limit | `2000m` |
| `api.resources.limits.memory` | Memory limit | `2Gi` |
| `api.autoscaling.enabled` | Enable HPA | `true` |
| `api.autoscaling.minReplicas` | Minimum replicas | `2` |
| `api.autoscaling.maxReplicas` | Maximum replicas | `10` |

### Worker Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `worker.enabled` | Enable worker service | `true` |
| `worker.replicaCount` | Number of worker replicas | `5` |
| `worker.resources.requests.cpu` | CPU request | `1000m` |
| `worker.resources.requests.memory` | Memory request | `2Gi` |
| `worker.resources.limits.cpu` | CPU limit | `4000m` |
| `worker.resources.limits.memory` | Memory limit | `8Gi` |
| `worker.autoscaling.enabled` | Enable HPA | `true` |
| `worker.autoscaling.minReplicas` | Minimum replicas | `3` |
| `worker.autoscaling.maxReplicas` | Maximum replicas | `20` |

### Dashboard Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dashboard.enabled` | Enable dashboard service | `true` |
| `dashboard.replicaCount` | Number of dashboard replicas | `2` |
| `dashboard.service.port` | Service port | `8050` |

### Redis Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `redis.enabled` | Enable Redis | `true` |
| `redis.auth.password` | Redis password | `changeme` |
| `redis.persistence.enabled` | Enable persistence | `true` |
| `redis.persistence.size` | PVC size | `10Gi` |

### PostgreSQL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Enable PostgreSQL | `true` |
| `postgresql.auth.database` | Database name | `neural_db` |
| `postgresql.auth.username` | Database user | `neural` |
| `postgresql.auth.password` | Database password | `changeme` |
| `postgresql.persistence.enabled` | Enable persistence | `true` |
| `postgresql.persistence.size` | PVC size | `20Gi` |

### Ingress Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hosts` | Ingress hosts configuration | See values.yaml |
| `ingress.tls` | Ingress TLS configuration | See values.yaml |

### Secrets Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.secretKey` | API secret key | `change-this` |
| `secrets.flowerUser` | Flower username | `admin` |
| `secrets.flowerPassword` | Flower password | `changeme` |

## Examples

### Development Environment

```yaml
# dev-values.yaml
api:
  replicaCount: 1
  autoscaling:
    enabled: false

worker:
  replicaCount: 2
  autoscaling:
    enabled: false

persistence:
  size: 10Gi

ingress:
  enabled: false
```

```bash
helm install neural-dsl . -f dev-values.yaml
```

### Production Environment

```yaml
# prod-values.yaml
api:
  replicaCount: 5
  resources:
    requests:
      cpu: 1000m
      memory: 1Gi
    limits:
      cpu: 4000m
      memory: 4Gi
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20

worker:
  replicaCount: 10
  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 8000m
      memory: 16Gi
  autoscaling:
    enabled: true
    minReplicas: 5
    maxReplicas: 50

persistence:
  size: 100Gi
  storageClass: fast-ssd

ingress:
  enabled: true
  hosts:
    - host: neural-dsl.example.com
  tls:
    - secretName: neural-dsl-tls
      hosts:
        - neural-dsl.example.com

secrets:
  secretKey: "your-production-secret-key"

redis:
  auth:
    password: "secure-redis-password"
  persistence:
    size: 20Gi

postgresql:
  auth:
    password: "secure-postgres-password"
  persistence:
    size: 50Gi
```

```bash
helm install neural-dsl . -f prod-values.yaml
```

## Upgrading

```bash
# Upgrade with new values
helm upgrade neural-dsl . -f new-values.yaml

# Upgrade with specific value
helm upgrade neural-dsl . --set api.replicaCount=10
```

## Rollback

```bash
# List releases
helm history neural-dsl

# Rollback to previous release
helm rollback neural-dsl

# Rollback to specific revision
helm rollback neural-dsl 2
```

## Monitoring

```bash
# Get release status
helm status neural-dsl

# Get release values
helm get values neural-dsl

# Get release manifest
helm get manifest neural-dsl
```

## Support

For issues and questions, please refer to the main repository:
https://github.com/Lemniscate-world/Neural
