# Environment Variables Reference

Complete reference for all environment variables used across Neural DSL services.

## Table of Contents

- [API Server](#api-server)
- [Dashboard Services](#dashboard-services)
- [Database Configuration](#database-configuration)
- [Redis Configuration](#redis-configuration)
- [Celery Workers](#celery-workers)
- [Security & Authentication](#security--authentication)
- [Storage & File Paths](#storage--file-paths)
- [Monitoring & Logging](#monitoring--logging)
- [Cloud Integrations](#cloud-integrations)
- [Feature Flags](#feature-flags)

## API Server

### Core Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | API server bind address |
| `API_PORT` | No | `8000` | API server port |
| `API_WORKERS` | No | `4` | Number of Gunicorn workers |
| `DEBUG` | No | `false` | Enable debug mode (never use in production) |
| `API_TITLE` | No | `Neural DSL API` | API documentation title |
| `API_VERSION` | No | `0.3.0` | API version string |

**Example:**
```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEBUG=false
```

**Production Recommendations:**
- Set `API_WORKERS` to `(2 * CPU_CORES) + 1`
- Always set `DEBUG=false` in production
- Use `0.0.0.0` for container deployments, `127.0.0.1` for local only

### Rate Limiting

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | No | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | No | `100` | Max requests per period |
| `RATE_LIMIT_PERIOD` | No | `60` | Time period in seconds |
| `RATE_LIMIT_STORAGE` | No | `redis` | Storage backend (redis/memory) |

**Example:**
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
RATE_LIMIT_STORAGE=redis
```

**Production Recommendations:**
- Always enable rate limiting in production
- Adjust limits based on expected traffic patterns
- Use Redis storage for multi-instance deployments

### CORS Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CORS_ORIGINS` | No | `["*"]` | Allowed CORS origins (JSON array) |
| `CORS_ALLOW_CREDENTIALS` | No | `true` | Allow credentials in CORS requests |
| `CORS_ALLOW_METHODS` | No | `["*"]` | Allowed HTTP methods |
| `CORS_ALLOW_HEADERS` | No | `["*"]` | Allowed request headers |

**Example:**
```bash
CORS_ORIGINS=["https://app.example.com","https://dashboard.example.com"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET","POST","PUT","DELETE","OPTIONS"]
CORS_ALLOW_HEADERS=["*"]
```

**Production Recommendations:**
- Never use `["*"]` in production - specify exact origins
- Limit methods to only what's needed
- Review headers regularly for security

## Dashboard Services

### NeuralDbg Dashboard (Port 8050)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHBOARD_HOST` | No | `0.0.0.0` | Dashboard bind address |
| `DASHBOARD_PORT` | No | `8050` | Dashboard port |
| `DASHBOARD_DEBUG` | No | `false` | Enable Dash debug mode |
| `DASHBOARD_TITLE` | No | `NeuralDbg` | Dashboard page title |

**Example:**
```bash
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=false
```

### No-Code GUI (Port 8051)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NOCODE_HOST` | No | `0.0.0.0` | No-code GUI bind address |
| `NOCODE_PORT` | No | `8051` | No-code GUI port |
| `NOCODE_DEBUG` | No | `false` | Enable debug mode |
| `NOCODE_AUTOSAVE` | No | `true` | Enable auto-save feature |

**Example:**
```bash
NOCODE_HOST=0.0.0.0
NOCODE_PORT=8051
NOCODE_DEBUG=false
NOCODE_AUTOSAVE=true
```

### Aquarium IDE (Optional)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AQUARIUM_HOST` | No | `0.0.0.0` | Aquarium IDE bind address |
| `AQUARIUM_PORT` | No | `8052` | Aquarium IDE port |
| `AQUARIUM_ENABLE_HPO` | No | `true` | Enable HPO features |
| `AQUARIUM_ENABLE_EXPORT` | No | `true` | Enable model export |

**Example:**
```bash
AQUARIUM_HOST=0.0.0.0
AQUARIUM_PORT=8052
AQUARIUM_ENABLE_HPO=true
AQUARIUM_ENABLE_EXPORT=true
```

## Database Configuration

### PostgreSQL (Production)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `DB_POOL_SIZE` | No | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | No | `20` | Max overflow connections |
| `DB_POOL_TIMEOUT` | No | `30` | Pool timeout in seconds |
| `DB_POOL_RECYCLE` | No | `3600` | Connection recycle time |
| `DB_ECHO` | No | `false` | Log SQL queries |

**Example:**
```bash
DATABASE_URL=postgresql://neural:${DB_PASSWORD}@postgres:5432/neural_api
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_ECHO=false
```

**Connection String Formats:**
```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@host:port/database

# PostgreSQL with SSL
DATABASE_URL=postgresql://user:password@host:port/database?sslmode=require

# SQLite (development only)
DATABASE_URL=sqlite:///./neural_api.db
```

**Production Recommendations:**
- Use connection pooling for high-traffic applications
- Enable SSL/TLS for database connections
- Set `DB_ECHO=false` to avoid performance overhead
- Use read replicas for read-heavy workloads

### Database Credentials

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_DB` | Yes | `neural_api` | Database name |
| `POSTGRES_USER` | Yes | `neural` | Database user |
| `POSTGRES_PASSWORD` | Yes | - | Database password |
| `DB_PASSWORD` | Yes | - | Application DB password |

**Example:**
```bash
POSTGRES_DB=neural_api
POSTGRES_USER=neural
POSTGRES_PASSWORD=<strong-random-password>
DB_PASSWORD=<strong-random-password>
```

## Redis Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_HOST` | Yes | `localhost` | Redis server hostname |
| `REDIS_PORT` | No | `6379` | Redis server port |
| `REDIS_DB` | No | `0` | Redis database number |
| `REDIS_PASSWORD` | No | - | Redis password (required for production) |
| `REDIS_URL` | No | - | Complete Redis URL (alternative to individual settings) |
| `REDIS_MAX_CONNECTIONS` | No | `50` | Max connection pool size |
| `REDIS_SOCKET_TIMEOUT` | No | `5` | Socket timeout in seconds |
| `REDIS_SOCKET_CONNECT_TIMEOUT` | No | `5` | Connection timeout in seconds |

**Example:**
```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=<strong-random-password>
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
```

**URL Format:**
```bash
# Without password
REDIS_URL=redis://redis:6379/0

# With password
REDIS_URL=redis://:password@redis:6379/0

# With SSL/TLS
REDIS_URL=rediss://:password@redis:6379/0
```

**Production Recommendations:**
- Always set `REDIS_PASSWORD` in production
- Use Redis Sentinel or Redis Cluster for high availability
- Enable persistence (AOF or RDB) for critical data
- Use SSL/TLS for encrypted connections

## Celery Workers

### Broker & Backend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CELERY_BROKER_URL` | Yes | - | Message broker URL |
| `CELERY_RESULT_BACKEND` | Yes | - | Result backend URL |
| `CELERY_TASK_SERIALIZER` | No | `json` | Task serialization format |
| `CELERY_RESULT_SERIALIZER` | No | `json` | Result serialization format |
| `CELERY_ACCEPT_CONTENT` | No | `["json"]` | Accepted content types |
| `CELERY_TIMEZONE` | No | `UTC` | Celery timezone |

**Example:**
```bash
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=["json"]
CELERY_TIMEZONE=UTC
```

### Worker Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CELERY_WORKER_CONCURRENCY` | No | `4` | Number of worker processes |
| `CELERY_WORKER_MAX_TASKS_PER_CHILD` | No | `1000` | Tasks before worker restart |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | No | `4` | Task prefetch multiplier |
| `CELERY_WORKER_POOL` | No | `prefork` | Worker pool type (prefork/gevent/solo) |
| `CELERY_TASK_TIME_LIMIT` | No | `3600` | Hard time limit (seconds) |
| `CELERY_TASK_SOFT_TIME_LIMIT` | No | `3000` | Soft time limit (seconds) |

**Example:**
```bash
CELERY_WORKER_CONCURRENCY=4
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
CELERY_WORKER_PREFETCH_MULTIPLIER=4
CELERY_WORKER_POOL=prefork
CELERY_TASK_TIME_LIMIT=3600
CELERY_TASK_SOFT_TIME_LIMIT=3000
```

**Production Recommendations:**
- Set concurrency based on CPU cores and task type
- Use `gevent` pool for I/O-bound tasks
- Set reasonable time limits to prevent hung tasks
- Enable worker monitoring with Flower

## Security & Authentication

### Secret Keys

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | - | Application secret key (JWT signing, sessions) |
| `API_KEY_HEADER` | No | `X-API-Key` | HTTP header for API key |
| `JWT_SECRET_KEY` | No | Same as SECRET_KEY | JWT signing key |
| `JWT_ALGORITHM` | No | `HS256` | JWT algorithm |
| `JWT_EXPIRATION` | No | `3600` | JWT expiration in seconds |

**Example:**
```bash
SECRET_KEY=<generate-with-openssl-rand-hex-32>
API_KEY_HEADER=X-API-Key
JWT_SECRET_KEY=<different-from-secret-key>
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600
```

**Generate Strong Keys:**
```bash
# Generate SECRET_KEY (64 characters)
openssl rand -hex 32

# Generate API keys (32 characters)
openssl rand -hex 16
```

### Authentication

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AUTH_ENABLED` | No | `true` | Enable authentication |
| `AUTH_TYPE` | No | `api_key` | Auth type (api_key/jwt/oauth2) |
| `ADMIN_API_KEY` | No | - | Admin API key |
| `REQUIRE_API_KEY` | No | `true` | Require API key for requests |

**Example:**
```bash
AUTH_ENABLED=true
AUTH_TYPE=api_key
ADMIN_API_KEY=<strong-random-key>
REQUIRE_API_KEY=true
```

### SSL/TLS

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SSL_ENABLED` | No | `false` | Enable SSL/TLS |
| `SSL_CERT_PATH` | No | - | Path to SSL certificate |
| `SSL_KEY_PATH` | No | - | Path to SSL private key |
| `SSL_CA_PATH` | No | - | Path to CA certificate |
| `FORCE_HTTPS` | No | `false` | Redirect HTTP to HTTPS |

**Example:**
```bash
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/neural.crt
SSL_KEY_PATH=/etc/ssl/private/neural.key
FORCE_HTTPS=true
```

## Storage & File Paths

### Local Storage

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STORAGE_PATH` | No | `./neural_storage` | Base storage directory |
| `EXPERIMENTS_PATH` | No | `./neural_experiments` | Experiment artifacts path |
| `MODELS_PATH` | No | `./neural_models` | Model storage path |
| `LOGS_PATH` | No | `./logs` | Application logs path |
| `TEMP_PATH` | No | `/tmp/neural` | Temporary files path |

**Example:**
```bash
STORAGE_PATH=/app/data/storage
EXPERIMENTS_PATH=/app/data/experiments
MODELS_PATH=/app/data/models
LOGS_PATH=/app/logs
TEMP_PATH=/tmp/neural
```

**Production Recommendations:**
- Use persistent volumes for storage paths
- Configure regular backups
- Monitor disk usage and set up alerts
- Use object storage (S3, GCS) for large files

### Object Storage (S3-Compatible)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_ENABLED` | No | `false` | Enable S3 storage |
| `S3_BUCKET` | No | - | S3 bucket name |
| `S3_REGION` | No | `us-east-1` | AWS region |
| `S3_ENDPOINT_URL` | No | - | Custom S3 endpoint (MinIO, DigitalOcean) |
| `AWS_ACCESS_KEY_ID` | No | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | No | - | AWS secret key |

**Example:**
```bash
S3_ENABLED=true
S3_BUCKET=neural-models
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
```

## Monitoring & Logging

### Logging Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `LOG_FORMAT` | No | `json` | Log format (json/text) |
| `LOG_FILE` | No | - | Log file path |
| `LOG_ROTATION` | No | `daily` | Log rotation (daily/size) |
| `LOG_RETENTION_DAYS` | No | `30` | Days to retain logs |

**Example:**
```bash
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/neural.log
LOG_ROTATION=daily
LOG_RETENTION_DAYS=30
```

### Prometheus Metrics

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PROMETHEUS_ENABLED` | No | `true` | Enable Prometheus metrics |
| `PROMETHEUS_PORT` | No | `9090` | Metrics endpoint port |
| `PROMETHEUS_PATH` | No | `/metrics` | Metrics endpoint path |

**Example:**
```bash
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
PROMETHEUS_PATH=/metrics
```

### Sentry Error Tracking

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SENTRY_ENABLED` | No | `false` | Enable Sentry error tracking |
| `SENTRY_DSN` | No | - | Sentry DSN URL |
| `SENTRY_ENVIRONMENT` | No | `production` | Environment name |
| `SENTRY_TRACES_SAMPLE_RATE` | No | `0.1` | Performance traces sample rate |

**Example:**
```bash
SENTRY_ENABLED=true
SENTRY_DSN=https://key@sentry.io/project
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

## Cloud Integrations

### AWS Services

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_REGION` | No | `us-east-1` | Default AWS region |
| `AWS_ACCESS_KEY_ID` | No | - | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | No | - | AWS secret access key |
| `AWS_SESSION_TOKEN` | No | - | AWS session token (temporary credentials) |

### Google Cloud

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_PROJECT_ID` | No | - | GCP project ID |
| `GCP_REGION` | No | `us-central1` | Default GCP region |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | - | Path to service account key JSON |

### Azure

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_SUBSCRIPTION_ID` | No | - | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | No | - | Resource group name |
| `AZURE_TENANT_ID` | No | - | Azure AD tenant ID |
| `AZURE_CLIENT_ID` | No | - | Service principal client ID |
| `AZURE_CLIENT_SECRET` | No | - | Service principal secret |

## Feature Flags

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENABLE_HPO` | No | `true` | Enable hyperparameter optimization |
| `ENABLE_AUTOML` | No | `true` | Enable AutoML features |
| `ENABLE_DISTRIBUTED` | No | `false` | Enable distributed training |
| `ENABLE_VISUALIZATION` | No | `true` | Enable visualization features |
| `ENABLE_WEBHOOKS` | No | `true` | Enable webhook notifications |
| `ENABLE_TEAMS` | No | `false` | Enable multi-tenancy features |

**Example:**
```bash
ENABLE_HPO=true
ENABLE_AUTOML=true
ENABLE_DISTRIBUTED=false
ENABLE_VISUALIZATION=true
ENABLE_WEBHOOKS=true
ENABLE_TEAMS=false
```

## Environment-Specific Examples

### Development

```bash
# .env.development
DEBUG=true
API_HOST=127.0.0.1
API_PORT=8000
DATABASE_URL=sqlite:///./neural_dev.db
REDIS_HOST=localhost
REDIS_PASSWORD=
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
LOG_LEVEL=DEBUG
RATE_LIMIT_ENABLED=false
AUTH_ENABLED=false
```

### Staging

```bash
# .env.staging
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DATABASE_URL=postgresql://neural:${DB_PASSWORD}@postgres:5432/neural_staging
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/0
SECRET_KEY=${SECRET_KEY}
LOG_LEVEL=INFO
RATE_LIMIT_ENABLED=true
AUTH_ENABLED=true
PROMETHEUS_ENABLED=true
SENTRY_ENABLED=true
SENTRY_ENVIRONMENT=staging
```

### Production

```bash
# .env.production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8
DATABASE_URL=postgresql://neural:${DB_PASSWORD}@postgres:5432/neural_production
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
REDIS_HOST=redis-cluster
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis-cluster:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis-cluster:6379/0
CELERY_WORKER_CONCURRENCY=8
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
LOG_LEVEL=INFO
LOG_FORMAT=json
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
AUTH_ENABLED=true
REQUIRE_API_KEY=true
SSL_ENABLED=true
FORCE_HTTPS=true
PROMETHEUS_ENABLED=true
SENTRY_ENABLED=true
SENTRY_ENVIRONMENT=production
S3_ENABLED=true
S3_BUCKET=neural-prod-models
```

## Validation Script

Use this script to validate your environment configuration:

```python
#!/usr/bin/env python3
import os
import sys

REQUIRED_VARS = {
    'production': [
        'SECRET_KEY',
        'DATABASE_URL',
        'REDIS_HOST',
        'REDIS_PASSWORD',
        'CELERY_BROKER_URL',
        'CELERY_RESULT_BACKEND',
    ],
    'staging': [
        'SECRET_KEY',
        'DATABASE_URL',
        'REDIS_HOST',
        'CELERY_BROKER_URL',
    ],
    'development': []
}

def validate_env(environment='production'):
    missing = []
    for var in REQUIRED_VARS.get(environment, []):
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing required environment variables for {environment}:")
        for var in missing:
            print(f"  - {var}")
        sys.exit(1)
    else:
        print(f"✅ All required environment variables present for {environment}")

if __name__ == '__main__':
    env = os.getenv('ENVIRONMENT', 'production')
    validate_env(env)
```

## Best Practices

1. **Never commit secrets** - Use `.env` files locally, secrets managers in production
2. **Use different keys per environment** - Don't reuse production keys in staging
3. **Rotate credentials regularly** - Change passwords and keys quarterly
4. **Use secrets management** - HashiCorp Vault, AWS Secrets Manager, etc.
5. **Validate on startup** - Check required variables before running
6. **Document custom variables** - Keep this file updated with project-specific vars
7. **Use infrastructure as code** - Terraform, CloudFormation, etc.
8. **Monitor configuration drift** - Ensure deployed config matches expected state
