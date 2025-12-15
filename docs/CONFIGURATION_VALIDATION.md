# Configuration Validation and Health Check System

Neural DSL includes a comprehensive configuration validation and health check system to ensure reliable deployment and operation of all services.

## Table of Contents

1. [Configuration Validation](#configuration-validation)
2. [Health Check System](#health-check-system)
3. [Startup Validation](#startup-validation)
4. [Kubernetes Integration](#kubernetes-integration)
5. [Configuration Migration](#configuration-migration)
6. [CLI Commands](#cli-commands)

## Configuration Validation

The configuration validation system checks all required and optional environment variables before deployment.

### Features

- **Required Variable Validation**: Ensures all required environment variables are set
- **Value Format Validation**: Validates variable formats (ports, URLs, keys, etc.)
- **Security Checks**: Detects dangerous default values
- **Port Conflict Detection**: Prevents multiple services from using the same port
- **Detailed Reporting**: Provides clear error messages and suggestions

### Usage

#### Validate All Services

```bash
neural config validate
```

#### Validate Specific Services

```bash
neural config validate -s api -s dashboard
```

#### Export Validation Report

```bash
neural config validate --report validation_report.txt
```

### Validation Rules

#### Required Variables

**API Service:**
- `SECRET_KEY`: Minimum 32 characters
- `DATABASE_URL`: Valid database connection string

**Dashboard Service:**
- `SECRET_KEY`: Minimum 32 characters

**Aquarium Service:**
- `SECRET_KEY`: Minimum 32 characters

**Marketplace Service:**
- `SECRET_KEY`: Minimum 32 characters

**Celery Service:**
- `REDIS_HOST`: Valid hostname or IP
- `CELERY_BROKER_URL`: Valid Redis URL
- `CELERY_RESULT_BACKEND`: Valid Redis URL

#### Optional Variables with Defaults

- `API_HOST`: Default `0.0.0.0`
- `API_PORT`: Default `8000`
- `API_WORKERS`: Default `4`
- `DASHBOARD_PORT`: Default `8050`
- `AQUARIUM_PORT`: Default `8051`
- `MARKETPLACE_PORT`: Default `5000`
- `REDIS_PORT`: Default `6379`

#### Security Validations

- Secret keys must be at least 32 characters
- Dangerous default values trigger errors:
  - `change-me-in-production`
  - `insecure-secret-key`
  - `development`
  - `test`

#### Port Validations

- Ports must be between 1024 and 65535
- No two services can use the same port
- Redis port must be between 1 and 65535

## Health Check System

Each service provides health check endpoints for monitoring and orchestration.

### Health Check Endpoints

#### Standard Health Check

All services expose `/health`:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.3.0",
  "services": {
    "api": "healthy",
    "redis": "healthy",
    "celery": "healthy"
  }
}
```

#### Liveness Probe

Kubernetes liveness probe at `/health/live`:

```bash
curl http://localhost:8000/health/live
```

Response:
```json
{
  "status": "alive"
}
```

Returns HTTP 503 if service is dead.

#### Readiness Probe

Kubernetes readiness probe at `/health/ready`:

```bash
curl http://localhost:8000/health/ready
```

Response:
```json
{
  "status": "ready"
}
```

Returns HTTP 503 if service is not ready.

#### Detailed Health Check

API service provides detailed health at `/health/detailed`:

```bash
curl http://localhost:8000/health/detailed
```

Response:
```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "services": {
    "api": {
      "name": "api",
      "status": "healthy",
      "message": "API is healthy",
      "response_time_ms": 1.23,
      "checked_at": "2024-01-15T10:30:00.000Z"
    },
    "redis": {
      "name": "redis",
      "status": "healthy",
      "message": "Redis is healthy",
      "details": {
        "version": "7.0.0",
        "uptime_days": 5,
        "connected_clients": 3,
        "used_memory_human": "2.5M"
      },
      "response_time_ms": 2.45
    },
    "celery": {
      "name": "celery",
      "status": "healthy",
      "message": "2 worker(s) active",
      "details": {
        "workers": 2,
        "active_tasks": 5
      },
      "response_time_ms": 15.67
    }
  }
}
```

### Service-Specific Health Checks

#### API Service
- Endpoint: `http://localhost:8000/health`
- Checks: API server, Redis connection, Celery workers

#### Dashboard Service
- Endpoint: `http://localhost:8050/health`
- Checks: Dashboard server status

#### Aquarium Service
- Endpoint: `http://localhost:8051/health`
- Checks: Aquarium IDE server status

#### Marketplace Service
- Endpoint: `http://localhost:5000/health`
- Checks: Marketplace API server, registry access

### CLI Health Check

Check service health from CLI:

```bash
neural config check-health
```

Check specific services:

```bash
neural config check-health -s api -s redis
```

Detailed health information:

```bash
neural config check-health --detailed
```

Output:
```
Service Health Status:

  ✓ api          healthy     API is healthy
      Response time: 1.23ms
  ✓ dashboard    healthy     Dashboard is healthy
      Response time: 2.45ms
  ✓ aquarium     healthy     Aquarium is healthy
      Response time: 3.12ms
  ✓ redis        healthy     Redis is healthy
      version: 7.0.0
      uptime_days: 5
      connected_clients: 3
      Response time: 2.45ms
  ⚠ celery       degraded    No workers available
      workers: 0
      active_tasks: 0
      Response time: 15.67ms

All services are healthy!
```

## Startup Validation

Services automatically validate configuration at startup.

### API Server Startup

The API server validates configuration on startup:

```python
from neural.config.validator import ConfigValidator

validator = ConfigValidator()
validator.validate_startup(services=['api', 'celery'])
```

If validation fails, the server will not start and will display clear error messages.

### Manual Startup Validation

Validate before starting services:

```bash
# Validate all services
neural config validate

# Start services if validation passes
docker-compose up -d
```

## Kubernetes Integration

### Deployment Manifests

Kubernetes deployment manifests are provided with liveness and readiness probes:

#### API Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-api
spec:
  template:
    spec:
      containers:
      - name: api
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
```

### Probe Configuration

**Liveness Probe:**
- Initial delay: 30 seconds
- Period: 10 seconds
- Timeout: 5 seconds
- Failure threshold: 3 (restart after 3 failures)

**Readiness Probe:**
- Initial delay: 10 seconds
- Period: 5 seconds
- Timeout: 3 seconds
- Failure threshold: 2 (mark unhealthy after 2 failures)

### Deploying to Kubernetes

1. **Create secrets:**
   ```bash
   # Edit kubernetes/neural-secrets.yaml with actual secrets
   kubectl apply -f kubernetes/neural-secrets.yaml
   ```

2. **Deploy Redis:**
   ```bash
   kubectl apply -f kubernetes/neural-redis-deployment.yaml
   ```

3. **Deploy services:**
   ```bash
   kubectl apply -f kubernetes/neural-api-deployment.yaml
   kubectl apply -f kubernetes/neural-dashboard-deployment.yaml
   kubectl apply -f kubernetes/neural-aquarium-deployment.yaml
   ```

4. **Check health:**
   ```bash
   kubectl get pods
   kubectl describe pod neural-api-xxxxx
   ```

## Configuration Migration

Migrate between YAML and environment variable configurations.

### YAML to .env

Convert YAML configuration to .env file:

```bash
neural config migrate config.yaml --direction yaml-to-env
```

With custom output:

```bash
neural config migrate config.yaml -o .env.production --overwrite
```

### .env to YAML

Convert .env file to YAML configuration:

```bash
neural config migrate .env --direction env-to-yaml -o config.yaml
```

### Generate Template

Generate a template .env file:

```bash
neural config template
```

Custom output:

```bash
neural config template -o .env.template
```

## CLI Commands

### Configuration Commands

```bash
# Validation
neural config validate                          # Validate all services
neural config validate -s api -s dashboard     # Validate specific services
neural config validate --report report.txt     # Export validation report
neural config validate --env-file .env.prod    # Validate custom .env file

# Health checks
neural config check-health                      # Check all services
neural config check-health -s api              # Check specific service
neural config check-health --detailed          # Show detailed information

# Migration
neural config migrate config.yaml              # YAML to .env
neural config migrate .env --direction env-to-yaml  # .env to YAML
neural config migrate config.yaml -o .env.prod --overwrite

# Templates
neural config template                         # Generate .env.example
neural config template -o .env.template        # Custom output
```

## Environment Variable Reference

### API Service

```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEBUG=false
SECRET_KEY=<secure-random-key>
DATABASE_URL=postgresql://user:pass@localhost:5432/neural
STORAGE_PATH=./neural_storage
EXPERIMENTS_PATH=./neural_experiments
MODELS_PATH=./neural_models
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

### Dashboard Service

```bash
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
SECRET_KEY=<secure-random-key>
```

### Aquarium Service

```bash
AQUARIUM_HOST=0.0.0.0
AQUARIUM_PORT=8051
SECRET_KEY=<secure-random-key>
```

### Redis & Celery

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Webhooks

```bash
WEBHOOK_TIMEOUT=30
WEBHOOK_RETRY_LIMIT=3
```

## Best Practices

1. **Always validate before deployment:**
   ```bash
   neural config validate && docker-compose up -d
   ```

2. **Use strong secret keys:**
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

3. **Monitor service health:**
   ```bash
   watch -n 5 'neural config check-health'
   ```

4. **Keep secrets secure:**
   - Never commit `.env` files to version control
   - Use Kubernetes secrets for production
   - Rotate keys regularly

5. **Export validation reports:**
   ```bash
   neural config validate --report validation_$(date +%Y%m%d).txt
   ```

## Troubleshooting

### Validation Fails

**Problem:** Required variables missing

**Solution:**
```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env

# Validate
neural config validate
```

**Problem:** Port conflicts

**Solution:**
Change conflicting ports in `.env`:
```bash
API_PORT=8000
DASHBOARD_PORT=8050
AQUARIUM_PORT=8051
MARKETPLACE_PORT=5000
```

### Health Check Fails

**Problem:** Service not responding

**Solution:**
1. Check if service is running:
   ```bash
   docker ps
   ```

2. Check service logs:
   ```bash
   docker logs neural-api
   ```

3. Verify ports are not blocked:
   ```bash
   netstat -an | grep 8000
   ```

**Problem:** Redis connection failed

**Solution:**
1. Check Redis is running:
   ```bash
   docker ps | grep redis
   ```

2. Verify Redis connection:
   ```bash
   redis-cli ping
   ```

3. Check REDIS_HOST and REDIS_PORT in `.env`

## API Integration

### Python

```python
from neural.config.validator import ConfigValidator
from neural.config.health import HealthChecker

# Validate configuration
validator = ConfigValidator()
result = validator.validate(services=['api'])

if result.has_errors():
    for issue in result.issues:
        print(f"Error: {issue.variable} - {issue.message}")
    raise RuntimeError("Configuration validation failed")

# Check service health
health_checker = HealthChecker()
api_health = health_checker.check_service('api')

if api_health.status != HealthStatus.HEALTHY:
    print(f"API is {api_health.status.value}: {api_health.message}")
```

### REST API

```bash
# Check API health
curl http://localhost:8000/health

# Check all services (detailed)
curl http://localhost:8000/health/detailed

# Check readiness
curl http://localhost:8000/health/ready

# Check liveness
curl http://localhost:8000/health/live
```

## See Also

- [Deployment Guide](DEPLOYMENT.md)
- [Kubernetes Guide](KUBERNETES.md)
- [API Documentation](API.md)
- [Security Best Practices](SECURITY.md)
