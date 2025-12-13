# Troubleshooting Guide

Common deployment issues and their solutions for Neural DSL production environments.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Application Issues](#application-issues)
- [Database Issues](#database-issues)
- [Redis/Cache Issues](#rediscache-issues)
- [Network Issues](#network-issues)
- [Performance Issues](#performance-issues)
- [Kubernetes Issues](#kubernetes-issues)
- [Security Issues](#security-issues)
- [Monitoring Issues](#monitoring-issues)

## Quick Diagnostics

### Health Check Commands

```bash
# Check API health
curl -f http://localhost:8000/health

# Check all services
docker-compose ps

# Check Kubernetes pods
kubectl get pods -n neural-prod

# Check logs
docker-compose logs --tail=100 api
kubectl logs -f deployment/neural-api -n neural-prod

# Check resources
docker stats
kubectl top nodes
kubectl top pods -n neural-prod
```

### Common Quick Fixes

```bash
# Restart services
docker-compose restart api
kubectl rollout restart deployment/neural-api -n neural-prod

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
kubectl exec -it redis-0 -n neural-prod -- redis-cli FLUSHALL

# Recreate pods
kubectl delete pod -l app=neural-api -n neural-prod

# Check connectivity
docker-compose exec api ping postgres
kubectl exec -it neural-api-xxx -n neural-prod -- ping postgres
```

## Application Issues

### Issue: API won't start

**Symptoms:**
- Container starts but immediately exits
- Health check fails
- Port already in use

**Diagnosis:**
```bash
# Check logs
docker-compose logs api
kubectl logs neural-api-xxx -n neural-prod

# Check port conflicts
netstat -tlnp | grep 8000
lsof -i :8000

# Check environment variables
docker-compose exec api env
kubectl exec neural-api-xxx -n neural-prod -- env
```

**Solutions:**

1. **Port conflict:**
```bash
# Change port in .env
API_PORT=8001

# Or kill process using port
kill -9 $(lsof -t -i:8000)
```

2. **Missing environment variables:**
```bash
# Check required variables
cat .env
# Ensure all required vars are set (see environment-variables.md)
```

3. **Database connection failure:**
```bash
# Test database connection
docker-compose exec api python -c "from neural.api.database import engine; engine.connect()"

# Check DATABASE_URL format
echo $DATABASE_URL
```

### Issue: Import errors / Module not found

**Symptoms:**
- `ModuleNotFoundError: No module named 'neural'`
- Import errors on startup

**Diagnosis:**
```bash
# Check Python path
docker-compose exec api python -c "import sys; print(sys.path)"

# Check installed packages
docker-compose exec api pip list

# Verify neural package
docker-compose exec api python -c "import neural; print(neural.__file__)"
```

**Solutions:**

1. **Rebuild image:**
```bash
docker-compose build --no-cache api
docker-compose up -d api
```

2. **Install in development mode:**
```bash
docker-compose exec api pip install -e .
```

3. **Check Dockerfile:**
```dockerfile
# Ensure these lines exist:
COPY neural/ /app/neural/
ENV PYTHONPATH=/app:$PYTHONPATH
```

### Issue: Workers not processing tasks

**Symptoms:**
- Tasks stuck in queue
- No worker logs
- Queue length increasing

**Diagnosis:**
```bash
# Check workers
docker-compose ps worker
kubectl get pods -l app=neural-worker -n neural-prod

# Check Celery status
docker-compose exec worker celery -A neural.api.celery_app inspect active
docker-compose exec worker celery -A neural.api.celery_app inspect stats

# Check queue length
docker-compose exec redis redis-cli LLEN celery
```

**Solutions:**

1. **Restart workers:**
```bash
docker-compose restart worker
kubectl rollout restart deployment/neural-worker -n neural-prod
```

2. **Check broker connection:**
```bash
# Verify Redis connection
docker-compose exec worker python -c "from neural.api.celery_app import celery; print(celery.connection().as_uri())"
```

3. **Increase worker concurrency:**
```bash
# In docker-compose.yml or deployment
command: celery -A neural.api.celery_app worker -c 8  # Increase from 4
```

4. **Check for stuck tasks:**
```bash
# Purge queue (WARNING: deletes all tasks)
docker-compose exec worker celery -A neural.api.celery_app purge

# Or selectively
docker-compose exec redis redis-cli DEL celery
```

## Database Issues

### Issue: Database connection refused

**Symptoms:**
- `Connection refused` errors
- `could not connect to server`
- API fails to start

**Diagnosis:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres
kubectl get pods -l app=postgres -n neural-prod

# Test connection
docker-compose exec api pg_isready -h postgres -p 5432
kubectl exec -it neural-api-xxx -n neural-prod -- pg_isready -h postgres -p 5432

# Check connection string
echo $DATABASE_URL
```

**Solutions:**

1. **Wait for database to be ready:**
```yaml
# In docker-compose.yml
depends_on:
  postgres:
    condition: service_healthy
```

2. **Check network:**
```bash
# Verify DNS resolution
docker-compose exec api nslookup postgres
kubectl exec neural-api-xxx -n neural-prod -- nslookup postgres

# Test connectivity
docker-compose exec api telnet postgres 5432
```

3. **Check credentials:**
```bash
# Verify credentials in secrets
docker-compose exec postgres psql -U neural -d neural_api
kubectl exec -it postgres-0 -n neural-prod -- psql -U neural -d neural_api
```

### Issue: Database connection pool exhausted

**Symptoms:**
- `OperationalError: (psycopg2.OperationalError) FATAL: sorry, too many clients already`
- Slow API responses

**Diagnosis:**
```bash
# Check active connections
docker-compose exec postgres psql -U neural -d neural_api -c "SELECT count(*) FROM pg_stat_activity;"

# Check max connections
docker-compose exec postgres psql -U neural -d neural_api -c "SHOW max_connections;"

# Check long-running queries
docker-compose exec postgres psql -U neural -d neural_api -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE state = 'active';"
```

**Solutions:**

1. **Increase pool size:**
```bash
# In .env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

2. **Increase max_connections:**
```bash
# In postgres.conf or command
docker-compose exec postgres psql -U postgres -c "ALTER SYSTEM SET max_connections = 200;"
docker-compose restart postgres
```

3. **Kill idle connections:**
```sql
-- Kill idle connections older than 5 minutes
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'neural_api'
  AND state = 'idle'
  AND state_change < NOW() - INTERVAL '5 minutes';
```

4. **Enable connection pooling with PgBouncer:**
```yaml
# docker-compose.yml
pgbouncer:
  image: pgbouncer/pgbouncer:latest
  environment:
    - DATABASES_HOST=postgres
    - DATABASES_PORT=5432
    - DATABASES_USER=neural
    - DATABASES_PASSWORD=${DB_PASSWORD}
    - DATABASES_DBNAME=neural_api
    - POOL_MODE=transaction
    - MAX_CLIENT_CONN=1000
    - DEFAULT_POOL_SIZE=25
```

### Issue: Database slow queries

**Symptoms:**
- High API latency
- Database CPU at 100%
- Queries taking >1s

**Diagnosis:**
```sql
-- Enable query logging
ALTER DATABASE neural_api SET log_min_duration_statement = 1000;

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check missing indexes
SELECT
  schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
  AND n_distinct > 100
ORDER BY n_distinct DESC;
```

**Solutions:**

1. **Add indexes:**
```sql
-- Create indexes on foreign keys
CREATE INDEX idx_experiments_model_id ON experiments(model_id);
CREATE INDEX idx_jobs_user_id ON jobs(user_id);

-- Create composite indexes for common queries
CREATE INDEX idx_experiments_user_created ON experiments(user_id, created_at);
```

2. **Optimize queries:**
```python
# Use select_related for foreign keys
experiments = Experiment.objects.select_related('model').all()

# Use prefetch_related for reverse foreign keys
users = User.objects.prefetch_related('experiments').all()

# Add pagination
experiments = Experiment.objects.all()[:100]
```

3. **Analyze and vacuum:**
```bash
# Analyze tables
docker-compose exec postgres psql -U neural -d neural_api -c "ANALYZE;"

# Vacuum
docker-compose exec postgres psql -U neural -d neural_api -c "VACUUM ANALYZE;"
```

## Redis/Cache Issues

### Issue: Redis connection errors

**Symptoms:**
- `ConnectionError: Error connecting to Redis`
- Workers can't connect to broker
- Cache not working

**Diagnosis:**
```bash
# Check Redis is running
docker-compose ps redis
kubectl get pods -l app=redis -n neural-prod

# Test connection
docker-compose exec redis redis-cli ping
kubectl exec -it redis-0 -n neural-prod -- redis-cli ping

# Check password
docker-compose exec redis redis-cli -a $REDIS_PASSWORD ping
```

**Solutions:**

1. **Check password:**
```bash
# Verify REDIS_PASSWORD is set
echo $REDIS_PASSWORD

# Test with password
docker-compose exec redis redis-cli -a $REDIS_PASSWORD ping
```

2. **Check network:**
```bash
# Test connectivity
docker-compose exec api telnet redis 6379
kubectl exec neural-api-xxx -n neural-prod -- telnet redis 6379
```

3. **Restart Redis:**
```bash
docker-compose restart redis
kubectl rollout restart statefulset/redis -n neural-prod
```

### Issue: Redis memory full

**Symptoms:**
- `OOM command not allowed when used memory > 'maxmemory'`
- Cache misses increasing
- Slow performance

**Diagnosis:**
```bash
# Check memory usage
docker-compose exec redis redis-cli INFO memory

# Check maxmemory
docker-compose exec redis redis-cli CONFIG GET maxmemory

# Check eviction policy
docker-compose exec redis redis-cli CONFIG GET maxmemory-policy
```

**Solutions:**

1. **Increase memory:**
```yaml
# docker-compose.yml
redis:
  command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
```

2. **Set eviction policy:**
```bash
# LRU (Least Recently Used)
docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# LFU (Least Frequently Used)
docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lfu
```

3. **Clear cache:**
```bash
# Flush all data (WARNING: clears everything)
docker-compose exec redis redis-cli FLUSHALL

# Delete specific keys
docker-compose exec redis redis-cli --scan --pattern "session:*" | xargs docker-compose exec redis redis-cli DEL
```

## Network Issues

### Issue: Services can't communicate

**Symptoms:**
- `Name or service not known`
- Connection timeouts
- Services can't reach each other

**Diagnosis:**
```bash
# Check DNS resolution
docker-compose exec api nslookup postgres
kubectl exec neural-api-xxx -n neural-prod -- nslookup postgres

# Check network connectivity
docker-compose exec api ping postgres
kubectl exec neural-api-xxx -n neural-prod -- ping postgres

# Check service endpoints
kubectl get endpoints -n neural-prod

# Check network policies
kubectl get networkpolicies -n neural-prod
```

**Solutions:**

1. **Use service names:**
```yaml
# docker-compose.yml - services communicate by service name
DATABASE_URL=postgresql://neural:password@postgres:5432/neural_api
REDIS_HOST=redis
```

2. **Check network:**
```bash
# Docker Compose
docker network ls
docker network inspect neural_default

# Kubernetes
kubectl get svc -n neural-prod
kubectl describe svc postgres -n neural-prod
```

3. **Fix NetworkPolicy:**
```yaml
# Allow all egress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-all-egress
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - {}
```

### Issue: External API calls fail

**Symptoms:**
- `Connection timed out`
- `SSL certificate verification failed`
- External integrations not working

**Diagnosis:**
```bash
# Test external connectivity
docker-compose exec api curl -v https://api.example.com
kubectl exec neural-api-xxx -n neural-prod -- curl -v https://api.example.com

# Check DNS
docker-compose exec api nslookup api.example.com
kubectl exec neural-api-xxx -n neural-prod -- nslookup api.example.com

# Check firewall/security groups
# Verify outbound rules allow HTTPS (port 443)
```

**Solutions:**

1. **Configure proxy:**
```bash
# In .env
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1,postgres,redis
```

2. **Trust SSL certificates:**
```bash
# Add CA certificate
docker-compose exec api update-ca-certificates
```

3. **Check timeout settings:**
```python
# Increase timeout
import requests
response = requests.get(url, timeout=30)  # 30 seconds
```

## Performance Issues

### Issue: High API latency

**Symptoms:**
- Requests take >2s
- p99 latency high
- User complaints about slowness

**Diagnosis:**
```bash
# Check metrics
curl http://localhost:8000/metrics | grep duration

# Profile request
time curl http://localhost:8000/api/compile

# Check resources
docker stats api
kubectl top pods -l app=neural-api -n neural-prod

# Check database query time
# See "Database slow queries" above
```

**Solutions:**

1. **Scale API instances:**
```bash
# Docker Compose
docker-compose up -d --scale api=5

# Kubernetes
kubectl scale deployment neural-api --replicas=5 -n neural-prod
```

2. **Enable caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_model(model_id):
    return Model.objects.get(id=model_id)
```

3. **Optimize queries:**
```python
# Use select_related and prefetch_related
# Add database indexes
# Implement pagination
```

4. **Increase worker count:**
```yaml
# docker-compose.yml
api:
  environment:
    - API_WORKERS=8  # Increase from 4
```

### Issue: High memory usage

**Symptoms:**
- OOM kills
- Pods/containers restarting
- Memory usage at 90%+

**Diagnosis:**
```bash
# Check memory usage
docker stats
kubectl top pods -n neural-prod

# Check memory limits
docker-compose exec api cat /sys/fs/cgroup/memory/memory.limit_in_bytes
kubectl describe pod neural-api-xxx -n neural-prod | grep -A 5 Limits

# Profile memory
docker-compose exec api python -m memory_profiler neural/api/main.py
```

**Solutions:**

1. **Increase memory limits:**
```yaml
# docker-compose.yml
api:
  deploy:
    resources:
      limits:
        memory: 4G  # Increase from 2G

# Kubernetes
resources:
  limits:
    memory: 4Gi
```

2. **Fix memory leaks:**
```python
# Close database connections
engine.dispose()

# Clear caches periodically
cache.clear()

# Limit batch sizes
for batch in chunks(data, size=100):
    process(batch)
```

3. **Enable garbage collection:**
```python
import gc
gc.collect()
```

## Kubernetes Issues

### Issue: Pods in CrashLoopBackOff

**Symptoms:**
- Pods restart repeatedly
- `CrashLoopBackOff` status
- Application won't stay running

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n neural-prod

# Check events
kubectl describe pod neural-api-xxx -n neural-prod

# Check logs
kubectl logs neural-api-xxx -n neural-prod
kubectl logs neural-api-xxx -n neural-prod --previous
```

**Solutions:**

1. **Fix application errors:**
```bash
# Check logs for errors
kubectl logs neural-api-xxx -n neural-prod | grep -i error
```

2. **Increase startup time:**
```yaml
# deployment.yaml
livenessProbe:
  initialDelaySeconds: 60  # Increase from 30
  periodSeconds: 10
readinessProbe:
  initialDelaySeconds: 30  # Increase from 10
  periodSeconds: 5
```

3. **Check resource limits:**
```yaml
resources:
  limits:
    memory: 2Gi  # Ensure not too low
    cpu: "2"
```

### Issue: ImagePullBackOff

**Symptoms:**
- Pods can't start
- `ImagePullBackOff` status
- Image pull errors

**Diagnosis:**
```bash
# Check pod events
kubectl describe pod neural-api-xxx -n neural-prod

# Check image
kubectl get pod neural-api-xxx -n neural-prod -o jsonpath='{.spec.containers[0].image}'
```

**Solutions:**

1. **Check image exists:**
```bash
# Docker Hub
docker pull neural-dsl:latest

# Private registry
docker login registry.example.com
docker pull registry.example.com/neural-dsl:latest
```

2. **Create image pull secret:**
```bash
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=password \
  -n neural-prod

# Add to deployment
spec:
  imagePullSecrets:
  - name: regcred
```

3. **Fix image tag:**
```yaml
# deployment.yaml
image: neural-dsl:v1.0.0  # Use specific tag, not 'latest'
```

### Issue: Service not accessible

**Symptoms:**
- Can't reach service externally
- `Connection refused` from outside cluster
- Ingress not working

**Diagnosis:**
```bash
# Check service
kubectl get svc -n neural-prod
kubectl describe svc neural-api -n neural-prod

# Check endpoints
kubectl get endpoints neural-api -n neural-prod

# Check ingress
kubectl get ingress -n neural-prod
kubectl describe ingress neural-ingress -n neural-prod

# Test from within cluster
kubectl run -it --rm debug --image=busybox --restart=Never -- wget -qO- http://neural-api:8000/health
```

**Solutions:**

1. **Check service type:**
```yaml
# For external access
apiVersion: v1
kind: Service
spec:
  type: LoadBalancer  # or NodePort
  ports:
  - port: 80
    targetPort: 8000
```

2. **Check ingress:**
```bash
# Verify ingress controller is running
kubectl get pods -n ingress-nginx

# Check ingress rules
kubectl get ingress neural-ingress -n neural-prod -o yaml
```

3. **Check labels:**
```yaml
# Ensure service selector matches pod labels
service:
  selector:
    app: neural-api  # Must match pod labels

deployment:
  template:
    metadata:
      labels:
        app: neural-api  # Must match
```

## Security Issues

### Issue: Secrets not loading

**Symptoms:**
- `Missing SECRET_KEY`
- Authentication fails
- Database password errors

**Diagnosis:**
```bash
# Check secrets exist
kubectl get secrets -n neural-prod
kubectl describe secret neural-secrets -n neural-prod

# Verify secret values
kubectl get secret neural-secrets -n neural-prod -o jsonpath='{.data.secret-key}' | base64 -d

# Check environment variables
kubectl exec neural-api-xxx -n neural-prod -- env | grep SECRET_KEY
```

**Solutions:**

1. **Create secrets:**
```bash
kubectl create secret generic neural-secrets \
  --from-literal=secret-key=$(openssl rand -hex 32) \
  -n neural-prod
```

2. **Mount secrets correctly:**
```yaml
env:
- name: SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: neural-secrets
      key: secret-key
```

3. **Use secrets manager:**
See [secrets-management.md](secrets-management.md) for HashiCorp Vault, AWS Secrets Manager, etc.

## Monitoring Issues

### Issue: Metrics not appearing

**Symptoms:**
- Grafana shows no data
- Prometheus not scraping
- Empty dashboards

**Diagnosis:**
```bash
# Check Prometheus targets
# Open http://localhost:9090/targets

# Check metrics endpoint
curl http://localhost:8000/metrics

# Check ServiceMonitor
kubectl get servicemonitor -n neural-prod
kubectl describe servicemonitor neural-api -n neural-prod
```

**Solutions:**

1. **Fix metrics endpoint:**
```python
from prometheus_client import generate_latest

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

2. **Fix ServiceMonitor:**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neural-api
spec:
  selector:
    matchLabels:
      app: neural-api  # Must match service labels
  endpoints:
  - port: http  # Must match service port name
    path: /metrics
```

3. **Check Prometheus config:**
```bash
kubectl get configmap prometheus-server -n monitoring -o yaml
```

## Emergency Procedures

### Complete System Restart

```bash
# Docker Compose
docker-compose down
docker-compose up -d

# Kubernetes
kubectl rollout restart deployment -n neural-prod
```

### Database Restore

```bash
# From backup
gunzip < backup.sql.gz | docker-compose exec -T postgres psql -U neural neural_api
```

### Clear All Cache

```bash
# Redis
docker-compose exec redis redis-cli FLUSHALL

# Application cache
docker-compose restart api worker
```

### Scale Down During Incident

```bash
# Reduce load
kubectl scale deployment neural-api --replicas=1 -n neural-prod

# Stop workers
kubectl scale deployment neural-worker --replicas=0 -n neural-prod
```

## Getting Help

If you can't resolve the issue:

1. Collect logs:
```bash
docker-compose logs > logs.txt
kubectl logs -n neural-prod --all-containers --timestamps > k8s-logs.txt
```

2. Collect metrics:
```bash
curl http://localhost:9090/api/v1/query?query=up > metrics.txt
```

3. Check [GitHub Issues](https://github.com/neural-dsl/neural/issues)
4. Ask in [Discussions](https://github.com/neural-dsl/neural/discussions)
5. Contact support with:
   - Error messages
   - Logs
   - Steps to reproduce
   - Environment details (OS, versions, deployment type)
