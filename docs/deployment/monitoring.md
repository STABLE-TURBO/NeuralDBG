# Monitoring & Observability

Comprehensive monitoring and observability setup for Neural DSL production deployments using Prometheus, Grafana, and logging solutions.

## Table of Contents

- [Monitoring Stack](#monitoring-stack)
- [Prometheus Setup](#prometheus-setup)
- [Grafana Dashboards](#grafana-dashboards)
- [Logging](#logging)
- [Tracing](#tracing)
- [Alerting](#alerting)
- [Metrics](#metrics)
- [Best Practices](#best-practices)

## Monitoring Stack

### Components

```
┌─────────────────────────────────────────────────────┐
│              Application Layer                       │
│  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │  API   │  │Workers │  │Services│               │
│  └───┬────┘  └───┬────┘  └───┬────┘               │
│      │           │           │                      │
│      └───────────┴───────────┘                      │
│                  │                                   │
│     ┌────────────▼──────────────┐                   │
│     │   Metrics Exporters        │                   │
│     │   (Prometheus Client)      │                   │
│     └────────────┬──────────────┘                   │
└──────────────────┼─────────────────────────────────┘
                   │
      ┌────────────▼──────────────┐
      │       Prometheus           │
      │   (Metrics Collection)     │
      └────────────┬──────────────┘
                   │
      ┌────────────▼──────────────┐
      │        Grafana             │
      │    (Visualization)         │
      └───────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              Logging Layer                           │
│  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │  API   │──│Workers │──│Services│               │
│  └────────┘  └────────┘  └────────┘               │
│      │           │           │                      │
│      └───────────┴───────────┘                      │
│                  │                                   │
│     ┌────────────▼──────────────┐                   │
│     │    Log Aggregation         │                   │
│     │  (Loki / ELK / Cloud)      │                   │
│     └────────────┬──────────────┘                   │
│                  │                                   │
│     ┌────────────▼──────────────┐                   │
│     │      Log Analysis          │                   │
│     │  (Grafana / Kibana)        │                   │
│     └───────────────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

## Prometheus Setup

### Installation with Helm

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create values file
cat > prometheus-values.yaml <<'EOF'
prometheus:
  prometheusSpec:
    retention: 30d
    retentionSize: "50GB"
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    serviceMonitorSelectorNilUsesHelmValues: false
    podMonitorSelectorNilUsesHelmValues: false
    
    additionalScrapeConfigs:
      - job_name: 'neural-api'
        static_configs:
          - targets: ['neural-api:8000']
        metrics_path: '/metrics'
        scrape_interval: 15s

alertmanager:
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi

grafana:
  adminPassword: <strong-password>
  persistence:
    enabled: true
    size: 10Gi
  
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
      - name: Prometheus
        type: prometheus
        url: http://prometheus-server
        access: proxy
        isDefault: true
      - name: Loki
        type: loki
        url: http://loki:3100
        access: proxy

  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/default
EOF

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values prometheus-values.yaml
```

### ServiceMonitor for Neural DSL

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neural-api
  namespace: neural-prod
  labels:
    app: neural-api
spec:
  selector:
    matchLabels:
      app: neural-api
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neural-workers
  namespace: neural-prod
  labels:
    app: neural-worker
spec:
  selector:
    matchLabels:
      app: neural-worker
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
```

### Prometheus Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'neural-prod'
    environment: 'production'

scrape_configs:
  - job_name: 'neural-api'
    static_configs:
      - targets: ['neural-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    
  - job_name: 'neural-workers'
    static_configs:
      - targets:
        - 'worker-1:9090'
        - 'worker-2:9090'
        - 'worker-3:9090'
    metrics_path: '/metrics'
    scrape_interval: 30s
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
  
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

rule_files:
  - '/etc/prometheus/rules/*.yml'

alerting:
  alertmanagers:
  - static_configs:
    - targets: ['alertmanager:9093']
```

## Grafana Dashboards

### API Performance Dashboard

```json
{
  "dashboard": {
    "title": "Neural DSL API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(neural_api_requests_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(neural_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(neural_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(neural_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(neural_api_requests_total{status=~\"5..\"}[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Requests",
        "targets": [
          {
            "expr": "neural_api_active_requests"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

Save as `neural-api-dashboard.json` and import to Grafana.

### Worker Performance Dashboard

```json
{
  "dashboard": {
    "title": "Neural DSL Workers",
    "panels": [
      {
        "title": "Queue Length",
        "targets": [
          {
            "expr": "celery_queue_length{queue=~\"training|compilation|export\"}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Task Processing Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(celery_task_duration_seconds_bucket[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Worker CPU Usage",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total{job=\"neural-workers\"}[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Worker Memory Usage",
        "targets": [
          {
            "expr": "process_resident_memory_bytes{job=\"neural-workers\"}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### System Overview Dashboard

```json
{
  "dashboard": {
    "title": "Neural DSL System Overview",
    "panels": [
      {
        "title": "API Instances",
        "targets": [
          {
            "expr": "count(up{job=\"neural-api\"})"
          }
        ],
        "type": "singlestat"
      },
      {
        "title": "Database Connections",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends{datname=\"neural_api\"}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Redis Memory Usage",
        "targets": [
          {
            "expr": "redis_memory_used_bytes"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Disk Usage",
        "targets": [
          {
            "expr": "(1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

## Logging

### Loki Setup

```bash
# Add Loki Helm repository
helm repo add grafana https://grafana.github.io/helm-charts

# Install Loki
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=100Gi \
  --set promtail.enabled=true \
  --set grafana.enabled=false

# Configure Promtail
cat > promtail-config.yaml <<'EOF'
promtail:
  config:
    clients:
      - url: http://loki:3100/loki/api/v1/push
    
    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - source_labels: [__meta_kubernetes_container_name]
            target_label: container
        pipeline_stages:
          - json:
              expressions:
                level: level
                message: message
                timestamp: timestamp
          - labels:
              level:
          - timestamp:
              source: timestamp
              format: RFC3339
EOF

helm upgrade loki grafana/loki-stack \
  --namespace monitoring \
  --values promtail-config.yaml
```

### Application Logging

```python
# neural/logging/logger.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['service'] = 'neural-api'
        log_record['environment'] = os.getenv('ENVIRONMENT', 'production')

def setup_logging():
    """Configure structured logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(service)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Usage
logger = setup_logging()
logger.info("API started", extra={"port": 8000, "workers": 4})
logger.error("Database connection failed", extra={"error": str(e), "retry_count": 3})
```

### Log Queries

```promql
# Search for errors in last hour
{namespace="neural-prod", level="ERROR"} |= "" |~ ".*"

# Count errors by service
sum by (container) (rate({namespace="neural-prod", level="ERROR"}[5m]))

# Search for specific error
{namespace="neural-prod"} |= "connection timeout" | json

# Slow requests
{namespace="neural-prod", container="api"} | json | duration > 1s
```

## Tracing

### Jaeger Setup

```bash
# Install Jaeger
kubectl create namespace observability

kubectl create -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/crds/jaegertracing.io_jaegers_crd.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/service_account.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/role.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/role_binding.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/operator.yaml

# Create Jaeger instance
cat > jaeger.yaml <<'EOF'
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: neural-jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
  ingress:
    enabled: true
EOF

kubectl apply -f jaeger.yaml
```

### Application Tracing

```python
# neural/tracing/tracer.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

def setup_tracing(app):
    """Setup distributed tracing."""
    resource = Resource(attributes={
        SERVICE_NAME: "neural-api"
    })
    
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831)),
    )
    
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    
    return trace.get_tracer(__name__)

# Usage
app = FastAPI()
tracer = setup_tracing(app)

@app.post("/api/compile")
async def compile_model(model: Model):
    with tracer.start_as_current_span("compile_model") as span:
        span.set_attribute("model.type", model.type)
        span.set_attribute("model.layers", len(model.layers))
        
        result = compile_model_service(model)
        span.set_attribute("compilation.success", True)
        return result
```

## Alerting

### AlertManager Configuration

```yaml
# alertmanager-config.yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-neural'
  routes:
  - match:
      severity: critical
    receiver: 'team-neural-critical'
  - match:
      severity: warning
    receiver: 'team-neural-warning'

receivers:
- name: 'team-neural'
  slack_configs:
  - channel: '#neural-alerts'
    title: 'Neural DSL Alert'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

- name: 'team-neural-critical'
  slack_configs:
  - channel: '#neural-critical'
    title: '🚨 CRITICAL: Neural DSL'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
  pagerduty_configs:
  - service_key: '<pagerduty-key>'

- name: 'team-neural-warning'
  slack_configs:
  - channel: '#neural-warnings'
    title: '⚠️ WARNING: Neural DSL'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

### Alert Rules

```yaml
# alert-rules.yaml
groups:
- name: neural_api_alerts
  interval: 30s
  rules:
  - alert: HighErrorRate
    expr: rate(neural_api_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(neural_api_request_duration_seconds_bucket[5m])) > 2
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      description: "P95 latency is {{ $value }}s (threshold: 2s)"

  - alert: APIDown
    expr: up{job="neural-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "API is down"
      description: "Neural API has been down for more than 1 minute"

- name: neural_worker_alerts
  interval: 30s
  rules:
  - alert: HighQueueLength
    expr: celery_queue_length > 100
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Celery queue length is high"
      description: "Queue {{ $labels.queue }} has {{ $value }} tasks waiting"

  - alert: WorkerDown
    expr: up{job="neural-workers"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Worker is down"
      description: "Worker {{ $labels.instance }} has been down for more than 2 minutes"

- name: neural_system_alerts
  interval: 30s
  rules:
  - alert: HighDiskUsage
    expr: (1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Disk usage is high"
      description: "Disk usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: HighMemoryUsage
    expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Memory usage is critical"
      description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: DatabaseConnectionPoolExhausted
    expr: pg_stat_database_numbackends{datname="neural_api"} > 90
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Database connection pool nearly exhausted"
      description: "{{ $value }} connections active (limit: 100)"
```

## Metrics

### Application Metrics

```python
# neural/metrics/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from fastapi import Request
import time

# Define metrics
REQUEST_COUNT = Counter(
    'neural_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'neural_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    'neural_api_active_requests',
    'Number of active requests'
)

COMPILATION_TIME = Histogram(
    'neural_compilation_duration_seconds',
    'Model compilation time',
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
)

TRAINING_TIME = Histogram(
    'neural_training_duration_seconds',
    'Model training time',
    ['model_type'],
    buckets=[60.0, 300.0, 600.0, 1800.0, 3600.0]
)

MODEL_SIZE = Gauge(
    'neural_model_size_bytes',
    'Model size in bytes',
    ['model_id', 'format']
)

CELERY_QUEUE_LENGTH = Gauge(
    'celery_queue_length',
    'Number of tasks in queue',
    ['queue']
)

# Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    ACTIVE_REQUESTS.dec()
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose metrics for Prometheus."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

## Best Practices

### Metric Naming

- Use consistent naming: `service_component_action_unit`
- Examples:
  - `neural_api_requests_total` (counter)
  - `neural_compilation_duration_seconds` (histogram)
  - `neural_model_size_bytes` (gauge)

### Labels

- Keep cardinality low (<100 unique combinations)
- Avoid high-cardinality labels (user IDs, timestamps)
- Use consistent label names across metrics

### Retention

- Short-term (15s resolution): 30 days
- Medium-term (5m resolution): 90 days
- Long-term (1h resolution): 1 year

### Alerting

- Alert on symptoms, not causes
- Set appropriate thresholds based on SLOs
- Include runbooks in alert annotations
- Group related alerts to reduce noise

### Dashboard Organization

1. **Executive Dashboard** - High-level KPIs
2. **Service Dashboard** - Per-service metrics
3. **Infrastructure Dashboard** - System resources
4. **Troubleshooting Dashboard** - Detailed diagnostics

### Documentation

Document all custom metrics:
- Name and description
- Type (counter/gauge/histogram)
- Labels and their meanings
- Expected values and thresholds
- Alert rules using this metric
