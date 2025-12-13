# Google Cloud Platform Deployment

Deploy Neural DSL on Google Cloud Platform using GKE, Cloud Run, and other GCP services.

## Table of Contents

- [Architecture Options](#architecture-options)
- [GKE Deployment](#gke-deployment)
- [Cloud Run Deployment](#cloud-run-deployment)
- [Cloud SQL and Memorystore](#cloud-sql-and-memorystore)
- [Load Balancing](#load-balancing)
- [Storage](#storage)
- [Monitoring](#monitoring)
- [Cost Optimization](#cost-optimization)

## Architecture Options

### Option 1: Cloud Run (Serverless)
- **Best for**: Variable traffic, simple deployments
- **Pros**: Fully managed, auto-scaling, pay-per-use
- **Cons**: Limited to HTTP/HTTPS, cold starts

### Option 2: GKE Autopilot
- **Best for**: Kubernetes without cluster management
- **Pros**: Managed nodes, auto-scaling, security hardened
- **Cons**: Less customization than Standard GKE

### Option 3: GKE Standard
- **Best for**: Full control, complex workloads
- **Pros**: Complete Kubernetes features, GPU support
- **Cons**: More operational overhead

## GKE Deployment

### Create GKE Cluster

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud config set project neural-project

# Enable APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable sqladmin.googleapis.com

# Create GKE Autopilot cluster (recommended)
gcloud container clusters create-auto neural-prod \
  --region=us-central1 \
  --release-channel=regular \
  --enable-private-nodes \
  --enable-private-endpoint \
  --master-ipv4-cidr=172.16.0.0/28

# Or create GKE Standard cluster
gcloud container clusters create neural-prod \
  --region=us-central1 \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --disk-size=100GB \
  --disk-type=pd-ssd \
  --enable-autoscaling \
  --min-nodes=3 \
  --max-nodes=20 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-ip-alias \
  --network=default \
  --subnetwork=default \
  --enable-stackdriver-kubernetes \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# Get cluster credentials
gcloud container clusters get-credentials neural-prod --region=us-central1
```

### Deploy to GKE

```bash
# Create namespace
kubectl create namespace neural-prod

# Create service account
gcloud iam service-accounts create neural-sa \
  --display-name="Neural DSL Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding neural-project \
  --member="serviceAccount:neural-sa@neural-project.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding neural-project \
  --member="serviceAccount:neural-sa@neural-project.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Enable Workload Identity
gcloud iam service-accounts add-iam-policy-binding neural-sa@neural-project.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:neural-project.svc.id.goog[neural-prod/neural-api]"

# Create Kubernetes service account
kubectl create serviceaccount neural-api -n neural-prod
kubectl annotate serviceaccount neural-api -n neural-prod \
  iam.gke.io/gcp-service-account=neural-sa@neural-project.iam.gserviceaccount.com

# Deploy application (see kubernetes.md for manifests)
kubectl apply -f k8s/ -n neural-prod
```

### Push to Artifact Registry

```bash
# Create repository
gcloud artifacts repositories create neural \
  --repository-format=docker \
  --location=us-central1 \
  --description="Neural DSL Docker images"

# Configure Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and push
docker build -t us-central1-docker.pkg.dev/neural-project/neural/neural-dsl:latest .
docker push us-central1-docker.pkg.dev/neural-project/neural/neural-dsl:latest
```

## Cloud Run Deployment

### Deploy API to Cloud Run

```bash
# Deploy from Artifact Registry
gcloud run deploy neural-api \
  --image=us-central1-docker.pkg.dev/neural-project/neural/neural-dsl:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --service-account=neural-sa@neural-project.iam.gserviceaccount.com \
  --set-env-vars="DATABASE_URL=postgresql://neural:password@/neural_api?host=/cloudsql/neural-project:us-central1:neural-db" \
  --add-cloudsql-instances=neural-project:us-central1:neural-db \
  --set-secrets=SECRET_KEY=secret-key:latest,DB_PASSWORD=db-password:latest \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=100 \
  --concurrency=80 \
  --timeout=300 \
  --ingress=all

# Deploy workers to Cloud Run Jobs
gcloud run jobs create neural-worker-training \
  --image=us-central1-docker.pkg.dev/neural-project/neural/neural-dsl:latest \
  --region=us-central1 \
  --service-account=neural-sa@neural-project.iam.gserviceaccount.com \
  --set-env-vars="CELERY_QUEUE=training" \
  --add-cloudsql-instances=neural-project:us-central1:neural-db \
  --set-secrets=SECRET_KEY=secret-key:latest \
  --memory=8Gi \
  --cpu=4 \
  --max-retries=3 \
  --task-timeout=3600 \
  --parallelism=4

# Execute job
gcloud run jobs execute neural-worker-training --region=us-central1
```

### Cloud Run YAML Configuration

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: neural-api
  labels:
    cloud.googleapis.com/location: us-central1
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cloudsql-instances: neural-project:us-central1:neural-db
        run.googleapis.com/cpu-throttling: "false"
    spec:
      serviceAccountName: neural-sa@neural-project.iam.gserviceaccount.com
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: us-central1-docker.pkg.dev/neural-project/neural/neural-dsl:latest
        ports:
        - name: http1
          containerPort: 8000
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
        - name: DATABASE_URL
          value: "postgresql://neural:$(DB_PASSWORD)@/neural_api?host=/cloudsql/neural-project:us-central1:neural-db"
        - name: REDIS_HOST
          value: "10.0.0.3"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: secret-key
              key: latest
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-password
              key: latest
        resources:
          limits:
            memory: 2Gi
            cpu: "2"
```

```bash
gcloud run services replace service.yaml --region=us-central1
```

## Cloud SQL and Memorystore

### Cloud SQL PostgreSQL

```bash
# Create instance
gcloud sql instances create neural-db \
  --database-version=POSTGRES_15 \
  --tier=db-custom-4-16384 \
  --region=us-central1 \
  --network=default \
  --no-assign-ip \
  --enable-bin-log \
  --backup-start-time=03:00 \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=4 \
  --availability-type=REGIONAL \
  --storage-type=SSD \
  --storage-size=100GB \
  --storage-auto-increase

# Create database
gcloud sql databases create neural_api --instance=neural-db

# Create user
gcloud sql users create neural \
  --instance=neural-db \
  --password=$(openssl rand -hex 32)

# Create read replica
gcloud sql instances create neural-db-replica \
  --master-instance-name=neural-db \
  --tier=db-custom-2-8192 \
  --region=us-central1 \
  --availability-type=ZONAL

# Enable automatic backups
gcloud sql instances patch neural-db \
  --backup-start-time=03:00 \
  --retained-backups-count=7
```

### Memorystore Redis

```bash
# Create Redis instance
gcloud redis instances create neural-cache \
  --size=5 \
  --region=us-central1 \
  --network=default \
  --redis-version=redis_7_0 \
  --tier=standard \
  --replica-count=1 \
  --read-replicas-mode=READ_REPLICAS_ENABLED \
  --persistence-mode=RDB \
  --rdb-snapshot-period=12h \
  --rdb-snapshot-start-time=2023-01-01T03:00:00Z

# Get connection info
gcloud redis instances describe neural-cache --region=us-central1
```

## Load Balancing

### Global Load Balancer

```bash
# Create static IP
gcloud compute addresses create neural-lb-ip --global

# Create managed instance group (for GKE backend)
# This is automatic with GKE Ingress

# Create health check
gcloud compute health-checks create http neural-health-check \
  --port=8000 \
  --request-path=/health \
  --check-interval=30s \
  --timeout=10s \
  --unhealthy-threshold=3 \
  --healthy-threshold=2

# Create backend service
gcloud compute backend-services create neural-backend \
  --protocol=HTTP \
  --health-checks=neural-health-check \
  --global \
  --enable-cdn \
  --connection-draining-timeout=300

# Create URL map
gcloud compute url-maps create neural-lb \
  --default-service=neural-backend

# Create SSL certificate
gcloud compute ssl-certificates create neural-ssl \
  --domains=neural.example.com \
  --global

# Create target HTTPS proxy
gcloud compute target-https-proxies create neural-https-proxy \
  --url-map=neural-lb \
  --ssl-certificates=neural-ssl

# Create forwarding rule
gcloud compute forwarding-rules create neural-https-rule \
  --address=neural-lb-ip \
  --target-https-proxy=neural-https-proxy \
  --global \
  --ports=443
```

### GKE Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-ingress
  namespace: neural-prod
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "neural-lb-ip"
    networking.gke.io/managed-certificates: "neural-ssl-cert"
    kubernetes.io/ingress.allow-http: "false"
spec:
  rules:
  - host: neural.example.com
    http:
      paths:
      - path: /api/*
        pathType: ImplementationSpecific
        backend:
          service:
            name: neural-api
            port:
              number: 8000
      - path: /dashboard/*
        pathType: ImplementationSpecific
        backend:
          service:
            name: neural-dashboard
            port:
              number: 8050
---
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: neural-ssl-cert
  namespace: neural-prod
spec:
  domains:
    - neural.example.com
```

## Storage

### Cloud Storage

```bash
# Create bucket
gcloud storage buckets create gs://neural-models-prod \
  --location=us-central1 \
  --uniform-bucket-level-access

# Enable versioning
gcloud storage buckets update gs://neural-models-prod \
  --versioning

# Set lifecycle policy
cat > lifecycle.json <<'EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 180}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 365, "matchesPrefix": ["temp/"]}
      }
    ]
  }
}
EOF

gcloud storage buckets update gs://neural-models-prod \
  --lifecycle-file=lifecycle.json

# Grant access to service account
gcloud storage buckets add-iam-policy-binding gs://neural-models-prod \
  --member=serviceAccount:neural-sa@neural-project.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin
```

### Persistent Disks

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neural-data
  namespace: neural-prod
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: standard-rwo
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: neural-shared
  namespace: neural-prod
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: filestore
  resources:
    requests:
      storage: 1Ti
```

## Monitoring

### Cloud Monitoring Setup

```bash
# Enable APIs
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com

# Create notification channel
gcloud alpha monitoring channels create \
  --display-name="Neural Alerts Email" \
  --type=email \
  --channel-labels=email_address=alerts@example.com

# Create uptime check
gcloud monitoring uptime-checks create neural-api-uptime \
  --resource-type=uptime-url \
  --host=neural.example.com \
  --path=/health \
  --port=443

# Create alert policy
cat > alert-policy.json <<'EOF'
{
  "displayName": "Neural API High Error Rate",
  "conditions": [{
    "displayName": "Error rate > 5%",
    "conditionThreshold": {
      "filter": "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"neural-prod\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 5,
      "duration": "300s",
      "aggregations": [{
        "alignmentPeriod": "60s",
        "perSeriesAligner": "ALIGN_RATE"
      }]
    }
  }],
  "notificationChannels": ["projects/neural-project/notificationChannels/xxxxx"],
  "alertStrategy": {
    "autoClose": "1800s"
  }
}
EOF

gcloud alpha monitoring policies create --policy-from-file=alert-policy.json
```

### Cloud Logging

```bash
# Create log sink to BigQuery
gcloud logging sinks create neural-logs-sink \
  bigquery.googleapis.com/projects/neural-project/datasets/neural_logs \
  --log-filter='resource.type="k8s_container" AND resource.labels.namespace_name="neural-prod"'

# Create metric from logs
gcloud logging metrics create neural_api_errors \
  --description="Count of API errors" \
  --log-filter='resource.type="k8s_container" AND jsonPayload.level="ERROR"'

# Export logs to Cloud Storage
gcloud logging sinks create neural-logs-archive \
  storage.googleapis.com/neural-logs-archive \
  --log-filter='resource.type="k8s_container"'
```

### Cloud Trace

```python
# neural/monitoring/gcp_trace.py
from google.cloud import trace_v1
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    """Setup Cloud Trace."""
    tracer_provider = TracerProvider()
    cloud_trace_exporter = CloudTraceSpanExporter()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(cloud_trace_exporter)
    )
    trace.set_tracer_provider(tracer_provider)
    
    return trace.get_tracer(__name__)

# Usage
tracer = setup_tracing()

@app.post("/api/compile")
async def compile_model(model: Model):
    with tracer.start_as_current_span("compile_model"):
        result = compile_model_service(model)
        return result
```

## Cost Optimization

### Committed Use Discounts

```bash
# Purchase compute commitment
gcloud compute commitments create neural-commitment \
  --region=us-central1 \
  --resources=vcpu=32,memory=128 \
  --plan=12-month

# View savings
gcloud billing accounts list
gcloud billing accounts describe ACCOUNT_ID
```

### Preemptible/Spot Instances

```bash
# Create node pool with Spot VMs
gcloud container node-pools create neural-spot-pool \
  --cluster=neural-prod \
  --region=us-central1 \
  --machine-type=e2-standard-4 \
  --spot \
  --num-nodes=3 \
  --min-nodes=0 \
  --max-nodes=20 \
  --enable-autoscaling
```

### Storage Classes

```bash
# Move old data to cheaper storage
gcloud storage objects update gs://neural-models-prod/** \
  --predefined-acl=private \
  --storage-class=NEARLINE

# Use autoclass
gcloud storage buckets update gs://neural-models-prod \
  --autoclass
```

### Cloud Run Concurrency

```yaml
# Increase concurrency to reduce instance count
spec:
  containerConcurrency: 100  # Up from default 80
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/target: "70"  # Target CPU utilization
```

## Complete Terraform Example

See [terraform-gcp](../examples/terraform/gcp/) for complete Infrastructure as Code examples.

## Best Practices

1. **Use Workload Identity** - Don't use service account keys
2. **Enable Binary Authorization** - Ensure only trusted images run
3. **Use VPC Service Controls** - Protect data from exfiltration
4. **Enable Cloud Armor** - DDoS protection and WAF
5. **Use Cloud KMS** - For encryption key management
6. **Enable audit logs** - Track all API calls
7. **Use resource labels** - For cost tracking and organization
8. **Implement least privilege** - Minimal IAM permissions
9. **Use private endpoints** - Reduce attack surface
10. **Regular security scans** - Use Container Analysis
