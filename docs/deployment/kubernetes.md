# Kubernetes Deployment

Deploy Neural DSL on Kubernetes for production-scale container orchestration.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Deployment](#deployment)
- [Storage](#storage)
- [Networking](#networking)
- [Scaling](#scaling)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Ingress Controller (Nginx)              │  │
│  │         (SSL Termination, Load Balancing)            │  │
│  └─────────┬────────────────────────────────────────────┘  │
│            │                                                │
│   ┌────────┴─────────┬──────────────┬──────────────┐      │
│   │                  │              │              │       │
│ ┌─▼──────────┐  ┌───▼────────┐ ┌──▼────────┐ ┌───▼───┐   │
│ │ API Pod 1  │  │ API Pod 2  │ │Dashboard │ │NoCode │   │
│ │ (2 cores)  │  │ (2 cores)  │ │ Pod      │ │ Pod   │   │
│ └────────────┘  └────────────┘ └──────────┘ └───────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │            StatefulSet: PostgreSQL                   │   │
│ │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│ │  │Primary   │  │Replica 1 │  │Replica 2 │          │   │
│ │  │(RW)      │  │(RO)      │  │(RO)      │          │   │
│ │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │   │
│ │       │             │             │                  │   │
│ │       └─────────────┴─────────────┘                  │   │
│ │         PersistentVolumeClaims (EBS/GCE PD)         │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │            StatefulSet: Redis Cluster                │   │
│ │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│ │  │Redis 1   │  │Redis 2   │  │Redis 3   │          │   │
│ │  └──────────┘  └──────────┘  └──────────┘          │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │            Deployment: Celery Workers                │   │
│ │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│ │  │Training  │  │Compile   │  │Export    │   ...    │   │
│ │  │Workers   │  │Workers   │  │Workers   │          │   │
│ │  └──────────┘  └──────────┘  └──────────┘          │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────── ┘
```

## Prerequisites

### Kubernetes Cluster

**Minimum cluster size:**
- 3 worker nodes
- 4 cores, 16 GB RAM per node
- Kubernetes 1.24+

**Managed Kubernetes options:**
- AWS EKS
- Google GKE
- Azure AKS
- DigitalOcean Kubernetes
- Linode Kubernetes Engine

### Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
kubectl version --client
helm version
```

### Storage Class

Ensure your cluster has a default StorageClass:

```bash
kubectl get storageclass

# AWS EBS
# GCP GCE Persistent Disk
# Azure Disk
# Or NFS/Ceph for on-premises
```

## Deployment

### 1. Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neural-prod
  labels:
    name: neural-prod
    environment: production
```

```bash
kubectl apply -f namespace.yaml
```

### 2. Create Secrets

```bash
# Generate secrets
export SECRET_KEY=$(openssl rand -hex 32)
export DB_PASSWORD=$(openssl rand -hex 32)
export REDIS_PASSWORD=$(openssl rand -hex 16)
export JWT_SECRET=$(openssl rand -hex 32)

# Create secret
kubectl create secret generic neural-secrets \
  --namespace=neural-prod \
  --from-literal=secret-key=$SECRET_KEY \
  --from-literal=db-password=$DB_PASSWORD \
  --from-literal=redis-password=$REDIS_PASSWORD \
  --from-literal=jwt-secret=$JWT_SECRET

# AWS credentials (if using S3)
kubectl create secret generic aws-credentials \
  --namespace=neural-prod \
  --from-literal=access-key-id=$AWS_ACCESS_KEY_ID \
  --from-literal=secret-access-key=$AWS_SECRET_ACCESS_KEY
```

### 3. Deploy PostgreSQL

```yaml
# postgres-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: neural-prod
spec:
  ports:
  - port: 5432
    name: postgres
  clusterIP: None
  selector:
    app: postgres
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: neural-prod
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          value: neural_api
        - name: POSTGRES_USER
          value: neural
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: db-password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - pg_isready -U neural
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - pg_isready -U neural
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "standard"
      resources:
        requests:
          storage: 50Gi
```

```bash
kubectl apply -f postgres-statefulset.yaml
```

### 4. Deploy Redis Cluster

```yaml
# redis-statefulset.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: neural-prod
data:
  redis.conf: |
    cluster-enabled yes
    cluster-require-full-coverage no
    cluster-node-timeout 15000
    cluster-config-file /data/nodes.conf
    cluster-migration-barrier 1
    appendonly yes
    protected-mode no
---
apiVersion: v1
kind: Service
metadata:
  name: redis-cluster
  namespace: neural-prod
spec:
  type: ClusterIP
  ports:
  - port: 6379
    targetPort: 6379
    name: client
  - port: 16379
    targetPort: 16379
    name: gossip
  selector:
    app: redis-cluster
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: neural-prod
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        command:
        - redis-server
        args:
        - /conf/redis.conf
        - --requirepass
        - $(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: redis-password
        volumeMounts:
        - name: conf
          mountPath: /conf
        - name: data
          mountPath: /data
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "1"
            memory: "2Gi"
      volumes:
      - name: conf
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

```bash
kubectl apply -f redis-statefulset.yaml

# Initialize cluster (after pods are running)
kubectl exec -it -n neural-prod redis-cluster-0 -- redis-cli --cluster create \
  $(kubectl get pods -n neural-prod -l app=redis-cluster -o jsonpath='{range.items[*]}{.status.podIP}:6379 ') \
  --cluster-replicas 1 \
  -a $REDIS_PASSWORD
```

### 5. Deploy API

```yaml
# api-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neural-config
  namespace: neural-prod
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "4"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql://neural:$(DB_PASSWORD)@postgres:5432/neural_api"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-api
  namespace: neural-prod
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: neural-api
  template:
    metadata:
      labels:
        app: neural-api
        version: v1
    spec:
      containers:
      - name: api
        image: neural-dsl:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: secret-key
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: db-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: redis-password
        - name: DATABASE_URL
          value: "postgresql://neural:$(DB_PASSWORD)@postgres:5432/neural_api"
        - name: REDIS_HOST
          value: "redis-cluster"
        - name: REDIS_PORT
          value: "6379"
        - name: CELERY_BROKER_URL
          value: "redis://:$(REDIS_PASSWORD)@redis-cluster:6379/0"
        - name: CELERY_RESULT_BACKEND
          value: "redis://:$(REDIS_PASSWORD)@redis-cluster:6379/0"
        envFrom:
        - configMapRef:
            name: neural-config
        resources:
          requests:
            cpu: "1"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: neural-api
  namespace: neural-prod
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: neural-api
```

```bash
kubectl apply -f api-deployment.yaml
```

### 6. Deploy Workers

```yaml
# workers-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-worker-training
  namespace: neural-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: neural-worker
      queue: training
  template:
    metadata:
      labels:
        app: neural-worker
        queue: training
    spec:
      containers:
      - name: worker
        image: neural-dsl:latest
        command: ["celery"]
        args:
        - "-A"
        - "neural.api.celery_app"
        - "worker"
        - "-Q"
        - "training"
        - "-c"
        - "2"
        - "--max-tasks-per-child=10"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: redis-password
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: db-password
        - name: DATABASE_URL
          value: "postgresql://neural:$(DB_PASSWORD)@postgres:5432/neural_api"
        - name: CELERY_BROKER_URL
          value: "redis://:$(REDIS_PASSWORD)@redis-cluster:6379/0"
        - name: CELERY_RESULT_BACKEND
          value: "redis://:$(REDIS_PASSWORD)@redis-cluster:6379/0"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-worker-compilation
  namespace: neural-prod
spec:
  replicas: 4
  selector:
    matchLabels:
      app: neural-worker
      queue: compilation
  template:
    metadata:
      labels:
        app: neural-worker
        queue: compilation
    spec:
      containers:
      - name: worker
        image: neural-dsl:latest
        command: ["celery"]
        args:
        - "-A"
        - "neural.api.celery_app"
        - "worker"
        - "-Q"
        - "compilation"
        - "-c"
        - "4"
        - "--max-tasks-per-child=50"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: redis-password
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-secrets
              key: db-password
        - name: DATABASE_URL
          value: "postgresql://neural:$(DB_PASSWORD)@postgres:5432/neural_api"
        - name: CELERY_BROKER_URL
          value: "redis://:$(REDIS_PASSWORD)@redis-cluster:6379/0"
        - name: CELERY_RESULT_BACKEND
          value: "redis://:$(REDIS_PASSWORD)@redis-cluster:6379/0"
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
```

```bash
kubectl apply -f workers-deployment.yaml
```

### 7. Configure Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-ingress
  namespace: neural-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - neural.example.com
    secretName: neural-tls
  rules:
  - host: neural.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: neural-api
            port:
              number: 8000
      - path: /dashboard
        pathType: Prefix
        backend:
          service:
            name: neural-dashboard
            port:
              number: 8050
      - path: /nocode
        pathType: Prefix
        backend:
          service:
            name: neural-nocode
            port:
              number: 8051
```

```bash
kubectl apply -f ingress.yaml
```

## Storage

### Persistent Volumes

**Using AWS EBS:**

```yaml
apiVersion: v1
kind: StorageClass
metadata:
  name: neural-ebs
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
```

**Using GCP Persistent Disk:**

```yaml
apiVersion: v1
kind: StorageClass
metadata:
  name: neural-pd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
allowVolumeExpansion: true
```

### Object Storage for Models

**ConfigMap for S3:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: storage-config
  namespace: neural-prod
data:
  S3_ENABLED: "true"
  S3_BUCKET: "neural-models-prod"
  S3_REGION: "us-east-1"
```

## Networking

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: neural-api-policy
  namespace: neural-prod
spec:
  podSelector:
    matchLabels:
      app: neural-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis-cluster
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## Scaling

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-api-hpa
  namespace: neural-prod
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

```bash
kubectl apply -f hpa.yaml
```

### Cluster Autoscaler

**AWS EKS:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  cluster-autoscaler.yaml: |
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: cluster-autoscaler
      namespace: kube-system
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: cluster-autoscaler
      template:
        metadata:
          labels:
            app: cluster-autoscaler
        spec:
          serviceAccountName: cluster-autoscaler
          containers:
          - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.26.0
            name: cluster-autoscaler
            command:
            - ./cluster-autoscaler
            - --v=4
            - --stderrthreshold=info
            - --cloud-provider=aws
            - --skip-nodes-with-local-storage=false
            - --expander=least-waste
            - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/neural-cluster
```

## Monitoring

### Deploy Prometheus Stack

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus + Grafana
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword=<strong-password>
```

### ServiceMonitor for API

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
    interval: 30s
```

## Troubleshooting

### Common Issues

**Pods not starting:**

```bash
kubectl describe pod -n neural-prod <pod-name>
kubectl logs -n neural-prod <pod-name>
```

**Resource constraints:**

```bash
kubectl top nodes
kubectl top pods -n neural-prod
```

**Network issues:**

```bash
# Test connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
# Inside pod:
wget -qO- http://neural-api:8000/health
```

**Storage issues:**

```bash
kubectl get pv
kubectl get pvc -n neural-prod
kubectl describe pvc -n neural-prod <pvc-name>
```

### Debug Commands

```bash
# Get all resources
kubectl get all -n neural-prod

# Describe deployment
kubectl describe deployment neural-api -n neural-prod

# View logs
kubectl logs -f deployment/neural-api -n neural-prod

# Execute commands in pod
kubectl exec -it -n neural-prod <pod-name> -- /bin/sh

# Port forward for debugging
kubectl port-forward -n neural-prod svc/neural-api 8000:8000
```

See [Troubleshooting Guide](troubleshooting.md) for more details.
