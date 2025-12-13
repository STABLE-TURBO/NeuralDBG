#!/bin/bash
# Deploy Neural DSL to Kubernetes

set -e

NAMESPACE="${NAMESPACE:-neural-dsl}"
CONTEXT="${CONTEXT:-}"

echo "Deploying Neural DSL to Kubernetes..."
echo "Namespace: ${NAMESPACE}"

# Set context if provided
if [ -n "$CONTEXT" ]; then
    echo "Using context: ${CONTEXT}"
    kubectl config use-context ${CONTEXT}
fi

# Create namespace
echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Apply secrets and configmaps
echo "Applying secrets and configmaps..."
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml

# Deploy infrastructure
echo "Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml

echo "Deploying PostgreSQL..."
kubectl apply -f k8s/postgres-deployment.yaml

# Wait for infrastructure to be ready
echo "Waiting for infrastructure services..."
kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s || true

# Deploy application services
echo "Deploying API..."
kubectl apply -f k8s/api-deployment.yaml

echo "Deploying Workers..."
kubectl apply -f k8s/worker-deployment.yaml

echo "Deploying Dashboard..."
kubectl apply -f k8s/dashboard-deployment.yaml

echo "Deploying No-Code interface..."
kubectl apply -f k8s/nocode-deployment.yaml

echo "Deploying Aquarium IDE..."
kubectl apply -f k8s/aquarium-deployment.yaml

# Apply ingress
echo "Applying Ingress..."
kubectl apply -f k8s/ingress.yaml

# Apply HPA
echo "Applying HorizontalPodAutoscaler..."
kubectl apply -f k8s/hpa.yaml

# Wait for deployments
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/neural-api -n ${NAMESPACE} --timeout=300s || true

# Show status
echo ""
echo "Deployment complete!"
echo ""
echo "Checking status..."
kubectl get all -n ${NAMESPACE}

echo ""
echo "Service URLs (configure DNS/Ingress):"
echo "  API: http://api.neural-dsl.example.com"
echo "  Dashboard: http://dashboard.neural-dsl.example.com"
echo "  No-Code: http://nocode.neural-dsl.example.com"
echo "  Aquarium IDE: http://aquarium.neural-dsl.example.com"
echo ""
echo "To check logs:"
echo "  kubectl logs -f deployment/neural-api -n ${NAMESPACE}"
echo ""
echo "To get services:"
echo "  kubectl get svc -n ${NAMESPACE}"
