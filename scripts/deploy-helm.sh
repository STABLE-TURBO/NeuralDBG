#!/bin/bash
# Deploy Neural DSL using Helm

set -e

RELEASE_NAME="${RELEASE_NAME:-neural-dsl}"
NAMESPACE="${NAMESPACE:-neural-dsl}"
VALUES_FILE="${VALUES_FILE:-helm/neural-dsl/values.yaml}"
CHART_PATH="${CHART_PATH:-helm/neural-dsl}"

echo "Deploying Neural DSL with Helm..."
echo "Release name: ${RELEASE_NAME}"
echo "Namespace: ${NAMESPACE}"
echo "Values file: ${VALUES_FILE}"
echo "Chart path: ${CHART_PATH}"
echo ""

# Check if Helm is installed
if ! command -v helm &> /dev/null; then
    echo "Error: Helm is not installed"
    echo "Install Helm from https://helm.sh/docs/intro/install/"
    exit 1
fi

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Check if release exists
if helm list -n ${NAMESPACE} | grep -q ${RELEASE_NAME}; then
    echo "Release ${RELEASE_NAME} already exists. Upgrading..."
    helm upgrade ${RELEASE_NAME} ${CHART_PATH} \
        --namespace ${NAMESPACE} \
        --values ${VALUES_FILE} \
        --wait \
        --timeout 10m
else
    echo "Installing new release..."
    helm install ${RELEASE_NAME} ${CHART_PATH} \
        --namespace ${NAMESPACE} \
        --values ${VALUES_FILE} \
        --create-namespace \
        --wait \
        --timeout 10m
fi

# Show status
echo ""
echo "Deployment complete!"
echo ""
helm status ${RELEASE_NAME} -n ${NAMESPACE}

echo ""
echo "To check pods:"
echo "  kubectl get pods -n ${NAMESPACE}"
echo ""
echo "To check services:"
echo "  kubectl get svc -n ${NAMESPACE}"
echo ""
echo "To get release values:"
echo "  helm get values ${RELEASE_NAME} -n ${NAMESPACE}"
echo ""
echo "To uninstall:"
echo "  helm uninstall ${RELEASE_NAME} -n ${NAMESPACE}"
