#!/bin/bash
# Push all Docker images for Neural DSL

set -e

# Default values
REGISTRY="${REGISTRY:-docker.io}"
REPO="${REPO:-neural-dsl}"
TAG="${TAG:-latest}"

echo "Pushing Neural DSL Docker images..."
echo "Registry: ${REGISTRY}"
echo "Repository: ${REPO}"
echo "Tag: ${TAG}"
echo ""

# Check if logged in to registry
if [ "$REGISTRY" != "docker.io" ] && [ "$REGISTRY" != "localhost" ]; then
    echo "Make sure you are logged in to ${REGISTRY}"
    echo "Run: docker login ${REGISTRY}"
    echo ""
fi

# Push images
echo "Pushing API image..."
docker push ${REGISTRY}/${REPO}/api:${TAG}

echo "Pushing Worker image..."
docker push ${REGISTRY}/${REPO}/worker:${TAG}

echo "Pushing Dashboard image..."
docker push ${REGISTRY}/${REPO}/dashboard:${TAG}

echo "Pushing No-Code image..."
docker push ${REGISTRY}/${REPO}/nocode:${TAG}

echo "Pushing Aquarium IDE image..."
docker push ${REGISTRY}/${REPO}/aquarium:${TAG}

echo ""
echo "All images pushed successfully!"
echo ""
echo "Images available at:"
echo "  - ${REGISTRY}/${REPO}/api:${TAG}"
echo "  - ${REGISTRY}/${REPO}/worker:${TAG}"
echo "  - ${REGISTRY}/${REPO}/dashboard:${TAG}"
echo "  - ${REGISTRY}/${REPO}/nocode:${TAG}"
echo "  - ${REGISTRY}/${REPO}/aquarium:${TAG}"
