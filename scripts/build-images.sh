#!/bin/bash
# Build all Docker images for Neural DSL

set -e

# Default values
REGISTRY="${REGISTRY:-docker.io}"
REPO="${REPO:-neural-dsl}"
TAG="${TAG:-latest}"

echo "Building Neural DSL Docker images..."
echo "Registry: ${REGISTRY}"
echo "Repository: ${REPO}"
echo "Tag: ${TAG}"

# Build API image
echo "Building API image..."
docker build -f dockerfiles/Dockerfile.api -t ${REGISTRY}/${REPO}/api:${TAG} .

# Build Worker image
echo "Building Worker image..."
docker build -f dockerfiles/Dockerfile.worker -t ${REGISTRY}/${REPO}/worker:${TAG} .

# Build Dashboard image
echo "Building Dashboard image..."
docker build -f dockerfiles/Dockerfile.dashboard -t ${REGISTRY}/${REPO}/dashboard:${TAG} .

# Build No-Code image
echo "Building No-Code image..."
docker build -f dockerfiles/Dockerfile.nocode -t ${REGISTRY}/${REPO}/nocode:${TAG} .

# Build Aquarium IDE image
echo "Building Aquarium IDE image..."
docker build -f dockerfiles/Dockerfile.aquarium -t ${REGISTRY}/${REPO}/aquarium:${TAG} .

echo "All images built successfully!"
echo ""
echo "Images:"
echo "  - ${REGISTRY}/${REPO}/api:${TAG}"
echo "  - ${REGISTRY}/${REPO}/worker:${TAG}"
echo "  - ${REGISTRY}/${REPO}/dashboard:${TAG}"
echo "  - ${REGISTRY}/${REPO}/nocode:${TAG}"
echo "  - ${REGISTRY}/${REPO}/aquarium:${TAG}"
echo ""
echo "To push images, run:"
echo "  docker push ${REGISTRY}/${REPO}/api:${TAG}"
echo "  docker push ${REGISTRY}/${REPO}/worker:${TAG}"
echo "  docker push ${REGISTRY}/${REPO}/dashboard:${TAG}"
echo "  docker push ${REGISTRY}/${REPO}/nocode:${TAG}"
echo "  docker push ${REGISTRY}/${REPO}/aquarium:${TAG}"
