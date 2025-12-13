# Production Deployment Documentation

Comprehensive guides for deploying Neural DSL to production environments.

## Quick Navigation

### Configuration
- [Environment Variables Reference](environment-variables.md) - Complete reference for all environment variables across services

### Deployment Architectures
- [Single-Server Deployment](single-server.md) - Deploy all services on one machine
- [Microservices Architecture](microservices.md) - Distributed deployment with service separation
- [Kubernetes Deployment](kubernetes.md) - Container orchestration with K8s

### Security
- [Security Best Practices](security.md) - Production security guidelines
- [Secrets Management](secrets-management.md) - HashiCorp Vault, AWS Secrets Manager, and more

### Cloud Providers
- [AWS Deployment](aws.md) - Deploy on AWS ECS, EKS, and other services
- [Google Cloud Deployment](gcp.md) - Deploy on GCP GKE and Cloud Run
- [Azure Deployment](azure.md) - Deploy on Azure AKS and Container Instances

### Operations
- [Monitoring & Observability](monitoring.md) - Prometheus, Grafana, and logging setup
- [Troubleshooting Guide](troubleshooting.md) - Common deployment issues and solutions

## Quick Start

For a rapid overview, see:
1. [Deployment Quick Start](../DEPLOYMENT_QUICK_START.md) - Model deployment basics
2. [Single-Server Setup](single-server.md) - Simplest production setup
3. [Environment Variables](environment-variables.md) - Required configuration

## Architecture Overview

Neural DSL consists of multiple services:

- **API Server** (Port 8000) - REST API for compilation, execution, and model management
- **Dashboard** (Port 8050) - NeuralDbg real-time debugger and visualization
- **No-Code GUI** (Port 8051) - Visual model builder interface
- **Celery Workers** - Asynchronous task processing (training, compilation)
- **Redis** (Port 6379) - Message broker and cache
- **PostgreSQL** (Port 5432) - Persistent data storage (production)
- **Flower** (Port 5555) - Celery task monitoring
- **Aquarium IDE** (Optional) - Advanced IDE features

## Deployment Decision Tree

### Choose Your Deployment Type

**Single Server** - Best for:
- Development/staging environments
- Low to medium traffic (<100 req/s)
- Budget constraints
- Simple operations requirements

**Microservices** - Best for:
- Production environments
- High availability requirements
- Independent service scaling
- Team separation

**Kubernetes** - Best for:
- Enterprise deployments
- Multi-region/multi-cloud
- Auto-scaling requirements
- Already using K8s infrastructure

### Choose Your Cloud Provider

**AWS** - Best for:
- Mature ecosystem and managed services
- Integration with AWS ML services (SageMaker)
- Enterprise compliance requirements

**GCP** - Best for:
- TensorFlow/PyTorch optimization
- Integration with Google Cloud AI Platform
- Kubernetes-native workloads (GKE)

**Azure** - Best for:
- Microsoft ecosystem integration
- Enterprise Azure customers
- Hybrid cloud scenarios

## Prerequisites

All deployment types require:

1. **Docker** (20.10+) and **Docker Compose** (2.0+)
2. **SSL/TLS certificates** for production
3. **Domain name** with DNS configured
4. **Secrets management** solution (Vault, cloud provider secrets)
5. **Monitoring infrastructure** (Prometheus/Grafana recommended)

## Security Checklist

Before deploying to production:

- [ ] Change all default passwords and secrets
- [ ] Enable HTTPS/TLS with valid certificates
- [ ] Configure firewall rules and security groups
- [ ] Set up secrets management (Vault/AWS Secrets Manager)
- [ ] Enable authentication and authorization
- [ ] Configure rate limiting
- [ ] Set up log aggregation and monitoring
- [ ] Enable database backups
- [ ] Configure CORS policies
- [ ] Review and harden container images
- [ ] Set up intrusion detection
- [ ] Enable audit logging

## Support and Resources

- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [GitHub Issues](https://github.com/neural-dsl/neural/issues) - Report bugs
- [Community Discussions](https://github.com/neural-dsl/neural/discussions) - Ask questions
- [Security Policy](../../SECURITY.md) - Report security issues

## Document Conventions

Throughout these guides:
- `<PLACEHOLDER>` - Replace with your actual values
- `${ENV_VAR}` - Environment variable reference
- **Bold** - Important warnings or required actions
- `code` - Commands, file paths, and configuration values
