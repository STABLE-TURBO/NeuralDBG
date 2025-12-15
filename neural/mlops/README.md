# Neural DSL MLOps Module

Basic MLOps capabilities for production machine learning workflows.

## Features

### 🏗️ Model Registry
- Versioned model storage with metadata tracking
- Simple stage promotion (Development → Staging → Production)
- Model comparison and tracking
- Centralized model repository

### 🚀 Basic Deployment
- Simple deployment tracking
- Deployment status management
- Environment-based organization

### 📋 Audit Logging
- Audit trail for compliance
- Event logging for model operations
- Compliance report generation
- Security violation tracking
- Flexible querying and export

### 🔧 CI/CD Templates
- GitHub Actions workflows
- GitLab CI pipelines
- Jenkins pipelines
- Azure Pipelines
- Ready-to-use configurations

## Installation

```bash
# Core dependencies only
pip install -e .

# With all MLOps dependencies
pip install -e ".[full]"
```

## Quick Start

```python
from neural.mlops.registry import ModelRegistry, ModelStage
from neural.mlops.deployment import DeploymentManager
from neural.mlops.audit import AuditLogger, EventType

# Initialize
registry = ModelRegistry("./models")
deployment = DeploymentManager("./deployments")
audit = AuditLogger("./audit_logs")

# Register model
metadata = registry.register_model(
    name="my_model",
    version="v1.0.0",
    model_path="./model.pt",
    framework="pytorch",
    created_by="user@company.com",
    metrics={"accuracy": 0.95}
)

# Create deployment
deploy = deployment.create_deployment(
    model_name="my_model",
    model_version="v1.0.0",
    environment="production",
    endpoint="http://localhost:8080/predict"
)

# Activate deployment
deployment.activate_deployment(deploy.deployment_id)

# Audit logging
audit.log_model_deployment(
    model_name="my_model",
    version="v1.0.0",
    user="user@company.com",
    environment="production"
)
```

## Architecture

```
neural/mlops/
├── registry.py          # Model registry with versioning
├── deployment.py        # Basic deployment tracking
├── audit.py            # Audit logging for compliance
├── ci_templates.py     # CI/CD template generator
└── ci_templates/       # Ready-to-use CI/CD templates
    ├── github_actions.yml
    ├── gitlab_ci.yml
    ├── Jenkinsfile
    └── azure_pipelines.yml
```

## Use Cases

### Production Model Deployment
1. Register model with metadata
2. Create deployment record
3. Activate deployment
4. Audit all actions for compliance

### Compliance and Governance
1. Log all model lifecycle events
2. Track who deployed what and when
3. Generate compliance reports
4. Export audit trails for regulators

## Documentation

- [Complete Documentation](../../docs/mlops/README.md)
- [Deployment Guide](../../docs/mlops/DEPLOYMENT_GUIDE.md)

## Examples

See [examples directory](../../examples/mlops/) for:
- Deployment workflows
- Audit logging patterns
- CI/CD integration

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
- name: Deploy model
  run: |
    python -c "from neural.mlops.deployment import DeploymentManager; \
               manager = DeploymentManager(); \
               # deployment logic..."
```

### GitLab CI

```yaml
# .gitlab-ci.yml
deploy:production:
  script:
    - python deploy_script.py
```

### Jenkins

```groovy
// Jenkinsfile
stage('Deploy') {
    steps {
        sh 'python deploy_script.py'
    }
}
```

## Configuration

### Model Registry

```python
registry = ModelRegistry(
    registry_path="./models"  # Storage location
)
```

### Deployment Manager

```python
deployment = DeploymentManager(
    storage_path="./deployments"
)
```

### Audit Logger

```python
audit = AuditLogger(
    storage_path="./audit_logs"
)
```

## Storage

All data stored as JSON files with structured directories:

```
./models/               # Model registry
./deployments/         # Deployment tracking
./audit_logs/          # Audit events (date-partitioned)
```

## Security

- Audit logging for tracking
- Security violation tracking
- Access tracking through events

## Compliance

Supports compliance requirements through audit logging:
- SOC 2
- GDPR
- HIPAA
- PCI DSS

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE.md](../../LICENSE.md)

## Support

- Documentation: [docs/mlops/](../../docs/mlops/)
- Issues: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- Discussions: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
