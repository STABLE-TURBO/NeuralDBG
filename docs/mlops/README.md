# Neural DSL MLOps Documentation

Basic MLOps capabilities for production ML workflows.

## Overview

The Neural DSL MLOps module provides essential tools for managing the machine learning lifecycle in production environments, including:

- **Model Registry**: Versioned model storage with stage promotion
- **Deployment Manager**: Basic deployment tracking
- **Audit Logging**: Compliance and governance tracking
- **CI/CD Templates**: Ready-to-use pipeline configurations

## Quick Start

### Installation

```bash
pip install -e ".[full]"
```

### Basic Usage

```python
from neural.mlops.registry import ModelRegistry, ModelStage
from neural.mlops.deployment import DeploymentManager
from neural.mlops.audit import AuditLogger

# Initialize components
registry = ModelRegistry("./models")
deployment = DeploymentManager("./deployments")
audit = AuditLogger("./audit_logs")
```

## Components

### 1. Model Registry

Centralized model storage with versioning.

```python
from neural.mlops.registry import ModelRegistry, ModelStage

registry = ModelRegistry("./models")

# Register a new model
metadata = registry.register_model(
    name="fraud_detector",
    version="v2.0.0",
    model_path="./model.pt",
    framework="pytorch",
    created_by="data_scientist@company.com",
    metrics={"accuracy": 0.96, "f1": 0.94},
    tags=["fraud", "production-ready"],
    description="Improved fraud detection model"
)

# Promote to production
registry.promote_model(
    name="fraud_detector",
    version="v2.0.0",
    target_stage=ModelStage.PRODUCTION
)

# Compare model versions
comparison = registry.compare_models(
    name="fraud_detector",
    version1="v1.0.0",
    version2="v2.0.0"
)
print(f"Accuracy improvement: {comparison['metrics_comparison']['accuracy']['difference']}")
```

### 2. Deployment Manager

Basic deployment tracking.

```python
from neural.mlops.deployment import DeploymentManager

manager = DeploymentManager("./deployments")

# Create deployment
deployment = manager.create_deployment(
    model_name="fraud_detector",
    model_version="v2.0.0",
    environment="production",
    endpoint="http://localhost:8080/predict"
)

# Activate deployment
manager.activate_deployment(deployment.deployment_id)

# List active deployments
deployments = manager.list_deployments(environment="production")
```

### 3. Audit Logging

Comprehensive audit logging for compliance.

```python
from neural.mlops.audit import AuditLogger, EventType, EventSeverity
from datetime import datetime, timedelta

logger = AuditLogger("./audit_logs")

# Log model deployment
logger.log_model_deployment(
    model_name="fraud_detector",
    version="v2.0.0",
    user="devops@company.com",
    environment="production"
)

# Log security violation
logger.log_security_violation(
    user="unknown@external.com",
    resource_type="model",
    resource_id="fraud_detector:v2.0.0",
    action="unauthorized_access",
    details={"ip_address": "192.168.1.1"},
    ip_address="192.168.1.1"
)

# Query events
events = logger.query_events(
    event_type=EventType.MODEL_DEPLOYED,
    start_date=datetime.now() - timedelta(days=30),
    severity=EventSeverity.INFO
)

# Generate compliance report
report = logger.generate_compliance_report(
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now()
)

print(f"Total events: {report.total_events}")
print(f"Security violations: {len(report.security_violations)}")

# Export audit trail
logger.export_audit_trail(
    output_path="audit_trail_Q1_2024.json",
    format="json",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31)
)
```

## CI/CD Integration

### GitHub Actions

See [GitHub Actions template](../../neural/mlops/ci_templates/github_actions.yml) for complete configuration.

### GitLab CI

See [GitLab CI template](../../neural/mlops/ci_templates/gitlab_ci.yml) for complete configuration.

### Jenkins

See [Jenkinsfile template](../../neural/mlops/ci_templates/Jenkinsfile) for complete configuration.

## Best Practices

### Model Registry

1. **Versioning**: Use semantic versioning (v1.0.0, v1.1.0, v2.0.0)
2. **Metadata**: Always include comprehensive metrics and tags
3. **Staging**: Test models in staging before production

### Deployment

1. **Monitoring**: Set up comprehensive monitoring and alerting
2. **Gradual Rollout**: Test thoroughly in staging first

### Audit Logging

1. **Compliance**: Log all model lifecycle events for compliance
2. **Security**: Track unauthorized access attempts
3. **Retention**: Maintain audit logs for required compliance period
4. **Regular Reports**: Generate and review compliance reports monthly

## Architecture

```
neural/mlops/
├── __init__.py
├── registry.py          # Model registry with versioning
├── deployment.py        # Basic deployment tracking
├── audit.py            # Audit logging for compliance
├── ci_templates.py     # CI/CD template generator
└── ci_templates/
    ├── github_actions.yml
    ├── gitlab_ci.yml
    ├── Jenkinsfile
    └── azure_pipelines.yml
```

## Storage Structure

```
./models/                    # Model registry
├── models/
│   └── fraud_detector/
│       ├── v1.0.0/
│       │   ├── model.pt
│       │   └── config.yaml
│       └── v2.0.0/
│           ├── model.pt
│           └── config.yaml
└── metadata/
    ├── fraud_detector_v1.0.0.json
    └── fraud_detector_v2.0.0.json

./deployments/               # Deployment tracking
└── deploy_20240115_120000_fraud_detector_v2.0.0.json

./audit_logs/                # Audit logging
├── events/
│   └── 2024/
│       └── 01/
│           └── 15/
│               └── event_20240115_120000_000001.json
└── reports/
    └── report_20240131_120000.json
```

## Support

For issues and questions:
- GitHub Issues: [Neural DSL Issues](https://github.com/Lemniscate-world/Neural/issues)
- Documentation: [Full Documentation](../README.md)
