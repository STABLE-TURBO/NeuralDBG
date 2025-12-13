# Neural DSL MLOps Module

Enterprise-grade MLOps capabilities for production machine learning workflows.

## Features

### 🏗️ Model Registry
- Versioned model storage with metadata tracking
- Multi-stage approval workflows (Development → Staging → Production)
- Model comparison and lineage tracking
- Centralized model repository

### 🧪 A/B Testing Framework
- Statistical hypothesis testing with confidence intervals
- Multiple traffic splitting strategies (random, hash-based, canary)
- Real-time metrics collection and analysis
- Automated significance testing

### 🚀 Deployment Manager
- Shadow deployment for risk-free testing
- Canary releases with gradual traffic shift
- Blue-green deployment support
- Automated rollback on performance degradation
- Comprehensive health monitoring

### 📋 Audit Logging
- Complete audit trail for compliance
- Tamper-evident event logging
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
from neural.mlops.deployment import DeploymentManager, DeploymentStrategy
from neural.mlops.ab_testing import ABTestManager, TrafficSplitStrategy
from neural.mlops.audit import AuditLogger, EventType

# Initialize
registry = ModelRegistry("./models")
deployment = DeploymentManager("./deployments")
ab_testing = ABTestManager("./ab_tests")
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

# Create A/B test
test = ab_testing.create_test(
    name="Model V2 Test",
    description="Testing new model",
    control_variant="v1.0.0",
    treatment_variant="v2.0.0",
    traffic_split=0.1,
    strategy=TrafficSplitStrategy.HASH_BASED,
    created_by="user@company.com"
)

# Shadow deployment
shadow = deployment.shadow_deploy(
    primary_model="my_model:v1.0.0",
    shadow_model="my_model:v2.0.0"
)

# Audit logging
audit.log_model_deployment(
    model_name="my_model",
    version="v2.0.0",
    user="user@company.com",
    environment="production",
    strategy="canary"
)
```

## Architecture

```
neural/mlops/
├── registry.py          # Model registry with approval workflows
├── ab_testing.py        # A/B testing with traffic splitting
├── deployment.py        # Deployment strategies and rollback
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
2. Run shadow deployment to validate
3. Create A/B test to compare performance
4. Deploy with canary strategy
5. Monitor and auto-rollback if needed
6. Audit all actions for compliance

### Compliance and Governance
1. Log all model lifecycle events
2. Require approval for production deployments
3. Track who deployed what and when
4. Generate compliance reports
5. Export audit trails for regulators

### Safe Experimentation
1. Shadow deploy new models without risk
2. A/B test with statistical significance
3. Compare metrics between variants
4. Roll out gradually with canary
5. Rollback automatically if performance degrades

## Documentation

- [Complete Documentation](../../docs/mlops/README.md)
- [Deployment Guide](../../docs/mlops/DEPLOYMENT_GUIDE.md)
- [API Reference](../../docs/mlops/)

## Examples

See [examples directory](../../examples/mlops/) for:
- Complete deployment workflows
- A/B testing examples
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
from neural.mlops.deployment import RollbackConfig

deployment = DeploymentManager(
    storage_path="./deployments",
    default_rollback_config=RollbackConfig(
        error_rate_threshold=0.01,
        latency_threshold_multiplier=1.5
    )
)
```

### A/B Testing

```python
ab_testing = ABTestManager(
    storage_path="./ab_tests"
)
```

### Audit Logger

```python
audit = AuditLogger(
    storage_path="./audit_logs"
)
```

## Performance

- **Model Registry**: O(1) model lookup, O(n) listing
- **A/B Testing**: O(1) variant assignment with hash-based splitting
- **Deployment**: O(1) health checks, O(log n) metrics analysis
- **Audit Logging**: O(1) event logging, O(n) querying with date partitioning

## Storage

All data stored as JSON files with structured directories:

```
./models/               # Model registry
./ab_tests/            # A/B test configurations and results
./deployments/         # Deployment tracking
./audit_logs/          # Audit events (date-partitioned)
```

## Security

- Tamper-evident audit logging
- Access control through approval workflows
- Security violation tracking
- Sensitive data handling in audit logs

## Compliance

Supports compliance requirements:
- SOC 2
- GDPR
- HIPAA
- PCI DSS
- Industry-specific regulations

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE.md](../../LICENSE.md)

## Support

- Documentation: [docs/mlops/](../../docs/mlops/)
- Issues: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- Discussions: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
