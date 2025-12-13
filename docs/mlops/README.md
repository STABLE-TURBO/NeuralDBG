# Neural DSL MLOps Documentation

Enterprise-grade MLOps capabilities for production ML workflows.

## Overview

The Neural DSL MLOps module provides comprehensive tools for managing the complete machine learning lifecycle in production environments, including:

- **Model Registry**: Versioned model storage with approval workflows
- **A/B Testing**: Statistical testing framework with traffic splitting
- **Deployment Manager**: Safe deployment strategies including shadow deployment
- **Audit Logging**: Complete compliance and governance tracking
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
from neural.mlops.ab_testing import ABTestManager
from neural.mlops.audit import AuditLogger

# Initialize components
registry = ModelRegistry("./models")
deployment = DeploymentManager("./deployments")
ab_testing = ABTestManager("./ab_tests")
audit = AuditLogger("./audit_logs")
```

## Components

### 1. Model Registry

Centralized model storage with versioning and approval workflows.

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
    description="Improved fraud detection model with XGBoost"
)

# Request promotion to production
registry.request_promotion(
    name="fraud_detector",
    version="v2.0.0",
    target_stage=ModelStage.PRODUCTION,
    requested_by="ml_engineer@company.com",
    justification="Model passed all validation tests with 96% accuracy",
    reviewers=["ml_manager@company.com"]
)

# Approve and promote
registry.approve_promotion(
    name="fraud_detector",
    version="v2.0.0",
    approver="ml_manager@company.com",
    comment="Metrics look good, approved for production"
)

# Compare model versions
comparison = registry.compare_models(
    name="fraud_detector",
    version1="v1.0.0",
    version2="v2.0.0"
)
print(f"Accuracy improvement: {comparison['metrics_comparison']['accuracy']['difference']}")
```

### 2. A/B Testing Framework

Statistical A/B testing with traffic splitting strategies.

```python
from neural.mlops.ab_testing import ABTestManager, TrafficSplitStrategy, TestStatus

manager = ABTestManager("./ab_tests")

# Create A/B test
test = manager.create_test(
    name="Model V2 A/B Test",
    description="Testing new model version against current production",
    control_variant="fraud_detector:v1.0.0",
    treatment_variant="fraud_detector:v2.0.0",
    traffic_split=0.1,  # 10% to treatment
    strategy=TrafficSplitStrategy.HASH_BASED,
    created_by="ml_engineer@company.com"
)

# Start test
manager.start_test(test.test_id)

# In your prediction service
variant = manager.get_variant(test.test_id, user_id="user123")
# Use the assigned variant for prediction

# Record results
manager.record_request(
    test_id=test.test_id,
    variant="treatment",
    success=True,
    latency=0.042,
    custom_metrics={"accuracy": 0.96}
)

# Analyze results
analysis = manager.analyze_test(test.test_id, confidence_level=0.95)

if analysis['statistically_significant']:
    print(f"Treatment variant is significantly better!")
    print(f"Improvement: {analysis['improvement']['relative_improvement']:.2f}%")
    print(f"Control success rate: {analysis['control']['success_rate']:.2%}")
    print(f"Treatment success rate: {analysis['treatment']['success_rate']:.2%}")
```

### 3. Deployment Manager

Safe deployment with shadow deployment and automated rollback.

```python
from neural.mlops.deployment import (
    DeploymentManager,
    DeploymentStrategy,
    PerformanceMetrics,
    RollbackConfig
)

manager = DeploymentManager("./deployments")

# Shadow deployment (no production impact)
shadow = manager.shadow_deploy(
    primary_model="fraud_detector:v1.0.0",
    shadow_model="fraud_detector:v2.0.0",
    traffic_percentage=100.0  # Copy 100% of traffic
)

# Record shadow comparisons
manager.record_shadow_comparison(
    shadow_id=shadow.shadow_id,
    primary_result={"prediction": 0},
    shadow_result={"prediction": 0},
    primary_latency=0.035,
    shadow_latency=0.042,
    agreement=True
)

# Analyze shadow results
comparison = manager.compare_shadow_deployment(shadow.shadow_id)
print(f"Agreement rate: {comparison['agreement_rate']:.2%}")
print(f"Shadow latency P95: {comparison['shadow_latency']['p95']:.3f}s")

# Production deployment with automated rollback
deployment = manager.create_deployment(
    model_name="fraud_detector",
    model_version="v2.0.0",
    strategy=DeploymentStrategy.CANARY,
    created_by="devops@company.com",
    baseline_metrics=PerformanceMetrics(
        latency_p95=0.050,
        error_rate=0.001
    ),
    rollback_config=RollbackConfig(
        error_rate_threshold=0.01,  # Rollback if error rate > 1%
        latency_threshold_multiplier=1.5,  # Rollback if latency 1.5x baseline
        min_requests_before_check=1000
    )
)

# Monitor and auto-rollback if performance degrades
current_metrics = PerformanceMetrics(
    latency_p95=0.080,  # Degraded performance
    error_rate=0.005
)
manager.update_metrics(deployment.deployment_id, current_metrics)

needs_rollback, reasons = manager.check_deployment_health(
    deployment.deployment_id,
    request_count=2000
)

if needs_rollback:
    print(f"Deployment rolled back: {reasons}")
```

### 4. Audit Logging

Comprehensive audit logging for compliance and governance.

```python
from neural.mlops.audit import AuditLogger, EventType, EventSeverity
from datetime import datetime, timedelta

logger = AuditLogger("./audit_logs")

# Log model deployment
logger.log_model_deployment(
    model_name="fraud_detector",
    version="v2.0.0",
    user="devops@company.com",
    environment="production",
    strategy="canary"
)

# Log approval
logger.log_approval_granted(
    model_name="fraud_detector",
    version="v2.0.0",
    approver="ml_manager@company.com",
    comment="Approved after successful A/B test"
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
print(f"Most active user: {report.summary['most_active_user']}")

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

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Model Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ -v
```

See [GitHub Actions template](../../neural/mlops/ci_templates/github_actions.yml) for complete configuration.

### GitLab CI

```yaml
# .gitlab-ci.yml
image: python:3.10

stages:
  - test
  - validate
  - deploy

test:unit:
  stage: test
  script:
    - pip install -e .
    - pytest tests/ -v
```

See [GitLab CI template](../../neural/mlops/ci_templates/gitlab_ci.yml) for complete configuration.

### Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'pip install -e .'
                sh 'pytest tests/ -v'
            }
        }
    }
}
```

See [Jenkinsfile template](../../neural/mlops/ci_templates/Jenkinsfile) for complete configuration.

## Best Practices

### Model Registry

1. **Versioning**: Use semantic versioning (v1.0.0, v1.1.0, v2.0.0)
2. **Metadata**: Always include comprehensive metrics and tags
3. **Approval Workflow**: Require approval for production deployments
4. **Staging**: Test models in staging before production

### A/B Testing

1. **Sample Size**: Ensure sufficient requests for statistical significance
2. **Duration**: Run tests for adequate time periods (typically 1-2 weeks)
3. **Metrics**: Track multiple metrics (latency, accuracy, business KPIs)
4. **Hash-based Splitting**: Use consistent user assignment with hash-based strategy

### Deployment

1. **Shadow Deployment**: Always test new models in shadow mode first
2. **Canary Release**: Start with small traffic percentage (5-10%)
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Rollback**: Configure automated rollback thresholds
5. **Gradual Rollout**: Increase traffic gradually over days/weeks

### Audit Logging

1. **Compliance**: Log all model lifecycle events for compliance
2. **Security**: Track unauthorized access attempts
3. **Retention**: Maintain audit logs for required compliance period
4. **Regular Reports**: Generate and review compliance reports monthly

## Architecture

```
neural/mlops/
├── __init__.py
├── registry.py          # Model registry with approval workflows
├── ab_testing.py        # A/B testing framework
├── deployment.py        # Deployment manager with rollback
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
├── metadata/
│   ├── fraud_detector_v1.0.0.json
│   └── fraud_detector_v2.0.0.json
└── approvals/
    └── fraud_detector_v2.0.0.json

./ab_tests/                  # A/B testing
├── test_20240115_120000_Model_V2_Test.json
└── test_20240116_140000_Another_Test.json

./deployments/               # Deployment tracking
├── deployments/
│   ├── deploy_20240115_120000_fraud_detector_v2.0.0.json
│   └── deploy_20240116_140000_fraud_detector_v2.1.0.json
└── shadows/
    └── shadow_20240115_120000_fraud_detector_v2.0.0.json

./audit_logs/                # Audit logging
├── events/
│   └── 2024/
│       └── 01/
│           └── 15/
│               ├── event_20240115_120000_000001.json
│               └── event_20240115_120500_000002.json
└── reports/
    └── report_20240131_120000.json
```

## API Reference

See individual module documentation:

- [Model Registry API](./registry.md)
- [A/B Testing API](./ab_testing.md)
- [Deployment API](./deployment.md)
- [Audit Logging API](./audit.md)
- [CI/CD Templates API](./ci_templates.md)

## Examples

See [examples directory](../../examples/mlops/) for complete examples:

- Model lifecycle management
- A/B testing workflow
- Shadow deployment
- Compliance reporting
- CI/CD integration

## Support

For issues and questions:
- GitHub Issues: [Neural DSL Issues](https://github.com/Lemniscate-world/Neural/issues)
- Documentation: [Full Documentation](../README.md)
