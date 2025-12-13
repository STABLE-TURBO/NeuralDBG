# Neural DSL MLOps - Quick Reference

Fast reference for common MLOps operations.

## Installation

```bash
pip install -e ".[full]"
```

## Imports

```python
from neural.mlops.registry import ModelRegistry, ModelStage
from neural.mlops.ab_testing import ABTestManager, TrafficSplitStrategy
from neural.mlops.deployment import (
    DeploymentManager, DeploymentStrategy,
    PerformanceMetrics, RollbackConfig
)
from neural.mlops.audit import AuditLogger, EventType, EventSeverity
from neural.mlops.ci_templates import CITemplateGenerator
```

## Model Registry

```python
# Initialize
registry = ModelRegistry("./models")

# Register model
metadata = registry.register_model(
    name="my_model", version="v1.0.0",
    model_path="./model.pt", framework="pytorch",
    created_by="user@company.com",
    metrics={"accuracy": 0.95}
)

# Get model
model = registry.get_model("my_model", "v1.0.0")

# List models
all_models = registry.list_models()
prod_models = registry.list_models(stage=ModelStage.PRODUCTION)

# Request promotion
registry.request_promotion(
    name="my_model", version="v1.0.0",
    target_stage=ModelStage.PRODUCTION,
    requested_by="user@company.com",
    justification="Ready for production"
)

# Approve promotion
registry.approve_promotion(
    name="my_model", version="v1.0.0",
    approver="manager@company.com"
)

# Compare versions
comparison = registry.compare_models("my_model", "v1.0.0", "v2.0.0")
```

## A/B Testing

```python
# Initialize
manager = ABTestManager("./ab_tests")

# Create test
test = manager.create_test(
    name="Model V2 Test",
    description="Testing new version",
    control_variant="v1.0.0",
    treatment_variant="v2.0.0",
    traffic_split=0.1,  # 10% to treatment
    strategy=TrafficSplitStrategy.HASH_BASED,
    created_by="user@company.com"
)

# Start test
manager.start_test(test.test_id)

# Get variant for user
variant = manager.get_variant(test.test_id, user_id="user123")

# Record result
manager.record_request(
    test_id=test.test_id,
    variant="treatment",
    success=True,
    latency=0.042,
    custom_metrics={"accuracy": 0.96}
)

# Analyze results
analysis = manager.analyze_test(test.test_id)
print(f"Significant: {analysis['statistically_significant']}")
print(f"Improvement: {analysis['improvement']['relative_improvement']:.2f}%")

# Complete test
manager.complete_test(test.test_id)
```

## Deployment

```python
# Initialize
manager = DeploymentManager("./deployments")

# Shadow deployment
shadow = manager.shadow_deploy(
    primary_model="model:v1.0.0",
    shadow_model="model:v2.0.0",
    traffic_percentage=100.0
)

# Record shadow comparison
manager.record_shadow_comparison(
    shadow_id=shadow.shadow_id,
    primary_result={"prediction": 0},
    shadow_result={"prediction": 0},
    primary_latency=0.035,
    shadow_latency=0.038,
    agreement=True
)

# Analyze shadow
comparison = manager.compare_shadow_deployment(shadow.shadow_id)

# Canary deployment
deployment = manager.create_deployment(
    model_name="my_model",
    model_version="v2.0.0",
    strategy=DeploymentStrategy.CANARY,
    created_by="user@company.com",
    baseline_metrics=PerformanceMetrics(
        latency_p95=0.050,
        error_rate=0.001
    ),
    rollback_config=RollbackConfig(
        error_rate_threshold=0.01,
        latency_threshold_multiplier=1.5,
        min_requests_before_check=1000
    )
)

# Start deployment
manager.start_deployment(deployment.deployment_id)

# Update metrics
manager.update_metrics(
    deployment.deployment_id,
    PerformanceMetrics(latency_p95=0.052, error_rate=0.002)
)

# Check health (auto-rollback if needed)
needs_rollback, reasons = manager.check_deployment_health(
    deployment.deployment_id,
    request_count=2000
)

# Complete deployment
manager.complete_deployment(deployment.deployment_id)

# Manual rollback if needed
manager.rollback_deployment(
    deployment.deployment_id,
    reason="Business decision",
    triggered_by="user@company.com"
)
```

## Audit Logging

```python
# Initialize
logger = AuditLogger("./audit_logs")

# Log model deployment
logger.log_model_deployment(
    model_name="my_model", version="v2.0.0",
    user="user@company.com",
    environment="production", strategy="canary"
)

# Log approval
logger.log_approval_granted(
    model_name="my_model", version="v2.0.0",
    approver="manager@company.com",
    comment="Approved"
)

# Log security violation
logger.log_security_violation(
    user="unknown@external.com",
    resource_type="model",
    resource_id="my_model:v2.0.0",
    action="unauthorized_access",
    ip_address="192.168.1.1"
)

# Query events
from datetime import datetime, timedelta
events = logger.query_events(
    event_type=EventType.MODEL_DEPLOYED,
    start_date=datetime.now() - timedelta(days=30),
    user="user@company.com"
)

# Generate compliance report
report = logger.generate_compliance_report(
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now()
)

# Export audit trail
logger.export_audit_trail(
    output_path="audit_trail.json",
    format="json"
)
```

## CI/CD Templates

```python
# Initialize
generator = CITemplateGenerator()

# GitHub Actions
github_config = generator.generate_github_actions(
    model_name="my_model",
    python_version="3.10",
    enable_gpu=True,
    deploy_environments=["staging", "production"]
)
generator.save_template(github_config, ".github/workflows/ml-pipeline.yml")

# GitLab CI
gitlab_config = generator.generate_gitlab_ci(
    model_name="my_model",
    python_version="3.10"
)
generator.save_template(gitlab_config, ".gitlab-ci.yml")

# Jenkins
jenkins_config = generator.generate_jenkins(
    model_name="my_model",
    python_version="3.10"
)
generator.save_template(jenkins_config, "Jenkinsfile")

# Azure Pipelines
azure_config = generator.generate_azure_pipelines(
    model_name="my_model",
    python_version="3.10"
)
generator.save_template(azure_config, "azure-pipelines.yml")
```

## Common Patterns

### Complete Deployment Pipeline

```python
# 1. Register model
metadata = registry.register_model(...)

# 2. Request and approve promotion
registry.request_promotion(...)
registry.approve_promotion(...)

# 3. Shadow deploy
shadow = deployment.shadow_deploy(...)
# Wait for data...
comparison = deployment.compare_shadow_deployment(shadow.shadow_id)

# 4. A/B test
test = ab_testing.create_test(...)
ab_testing.start_test(...)
# Wait for data...
analysis = ab_testing.analyze_test(test.test_id)

# 5. Canary deploy
deployment = deployment.create_deployment(
    strategy=DeploymentStrategy.CANARY,
    rollback_config=RollbackConfig(...)
)
deployment.start_deployment(...)

# 6. Monitor and complete
deployment.check_deployment_health(...)
deployment.complete_deployment(...)

# 7. Audit log
audit.log_model_deployment(...)
```

### Emergency Rollback

```python
# Immediate rollback
manager.rollback_deployment(
    deployment_id=deployment_id,
    reason="EMERGENCY: Critical issue in production",
    triggered_by="on-call@company.com"
)

# Log critical event
logger.log_event(
    event_type=EventType.MODEL_ROLLED_BACK,
    user="on-call@company.com",
    resource_type="deployment",
    resource_id=deployment_id,
    action="emergency_rollback",
    severity=EventSeverity.CRITICAL
)
```

## Enums and Constants

### ModelStage
```python
ModelStage.DEVELOPMENT
ModelStage.STAGING
ModelStage.PRODUCTION
ModelStage.ARCHIVED
```

### DeploymentStrategy
```python
DeploymentStrategy.DIRECT
DeploymentStrategy.BLUE_GREEN
DeploymentStrategy.CANARY
DeploymentStrategy.SHADOW
DeploymentStrategy.ROLLING
```

### TrafficSplitStrategy
```python
TrafficSplitStrategy.RANDOM
TrafficSplitStrategy.HASH_BASED
TrafficSplitStrategy.PERCENTAGE
TrafficSplitStrategy.CANARY
```

### EventType
```python
EventType.MODEL_REGISTERED
EventType.MODEL_DEPLOYED
EventType.MODEL_ROLLED_BACK
EventType.APPROVAL_REQUESTED
EventType.APPROVAL_GRANTED
EventType.APPROVAL_REJECTED
EventType.AB_TEST_CREATED
EventType.SECURITY_VIOLATION
# ... and more
```

### EventSeverity
```python
EventSeverity.INFO
EventSeverity.WARNING
EventSeverity.ERROR
EventSeverity.CRITICAL
```

## Configuration

### Rollback Configuration
```python
RollbackConfig(
    enabled=True,
    error_rate_threshold=0.01,        # 1%
    latency_threshold_multiplier=1.5, # 50% increase
    min_requests_before_check=1000,
    check_interval_seconds=60
)
```

### Performance Metrics
```python
PerformanceMetrics(
    latency_p50=0.035,
    latency_p95=0.050,
    latency_p99=0.075,
    error_rate=0.001,
    requests_per_second=1000.0,
    cpu_usage=0.65,
    memory_usage=0.80,
    custom_metrics={"business_metric": 0.95}
)
```

## File Locations

```
./models/           # Model registry
./ab_tests/         # A/B testing
./deployments/      # Deployment tracking
./audit_logs/       # Audit logging
```

## CLI Examples

### Using in Scripts

```bash
# Register model
python -c "
from neural.mlops.registry import ModelRegistry
registry = ModelRegistry('./models')
registry.register_model('my_model', 'v1.0.0', './model.pt', 'pytorch', 'user@company.com')
"

# List models
python -c "
from neural.mlops.registry import ModelRegistry
registry = ModelRegistry('./models')
models = registry.list_models()
for m in models:
    print(f'{m.name} {m.version} - {m.stage.value}')
"
```

## Troubleshooting

```python
# Check if model exists
try:
    model = registry.get_model("my_model", "v1.0.0")
except FileNotFoundError:
    print("Model not found")

# List pending approvals
approvals = registry.approval_workflow.list_pending_approvals()

# Get deployment status
deployment = manager.get_deployment(deployment_id)
print(f"Status: {deployment.status.value}")

# Query recent audit events
events = logger.query_events(
    start_date=datetime.now() - timedelta(hours=24)
)
print(f"Events in last 24h: {len(events)}")
```

## Best Practices

1. **Always shadow deploy before canary**
2. **Start canary with 5% traffic**
3. **Set conservative rollback thresholds**
4. **Log all production changes**
5. **Generate monthly compliance reports**
6. **Use semantic versioning (v1.0.0)**
7. **Require approval for production**
8. **Monitor metrics continuously**

## Links

- [Full Documentation](./README.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)
- [Feature Summary](./MLOPS_FEATURES.md)
- [Examples](../../examples/mlops/)
