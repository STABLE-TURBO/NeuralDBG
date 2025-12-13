# Neural DSL MLOps Examples

Complete examples demonstrating Neural DSL's MLOps capabilities.

## Available Examples

### complete_workflow.py

Comprehensive end-to-end example demonstrating:
- Model registration with metadata
- Approval workflow for production deployment
- Shadow deployment for validation
- A/B testing with statistical analysis
- Canary deployment with automated rollback
- Audit logging and compliance reporting

**Run it:**
```bash
python examples/mlops/complete_workflow.py
```

**What it demonstrates:**
1. Training and registering a new model version
2. Requesting and approving promotion to staging
3. Shadow deploying to validate with production traffic
4. A/B testing control vs treatment variants
5. Canary deployment with health monitoring
6. Automated rollback configuration
7. Compliance report generation
8. Model version comparison

## Quick Start Examples

### Model Registry

```python
from neural.mlops.registry import ModelRegistry, ModelStage

registry = ModelRegistry("./models")

# Register model
metadata = registry.register_model(
    name="my_model",
    version="v1.0.0",
    model_path="./model.pt",
    framework="pytorch",
    created_by="user@company.com",
    metrics={"accuracy": 0.95}
)

# Request promotion
registry.request_promotion(
    name="my_model",
    version="v1.0.0",
    target_stage=ModelStage.PRODUCTION,
    requested_by="user@company.com",
    justification="Model meets requirements"
)

# Approve
registry.approve_promotion(
    name="my_model",
    version="v1.0.0",
    approver="manager@company.com"
)

# List production models
models = registry.list_models(stage=ModelStage.PRODUCTION)
```

### A/B Testing

```python
from neural.mlops.ab_testing import ABTestManager, TrafficSplitStrategy

manager = ABTestManager("./ab_tests")

# Create test
test = manager.create_test(
    name="Model V2 Test",
    description="Testing new model version",
    control_variant="v1.0.0",
    treatment_variant="v2.0.0",
    traffic_split=0.1,
    strategy=TrafficSplitStrategy.HASH_BASED,
    created_by="user@company.com"
)

# Start test
manager.start_test(test.test_id)

# Record results
manager.record_request(
    test_id=test.test_id,
    variant="treatment",
    success=True,
    latency=0.042
)

# Analyze
analysis = manager.analyze_test(test.test_id)
print(f"Significant: {analysis['statistically_significant']}")
```

### Shadow Deployment

```python
from neural.mlops.deployment import DeploymentManager

manager = DeploymentManager("./deployments")

# Create shadow
shadow = manager.shadow_deploy(
    primary_model="model:v1.0.0",
    shadow_model="model:v2.0.0",
    traffic_percentage=100.0
)

# Record comparison
manager.record_shadow_comparison(
    shadow_id=shadow.shadow_id,
    primary_result={"prediction": 0},
    shadow_result={"prediction": 0},
    primary_latency=0.035,
    shadow_latency=0.038,
    agreement=True
)

# Analyze
comparison = manager.compare_shadow_deployment(shadow.shadow_id)
print(f"Agreement rate: {comparison['agreement_rate']:.2%}")
```

### Canary Deployment

```python
from neural.mlops.deployment import (
    DeploymentManager,
    DeploymentStrategy,
    PerformanceMetrics,
    RollbackConfig
)

manager = DeploymentManager("./deployments")

# Create deployment
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
        latency_threshold_multiplier=1.5
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
```

### Audit Logging

```python
from neural.mlops.audit import AuditLogger, EventType, EventSeverity
from datetime import datetime, timedelta

logger = AuditLogger("./audit_logs")

# Log deployment
logger.log_model_deployment(
    model_name="my_model",
    version="v2.0.0",
    user="user@company.com",
    environment="production",
    strategy="canary"
)

# Log security violation
logger.log_security_violation(
    user="unknown@external.com",
    resource_type="model",
    resource_id="my_model:v2.0.0",
    action="unauthorized_access",
    ip_address="192.168.1.1"
)

# Generate compliance report
report = logger.generate_compliance_report(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)

# Export audit trail
logger.export_audit_trail(
    output_path="audit_trail.json",
    format="json"
)
```

### CI/CD Template Generation

```python
from neural.mlops.ci_templates import CITemplateGenerator

generator = CITemplateGenerator()

# Generate GitHub Actions workflow
github_config = generator.generate_github_actions(
    model_name="my_model",
    python_version="3.10",
    enable_gpu=True,
    deploy_environments=["staging", "production"]
)

# Save to file
generator.save_template(
    github_config,
    ".github/workflows/ml-pipeline.yml"
)

# Generate GitLab CI
gitlab_config = generator.generate_gitlab_ci(
    model_name="my_model",
    python_version="3.10"
)

generator.save_template(gitlab_config, ".gitlab-ci.yml")

# Generate Jenkins pipeline
jenkins_config = generator.generate_jenkins(
    model_name="my_model",
    python_version="3.10"
)

generator.save_template(jenkins_config, "Jenkinsfile")
```

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from neural.mlops.registry import ModelRegistry
from neural.mlops.deployment import DeploymentManager

app = Flask(__name__)
registry = ModelRegistry("./models")
deployment = DeploymentManager("./deployments")

@app.route("/models", methods=["GET"])
def list_models():
    models = registry.list_models()
    return jsonify([m.to_dict() for m in models])

@app.route("/models/<name>/<version>", methods=["GET"])
def get_model(name, version):
    metadata = registry.get_model(name, version)
    return jsonify(metadata.to_dict())

@app.route("/deployments", methods=["POST"])
def create_deployment():
    data = request.json
    deployment = deployment.create_deployment(**data)
    return jsonify(deployment.to_dict()), 201

if __name__ == "__main__":
    app.run(debug=True)
```

### FastAPI

```python
from fastapi import FastAPI, HTTPException
from neural.mlops.registry import ModelRegistry, ModelStage
from pydantic import BaseModel

app = FastAPI()
registry = ModelRegistry("./models")

class ModelRegistration(BaseModel):
    name: str
    version: str
    framework: str
    created_by: str
    metrics: dict

@app.post("/models/register")
def register_model(model: ModelRegistration):
    metadata = registry.register_model(**model.dict())
    return metadata.to_dict()

@app.get("/models/{name}/{version}")
def get_model(name: str, version: str):
    try:
        metadata = registry.get_model(name, version)
        return metadata.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get("/models")
def list_models(stage: str = None):
    stage_enum = ModelStage(stage) if stage else None
    models = registry.list_models(stage=stage_enum)
    return [m.to_dict() for m in models]
```

## Testing Examples

### Unit Tests

```python
import unittest
from neural.mlops.registry import ModelRegistry
from neural.mlops.deployment import DeploymentManager

class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry("./test_models")
    
    def test_register_model(self):
        metadata = self.registry.register_model(
            name="test_model",
            version="v1.0.0",
            model_path="./dummy.pt",
            framework="pytorch",
            created_by="test@example.com",
            metrics={"accuracy": 0.95}
        )
        self.assertEqual(metadata.name, "test_model")
        self.assertEqual(metadata.metrics["accuracy"], 0.95)
    
    def test_model_comparison(self):
        # Register two versions
        self.registry.register_model(
            name="test_model", version="v1.0.0",
            model_path="./dummy.pt", framework="pytorch",
            created_by="test@example.com",
            metrics={"accuracy": 0.94}
        )
        self.registry.register_model(
            name="test_model", version="v2.0.0",
            model_path="./dummy.pt", framework="pytorch",
            created_by="test@example.com",
            metrics={"accuracy": 0.96}
        )
        
        # Compare
        comparison = self.registry.compare_models(
            "test_model", "v1.0.0", "v2.0.0"
        )
        self.assertEqual(
            comparison["metrics_comparison"]["accuracy"]["difference"],
            0.02
        )
```

## Production Patterns

### Deployment Pipeline

```python
def deploy_model(model_name: str, version: str):
    """Complete deployment pipeline."""
    # 1. Shadow deploy
    shadow = deployment_mgr.shadow_deploy(
        primary_model=f"{model_name}:v1.0.0",
        shadow_model=f"{model_name}:{version}"
    )
    
    # Wait for shadow data collection
    time.sleep(3600)  # 1 hour
    
    # 2. Analyze shadow results
    comparison = deployment_mgr.compare_shadow_deployment(shadow.shadow_id)
    
    if comparison['agreement_rate'] < 0.95:
        raise Exception("Shadow agreement rate too low")
    
    # 3. A/B test
    test = ab_manager.create_test(
        name=f"{model_name} {version} Test",
        control_variant=f"{model_name}:v1.0.0",
        treatment_variant=f"{model_name}:{version}",
        traffic_split=0.1,
        strategy=TrafficSplitStrategy.HASH_BASED,
        created_by="system"
    )
    
    ab_manager.start_test(test.test_id)
    
    # Wait for A/B test data
    time.sleep(86400)  # 24 hours
    
    # 4. Analyze A/B test
    analysis = ab_manager.analyze_test(test.test_id)
    
    if not analysis['statistically_significant']:
        raise Exception("A/B test not significant")
    
    # 5. Canary deploy
    deployment = deployment_mgr.create_deployment(
        model_name=model_name,
        model_version=version,
        strategy=DeploymentStrategy.CANARY,
        created_by="system",
        rollback_config=RollbackConfig(
            error_rate_threshold=0.01,
            latency_threshold_multiplier=1.5
        )
    )
    
    deployment_mgr.start_deployment(deployment.deployment_id)
    
    # 6. Audit log
    audit_logger.log_model_deployment(
        model_name=model_name,
        version=version,
        user="system",
        environment="production",
        strategy="canary"
    )
    
    return deployment.deployment_id
```

## Documentation

- [Complete MLOps Documentation](../../docs/mlops/README.md)
- [Deployment Guide](../../docs/mlops/DEPLOYMENT_GUIDE.md)
- [Feature Summary](../../docs/mlops/MLOPS_FEATURES.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
