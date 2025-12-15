# Neural DSL MLOps Examples

Basic examples demonstrating Neural DSL's MLOps capabilities.

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

# Promote to production
registry.promote_model(
    name="my_model",
    version="v1.0.0",
    target_stage=ModelStage.PRODUCTION
)

# List production models
models = registry.list_models(stage=ModelStage.PRODUCTION)
```

### Basic Deployment

```python
from neural.mlops.deployment import DeploymentManager

manager = DeploymentManager("./deployments")

# Create deployment
deployment = manager.create_deployment(
    model_name="my_model",
    model_version="v1.0.0",
    environment="production",
    endpoint="http://localhost:8080/predict"
)

# Activate deployment
manager.activate_deployment(deployment.deployment_id)

# List deployments
deployments = manager.list_deployments(environment="production")
```

### Audit Logging

```python
from neural.mlops.audit import AuditLogger, EventType, EventSeverity
from datetime import datetime, timedelta

logger = AuditLogger("./audit_logs")

# Log deployment
logger.log_model_deployment(
    model_name="my_model",
    version="v1.0.0",
    user="user@company.com",
    environment="production"
)

# Log security violation
logger.log_security_violation(
    user="unknown@external.com",
    resource_type="model",
    resource_id="my_model:v1.0.0",
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
    deploy = deployment.create_deployment(**data)
    return jsonify(deploy.to_dict()), 201

if __name__ == "__main__":
    app.run(debug=True)
```

## Testing Examples

### Unit Tests

```python
import unittest
from neural.mlops.registry import ModelRegistry

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

## Documentation

- [Complete MLOps Documentation](../../docs/mlops/README.md)
- [Feature Summary](../../docs/mlops/MLOPS_FEATURES.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/Lemniscate-world/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lemniscate-world/Neural/discussions)
