"""
Neural DSL MLOps Module - Basic Model Operations Support.

This module provides essential MLOps capabilities including:
- Model registry with versioning
- Basic deployment tracking
- Audit logging for compliance
- CI/CD pipeline templates

Example Usage:
    # Model Registry
    from neural.mlops.registry import ModelRegistry
    registry = ModelRegistry("./models")
    registry.register_model("my_model", "v1.0.0", "./model.pt", "pytorch", "user@example.com")
    
    # Basic Deployment
    from neural.mlops.deployment import DeploymentManager
    deploy_mgr = DeploymentManager()
    deploy_mgr.create_deployment("my_model", "v1.0.0", environment="production")
    
    # Audit Logging
    from neural.mlops.audit import AuditLogger
    logger = AuditLogger()
    logger.log_deployment("model_v1", user="admin", status="success")
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .audit import AuditEvent, AuditLogger, ComplianceReport
    from .ci_templates import CITemplateGenerator
    from .deployment import DeploymentManager, Deployment, ModelDeployment
    from .registry import ModelMetadata, ModelRegistry

__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "DeploymentManager",
    "Deployment",
    "ModelDeployment",
    "AuditLogger",
    "AuditEvent",
    "ComplianceReport",
    "CITemplateGenerator",
]
