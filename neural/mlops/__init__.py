"""
Neural DSL MLOps Module - Enterprise ML Operations Support.

This module provides comprehensive MLOps capabilities including:
- Model registry with approval workflows
- A/B testing and shadow deployment
- Automated rollback on performance degradation
- Audit logging for compliance
- CI/CD pipeline templates

Example Usage:
    # Model Registry
    from neural.mlops.registry import ModelRegistry
    registry = ModelRegistry("./models")
    registry.register_model("my_model", "v1.0.0", {"accuracy": 0.95})
    
    # A/B Testing
    from neural.mlops.ab_testing import ABTestManager
    ab_manager = ABTestManager()
    ab_manager.create_test("model_v1", "model_v2", traffic_split=0.1)
    
    # Shadow Deployment
    from neural.mlops.deployment import DeploymentManager
    deploy_mgr = DeploymentManager()
    deploy_mgr.shadow_deploy("model_v2", primary_model="model_v1")
    
    # Audit Logging
    from neural.mlops.audit import AuditLogger
    logger = AuditLogger()
    logger.log_deployment("model_v1", user="admin", status="success")
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .ab_testing import ABTest, ABTestManager, TrafficSplitter
    from .audit import AuditEvent, AuditLogger, ComplianceReport
    from .ci_templates import CITemplateGenerator
    from .deployment import DeploymentManager, RollbackManager, ShadowDeployment
    from .registry import ApprovalWorkflow, ModelMetadata, ModelRegistry

__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "ApprovalWorkflow",
    "ABTestManager",
    "ABTest",
    "TrafficSplitter",
    "DeploymentManager",
    "ShadowDeployment",
    "RollbackManager",
    "AuditLogger",
    "AuditEvent",
    "ComplianceReport",
    "CITemplateGenerator",
]
