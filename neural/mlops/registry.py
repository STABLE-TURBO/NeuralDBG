"""
Model Registry with Approval Workflows.

Provides versioned model storage, metadata tracking, and multi-stage
approval workflows for enterprise ML governance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import yaml


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ApprovalStatus(Enum):
    """Approval workflow status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    version: str
    stage: ModelStage
    created_at: str
    created_by: str
    framework: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    model_path: str = ""
    config_path: str = ""
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['stage'] = self.stage.value
        data['approval_status'] = self.approval_status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        data = data.copy()
        data['stage'] = ModelStage(data['stage'])
        data['approval_status'] = ApprovalStatus(data['approval_status'])
        return cls(**data)


@dataclass
class ApprovalRequest:
    """Model approval request."""
    model_name: str
    version: str
    requested_by: str
    requested_at: str
    target_stage: ModelStage
    justification: str
    reviewers: List[str] = field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    comments: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['target_stage'] = self.target_stage.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApprovalRequest:
        """Create from dictionary."""
        data = data.copy()
        data['target_stage'] = ModelStage(data['target_stage'])
        data['status'] = ApprovalStatus(data['status'])
        return cls(**data)


class ApprovalWorkflow:
    """Manages model approval workflows."""
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.approvals_path = self.registry_path / "approvals"
        self.approvals_path.mkdir(parents=True, exist_ok=True)
    
    def create_approval_request(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
        requested_by: str,
        justification: str,
        reviewers: Optional[List[str]] = None
    ) -> ApprovalRequest:
        """Create a new approval request."""
        request = ApprovalRequest(
            model_name=model_name,
            version=version,
            requested_by=requested_by,
            requested_at=datetime.now().isoformat(),
            target_stage=target_stage,
            justification=justification,
            reviewers=reviewers or [],
        )
        
        self._save_approval_request(request)
        return request
    
    def approve_request(
        self,
        model_name: str,
        version: str,
        approver: str,
        comment: Optional[str] = None
    ) -> None:
        """Approve a model promotion request."""
        request = self.get_approval_request(model_name, version)
        request.status = ApprovalStatus.APPROVED
        request.comments.append({
            "user": approver,
            "timestamp": datetime.now().isoformat(),
            "action": "approved",
            "comment": comment or "Approved"
        })
        self._save_approval_request(request)
    
    def reject_request(
        self,
        model_name: str,
        version: str,
        reviewer: str,
        reason: str
    ) -> None:
        """Reject a model promotion request."""
        request = self.get_approval_request(model_name, version)
        request.status = ApprovalStatus.REJECTED
        request.comments.append({
            "user": reviewer,
            "timestamp": datetime.now().isoformat(),
            "action": "rejected",
            "comment": reason
        })
        self._save_approval_request(request)
    
    def get_approval_request(self, model_name: str, version: str) -> ApprovalRequest:
        """Get approval request for a model version."""
        request_path = self.approvals_path / f"{model_name}_{version}.json"
        if not request_path.exists():
            raise FileNotFoundError(f"No approval request found for {model_name} v{version}")
        
        with open(request_path, 'r') as f:
            data = json.load(f)
        return ApprovalRequest.from_dict(data)
    
    def list_pending_approvals(self) -> List[ApprovalRequest]:
        """List all pending approval requests."""
        pending = []
        for request_file in self.approvals_path.glob("*.json"):
            with open(request_file, 'r') as f:
                data = json.load(f)
            request = ApprovalRequest.from_dict(data)
            if request.status == ApprovalStatus.PENDING:
                pending.append(request)
        return pending
    
    def _save_approval_request(self, request: ApprovalRequest) -> None:
        """Save approval request to disk."""
        request_path = self.approvals_path / f"{request.model_name}_{request.version}.json"
        with open(request_path, 'w') as f:
            json.dump(request.to_dict(), f, indent=2)


class ModelRegistry:
    """
    Enterprise model registry with versioning and approval workflows.
    
    Provides centralized model storage, metadata tracking, and lifecycle
    management with approval workflows for production deployment.
    
    Example:
        registry = ModelRegistry("./models")
        
        # Register a new model
        metadata = registry.register_model(
            name="fraud_detector",
            version="v1.0.0",
            model_path="./model.pt",
            framework="pytorch",
            metrics={"accuracy": 0.95, "f1": 0.93},
            created_by="data_scientist@company.com"
        )
        
        # Request production promotion
        registry.request_promotion(
            name="fraud_detector",
            version="v1.0.0",
            target_stage=ModelStage.PRODUCTION,
            requested_by="ml_engineer@company.com",
            justification="Model passed all validation tests"
        )
        
        # Approve and promote
        registry.approve_promotion("fraud_detector", "v1.0.0", "ml_manager@company.com")
    """
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        self.approval_workflow = ApprovalWorkflow(str(self.registry_path))
    
    def register_model(
        self,
        name: str,
        version: str,
        model_path: str,
        framework: str,
        created_by: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> ModelMetadata:
        """Register a new model version."""
        model_dir = self.models_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        target_model_path = model_dir / Path(model_path).name
        if Path(model_path).exists():
            shutil.copy2(model_path, target_model_path)
        
        config_path = ""
        if config:
            config_path = str(model_dir / "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        
        metadata = ModelMetadata(
            name=name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            framework=framework,
            metrics=metrics or {},
            tags=tags or [],
            description=description,
            model_path=str(target_model_path),
            config_path=config_path,
        )
        
        self._save_metadata(metadata)
        return metadata
    
    def get_model(self, name: str, version: Optional[str] = None) -> ModelMetadata:
        """Get model metadata by name and version."""
        if version is None:
            version = self.get_latest_version(name)
        
        metadata_file = self.metadata_path / f"{name}_{version}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Model {name} version {version} not found")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        return ModelMetadata.from_dict(data)
    
    def list_models(
        self,
        stage: Optional[ModelStage] = None,
        name_filter: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List all registered models with optional filtering."""
        models = []
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            metadata = ModelMetadata.from_dict(data)
            
            if stage and metadata.stage != stage:
                continue
            if name_filter and name_filter not in metadata.name:
                continue
            
            models.append(metadata)
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def get_latest_version(self, name: str) -> str:
        """Get the latest version of a model."""
        versions = []
        for metadata_file in self.metadata_path.glob(f"{name}_*.json"):
            version = metadata_file.stem.split('_', 1)[1]
            versions.append(version)
        
        if not versions:
            raise FileNotFoundError(f"No versions found for model {name}")
        
        return sorted(versions)[-1]
    
    def request_promotion(
        self,
        name: str,
        version: str,
        target_stage: ModelStage,
        requested_by: str,
        justification: str,
        reviewers: Optional[List[str]] = None
    ) -> ApprovalRequest:
        """Request promotion of a model to a higher stage."""
        metadata = self.get_model(name, version)
        
        if target_stage == ModelStage.PRODUCTION:
            if metadata.stage not in [ModelStage.STAGING]:
                raise ValueError("Model must be in STAGING before promoting to PRODUCTION")
        
        return self.approval_workflow.create_approval_request(
            model_name=name,
            version=version,
            target_stage=target_stage,
            requested_by=requested_by,
            justification=justification,
            reviewers=reviewers
        )
    
    def approve_promotion(
        self,
        name: str,
        version: str,
        approver: str,
        comment: Optional[str] = None
    ) -> None:
        """Approve and execute model promotion."""
        self.approval_workflow.approve_request(name, version, approver, comment)
        
        request = self.approval_workflow.get_approval_request(name, version)
        if request.status == ApprovalStatus.APPROVED:
            self._promote_model(name, version, request.target_stage, approver)
    
    def reject_promotion(
        self,
        name: str,
        version: str,
        reviewer: str,
        reason: str
    ) -> None:
        """Reject model promotion request."""
        self.approval_workflow.reject_request(name, version, reviewer, reason)
    
    def archive_model(self, name: str, version: str, archived_by: str) -> None:
        """Archive a model version."""
        metadata = self.get_model(name, version)
        metadata.stage = ModelStage.ARCHIVED
        self._save_metadata(metadata)
    
    def compare_models(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare metrics between two model versions."""
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {},
            "stage_comparison": {
                "version1": model1.stage.value,
                "version2": model2.stage.value
            }
        }
        
        all_metrics = set(model1.metrics.keys()) | set(model2.metrics.keys())
        for metric in all_metrics:
            val1 = model1.metrics.get(metric)
            val2 = model2.metrics.get(metric)
            diff = None
            if val1 is not None and val2 is not None:
                diff = val2 - val1
            
            comparison["metrics_comparison"][metric] = {
                "version1": val1,
                "version2": val2,
                "difference": diff
            }
        
        return comparison
    
    def _promote_model(
        self,
        name: str,
        version: str,
        target_stage: ModelStage,
        promoted_by: str
    ) -> None:
        """Internal method to promote model stage."""
        metadata = self.get_model(name, version)
        metadata.stage = target_stage
        metadata.approved_by = promoted_by
        metadata.approved_at = datetime.now().isoformat()
        metadata.approval_status = ApprovalStatus.APPROVED
        self._save_metadata(metadata)
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to disk."""
        metadata_file = self.metadata_path / f"{metadata.name}_{metadata.version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
