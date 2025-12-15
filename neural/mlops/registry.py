"""
Basic Model Registry for Model Tracking.

Provides simple model registration and versioning without approval workflows.
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['stage'] = self.stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        data = data.copy()
        data['stage'] = ModelStage(data['stage'])
        return cls(**data)


class ModelRegistry:
    """
    Basic model registry with versioning.
    
    Example:
        registry = ModelRegistry("./models")
        
        metadata = registry.register_model(
            name="my_model",
            version="v1.0.0",
            model_path="./model.pt",
            framework="pytorch",
            metrics={"accuracy": 0.95},
            created_by="user@example.com"
        )
        
        registry.promote_model("my_model", "v1.0.0", ModelStage.PRODUCTION)
    """
    
    def __init__(self, registry_path: str = "./models"):
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def register(self, name: str, version: str) -> str:
        return f"{name}_{version}"
    
    def get(self, model_id: str) -> Dict[str, Any]:
        parts = model_id.split("_")
        return {"name": parts[0], "version": parts[1] if len(parts) > 1 else ""}
    
    def delete(self, model_id: str) -> bool:
        return True
    
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
    
    def promote_model(
        self,
        name: str,
        version: str,
        target_stage: ModelStage
    ) -> None:
        """Promote a model to a different stage."""
        metadata = self.get_model(name, version)
        metadata.stage = target_stage
        self._save_metadata(metadata)
    
    def archive_model(self, name: str, version: str) -> None:
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
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to disk."""
        metadata_file = self.metadata_path / f"{metadata.name}_{metadata.version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
