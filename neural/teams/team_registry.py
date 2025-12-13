"""
Team-wide model registry for shared models.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .access_control import AccessController, Permission
from .models import Team


class TeamModelRegistry:
    """Manages models shared within a team."""
    
    def __init__(self, base_dir: str = "neural_organizations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_team_registry_dir(self, team_id: str) -> Path:
        """Get the registry directory for a team."""
        registry_dir = self.base_dir / "registries" / team_id
        registry_dir.mkdir(parents=True, exist_ok=True)
        return registry_dir
    
    def _get_registry_file(self, team_id: str) -> Path:
        """Get the registry metadata file for a team."""
        return self._get_team_registry_dir(team_id) / "models.json"
    
    def _load_registry(self, team_id: str) -> Dict[str, Any]:
        """Load registry metadata for a team."""
        registry_file = self._get_registry_file(team_id)
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self, team_id: str, registry: Dict[str, Any]) -> None:
        """Save registry metadata for a team."""
        registry_file = self._get_registry_file(team_id)
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def register_model(
        self,
        team: Team,
        user_id: str,
        model_id: str,
        name: str,
        model_path: str,
        description: str = "",
        version: str = "1.0.0",
        framework: str = "neural-dsl",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a model in the team registry."""
        AccessController.require_permission(user_id, team, Permission.CREATE_MODELS)
        
        if not team.quota.check_quota('models'):
            raise ValueError(
                f"Team has reached the maximum number of models "
                f"({team.quota.max_models})"
            )
        
        registry = self._load_registry(team.team_id)
        
        if model_id in registry:
            raise ValueError(f"Model {model_id} already exists in registry")
        
        registry_dir = self._get_team_registry_dir(team.team_id)
        model_storage_path = registry_dir / model_id
        model_storage_path.mkdir(parents=True, exist_ok=True)
        
        if Path(model_path).exists():
            dest_path = model_storage_path / Path(model_path).name
            shutil.copy2(model_path, dest_path)
            stored_path = str(dest_path)
        else:
            stored_path = model_path
        
        model_info = {
            'model_id': model_id,
            'name': name,
            'description': description,
            'version': version,
            'framework': framework,
            'path': stored_path,
            'owner_id': user_id,
            'team_id': team.team_id,
            'tags': tags or [],
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'downloads': 0,
            'views': 0,
        }
        
        registry[model_id] = model_info
        self._save_registry(team.team_id, registry)
        
        team.quota.current_models = len(registry)
        
        return model_info
    
    def get_model(
        self,
        team: Team,
        user_id: str,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a model from the team registry."""
        AccessController.require_permission(user_id, team, Permission.VIEW_MODELS)
        
        registry = self._load_registry(team.team_id)
        model_info = registry.get(model_id)
        
        if model_info:
            model_info['views'] = model_info.get('views', 0) + 1
            self._save_registry(team.team_id, registry)
        
        return model_info
    
    def list_models(
        self,
        team: Team,
        user_id: str,
        tags: Optional[List[str]] = None,
        framework: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all models in the team registry."""
        AccessController.require_permission(user_id, team, Permission.VIEW_MODELS)
        
        registry = self._load_registry(team.team_id)
        models = list(registry.values())
        
        if tags:
            models = [
                m for m in models
                if any(tag in m.get('tags', []) for tag in tags)
            ]
        
        if framework:
            models = [m for m in models if m.get('framework') == framework]
        
        if owner_id:
            models = [m for m in models if m.get('owner_id') == owner_id]
        
        return sorted(models, key=lambda x: x['registered_at'], reverse=True)
    
    def update_model(
        self,
        team: Team,
        user_id: str,
        model_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a model in the team registry."""
        AccessController.require_permission(user_id, team, Permission.EDIT_MODELS)
        
        registry = self._load_registry(team.team_id)
        model_info = registry.get(model_id)
        
        if not model_info:
            return None
        
        if name is not None:
            model_info['name'] = name
        if description is not None:
            model_info['description'] = description
        if version is not None:
            model_info['version'] = version
        if tags is not None:
            model_info['tags'] = tags
        if metadata is not None:
            model_info['metadata'].update(metadata)
        
        model_info['updated_at'] = datetime.now().isoformat()
        
        registry[model_id] = model_info
        self._save_registry(team.team_id, registry)
        
        return model_info
    
    def delete_model(
        self,
        team: Team,
        user_id: str,
        model_id: str,
    ) -> bool:
        """Delete a model from the team registry."""
        AccessController.require_permission(user_id, team, Permission.DELETE_MODELS)
        
        registry = self._load_registry(team.team_id)
        
        if model_id not in registry:
            return False
        
        model_storage_path = self._get_team_registry_dir(team.team_id) / model_id
        if model_storage_path.exists():
            shutil.rmtree(model_storage_path)
        
        del registry[model_id]
        self._save_registry(team.team_id, registry)
        
        team.quota.current_models = len(registry)
        
        return True
    
    def download_model(
        self,
        team: Team,
        user_id: str,
        model_id: str,
        output_dir: str = ".",
    ) -> str:
        """Download a model from the team registry."""
        AccessController.require_permission(user_id, team, Permission.VIEW_MODELS)
        
        registry = self._load_registry(team.team_id)
        model_info = registry.get(model_id)
        
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        source_path = Path(model_info['path'])
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {source_path}")
        
        output_path = Path(output_dir) / source_path.name
        shutil.copy2(source_path, output_path)
        
        model_info['downloads'] = model_info.get('downloads', 0) + 1
        registry[model_id] = model_info
        self._save_registry(team.team_id, registry)
        
        return str(output_path)
