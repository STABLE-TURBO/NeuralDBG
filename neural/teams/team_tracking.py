"""
Team-wide experiment tracking functionality.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .access_control import AccessController, Permission
from .models import Team


class TeamExperimentTracker:
    """Manages experiments shared within a team."""
    
    def __init__(self, base_dir: str = "neural_organizations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_team_experiments_dir(self, team_id: str) -> Path:
        """Get the experiments directory for a team."""
        exp_dir = self.base_dir / "experiments" / team_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def _get_experiments_file(self, team_id: str) -> Path:
        """Get the experiments metadata file for a team."""
        return self._get_team_experiments_dir(team_id) / "experiments.json"
    
    def _load_experiments(self, team_id: str) -> Dict[str, Any]:
        """Load experiments metadata for a team."""
        exp_file = self._get_experiments_file(team_id)
        if exp_file.exists():
            with open(exp_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_experiments(self, team_id: str, experiments: Dict[str, Any]) -> None:
        """Save experiments metadata for a team."""
        exp_file = self._get_experiments_file(team_id)
        with open(exp_file, 'w') as f:
            json.dump(experiments, f, indent=2)
    
    def create_experiment(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new experiment in the team."""
        AccessController.require_permission(user_id, team, Permission.CREATE_EXPERIMENTS)
        
        if not team.quota.check_quota('experiments'):
            raise ValueError(
                f"Team has reached the maximum number of experiments "
                f"({team.quota.max_experiments})"
            )
        
        experiments = self._load_experiments(team.team_id)
        
        if experiment_id in experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
        
        experiment_info = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'owner_id': user_id,
            'team_id': team.team_id,
            'tags': tags or [],
            'metadata': metadata or {},
            'status': 'created',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'start_time': None,
            'end_time': None,
            'hyperparameters': {},
            'metrics': {},
            'artifacts': [],
        }
        
        experiments[experiment_id] = experiment_info
        self._save_experiments(team.team_id, experiments)
        
        team.quota.current_experiments = len(experiments)
        
        exp_dir = self._get_team_experiments_dir(team.team_id) / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        return experiment_info
    
    def get_experiment(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get an experiment from the team."""
        AccessController.require_permission(user_id, team, Permission.VIEW_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        return experiments.get(experiment_id)
    
    def list_experiments(
        self,
        team: Team,
        user_id: str,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all experiments in the team."""
        AccessController.require_permission(user_id, team, Permission.VIEW_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        exp_list = list(experiments.values())
        
        if status:
            exp_list = [e for e in exp_list if e.get('status') == status]
        
        if tags:
            exp_list = [
                e for e in exp_list
                if any(tag in e.get('tags', []) for tag in tags)
            ]
        
        if owner_id:
            exp_list = [e for e in exp_list if e.get('owner_id') == owner_id]
        
        return sorted(exp_list, key=lambda x: x['created_at'], reverse=True)
    
    def update_experiment(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an experiment in the team."""
        AccessController.require_permission(user_id, team, Permission.EDIT_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        exp_info = experiments.get(experiment_id)
        
        if not exp_info:
            return None
        
        if name is not None:
            exp_info['name'] = name
        if description is not None:
            exp_info['description'] = description
        if status is not None:
            exp_info['status'] = status
            if status == 'running' and not exp_info['start_time']:
                exp_info['start_time'] = datetime.now().isoformat()
            elif status in ['completed', 'failed'] and not exp_info['end_time']:
                exp_info['end_time'] = datetime.now().isoformat()
        if tags is not None:
            exp_info['tags'] = tags
        if metadata is not None:
            exp_info['metadata'].update(metadata)
        
        exp_info['updated_at'] = datetime.now().isoformat()
        
        experiments[experiment_id] = exp_info
        self._save_experiments(team.team_id, experiments)
        
        return exp_info
    
    def log_hyperparameters(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
        hyperparameters: Dict[str, Any],
    ) -> None:
        """Log hyperparameters for an experiment."""
        AccessController.require_permission(user_id, team, Permission.EDIT_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        exp_info = experiments.get(experiment_id)
        
        if not exp_info:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp_info['hyperparameters'].update(hyperparameters)
        exp_info['updated_at'] = datetime.now().isoformat()
        
        experiments[experiment_id] = exp_info
        self._save_experiments(team.team_id, experiments)
    
    def log_metrics(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics for an experiment."""
        AccessController.require_permission(user_id, team, Permission.EDIT_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        exp_info = experiments.get(experiment_id)
        
        if not exp_info:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        for metric_name, value in metrics.items():
            if metric_name not in exp_info['metrics']:
                exp_info['metrics'][metric_name] = []
            
            exp_info['metrics'][metric_name].append({
                'step': step,
                'value': value,
                'timestamp': datetime.now().isoformat(),
            })
        
        exp_info['updated_at'] = datetime.now().isoformat()
        
        experiments[experiment_id] = exp_info
        self._save_experiments(team.team_id, experiments)
    
    def log_artifact(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
        artifact_path: str,
        artifact_name: Optional[str] = None,
    ) -> None:
        """Log an artifact for an experiment."""
        AccessController.require_permission(user_id, team, Permission.EDIT_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        exp_info = experiments.get(experiment_id)
        
        if not exp_info:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        artifact_info = {
            'path': artifact_path,
            'name': artifact_name or Path(artifact_path).name,
            'logged_at': datetime.now().isoformat(),
        }
        
        exp_info['artifacts'].append(artifact_info)
        exp_info['updated_at'] = datetime.now().isoformat()
        
        experiments[experiment_id] = exp_info
        self._save_experiments(team.team_id, experiments)
    
    def delete_experiment(
        self,
        team: Team,
        user_id: str,
        experiment_id: str,
    ) -> bool:
        """Delete an experiment from the team."""
        AccessController.require_permission(user_id, team, Permission.DELETE_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        
        if experiment_id not in experiments:
            return False
        
        del experiments[experiment_id]
        self._save_experiments(team.team_id, experiments)
        
        team.quota.current_experiments = len(experiments)
        
        return True
    
    def get_experiment_summary(
        self,
        team: Team,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get a summary of all experiments in the team."""
        AccessController.require_permission(user_id, team, Permission.VIEW_EXPERIMENTS)
        
        experiments = self._load_experiments(team.team_id)
        
        total = len(experiments)
        by_status = {}
        by_owner = {}
        
        for exp in experiments.values():
            status = exp.get('status', 'unknown')
            by_status[status] = by_status.get(status, 0) + 1
            
            owner = exp.get('owner_id', 'unknown')
            by_owner[owner] = by_owner.get(owner, 0) + 1
        
        return {
            'total_experiments': total,
            'by_status': by_status,
            'by_owner': by_owner,
            'quota_usage': team.quota.usage_percentage('experiments'),
        }
