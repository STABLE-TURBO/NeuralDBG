"""
Core data models for multi-tenancy and team management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class Role(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class BillingPlan(str, Enum):
    """Billing plan types."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class ResourceQuota:
    """Resource quotas for a team."""
    max_models: int = 10
    max_experiments: int = 100
    max_storage_gb: float = 10.0
    max_compute_hours: float = 100.0
    max_team_members: int = 5
    max_api_calls_per_day: int = 10000
    max_concurrent_runs: int = 5
    
    current_models: int = 0
    current_experiments: int = 0
    current_storage_gb: float = 0.0
    current_compute_hours: float = 0.0
    current_team_members: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_models': self.max_models,
            'max_experiments': self.max_experiments,
            'max_storage_gb': self.max_storage_gb,
            'max_compute_hours': self.max_compute_hours,
            'max_team_members': self.max_team_members,
            'max_api_calls_per_day': self.max_api_calls_per_day,
            'max_concurrent_runs': self.max_concurrent_runs,
            'current_models': self.current_models,
            'current_experiments': self.current_experiments,
            'current_storage_gb': self.current_storage_gb,
            'current_compute_hours': self.current_compute_hours,
            'current_team_members': self.current_team_members,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResourceQuota:
        """Create from dictionary."""
        return cls(**data)
    
    def check_quota(self, resource: str) -> bool:
        """Check if resource quota is available."""
        if resource == 'models':
            return self.current_models < self.max_models
        elif resource == 'experiments':
            return self.current_experiments < self.max_experiments
        elif resource == 'storage':
            return self.current_storage_gb < self.max_storage_gb
        elif resource == 'compute':
            return self.current_compute_hours < self.max_compute_hours
        elif resource == 'team_members':
            return self.current_team_members < self.max_team_members
        return False
    
    def usage_percentage(self, resource: str) -> float:
        """Get usage percentage for a resource."""
        if resource == 'models':
            return (self.current_models / self.max_models) * 100 if self.max_models > 0 else 0
        elif resource == 'experiments':
            return (self.current_experiments / self.max_experiments) * 100 if self.max_experiments > 0 else 0
        elif resource == 'storage':
            return (self.current_storage_gb / self.max_storage_gb) * 100 if self.max_storage_gb > 0 else 0
        elif resource == 'compute':
            return (self.current_compute_hours / self.max_compute_hours) * 100 if self.max_compute_hours > 0 else 0
        elif resource == 'team_members':
            return (self.current_team_members / self.max_team_members) * 100 if self.max_team_members > 0 else 0
        return 0.0


@dataclass
class User:
    """User model."""
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    full_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> User:
        """Create from dictionary."""
        user = cls(
            user_id=data['user_id'],
            username=data['username'],
            email=data['email'],
            full_name=data['full_name'],
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data.get('is_active', True),
            metadata=data.get('metadata', {}),
        )
        if data.get('last_login'):
            user.last_login = datetime.fromisoformat(data['last_login'])
        return user


@dataclass
class Team:
    """Team model."""
    team_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    organization_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    members: Dict[str, Role] = field(default_factory=dict)
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'team_id': self.team_id,
            'name': self.name,
            'description': self.description,
            'organization_id': self.organization_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'members': {user_id: role.value for user_id, role in self.members.items()},
            'quota': self.quota.to_dict(),
            'settings': self.settings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Team:
        """Create from dictionary."""
        return cls(
            team_id=data['team_id'],
            name=data['name'],
            description=data['description'],
            organization_id=data['organization_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            members={user_id: Role(role) for user_id, role in data.get('members', {}).items()},
            quota=ResourceQuota.from_dict(data.get('quota', {})),
            settings=data.get('settings', {}),
        )
    
    def add_member(self, user_id: str, role: Role) -> bool:
        """Add a member to the team."""
        if self.quota.check_quota('team_members'):
            self.members[user_id] = role
            self.quota.current_team_members = len(self.members)
            self.updated_at = datetime.now()
            return True
        return False
    
    def remove_member(self, user_id: str) -> bool:
        """Remove a member from the team."""
        if user_id in self.members:
            del self.members[user_id]
            self.quota.current_team_members = len(self.members)
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_member_role(self, user_id: str) -> Optional[Role]:
        """Get the role of a member."""
        return self.members.get(user_id)
    
    def has_member(self, user_id: str) -> bool:
        """Check if user is a member of the team."""
        return user_id in self.members


@dataclass
class Organization:
    """Organization model for multi-tenancy."""
    org_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    slug: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner_id: str = ""
    billing_plan: BillingPlan = BillingPlan.FREE
    billing_email: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'org_id': self.org_id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'owner_id': self.owner_id,
            'billing_plan': self.billing_plan.value,
            'billing_email': self.billing_email,
            'settings': self.settings,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Organization:
        """Create from dictionary."""
        return cls(
            org_id=data['org_id'],
            name=data['name'],
            slug=data['slug'],
            description=data['description'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            owner_id=data['owner_id'],
            billing_plan=BillingPlan(data.get('billing_plan', 'free')),
            billing_email=data.get('billing_email', ''),
            settings=data.get('settings', {}),
            metadata=data.get('metadata', {}),
        )
    
    def get_plan_quotas(self) -> ResourceQuota:
        """Get resource quotas based on billing plan."""
        quotas = {
            BillingPlan.FREE: ResourceQuota(
                max_models=10,
                max_experiments=100,
                max_storage_gb=10.0,
                max_compute_hours=100.0,
                max_team_members=5,
                max_api_calls_per_day=10000,
                max_concurrent_runs=5,
            ),
            BillingPlan.STARTER: ResourceQuota(
                max_models=50,
                max_experiments=500,
                max_storage_gb=100.0,
                max_compute_hours=1000.0,
                max_team_members=10,
                max_api_calls_per_day=100000,
                max_concurrent_runs=10,
            ),
            BillingPlan.PROFESSIONAL: ResourceQuota(
                max_models=200,
                max_experiments=2000,
                max_storage_gb=500.0,
                max_compute_hours=5000.0,
                max_team_members=50,
                max_api_calls_per_day=1000000,
                max_concurrent_runs=25,
            ),
            BillingPlan.ENTERPRISE: ResourceQuota(
                max_models=999999,
                max_experiments=999999,
                max_storage_gb=99999.0,
                max_compute_hours=99999.0,
                max_team_members=999,
                max_api_calls_per_day=99999999,
                max_concurrent_runs=100,
            ),
        }
        return quotas.get(self.billing_plan, quotas[BillingPlan.FREE])
