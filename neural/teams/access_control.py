"""
Role-based access control (RBAC) implementation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .models import Role, Team, User


class Permission(str, Enum):
    """System permissions."""
    VIEW_MODELS = "view_models"
    CREATE_MODELS = "create_models"
    EDIT_MODELS = "edit_models"
    DELETE_MODELS = "delete_models"
    
    VIEW_EXPERIMENTS = "view_experiments"
    CREATE_EXPERIMENTS = "create_experiments"
    EDIT_EXPERIMENTS = "edit_experiments"
    DELETE_EXPERIMENTS = "delete_experiments"
    
    VIEW_TEAM = "view_team"
    MANAGE_TEAM = "manage_team"
    INVITE_MEMBERS = "invite_members"
    REMOVE_MEMBERS = "remove_members"
    
    VIEW_BILLING = "view_billing"
    MANAGE_BILLING = "manage_billing"
    
    VIEW_SETTINGS = "view_settings"
    MANAGE_SETTINGS = "manage_settings"
    
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_QUOTAS = "manage_quotas"


class AccessController:
    """Controls access based on roles and permissions."""
    
    ROLE_PERMISSIONS = {
        Role.VIEWER: {
            Permission.VIEW_MODELS,
            Permission.VIEW_EXPERIMENTS,
            Permission.VIEW_TEAM,
            Permission.VIEW_ANALYTICS,
        },
        Role.DEVELOPER: {
            Permission.VIEW_MODELS,
            Permission.CREATE_MODELS,
            Permission.EDIT_MODELS,
            Permission.VIEW_EXPERIMENTS,
            Permission.CREATE_EXPERIMENTS,
            Permission.EDIT_EXPERIMENTS,
            Permission.VIEW_TEAM,
            Permission.VIEW_ANALYTICS,
        },
        Role.ADMIN: {
            Permission.VIEW_MODELS,
            Permission.CREATE_MODELS,
            Permission.EDIT_MODELS,
            Permission.DELETE_MODELS,
            Permission.VIEW_EXPERIMENTS,
            Permission.CREATE_EXPERIMENTS,
            Permission.EDIT_EXPERIMENTS,
            Permission.DELETE_EXPERIMENTS,
            Permission.VIEW_TEAM,
            Permission.MANAGE_TEAM,
            Permission.INVITE_MEMBERS,
            Permission.REMOVE_MEMBERS,
            Permission.VIEW_BILLING,
            Permission.MANAGE_BILLING,
            Permission.VIEW_SETTINGS,
            Permission.MANAGE_SETTINGS,
            Permission.VIEW_ANALYTICS,
            Permission.MANAGE_QUOTAS,
        },
    }
    
    @classmethod
    def has_permission(
        cls,
        user_id: str,
        team: Team,
        permission: Permission,
    ) -> bool:
        """Check if a user has a specific permission in a team."""
        role = team.get_member_role(user_id)
        if not role:
            return False
        
        allowed_permissions = cls.ROLE_PERMISSIONS.get(role, set())
        return permission in allowed_permissions
    
    @classmethod
    def get_role_permissions(cls, role: Role) -> set[Permission]:
        """Get all permissions for a role."""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def can_view_resource(cls, user_id: str, team: Team, resource_type: str) -> bool:
        """Check if user can view a resource type."""
        permission_map = {
            'model': Permission.VIEW_MODELS,
            'experiment': Permission.VIEW_EXPERIMENTS,
            'team': Permission.VIEW_TEAM,
            'billing': Permission.VIEW_BILLING,
            'settings': Permission.VIEW_SETTINGS,
            'analytics': Permission.VIEW_ANALYTICS,
        }
        
        permission = permission_map.get(resource_type)
        if not permission:
            return False
        
        return cls.has_permission(user_id, team, permission)
    
    @classmethod
    def can_create_resource(cls, user_id: str, team: Team, resource_type: str) -> bool:
        """Check if user can create a resource type."""
        permission_map = {
            'model': Permission.CREATE_MODELS,
            'experiment': Permission.CREATE_EXPERIMENTS,
        }
        
        permission = permission_map.get(resource_type)
        if not permission:
            return False
        
        return cls.has_permission(user_id, team, permission)
    
    @classmethod
    def can_edit_resource(cls, user_id: str, team: Team, resource_type: str) -> bool:
        """Check if user can edit a resource type."""
        permission_map = {
            'model': Permission.EDIT_MODELS,
            'experiment': Permission.EDIT_EXPERIMENTS,
        }
        
        permission = permission_map.get(resource_type)
        if not permission:
            return False
        
        return cls.has_permission(user_id, team, permission)
    
    @classmethod
    def can_delete_resource(cls, user_id: str, team: Team, resource_type: str) -> bool:
        """Check if user can delete a resource type."""
        permission_map = {
            'model': Permission.DELETE_MODELS,
            'experiment': Permission.DELETE_EXPERIMENTS,
        }
        
        permission = permission_map.get(resource_type)
        if not permission:
            return False
        
        return cls.has_permission(user_id, team, permission)
    
    @classmethod
    def require_permission(
        cls,
        user_id: str,
        team: Team,
        permission: Permission,
    ) -> None:
        """Require a permission or raise an exception."""
        if not cls.has_permission(user_id, team, permission):
            role = team.get_member_role(user_id)
            raise PermissionError(
                f"User {user_id} with role {role.value if role else 'None'} "
                f"does not have permission {permission.value}"
            )
