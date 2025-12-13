"""
Access Control - Manages permissions and authentication for collaborative workspaces.

Implements role-based access control (RBAC) for workspace permissions.
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

from neural.exceptions import AccessControlError


class Permission(Enum):
    """Workspace permissions."""
    READ = 'read'
    WRITE = 'write'
    ADMIN = 'admin'
    OWNER = 'owner'


class Role(Enum):
    """User roles in workspace."""
    VIEWER = 'viewer'
    MEMBER = 'member'
    ADMIN = 'admin'
    OWNER = 'owner'


ROLE_PERMISSIONS = {
    Role.VIEWER: {Permission.READ},
    Role.MEMBER: {Permission.READ, Permission.WRITE},
    Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.ADMIN},
    Role.OWNER: {Permission.READ, Permission.WRITE, Permission.ADMIN, Permission.OWNER}
}


@dataclass
class AccessToken:
    """Access token for workspace authentication."""
    token: str
    user_id: str
    workspace_id: str
    created_at: datetime
    expires_at: datetime
    
    def is_valid(self) -> bool:
        """Check if token is still valid."""
        return datetime.utcnow() < self.expires_at


class AccessController:
    """
    Manages access control for collaborative workspaces.
    
    Implements role-based access control with token-based authentication.
    """
    
    def __init__(self):
        """Initialize access controller."""
        self.workspace_roles: Dict[str, Dict[str, Role]] = {}
        self.tokens: Dict[str, AccessToken] = {}
    
    def set_role(self, workspace_id: str, user_id: str, role: Role):
        """
        Set a user's role in a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
        role : Role
            Role to assign
        """
        if workspace_id not in self.workspace_roles:
            self.workspace_roles[workspace_id] = {}
        
        self.workspace_roles[workspace_id][user_id] = role
    
    def get_role(self, workspace_id: str, user_id: str) -> Optional[Role]:
        """
        Get a user's role in a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
            
        Returns
        -------
        Optional[Role]
            User's role or None if not found
        """
        return self.workspace_roles.get(workspace_id, {}).get(user_id)
    
    def has_permission(
        self,
        workspace_id: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if a user has a specific permission.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
        permission : Permission
            Permission to check
            
        Returns
        -------
        bool
            True if user has permission
        """
        role = self.get_role(workspace_id, user_id)
        if not role:
            return False
        
        return permission in ROLE_PERMISSIONS[role]
    
    def require_permission(
        self,
        workspace_id: str,
        user_id: str,
        permission: Permission
    ):
        """
        Require a user to have a specific permission.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
        permission : Permission
            Required permission
            
        Raises
        ------
        AccessControlError
            If user does not have permission
        """
        if not self.has_permission(workspace_id, user_id, permission):
            raise AccessControlError(
                f"User {user_id} does not have {permission.value} permission "
                f"in workspace {workspace_id}"
            )
    
    def create_token(
        self,
        user_id: str,
        workspace_id: str,
        expires_in: int = 3600
    ) -> str:
        """
        Create an access token.
        
        Parameters
        ----------
        user_id : str
            User identifier
        workspace_id : str
            Workspace identifier
        expires_in : int
            Token expiration time in seconds
            
        Returns
        -------
        str
            Access token
        """
        token = secrets.token_urlsafe(32)
        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=expires_in)
        
        access_token = AccessToken(
            token=token,
            user_id=user_id,
            workspace_id=workspace_id,
            created_at=created_at,
            expires_at=expires_at
        )
        
        self.tokens[token] = access_token
        
        return token
    
    def verify_token(self, token: str) -> Optional[AccessToken]:
        """
        Verify an access token.
        
        Parameters
        ----------
        token : str
            Access token to verify
            
        Returns
        -------
        Optional[AccessToken]
            Access token if valid, None otherwise
        """
        access_token = self.tokens.get(token)
        if not access_token or not access_token.is_valid():
            return None
        
        return access_token
    
    def revoke_token(self, token: str):
        """
        Revoke an access token.
        
        Parameters
        ----------
        token : str
            Token to revoke
        """
        if token in self.tokens:
            del self.tokens[token]
    
    def revoke_user_tokens(self, user_id: str):
        """
        Revoke all tokens for a user.
        
        Parameters
        ----------
        user_id : str
            User identifier
        """
        tokens_to_revoke = [
            token for token, access_token in self.tokens.items()
            if access_token.user_id == user_id
        ]
        
        for token in tokens_to_revoke:
            del self.tokens[token]
    
    def cleanup_expired_tokens(self):
        """Remove expired tokens."""
        now = datetime.utcnow()
        expired_tokens = [
            token for token, access_token in self.tokens.items()
            if access_token.expires_at < now
        ]
        
        for token in expired_tokens:
            del self.tokens[token]
    
    def get_workspace_users(self, workspace_id: str) -> List[str]:
        """
        Get list of users with access to a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
            
        Returns
        -------
        List[str]
            List of user IDs
        """
        return list(self.workspace_roles.get(workspace_id, {}).keys())
    
    def remove_user_from_workspace(self, workspace_id: str, user_id: str):
        """
        Remove a user from a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
        """
        if workspace_id in self.workspace_roles:
            if user_id in self.workspace_roles[workspace_id]:
                del self.workspace_roles[workspace_id][user_id]
        
        self.revoke_user_tokens(user_id)
