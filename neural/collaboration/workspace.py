"""
Workspace Management - Shared workspace management with access controls.

Manages collaborative workspaces where multiple users can edit Neural DSL files.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from neural.exceptions import WorkspaceError, AccessControlError


class Workspace:
    """
    Represents a collaborative workspace.
    
    A workspace contains Neural DSL files that can be edited by multiple users
    with appropriate permissions.
    """
    
    def __init__(
        self,
        workspace_id: str,
        name: str,
        owner: str,
        workspace_dir: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Unique workspace identifier
        name : str
            Workspace name
        owner : str
            Workspace owner user ID
        workspace_dir : Path
            Directory containing workspace files
        metadata : Optional[Dict]
            Additional workspace metadata
        """
        self.workspace_id = workspace_id
        self.name = name
        self.owner = owner
        self.workspace_dir = workspace_dir
        self.metadata = metadata or {}
        self.members: Set[str] = {owner}
        self.files: Dict[str, str] = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = self.created_at.isoformat()
    
    def add_member(self, user_id: str, role: str = 'member'):
        """
        Add a member to the workspace.
        
        Parameters
        ----------
        user_id : str
            User identifier
        role : str
            User role (owner, admin, member, viewer)
        """
        self.members.add(user_id)
        if 'roles' not in self.metadata:
            self.metadata['roles'] = {}
        self.metadata['roles'][user_id] = role
        self.updated_at = datetime.utcnow()
    
    def remove_member(self, user_id: str):
        """
        Remove a member from the workspace.
        
        Parameters
        ----------
        user_id : str
            User identifier
        """
        if user_id == self.owner:
            raise WorkspaceError("Cannot remove workspace owner")
        
        self.members.discard(user_id)
        if 'roles' in self.metadata and user_id in self.metadata['roles']:
            del self.metadata['roles'][user_id]
        self.updated_at = datetime.utcnow()
    
    def has_member(self, user_id: str) -> bool:
        """
        Check if a user is a member of the workspace.
        
        Parameters
        ----------
        user_id : str
            User identifier
            
        Returns
        -------
        bool
            True if user is a member
        """
        return user_id in self.members
    
    def get_role(self, user_id: str) -> Optional[str]:
        """
        Get a user's role in the workspace.
        
        Parameters
        ----------
        user_id : str
            User identifier
            
        Returns
        -------
        Optional[str]
            User role or None if not a member
        """
        if not self.has_member(user_id):
            return None
        
        if user_id == self.owner:
            return 'owner'
        
        return self.metadata.get('roles', {}).get(user_id, 'member')
    
    def add_file(self, filename: str, file_path: str):
        """
        Add a file to the workspace.
        
        Parameters
        ----------
        filename : str
            File name
        file_path : str
            Path to file
        """
        self.files[filename] = file_path
        self.updated_at = datetime.utcnow()
    
    def remove_file(self, filename: str):
        """
        Remove a file from the workspace.
        
        Parameters
        ----------
        filename : str
            File name
        """
        if filename in self.files:
            del self.files[filename]
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """
        Convert workspace to dictionary.
        
        Returns
        -------
        Dict
            Workspace data as dictionary
        """
        return {
            'workspace_id': self.workspace_id,
            'name': self.name,
            'owner': self.owner,
            'members': list(self.members),
            'files': self.files,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict, workspace_dir: Path) -> 'Workspace':
        """
        Create workspace from dictionary.
        
        Parameters
        ----------
        data : Dict
            Workspace data dictionary
        workspace_dir : Path
            Workspace directory
            
        Returns
        -------
        Workspace
            Workspace instance
        """
        workspace = cls(
            workspace_id=data['workspace_id'],
            name=data['name'],
            owner=data['owner'],
            workspace_dir=workspace_dir,
            metadata=data.get('metadata', {})
        )
        workspace.members = set(data.get('members', [workspace.owner]))
        workspace.files = data.get('files', {})
        
        if 'created_at' in data:
            workspace.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            workspace.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return workspace


class WorkspaceManager:
    """
    Manages collaborative workspaces.
    
    Handles workspace creation, deletion, and access control.
    """
    
    def __init__(self, base_dir: str = 'neural_workspaces'):
        """
        Initialize workspace manager.
        
        Parameters
        ----------
        base_dir : str
            Base directory for workspaces
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces: Dict[str, Workspace] = {}
        self._load_workspaces()
    
    def _load_workspaces(self):
        """Load existing workspaces from disk."""
        if not self.base_dir.exists():
            return
        
        for workspace_dir in self.base_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            metadata_file = workspace_dir / 'workspace.json'
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                workspace = Workspace.from_dict(data, workspace_dir)
                self.workspaces[workspace.workspace_id] = workspace
            except Exception as e:
                print(f"Warning: Failed to load workspace {workspace_dir}: {e}")
    
    def create_workspace(
        self,
        name: str,
        owner: str,
        description: Optional[str] = None
    ) -> Workspace:
        """
        Create a new workspace.
        
        Parameters
        ----------
        name : str
            Workspace name
        owner : str
            Owner user ID
        description : Optional[str]
            Workspace description
            
        Returns
        -------
        Workspace
            Created workspace
        """
        workspace_id = str(uuid.uuid4())
        workspace_dir = self.base_dir / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {}
        if description:
            metadata['description'] = description
        
        workspace = Workspace(workspace_id, name, owner, workspace_dir, metadata)
        self.workspaces[workspace_id] = workspace
        
        self._save_workspace(workspace)
        
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get a workspace by ID.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
            
        Returns
        -------
        Optional[Workspace]
            Workspace or None if not found
        """
        return self.workspaces.get(workspace_id)
    
    def delete_workspace(self, workspace_id: str, user_id: str):
        """
        Delete a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User requesting deletion
        """
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            raise WorkspaceError(f"Workspace not found: {workspace_id}")
        
        if workspace.owner != user_id:
            raise AccessControlError("Only workspace owner can delete the workspace")
        
        import shutil
        if workspace.workspace_dir.exists():
            shutil.rmtree(workspace.workspace_dir)
        
        del self.workspaces[workspace_id]
    
    def list_workspaces(self, user_id: Optional[str] = None) -> List[Workspace]:
        """
        List workspaces.
        
        Parameters
        ----------
        user_id : Optional[str]
            Filter by user ID (show only workspaces user has access to)
            
        Returns
        -------
        List[Workspace]
            List of workspaces
        """
        if user_id:
            return [
                ws for ws in self.workspaces.values()
                if ws.has_member(user_id)
            ]
        return list(self.workspaces.values())
    
    def _save_workspace(self, workspace: Workspace):
        """
        Save workspace metadata to disk.
        
        Parameters
        ----------
        workspace : Workspace
            Workspace to save
        """
        metadata_file = workspace.workspace_dir / 'workspace.json'
        with open(metadata_file, 'w') as f:
            json.dump(workspace.to_dict(), f, indent=2)
    
    def update_workspace(self, workspace: Workspace):
        """
        Update workspace metadata.
        
        Parameters
        ----------
        workspace : Workspace
            Workspace to update
        """
        workspace.updated_at = datetime.utcnow()
        self._save_workspace(workspace)
