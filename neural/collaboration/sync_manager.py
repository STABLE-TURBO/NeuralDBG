"""
Sync Manager - Manages synchronization between clients and workspace.

Coordinates file synchronization, change tracking, and conflict detection.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from neural.exceptions import SyncError
from .conflict_resolution import ConflictResolver, EditOperation


class FileVersion:
    """Represents a version of a file."""
    
    def __init__(self, content: str, version: int, timestamp: str, user_id: str):
        """
        Initialize file version.
        
        Parameters
        ----------
        content : str
            File content
        version : int
            Version number
        timestamp : str
            Version timestamp
        user_id : str
            User who created this version
        """
        self.content = content
        self.version = version
        self.timestamp = timestamp
        self.user_id = user_id
        self.hash = self._compute_hash(content)
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'version': self.version,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FileVersion':
        """Create from dictionary."""
        return cls(
            content=data['content'],
            version=data['version'],
            timestamp=data['timestamp'],
            user_id=data['user_id']
        )


class SyncManager:
    """
    Manages file synchronization in collaborative workspaces.
    
    Tracks file versions, detects changes, and coordinates synchronization
    between multiple clients.
    """
    
    def __init__(self, workspace_dir: Path):
        """
        Initialize sync manager.
        
        Parameters
        ----------
        workspace_dir : Path
            Workspace directory
        """
        self.workspace_dir = Path(workspace_dir)
        self.sync_dir = self.workspace_dir / '.neural_sync'
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_versions: Dict[str, List[FileVersion]] = {}
        self.active_locks: Dict[str, str] = {}
        self.conflict_resolver = ConflictResolver()
        
        self._load_sync_state()
    
    def _load_sync_state(self):
        """Load synchronization state from disk."""
        state_file = self.sync_dir / 'sync_state.json'
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            
            for filename, versions_data in data.get('file_versions', {}).items():
                self.file_versions[filename] = [
                    FileVersion.from_dict(v) for v in versions_data
                ]
            
            self.active_locks = data.get('active_locks', {})
        except Exception as e:
            print(f"Warning: Failed to load sync state: {e}")
    
    def _save_sync_state(self):
        """Save synchronization state to disk."""
        state_file = self.sync_dir / 'sync_state.json'
        
        data = {
            'file_versions': {
                filename: [v.to_dict() for v in versions]
                for filename, versions in self.file_versions.items()
            },
            'active_locks': self.active_locks
        }
        
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def track_file(self, filename: str, content: str, user_id: str) -> int:
        """
        Start tracking a file.
        
        Parameters
        ----------
        filename : str
            File name
        content : str
            File content
        user_id : str
            User ID
            
        Returns
        -------
        int
            Version number
        """
        if filename not in self.file_versions:
            self.file_versions[filename] = []
        
        version = len(self.file_versions[filename]) + 1
        file_version = FileVersion(
            content=content,
            version=version,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id
        )
        
        self.file_versions[filename].append(file_version)
        self._save_sync_state()
        
        return version
    
    def get_file_version(
        self,
        filename: str,
        version: Optional[int] = None
    ) -> Optional[FileVersion]:
        """
        Get a specific version of a file.
        
        Parameters
        ----------
        filename : str
            File name
        version : Optional[int]
            Version number (None for latest)
            
        Returns
        -------
        Optional[FileVersion]
            File version or None if not found
        """
        if filename not in self.file_versions:
            return None
        
        versions = self.file_versions[filename]
        if not versions:
            return None
        
        if version is None:
            return versions[-1]
        
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    def update_file(
        self,
        filename: str,
        content: str,
        user_id: str,
        base_version: int
    ) -> Tuple[int, Optional[str]]:
        """
        Update a file with conflict detection.
        
        Parameters
        ----------
        filename : str
            File name
        content : str
            New content
        user_id : str
            User ID
        base_version : int
            Base version for update
            
        Returns
        -------
        Tuple[int, Optional[str]]
            New version number and conflict message if any
        """
        current = self.get_file_version(filename)
        
        if not current:
            return self.track_file(filename, content, user_id), None
        
        if current.version == base_version:
            return self.track_file(filename, content, user_id), None
        
        base = self.get_file_version(filename, base_version)
        if not base:
            raise SyncError(f"Base version {base_version} not found for {filename}")
        
        merged_content, conflicts = self.conflict_resolver.three_way_merge(
            base.content,
            content,
            current.content
        )
        
        new_version = self.track_file(filename, merged_content, user_id)
        
        conflict_msg = None
        if conflicts:
            conflict_msg = f"Conflicts detected: {len(conflicts)} conflicts"
        
        return new_version, conflict_msg
    
    def acquire_lock(self, filename: str, user_id: str) -> bool:
        """
        Acquire an exclusive lock on a file.
        
        Parameters
        ----------
        filename : str
            File name
        user_id : str
            User ID
            
        Returns
        -------
        bool
            True if lock acquired
        """
        if filename in self.active_locks:
            return self.active_locks[filename] == user_id
        
        self.active_locks[filename] = user_id
        self._save_sync_state()
        return True
    
    def release_lock(self, filename: str, user_id: str) -> bool:
        """
        Release a file lock.
        
        Parameters
        ----------
        filename : str
            File name
        user_id : str
            User ID
            
        Returns
        -------
        bool
            True if lock released
        """
        if filename not in self.active_locks:
            return True
        
        if self.active_locks[filename] != user_id:
            return False
        
        del self.active_locks[filename]
        self._save_sync_state()
        return True
    
    def is_locked(self, filename: str) -> bool:
        """
        Check if a file is locked.
        
        Parameters
        ----------
        filename : str
            File name
            
        Returns
        -------
        bool
            True if file is locked
        """
        return filename in self.active_locks
    
    def get_lock_owner(self, filename: str) -> Optional[str]:
        """
        Get the owner of a file lock.
        
        Parameters
        ----------
        filename : str
            File name
            
        Returns
        -------
        Optional[str]
            User ID of lock owner or None
        """
        return self.active_locks.get(filename)
    
    def get_file_history(self, filename: str) -> List[FileVersion]:
        """
        Get version history of a file.
        
        Parameters
        ----------
        filename : str
            File name
            
        Returns
        -------
        List[FileVersion]
            List of file versions
        """
        return self.file_versions.get(filename, []).copy()
    
    def diff_versions(
        self,
        filename: str,
        version1: int,
        version2: int
    ) -> List[str]:
        """
        Get diff between two versions.
        
        Parameters
        ----------
        filename : str
            File name
        version1 : int
            First version
        version2 : int
            Second version
            
        Returns
        -------
        List[str]
            Diff lines
        """
        import difflib
        
        v1 = self.get_file_version(filename, version1)
        v2 = self.get_file_version(filename, version2)
        
        if not v1 or not v2:
            raise SyncError(f"Version not found for {filename}")
        
        diff = difflib.unified_diff(
            v1.content.splitlines(),
            v2.content.splitlines(),
            fromfile=f'{filename} (v{version1})',
            tofile=f'{filename} (v{version2})',
            lineterm=''
        )
        
        return list(diff)
    
    def get_changes_since(
        self,
        filename: str,
        version: int
    ) -> List[FileVersion]:
        """
        Get all changes since a specific version.
        
        Parameters
        ----------
        filename : str
            File name
        version : int
            Starting version
            
        Returns
        -------
        List[FileVersion]
            List of versions since specified version
        """
        if filename not in self.file_versions:
            return []
        
        return [
            v for v in self.file_versions[filename]
            if v.version > version
        ]
    
    def cleanup_old_versions(self, filename: str, keep_last: int = 10):
        """
        Clean up old versions of a file.
        
        Parameters
        ----------
        filename : str
            File name
        keep_last : int
            Number of recent versions to keep
        """
        if filename not in self.file_versions:
            return
        
        versions = self.file_versions[filename]
        if len(versions) <= keep_last:
            return
        
        self.file_versions[filename] = versions[-keep_last:]
        self._save_sync_state()
