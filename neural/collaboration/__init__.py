"""
Neural Collaboration Module - Real-time collaborative DSL editing.

This module provides infrastructure for real-time collaborative editing of Neural DSL files
with conflict resolution, access control, and version control integration.

.. deprecated:: 0.3.0
    The collaboration module will be removed in v0.4.0.
    Use Git and GitHub/GitLab for version control and collaboration instead.
    See docs/DEPRECATIONS.md for migration guide.

Features:
- Real-time DSL editing using WebSockets
- Conflict resolution for concurrent edits
- Shared workspace management with access controls
- Version control integration (Git API)

Usage:
    from neural.collaboration import CollaborationServer, WorkspaceManager
    
    # Start collaboration server
    server = CollaborationServer(host='localhost', port=8080)
    server.start()
    
    # Manage workspaces
    manager = WorkspaceManager()
    workspace = manager.create_workspace('my-project', owner='user1')
"""

import warnings
from typing import Optional

warnings.warn(
    "neural.collaboration is deprecated and will be removed in v0.4.0. "
    "Use Git and GitHub/GitLab for version control and collaboration instead. "
    "See docs/DEPRECATIONS.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

try:
    from .server import CollaborationServer
    from .workspace import WorkspaceManager, Workspace
    from .conflict_resolution import ConflictResolver
    from .access_control import AccessController
    from .git_integration import GitIntegration
    from .sync_manager import SyncManager
    from .client import CollaborationClient
except ImportError as e:
    import warnings
    warnings.warn(f"Collaboration module dependencies not fully available: {e}")
    CollaborationServer = None
    WorkspaceManager = None
    Workspace = None
    ConflictResolver = None
    AccessController = None
    GitIntegration = None
    SyncManager = None
    CollaborationClient = None

__all__ = [
    'CollaborationServer',
    'WorkspaceManager',
    'Workspace',
    'ConflictResolver',
    'AccessController',
    'GitIntegration',
    'SyncManager',
    'CollaborationClient',
]
