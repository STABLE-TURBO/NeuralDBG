"""
Collaboration Client - Client library for connecting to collaboration server.

Provides a high-level API for connecting to collaboration workspaces and
handling real-time editing.
"""

import asyncio
import json
from typing import Callable, Dict, Optional

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from neural.exceptions import CollaborationException


class CollaborationClient:
    """
    Client for connecting to collaboration server.
    
    Provides a high-level API for real-time collaborative editing.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8080):
        """
        Initialize collaboration client.
        
        Parameters
        ----------
        host : str
            Server host
        port : int
            Server port
        """
        if not WEBSOCKETS_AVAILABLE:
            raise CollaborationException(
                "websockets package required. Install with: pip install websockets"
            )
        
        self.host = host
        self.port = port
        self.websocket = None
        self.client_id = None
        self.workspace_id = None
        self.connected = False
        
        self.on_user_joined: Optional[Callable] = None
        self.on_user_left: Optional[Callable] = None
        self.on_edit: Optional[Callable] = None
        self.on_cursor: Optional[Callable] = None
        self.on_selection: Optional[Callable] = None
    
    async def connect(
        self,
        workspace_id: str,
        user_id: str,
        username: str
    ):
        """
        Connect to workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
        username : str
            User display name
        """
        uri = f"ws://{self.host}:{self.port}"
        
        self.websocket = await websockets.connect(uri)
        
        auth_message = {
            'type': 'auth',
            'workspace_id': workspace_id,
            'user_id': user_id,
            'username': username
        }
        
        await self.websocket.send(json.dumps(auth_message))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data.get('type') == 'auth_success':
            self.client_id = data.get('client_id')
            self.workspace_id = workspace_id
            self.connected = True
        else:
            raise CollaborationException(f"Authentication failed: {data.get('message')}")
    
    async def disconnect(self):
        """Disconnect from workspace."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
    
    async def send_edit(self, operation: Dict):
        """
        Send an edit operation.
        
        Parameters
        ----------
        operation : Dict
            Edit operation
        """
        if not self.connected:
            raise CollaborationException("Not connected to workspace")
        
        message = {
            'type': 'edit',
            'operation': operation
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def send_cursor_position(self, position: Dict):
        """
        Send cursor position.
        
        Parameters
        ----------
        position : Dict
            Cursor position (line, column)
        """
        if not self.connected:
            raise CollaborationException("Not connected to workspace")
        
        message = {
            'type': 'cursor',
            'position': position
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def send_selection(self, range_data: Dict):
        """
        Send selection range.
        
        Parameters
        ----------
        range_data : Dict
            Selection range (start, end)
        """
        if not self.connected:
            raise CollaborationException("Not connected to workspace")
        
        message = {
            'type': 'selection',
            'range': range_data
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def listen(self):
        """
        Listen for messages from server.
        
        Handles incoming messages and calls registered callbacks.
        """
        if not self.connected:
            raise CollaborationException("Not connected to workspace")
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'user_joined' and self.on_user_joined:
                    self.on_user_joined(data)
                elif message_type == 'user_left' and self.on_user_left:
                    self.on_user_left(data)
                elif message_type == 'edit' and self.on_edit:
                    self.on_edit(data)
                elif message_type == 'cursor' and self.on_cursor:
                    self.on_cursor(data)
                elif message_type == 'selection' and self.on_selection:
                    self.on_selection(data)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
    
    def set_on_user_joined(self, callback: Callable):
        """Set callback for user joined events."""
        self.on_user_joined = callback
    
    def set_on_user_left(self, callback: Callable):
        """Set callback for user left events."""
        self.on_user_left = callback
    
    def set_on_edit(self, callback: Callable):
        """Set callback for edit events."""
        self.on_edit = callback
    
    def set_on_cursor(self, callback: Callable):
        """Set callback for cursor position events."""
        self.on_cursor = callback
    
    def set_on_selection(self, callback: Callable):
        """Set callback for selection events."""
        self.on_selection = callback


async def connect_to_workspace(
    workspace_id: str,
    user_id: str,
    username: str,
    host: str = 'localhost',
    port: int = 8080
) -> CollaborationClient:
    """
    Connect to a collaboration workspace.
    
    Parameters
    ----------
    workspace_id : str
        Workspace identifier
    user_id : str
        User identifier
    username : str
        User display name
    host : str
        Server host
    port : int
        Server port
        
    Returns
    -------
    CollaborationClient
        Connected client
    """
    client = CollaborationClient(host=host, port=port)
    await client.connect(workspace_id, user_id, username)
    return client
