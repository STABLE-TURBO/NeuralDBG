"""
Collaboration Server - WebSocket-based real-time collaboration server.

Handles WebSocket connections for real-time collaborative editing of Neural DSL files.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Set, Optional, Any

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any

from neural.exceptions import CollaborationException, AccessControlError

logger = logging.getLogger(__name__)


class CollaborationServer:
    """
    WebSocket-based collaboration server for real-time DSL editing.
    
    Manages WebSocket connections, broadcasts changes to clients,
    and coordinates edit operations across multiple users.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 8080):
        """
        Initialize the collaboration server.
        
        Parameters
        ----------
        host : str
            Server host address
        port : int
            Server port
        """
        if not WEBSOCKETS_AVAILABLE:
            raise CollaborationException(
                "websockets package is required for collaboration server. "
                "Install with: pip install websockets"
            )
        
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.workspace_clients: Dict[str, Set[str]] = {}
        self.client_info: Dict[str, Dict[str, Any]] = {}
        self.running = False
    
    async def register_client(
        self,
        websocket: WebSocketServerProtocol,
        workspace_id: str,
        user_id: str,
        username: str
    ) -> str:
        """
        Register a new client connection.
        
        Parameters
        ----------
        websocket : WebSocketServerProtocol
            WebSocket connection
        workspace_id : str
            Workspace identifier
        user_id : str
            User identifier
        username : str
            User display name
            
        Returns
        -------
        str
            Client connection ID
        """
        client_id = str(uuid.uuid4())
        
        self.clients[client_id] = websocket
        self.client_info[client_id] = {
            'user_id': user_id,
            'username': username,
            'workspace_id': workspace_id,
            'connected_at': datetime.utcnow().isoformat()
        }
        
        if workspace_id not in self.workspace_clients:
            self.workspace_clients[workspace_id] = set()
        self.workspace_clients[workspace_id].add(client_id)
        
        logger.info(f"Client {client_id} ({username}) connected to workspace {workspace_id}")
        
        await self.broadcast_to_workspace(
            workspace_id,
            {
                'type': 'user_joined',
                'user_id': user_id,
                'username': username,
                'timestamp': datetime.utcnow().isoformat()
            },
            exclude_client=client_id
        )
        
        return client_id
    
    async def unregister_client(self, client_id: str):
        """
        Unregister a client connection.
        
        Parameters
        ----------
        client_id : str
            Client connection ID
        """
        if client_id not in self.clients:
            return
        
        client_info = self.client_info.get(client_id, {})
        workspace_id = client_info.get('workspace_id')
        username = client_info.get('username', 'Unknown')
        user_id = client_info.get('user_id')
        
        del self.clients[client_id]
        del self.client_info[client_id]
        
        if workspace_id and workspace_id in self.workspace_clients:
            self.workspace_clients[workspace_id].discard(client_id)
            if not self.workspace_clients[workspace_id]:
                del self.workspace_clients[workspace_id]
        
        logger.info(f"Client {client_id} ({username}) disconnected from workspace {workspace_id}")
        
        if workspace_id:
            await self.broadcast_to_workspace(
                workspace_id,
                {
                    'type': 'user_left',
                    'user_id': user_id,
                    'username': username,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
    
    async def broadcast_to_workspace(
        self,
        workspace_id: str,
        message: Dict[str, Any],
        exclude_client: Optional[str] = None
    ):
        """
        Broadcast a message to all clients in a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
        message : Dict[str, Any]
            Message to broadcast
        exclude_client : Optional[str]
            Client ID to exclude from broadcast
        """
        if workspace_id not in self.workspace_clients:
            return
        
        message_json = json.dumps(message)
        
        for client_id in self.workspace_clients[workspace_id]:
            if exclude_client and client_id == exclude_client:
                continue
            
            if client_id in self.clients:
                try:
                    await self.clients[client_id].send(message_json)
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle a client connection.
        
        Parameters
        ----------
        websocket : WebSocketServerProtocol
            WebSocket connection
        path : str
            WebSocket path
        """
        client_id = None
        
        try:
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            if auth_data.get('type') != 'auth':
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Authentication required'
                }))
                return
            
            workspace_id = auth_data.get('workspace_id')
            user_id = auth_data.get('user_id')
            username = auth_data.get('username')
            
            if not all([workspace_id, user_id, username]):
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Missing required authentication fields'
                }))
                return
            
            client_id = await self.register_client(websocket, workspace_id, user_id, username)
            
            await websocket.send(json.dumps({
                'type': 'auth_success',
                'client_id': client_id,
                'workspace_id': workspace_id
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(client_id, data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message from client {client_id}: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client connection closed")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            if client_id:
                await self.unregister_client(client_id)
    
    async def handle_message(self, client_id: str, data: Dict[str, Any]):
        """
        Handle a message from a client.
        
        Parameters
        ----------
        client_id : str
            Client connection ID
        data : Dict[str, Any]
            Message data
        """
        message_type = data.get('type')
        client_info = self.client_info.get(client_id, {})
        workspace_id = client_info.get('workspace_id')
        
        if message_type == 'edit':
            await self.broadcast_to_workspace(
                workspace_id,
                {
                    'type': 'edit',
                    'user_id': client_info.get('user_id'),
                    'username': client_info.get('username'),
                    'operation': data.get('operation'),
                    'timestamp': datetime.utcnow().isoformat()
                },
                exclude_client=client_id
            )
        
        elif message_type == 'cursor':
            await self.broadcast_to_workspace(
                workspace_id,
                {
                    'type': 'cursor',
                    'user_id': client_info.get('user_id'),
                    'username': client_info.get('username'),
                    'position': data.get('position'),
                    'timestamp': datetime.utcnow().isoformat()
                },
                exclude_client=client_id
            )
        
        elif message_type == 'selection':
            await self.broadcast_to_workspace(
                workspace_id,
                {
                    'type': 'selection',
                    'user_id': client_info.get('user_id'),
                    'username': client_info.get('username'),
                    'range': data.get('range'),
                    'timestamp': datetime.utcnow().isoformat()
                },
                exclude_client=client_id
            )
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def start_async(self):
        """Start the collaboration server (async)."""
        self.running = True
        logger.info(f"Starting collaboration server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()
    
    def start(self):
        """Start the collaboration server."""
        try:
            asyncio.run(self.start_async())
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            self.running = False
    
    def get_workspace_users(self, workspace_id: str) -> list:
        """
        Get list of users currently in a workspace.
        
        Parameters
        ----------
        workspace_id : str
            Workspace identifier
            
        Returns
        -------
        list
            List of user information dictionaries
        """
        if workspace_id not in self.workspace_clients:
            return []
        
        users = []
        for client_id in self.workspace_clients[workspace_id]:
            if client_id in self.client_info:
                info = self.client_info[client_id]
                users.append({
                    'user_id': info.get('user_id'),
                    'username': info.get('username'),
                    'connected_at': info.get('connected_at')
                })
        
        return users
