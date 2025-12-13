# Neural Collaboration Module

Real-time collaborative editing infrastructure for Neural DSL files.

## Features

- **Real-time WebSocket Server**: WebSocket-based server for real-time DSL editing
- **Conflict Resolution**: Operational transformation for concurrent edits
- **Workspace Management**: Shared workspaces with access controls
- **Version Control**: Git integration for workspace synchronization
- **Access Control**: Role-based permissions (owner, admin, member, viewer)
- **Sync Manager**: File versioning and change tracking

## Installation

```bash
# Install collaboration dependencies
pip install websockets

# Or install full Neural package with all dependencies
pip install -e ".[full]"
```

## Quick Start

### 1. Start the Collaboration Server

```bash
neural collab server --host localhost --port 8080
```

### 2. Create a Workspace

```bash
neural collab create my-project --user-id user1 --description "My collaborative project"
```

### 3. Join the Workspace

In another terminal:

```bash
neural collab join <workspace-id> --user-id user2 --username "User 2"
```

### 4. Add Members

```bash
neural collab add-member <workspace-id> user3 --role member --owner-id user1
```

### 5. Sync Changes

```bash
neural collab sync <workspace-id> --user-id user1
```

## CLI Commands

### Workspace Management

- `neural collab create <name>` - Create a new workspace
- `neural collab list` - List all workspaces
- `neural collab info <workspace-id>` - Show workspace details
- `neural collab add-member <workspace-id> <user-id>` - Add a member
- `neural collab remove-member <workspace-id> <user-id>` - Remove a member

### Real-time Collaboration

- `neural collab server` - Start collaboration server
- `neural collab join <workspace-id>` - Join a workspace session

### Version Control

- `neural collab sync <workspace-id>` - Sync workspace with Git

## Python API

### Server

```python
from neural.collaboration import CollaborationServer

# Start server
server = CollaborationServer(host='localhost', port=8080)
server.start()
```

### Workspace Management

```python
from neural.collaboration import WorkspaceManager

# Create workspace manager
manager = WorkspaceManager(base_dir='neural_workspaces')

# Create workspace
workspace = manager.create_workspace(
    name='my-project',
    owner='user1',
    description='Collaborative Neural DSL project'
)

# Add member
workspace.add_member('user2', role='member')
manager.update_workspace(workspace)

# List workspaces
workspaces = manager.list_workspaces(user_id='user1')
```

### Client Connection

```python
from neural.collaboration import CollaborationClient
import asyncio

async def main():
    # Connect to workspace
    client = CollaborationClient(host='localhost', port=8080)
    await client.connect(
        workspace_id='<workspace-id>',
        user_id='user1',
        username='User 1'
    )
    
    # Set up event handlers
    def on_edit(data):
        print(f"Edit by {data['username']}: {data['operation']}")
    
    client.set_on_edit(on_edit)
    
    # Listen for events
    await client.listen()

asyncio.run(main())
```

### Git Integration

```python
from neural.collaboration import GitIntegration
from pathlib import Path

# Initialize Git for workspace
git = GitIntegration(Path('neural_workspaces/workspace-id'))

# Initialize repo
git.init_repo()

# Check status
status = git.get_status()
print(f"Modified: {status['modified']}")
print(f"Untracked: {status['untracked']}")

# Add and commit
git.add_files(['.'])
git.commit('Update workspace', author_name='user1')

# Get log
commits = git.get_log(n=5)
for commit in commits:
    print(f"{commit['hash'][:7]} - {commit['message']}")
```

### Conflict Resolution

```python
from neural.collaboration import ConflictResolver
from neural.collaboration.conflict_resolution import EditOperation, OperationType

resolver = ConflictResolver()

# Create operations
op1 = EditOperation(
    type=OperationType.INSERT,
    position=10,
    length=5,
    content='hello',
    user_id='user1'
)

op2 = EditOperation(
    type=OperationType.INSERT,
    position=12,
    length=6,
    content='world',
    user_id='user2'
)

# Transform operations
op1_prime, op2_prime = resolver.transform_operations(op1, op2)

# Apply operations
text = "Original text"
text = resolver.apply_operation(text, op1_prime)
text = resolver.apply_operation(text, op2_prime)

# Three-way merge
merged_text, conflicts = resolver.three_way_merge(
    base="base version",
    local="local changes",
    remote="remote changes"
)
```

### Sync Manager

```python
from neural.collaboration import SyncManager
from pathlib import Path

# Initialize sync manager
sync = SyncManager(Path('neural_workspaces/workspace-id'))

# Track file
version = sync.track_file('model.neural', content, user_id='user1')

# Update file with conflict detection
new_version, conflict_msg = sync.update_file(
    'model.neural',
    new_content,
    user_id='user2',
    base_version=version
)

if conflict_msg:
    print(f"Conflict: {conflict_msg}")

# Get file history
history = sync.get_file_history('model.neural')
for version in history:
    print(f"v{version.version} by {version.user_id} at {version.timestamp}")

# Get diff between versions
diff = sync.diff_versions('model.neural', version1=1, version2=2)
print('\n'.join(diff))
```

### Access Control

```python
from neural.collaboration import AccessController
from neural.collaboration.access_control import Role, Permission

controller = AccessController()

# Set roles
controller.set_role('workspace-id', 'user1', Role.OWNER)
controller.set_role('workspace-id', 'user2', Role.MEMBER)
controller.set_role('workspace-id', 'user3', Role.VIEWER)

# Check permissions
if controller.has_permission('workspace-id', 'user2', Permission.WRITE):
    print("User can write")

# Create access token
token = controller.create_token('user2', 'workspace-id', expires_in=3600)

# Verify token
access_token = controller.verify_token(token)
if access_token:
    print(f"Valid token for {access_token.user_id}")
```

## Architecture

### Components

1. **CollaborationServer**: WebSocket server handling client connections
2. **WorkspaceManager**: Manages workspace creation and membership
3. **ConflictResolver**: Operational transformation for concurrent edits
4. **AccessController**: Role-based access control with token authentication
5. **GitIntegration**: Version control operations
6. **SyncManager**: File versioning and synchronization

### Message Flow

```
Client A                Server                Client B
   |                      |                      |
   |--AUTH-------------->|                      |
   |<--AUTH_SUCCESS------|                      |
   |                      |<--AUTH--------------|
   |                      |--AUTH_SUCCESS------>|
   |                      |                      |
   |--EDIT-------------->|                      |
   |                      |--EDIT-------------->|
   |                      |<--EDIT--------------|
   |<--EDIT--------------|                      |
```

### Operational Transformation

The conflict resolver implements operational transformation (OT) to handle concurrent edits:

1. **Insert-Insert**: Adjust positions based on insertion order
2. **Insert-Delete**: Adjust positions based on operation order
3. **Delete-Delete**: Handle overlapping deletions
4. **Replace**: Convert to delete + insert and apply transformations

## Security

- Token-based authentication
- Role-based access control
- Workspace-level permissions
- Secure WebSocket connections (use WSS in production)

## Best Practices

1. **Always use access tokens** in production
2. **Enable WSS** (WebSocket Secure) for production
3. **Regularly sync** workspaces with Git
4. **Monitor conflicts** and resolve them promptly
5. **Set appropriate roles** for team members
6. **Clean up old versions** periodically

## Examples

See `examples/collaboration/` for complete examples:

- `basic_server.py` - Simple collaboration server
- `client_demo.py` - Client connection example
- `workspace_setup.py` - Workspace creation and management
- `conflict_demo.py` - Conflict resolution examples

## Troubleshooting

### Connection Issues

```bash
# Check if server is running
neural collab server --port 8080

# Test connection
neural collab join <workspace-id> --user-id test --username Test
```

### Sync Issues

```bash
# Check Git status
neural collab sync <workspace-id> --user-id user1

# View workspace info
neural collab info <workspace-id>
```

### Permission Issues

```bash
# Check workspace members
neural collab info <workspace-id>

# Add member with correct role
neural collab add-member <workspace-id> <user-id> --role member --owner-id <owner>
```

## License

MIT License - See LICENSE file for details
