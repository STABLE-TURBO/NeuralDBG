# Collaboration Examples

This directory contains examples demonstrating the Neural DSL collaborative editing features.

## Prerequisites

Install collaboration dependencies:

```bash
pip install websockets
```

Or install the full Neural package:

```bash
pip install -e ".[collaboration]"
```

## Examples

### 1. Basic Workspace Usage

**File:** `basic_usage.py`

Demonstrates creating and managing collaborative workspaces.

```bash
python examples/collaboration/basic_usage.py
```

Features:
- Create a workspace
- Add team members with roles
- List workspaces
- View workspace details

### 2. Conflict Resolution

**File:** `conflict_demo.py`

Demonstrates operational transformation for concurrent edits.

```bash
python examples/collaboration/conflict_demo.py
```

Features:
- Transform concurrent insert/delete operations
- Three-way merge for Neural DSL files
- Conflict detection and resolution

### 3. Git Integration

**File:** `git_integration_demo.py`

Demonstrates version control integration for workspaces.

```bash
python examples/collaboration/git_integration_demo.py
```

Features:
- Initialize Git repository
- Commit changes
- View commit history
- Create and manage branches
- View diffs

## Real-World Workflow

### Step 1: Create Workspace

```bash
# Owner creates workspace
neural collab create ml-project --user-id alice --description "Team ML project"
```

### Step 2: Start Server

```bash
# Start collaboration server
neural collab server --host localhost --port 8080
```

### Step 3: Join Workspace

```bash
# Team members join
neural collab join <workspace-id> --user-id bob --username "Bob Smith"
```

### Step 4: Add Members

```bash
# Owner adds team members
neural collab add-member <workspace-id> charlie --role member --owner-id alice
```

### Step 5: Collaborate

Team members can now edit files in real-time. The server broadcasts changes to all connected clients.

### Step 6: Sync Changes

```bash
# Sync workspace with Git
neural collab sync <workspace-id> --user-id alice
```

## Testing Locally

### Terminal 1: Server

```bash
neural collab server
```

### Terminal 2: User 1

```bash
neural collab create test-workspace --user-id user1
neural collab join <workspace-id> --user-id user1 --username "User 1"
```

### Terminal 3: User 2

```bash
neural collab join <workspace-id> --user-id user2 --username "User 2"
```

Now both users can see each other's presence in the workspace!

## Advanced Features

### Custom Server Configuration

```python
from neural.collaboration import CollaborationServer

server = CollaborationServer(host='0.0.0.0', port=8080)
server.start()
```

### Programmatic Client

```python
import asyncio
from neural.collaboration import CollaborationClient

async def main():
    client = CollaborationClient(host='localhost', port=8080)
    
    await client.connect(
        workspace_id='<workspace-id>',
        user_id='alice',
        username='Alice'
    )
    
    # Set up event handlers
    client.set_on_edit(lambda data: print(f"Edit: {data}"))
    
    # Listen for events
    await client.listen()

asyncio.run(main())
```

### Conflict Resolution

```python
from neural.collaboration import ConflictResolver

resolver = ConflictResolver()

# Three-way merge
merged, conflicts = resolver.three_way_merge(
    base="base version",
    local="local changes",
    remote="remote changes"
)

if conflicts:
    print(f"Found {len(conflicts)} conflicts")
```

## Tips

1. **Always sync before making changes** to avoid conflicts
2. **Use descriptive commit messages** for easier tracking
3. **Set appropriate roles** (owner, admin, member, viewer)
4. **Enable WSS** for production deployments
5. **Monitor workspace activity** through the server logs

## Troubleshooting

### Connection Refused

Make sure the server is running:

```bash
neural collab server
```

### Permission Denied

Check your role in the workspace:

```bash
neural collab info <workspace-id>
```

### Git Conflicts

Sync your workspace and resolve conflicts:

```bash
neural collab sync <workspace-id> --user-id <your-id>
```

## See Also

- [Collaboration Module README](../../neural/collaboration/README.md)
- [AGENTS.md](../../AGENTS.md) - Development guide
