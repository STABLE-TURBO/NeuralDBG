"""
Basic Collaboration Example

Demonstrates creating a workspace and managing members.
"""

from neural.collaboration import WorkspaceManager

def main():
    # Create workspace manager
    manager = WorkspaceManager(base_dir='neural_workspaces')
    
    # Create a new workspace
    print("Creating workspace...")
    workspace = manager.create_workspace(
        name='neural-mnist-project',
        owner='alice',
        description='Collaborative MNIST classifier project'
    )
    
    print(f"✓ Created workspace: {workspace.name}")
    print(f"  ID: {workspace.workspace_id}")
    print(f"  Owner: {workspace.owner}")
    
    # Add team members
    print("\nAdding team members...")
    workspace.add_member('bob', role='member')
    workspace.add_member('charlie', role='viewer')
    manager.update_workspace(workspace)
    
    print(f"✓ Added 2 members")
    
    # List all workspaces
    print("\nAll workspaces:")
    for ws in manager.list_workspaces():
        print(f"  - {ws.name} (Owner: {ws.owner}, Members: {len(ws.members)})")
    
    # Get workspace info
    print(f"\nWorkspace '{workspace.name}' details:")
    print(f"  Members:")
    for member in workspace.members:
        role = workspace.get_role(member)
        print(f"    - {member}: {role}")
    
    # Demonstrate access control
    print(f"\n✓ Workspace ready for collaboration!")
    print(f"  Server: neural collab server")
    print(f"  Join: neural collab join {workspace.workspace_id} --user-id bob --username Bob")

if __name__ == '__main__':
    main()
