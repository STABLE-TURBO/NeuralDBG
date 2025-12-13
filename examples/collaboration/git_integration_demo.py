"""
Git Integration Example

Demonstrates version control integration for collaborative workspaces.
"""

import tempfile
from pathlib import Path
from neural.collaboration import GitIntegration

def main():
    # Create temporary workspace directory
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_dir = Path(temp_dir) / 'workspace'
        workspace_dir.mkdir()
        
        print(f"Workspace directory: {workspace_dir}\n")
        
        # Initialize Git
        print("Initializing Git repository...")
        git = GitIntegration(workspace_dir)
        git.init_repo()
        print("✓ Git repository initialized\n")
        
        # Create a sample Neural DSL file
        model_file = workspace_dir / 'mnist_classifier.neural'
        model_file.write_text("""network MNIST {
    input: shape=(1, 28, 28, 1)
    
    Conv2D(32, 3, activation='relu')
    MaxPool2D(2)
    Conv2D(64, 3, activation='relu')
    MaxPool2D(2)
    Flatten()
    Dense(128, activation='relu')
    Output(10, activation='softmax')
}
""")
        print(f"Created model file: {model_file.name}")
        
        # Check status
        print("\nChecking Git status...")
        status = git.get_status()
        print(f"  Untracked files: {status['untracked']}")
        
        # Add and commit
        print("\nAdding files to staging...")
        git.add_files(['mnist_classifier.neural'])
        
        print("Committing changes...")
        commit_hash = git.commit(
            message="Initial commit: MNIST classifier",
            author_name="Alice",
            author_email="alice@example.com"
        )
        print(f"✓ Committed: {commit_hash[:7]}\n")
        
        # Make changes
        print("Making changes to the model...")
        model_file.write_text("""network MNIST {
    input: shape=(1, 28, 28, 1)
    
    Conv2D(64, 3, activation='relu')  # Increased filters
    MaxPool2D(2)
    Dropout(0.25)  # Added dropout
    Conv2D(128, 3, activation='relu')  # Increased filters
    MaxPool2D(2)
    Dropout(0.25)  # Added dropout
    Flatten()
    Dense(256, activation='relu')  # Increased units
    Dropout(0.5)  # Added dropout
    Output(10, activation='softmax')
}
""")
        
        # Check status again
        status = git.get_status()
        print(f"  Modified files: {status['modified']}")
        
        # Commit changes
        git.add_files(['mnist_classifier.neural'])
        commit_hash = git.commit(
            message="Improve model: add dropout and increase capacity",
            author_name="Bob",
            author_email="bob@example.com"
        )
        print(f"✓ Committed: {commit_hash[:7]}\n")
        
        # View commit history
        print("Commit history:")
        commits = git.get_log(n=5)
        for commit in commits:
            print(f"  {commit['hash'][:7]} - {commit['author_name']}: {commit['message']}")
        
        # Create branch
        print("\nCreating feature branch...")
        git.create_branch('feature/add-batch-norm')
        git.checkout_branch('feature/add-batch-norm')
        print("✓ Created and checked out 'feature/add-batch-norm'")
        
        # List branches
        print("\nBranches:")
        branches = git.list_branches()
        for branch in branches:
            current = " (current)" if branch == git.get_current_branch() else ""
            print(f"  - {branch}{current}")
        
        # Show diff
        print("\nShowing diff between commits:")
        diff = git.diff(commits[1]['hash'], commits[0]['hash'])
        if diff:
            print(diff[:500] + "..." if len(diff) > 500 else diff)

if __name__ == '__main__':
    main()
