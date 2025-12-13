# Aquarium Project Management System

A comprehensive project management system for the Neural DSL IDE with file explorer tree view, workspace configuration, and multi-file editing capabilities.

## Features

### 1. File Explorer Tree View
- Hierarchical file tree visualization
- Support for `.neural` file filtering
- Directory expansion/collapse
- File and directory node management
- Automatic sorting (directories first, then files)
- Ignore patterns support (e.g., `__pycache__`, `.git`, `.venv`)

### 2. Project Workspace Configuration
- Compiler settings (backend, optimization, output directory)
- Linter configuration (enabled, shape checking, strict mode)
- Editor preferences (rulers, minimap, bracket colorization)
- File associations and exclusion patterns
- Search configuration

### 3. Recent Projects List
- Track recently opened projects
- Automatic cleanup of non-existing projects
- Last opened timestamp tracking
- Project descriptions
- Configurable maximum recent projects (default: 20)

### 4. File Operations
- **New**: Create new files and directories
- **Open**: Open files for editing
- **Save**: Save file changes
- **Save As**: Save file with a new name
- **Close**: Close open files
- **Delete**: Remove files and directories
- **Rename**: Rename files and directories
- **Copy**: Copy files and directories
- **Move**: Move files and directories

### 5. Multi-File Editing with Tabs
- Multiple file tabs support
- Active tab tracking
- Tab navigation (next/previous)
- Modified state tracking
- Cursor and scroll position persistence
- Tab metadata support
- Close all/close others functionality

### 6. Project Metadata (.aquarium-project)
- Project settings persistence
- Editor preferences (font size, theme, tab size, etc.)
- Open files and active file restoration
- Bookmarks management
- Breakpoints tracking
- Recent searches history
- Workspace layout configuration

## Architecture

### Core Components

#### ProjectManager
Main controller that orchestrates all project management functionality.

```python
from neural.aquarium.src.services.project import ProjectManager

pm = ProjectManager()
pm.open_project(Path("./my_project"))
```

#### FileTree & FileNode
Hierarchical file system representation.

```python
from neural.aquarium.src.services.project import FileTree, FileNode, FileNodeType

tree = FileTree(Path("./my_project"))
neural_files = tree.get_all_neural_files()
```

#### TabManager & EditorTab
Multi-file editing with tab management.

```python
from neural.aquarium.src.services.project import TabManager

tab_manager = TabManager()
tab = tab_manager.open_tab(file_path, content="...")
tab_manager.activate_next_tab()
```

#### FileOperations
File system operations with callbacks.

```python
from neural.aquarium.src.services.project import FileOperations

file_ops = FileOperations()
result = file_ops.new_file(directory, "model.neural", content="...")
```

#### WorkspaceConfig
Project-specific configuration management.

```python
from neural.aquarium.src.services.project import WorkspaceConfig

config = WorkspaceConfig(project_path)
backend = config.get_compiler_backend()
config.set_compiler_backend("pytorch")
```

#### ProjectMetadata
IDE settings and state persistence.

```python
from neural.aquarium.src.services.project import ProjectMetadata

metadata = ProjectMetadata(project_path)
metadata.add_bookmark("model.neural", 42, "Important layer")
metadata.save()
```

#### RecentProjectsManager
Recently opened projects tracking.

```python
from neural.aquarium.src.services.project import RecentProjectsManager

recent_mgr = RecentProjectsManager(config_dir)
recent_projects = recent_mgr.get_recent(10)
```

## Usage Examples

### Creating a New Project

```python
from pathlib import Path
from neural.aquarium.src.services.project import ProjectManager

pm = ProjectManager()
project_path = Path("./my_neural_project")

if pm.create_project(project_path, "My Neural Project"):
    print(f"Project created: {project_path}")
```

### Opening and Editing Files

```python
pm.open_project(project_path)

result = pm.new_file("model.neural")
if result.success:
    pm.open_file(result.path)
    
    active_tab = pm.tab_manager.get_active_tab()
    if active_tab:
        active_tab.content = "model MyModel { ... }"
        pm.save_file(active_tab)
```

### Working with File Tree

```python
if pm.file_tree and pm.file_tree.root:
    root = pm.file_tree.root
    
    for child in root.children:
        print(f"{child.name} - {child.node_type.value}")
        
        if child.node_type == FileNodeType.DIRECTORY:
            pm.file_tree.toggle_expand(child)
```

### Managing Multiple Tabs

```python
file1 = project_path / "model1.neural"
file2 = project_path / "model2.neural"
file3 = project_path / "model3.neural"

pm.open_file(file1)
pm.open_file(file2)
pm.open_file(file3)

pm.tab_manager.activate_next_tab()

modified_tabs = pm.tab_manager.get_modified_tabs()
for tab in modified_tabs:
    pm.save_file(tab)
```

### Configuring Workspace

```python
if pm.workspace_config:
    pm.workspace_config.set_compiler_backend("tensorflow")
    pm.workspace_config.add_excluded_pattern("**/temp")
    
    config_dict = pm.workspace_config.to_dict()
```

### Using Project Metadata

```python
if pm.project_metadata:
    pm.project_metadata.add_bookmark("model.neural", 42, "Key layer")
    pm.project_metadata.set_breakpoints("model.neural", [10, 25, 42])
    pm.project_metadata.add_recent_search("Conv2D")
    pm.project_metadata.set_setting("editor", "font_size", value=16)
    pm.project_metadata.save()
```

### Event Callbacks

```python
def on_project_opened(path):
    print(f"Project opened: {path}")

def on_tab_modified(tab):
    print(f"Tab modified: {tab.title}")

pm.on_project_opened = on_project_opened
pm.tab_manager.on_tab_modified = on_tab_modified
```

## File Structure

```
neural/aquarium/src/services/project/
├── __init__.py                 # Module exports
├── project_manager.py          # Main project manager
├── file_tree.py                # File tree implementation
├── file_node.py                # File/directory node
├── tab_manager.py              # Tab management
├── file_operations.py          # File system operations
├── workspace_config.py         # Workspace configuration
├── project_metadata.py         # Project metadata (.aquarium-project)
├── recent_projects.py          # Recent projects tracking
├── project_utils.py            # Utility functions
├── examples.py                 # Usage examples
└── README.md                   # This file
```

## Configuration Files

### .aquarium-project
JSON file storing project-specific IDE settings:

```json
{
  "version": "1.0",
  "name": "My Neural Project",
  "created_at": "2024-01-01T00:00:00",
  "last_modified": "2024-01-01T12:00:00",
  "settings": {
    "editor": {
      "font_size": 14,
      "theme": "dark",
      "tab_size": 4,
      "auto_save": true
    },
    "project": {
      "default_backend": "tensorflow",
      "auto_compile": false
    }
  },
  "open_files": ["model.neural", "main.neural"],
  "active_file": "model.neural",
  "bookmarks": [],
  "breakpoints": {}
}
```

### recent_projects.json
Stored in user config directory (`~/.aquarium/`):

```json
{
  "projects": [
    {
      "path": "/path/to/project",
      "name": "Project Name",
      "last_opened": "2024-01-01T12:00:00",
      "description": "Project description"
    }
  ]
}
```

## API Reference

See individual module docstrings and type hints for detailed API documentation.

## Integration

This project management system is designed to integrate with:
- Neural DSL parser and compiler
- Code editors and IDEs
- Version control systems
- Build and deployment tools

## Best Practices

1. **Always save project state**: Call `pm.close_project(save_state=True)` before exiting
2. **Check operation results**: Verify `FileOperationResult.success` before proceeding
3. **Handle unsaved changes**: Check `tab_manager.has_unsaved_changes()` before closing
4. **Use callbacks**: Set up callbacks for project and file events
5. **Sanitize filenames**: Use `project_utils.sanitize_filename()` for user input
6. **Regular cleanup**: Call `recent_projects.clean_invalid_projects()` periodically

## Future Enhancements

- Undo/redo support for file operations
- File watching for external changes
- Conflict resolution for concurrent edits
- Project templates
- Workspace presets
- Git integration
- Search and replace across files
- File comparison and diff
- Project export/import
