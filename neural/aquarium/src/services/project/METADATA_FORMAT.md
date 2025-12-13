# .aquarium-project Metadata Format

The `.aquarium-project` file is a JSON-formatted metadata file that stores IDE-specific settings, state, and preferences for a Neural DSL project. This file should be committed to version control to share IDE settings across team members.

## File Location

The `.aquarium-project` file is located in the root directory of each Neural DSL project:

```
my_neural_project/
├── .aquarium-project          # IDE metadata (this file)
├── models/
│   ├── cnn.neural
│   └── rnn.neural
└── main.neural
```

## Format Specification

### Version 1.0

```json
{
  "version": "1.0",
  "name": "Project Name",
  "created_at": "2024-01-01T00:00:00.000000",
  "last_modified": "2024-01-01T12:00:00.000000",
  "settings": {
    "editor": {
      "font_size": 14,
      "theme": "dark",
      "tab_size": 4,
      "auto_save": true,
      "auto_save_delay": 1000,
      "show_line_numbers": true,
      "word_wrap": false
    },
    "project": {
      "default_backend": "tensorflow",
      "auto_compile": false,
      "show_hidden_files": false
    }
  },
  "open_files": [
    "models/cnn.neural",
    "main.neural"
  ],
  "active_file": "main.neural",
  "workspace": {
    "layout": "default",
    "sidebar_width": 250,
    "panel_height": 200
  },
  "recent_searches": [
    "Conv2D",
    "Dense",
    "activation"
  ],
  "bookmarks": [
    {
      "file": "models/cnn.neural",
      "line": 42,
      "description": "Important layer definition",
      "created_at": "2024-01-01T10:00:00.000000"
    }
  ],
  "breakpoints": {
    "models/cnn.neural": [10, 25, 42],
    "main.neural": [5]
  }
}
```

## Field Descriptions

### Root Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Metadata format version (currently "1.0") |
| `name` | string | Yes | Human-readable project name |
| `created_at` | string | Yes | ISO 8601 timestamp of project creation |
| `last_modified` | string | Yes | ISO 8601 timestamp of last modification |
| `settings` | object | Yes | Project and editor settings |
| `open_files` | array | Yes | List of file paths that were open |
| `active_file` | string/null | Yes | Path to the currently active file |
| `workspace` | object | Yes | Workspace layout configuration |
| `recent_searches` | array | Yes | Recently used search queries |
| `bookmarks` | array | Yes | Code bookmarks |
| `breakpoints` | object | Yes | Debug breakpoints by file |

### settings.editor

Editor-specific preferences:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `font_size` | integer | 14 | Editor font size in pixels |
| `theme` | string | "dark" | Color theme ("dark", "light", "auto") |
| `tab_size` | integer | 4 | Number of spaces per tab |
| `auto_save` | boolean | true | Enable automatic file saving |
| `auto_save_delay` | integer | 1000 | Auto-save delay in milliseconds |
| `show_line_numbers` | boolean | true | Show line numbers in editor |
| `word_wrap` | boolean | false | Enable word wrapping |

### settings.project

Project-specific settings:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_backend` | string | "tensorflow" | Default backend for compilation ("tensorflow", "pytorch", "onnx") |
| `auto_compile` | boolean | false | Automatically compile on save |
| `show_hidden_files` | boolean | false | Show hidden files in file tree |

### workspace

Workspace layout configuration:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `layout` | string | "default" | Workspace layout preset |
| `sidebar_width` | integer | 250 | File explorer sidebar width in pixels |
| `panel_height` | integer | 200 | Bottom panel height in pixels |

### open_files

Array of file paths (relative to project root) that were open when the project was last closed. Files will be restored when the project is reopened.

```json
"open_files": [
  "models/cnn.neural",
  "models/rnn.neural",
  "main.neural"
]
```

### active_file

Path (relative to project root) to the file that was active/focused when the project was last closed. Can be `null` if no file was active.

```json
"active_file": "models/cnn.neural"
```

### recent_searches

Array of recent search queries, ordered from most recent to oldest. Limited to 20 entries.

```json
"recent_searches": [
  "Conv2D",
  "Dense",
  "activation"
]
```

### bookmarks

Array of code bookmarks:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | Yes | File path (relative to project root) |
| `line` | integer | Yes | Line number (0-based) |
| `description` | string | No | Optional bookmark description |
| `created_at` | string | Yes | ISO 8601 timestamp |

Example:

```json
"bookmarks": [
  {
    "file": "models/cnn.neural",
    "line": 42,
    "description": "Important layer definition",
    "created_at": "2024-01-01T10:00:00.000000"
  },
  {
    "file": "main.neural",
    "line": 10,
    "description": "",
    "created_at": "2024-01-01T11:00:00.000000"
  }
]
```

### breakpoints

Object mapping file paths to arrays of line numbers where breakpoints are set:

```json
"breakpoints": {
  "models/cnn.neural": [10, 25, 42],
  "models/rnn.neural": [15, 30],
  "main.neural": [5]
}
```

## Usage Examples

### Creating a New Project Metadata File

```python
from pathlib import Path
from neural.aquarium.src.services.project import ProjectMetadata

project_path = Path("./my_project")
metadata = ProjectMetadata(project_path)
metadata.save()
```

### Loading and Modifying Metadata

```python
metadata = ProjectMetadata(project_path)
metadata.load()

# Update editor settings
metadata.set_setting("editor", "font_size", value=16)
metadata.set_setting("editor", "theme", value="light")

# Add bookmark
metadata.add_bookmark("model.neural", 42, "Check this layer")

# Set breakpoints
metadata.set_breakpoints("model.neural", [10, 25, 42])

# Save changes
metadata.save()
```

### Reading Settings

```python
font_size = metadata.get_setting("editor", "font_size")
backend = metadata.get_setting("project", "default_backend")
```

## Version Control

### Should I Commit .aquarium-project?

**Yes**, the `.aquarium-project` file should be committed to version control. This allows team members to:

- Share consistent IDE settings
- Maintain consistent formatting and style
- Share bookmarks for important code locations
- Keep consistent compiler and project configurations

### What About .aquarium Directory?

The `.aquarium` directory in the user's home directory (`~/.aquarium/`) contains user-specific data and should **not** be committed:

- `recent_projects.json` - User's recent project history
- `*.cache` - Temporary cache files

These are already excluded in `.gitignore`:

```gitignore
# Aquarium IDE user-specific files
.aquarium/recent_projects.json
.aquarium/*.cache
```

## Migration and Compatibility

### Future Version Changes

When the metadata format version changes:

1. The `version` field will be updated (e.g., "1.1", "2.0")
2. The system will automatically migrate older formats
3. New fields will have sensible defaults
4. Deprecated fields will be preserved for one major version

### Handling Unknown Fields

The metadata system preserves unknown fields when loading and saving, allowing:

- Forward compatibility with newer versions
- Custom extensions without breaking existing functionality
- Safe collaboration across different IDE versions

## Best Practices

1. **Commit the file**: Include `.aquarium-project` in version control
2. **Review changes**: Check metadata changes before committing
3. **Resolve conflicts**: Merge conflicts in settings carefully
4. **Document custom fields**: If extending, document your additions
5. **Preserve structure**: Don't manually edit unless necessary

## Security Considerations

- Never store secrets, API keys, or passwords in `.aquarium-project`
- File paths are relative to prevent exposing system structure
- User-specific data is stored separately in `~/.aquarium/`

## Schema Validation

The metadata format can be validated against a JSON schema (future enhancement):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "name", "created_at", "last_modified"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+$"
    },
    "name": {
      "type": "string",
      "minLength": 1
    }
  }
}
```

## See Also

- [Project Management README](README.md)
- [Workspace Configuration](workspace_config.py)
- [Project Metadata API](project_metadata.py)
