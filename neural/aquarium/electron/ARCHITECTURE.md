# Neural Aquarium IDE - Architecture Documentation

## Overview

The Neural Aquarium IDE is built using Electron, providing a native desktop application for Neural DSL development. The architecture follows a multi-process model with strict separation of concerns and secure IPC communication.

## Process Architecture

### Main Process (main.js)

The main process is the heart of the application, running Node.js with full system access.

**Responsibilities:**
- Window lifecycle management
- Menu system
- IPC communication coordination
- File system operations
- Python subprocess management
- Configuration persistence

**Components:**
- `AquariumIDE`: Main application class
- `WindowManager`: Window creation and management
- `IPCController`: IPC message routing
- `MenuBuilder`: Application menu construction
- `ConfigManager`: User preferences storage

### Renderer Process (renderer/)

The renderer process runs the UI in a sandboxed Chromium environment.

**Responsibilities:**
- User interface rendering
- User input handling
- DSL code editing
- Visual feedback
- Communication with main process via IPC

**Security:**
- `nodeIntegration: false`
- `contextIsolation: true`
- `enableRemoteModule: false`
- Content Security Policy enforced

### Preload Script (preload.js)

The preload script bridges main and renderer processes securely.

**Responsibilities:**
- Expose safe IPC APIs to renderer
- Use `contextBridge` to create `window.aquarium` API
- Sanitize data between processes

## Component Architecture

### 1. Window Management

**File:** `window/window-manager.js`

Manages all application windows including:
- Main IDE window
- Debugger window
- Visualizer window
- Preferences window

**Features:**
- Window state persistence (position, size)
- Multi-window coordination
- Window lifecycle events

### 2. IPC Communication Layer

**File:** `ipc/ipc-controller.js`

Central hub for inter-process communication.

**Handlers:**
- Project operations (create, open, save, close)
- File operations (open, save, read)
- DSL operations (parse, compile, validate)
- Execution management (run, stop, status)
- Debugger control (start, stop, breakpoints)
- Python integration (version, packages)
- Backend communication (WebSocket)
- Configuration management
- Window control
- Dialogs and notifications

### 3. Menu System

**File:** `menu/menu-builder.js`

Builds native application menus with keyboard shortcuts.

**Menus:**
- **File**: Project/file management, recent items
- **Edit**: Text operations, find/replace, preferences
- **View**: UI controls, zoom, panels
- **Run**: DSL compilation, training execution, metrics
- **Debug**: Debugger controls, breakpoints, inspection
- **Help**: Documentation, examples, about

### 4. Configuration Management

**File:** `config/config-manager.js`

Persistent configuration storage using `electron-store`.

**Settings:**
- Theme and appearance
- Editor preferences
- Python environment
- Backend selection
- Recent projects/files
- Window states
- Debugger configuration

### 5. Project Management

**File:** `ipc/handlers/project-handler.js`

Handles Neural DSL project operations.

**Features:**
- Project creation with directory structure
- Project opening and validation
- Project metadata (neural-project.json)
- Recent projects tracking
- Project configuration updates

**Project Structure:**
```
MyProject/
├── neural-project.json    # Project configuration
├── models/                # DSL model definitions
├── scripts/               # Generated training scripts
└── data/                  # Dataset storage
```

### 6. File Management

**File:** `ipc/handlers/file-handler.js`

Handles file I/O operations.

**Supported Formats:**
- `.ndsl`, `.neural` - Neural DSL files
- `.py` - Python scripts
- All text files

**Features:**
- File dialogs (open, save, save-as)
- File content reading/writing
- Current file tracking

### 7. DSL Handler

**File:** `ipc/handlers/dsl-handler.js`

Interfaces with Python backend for DSL operations.

**Operations:**
- Parse DSL code
- Compile to target backend
- Validate syntax and semantics
- Load example models

**Communication:**
- Spawns Python processes
- Passes data via JSON
- Captures stdout/stderr
- Returns structured responses

### 8. Execution Handler

**File:** `ipc/handlers/execution-handler.js`

Manages training script execution.

**Features:**
- Temporary script creation
- Python subprocess management
- Real-time output streaming
- Metrics parsing (loss, accuracy)
- Process cleanup
- Execution history

**Metrics Parsing:**
```javascript
// Parses: Epoch 1/10 loss: 0.5432 accuracy: 0.8765
const epochRegex = /Epoch (\d+)\/(\d+).*loss: ([\d.]+).*accuracy: ([\d.]+)/;
```

### 9. Debugger Handler

**File:** `ipc/handlers/debugger-handler.js`

Integrates with Neural DSL debugger.

**Features:**
- Debugger process management
- WebSocket communication
- Breakpoint management
- Step-by-step execution
- Variable inspection
- Call stack navigation

**Protocol:**
```javascript
// WebSocket messages
{
  type: 'setBreakpoint' | 'step' | 'continue' | 'getVariables',
  data: { ... }
}
```

### 10. Python Handler

**File:** `ipc/handlers/python-handler.js`

Python environment management.

**Features:**
- Python version detection
- Python path resolution
- Package installation
- Package listing

### 11. Backend Handler

**File:** `ipc/handlers/backend-handler.js`

Manages Python backend server connection.

**Features:**
- Backend server startup
- WebSocket connection
- Message passing
- Connection status monitoring
- Automatic reconnection

**Architecture:**
```
Electron Main Process
    ↓
Python Backend Server (Flask/FastAPI)
    ↓
WebSocket (/ws endpoint)
    ↓
Neural DSL Core
```

## Data Flow

### DSL Compilation Flow

```
1. User writes DSL in renderer
2. Renderer → IPC → Main: compile request
3. Main → Python: spawn CLI process
4. Python: parse + compile DSL
5. Python → Main: compiled code (JSON)
6. Main → Renderer: compilation result
7. Renderer: display code/errors
```

### Training Execution Flow

```
1. User clicks "Run"
2. Renderer → IPC → Main: run request
3. Main: create temp script file
4. Main: spawn Python process
5. Python: execute training
6. Python → Main: stdout/stderr streams
7. Main → Renderer: real-time output
8. Main: parse metrics from output
9. Main → Renderer: metrics updates
10. Python: training complete
11. Main: cleanup temp files
12. Main → Renderer: completion status
```

### Debugging Flow

```
1. User starts debugger
2. Renderer → IPC → Main: start debug
3. Main: spawn debugger process
4. Main: establish WebSocket
5. User sets breakpoints
6. Renderer → Main → Debugger: breakpoints
7. Debugger: execution paused
8. Debugger → Main → Renderer: state update
9. User inspects variables
10. Renderer → Main → Debugger: get vars
11. Debugger → Main → Renderer: variables
```

## Security Model

### Process Isolation

- Main process: Full system access (Node.js)
- Renderer process: Sandboxed (Chromium)
- Preload: Limited bridge between processes

### IPC Security

- All IPC channels explicitly defined
- No dynamic channel creation
- Input validation on all handlers
- Error handling and sanitization

### Context Isolation

```javascript
// Renderer CANNOT access:
require()
process
__dirname
Node.js APIs

// Renderer CAN access:
window.aquarium.* (exposed APIs only)
```

### Content Security Policy

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' 'unsafe-inline'; 
               style-src 'self' 'unsafe-inline'">
```

## Extension Points

### Adding New IPC Handlers

1. Create handler class in `ipc/handlers/`
2. Implement async methods
3. Register in `IPCController`
4. Expose in `preload.js`
5. Document in API

### Adding New Windows

1. Add creation method to `WindowManager`
2. Configure window options
3. Load appropriate content
4. Handle window events

### Adding Menu Items

1. Add to appropriate menu in `MenuBuilder`
2. Define keyboard shortcut
3. Implement click handler
4. Send IPC message to renderer

## Performance Considerations

### IPC Optimization

- Use `invoke/handle` for request-response
- Use `send/on` for streaming data
- Batch updates when possible
- Avoid large data transfers

### Process Management

- Limit concurrent Python processes
- Clean up temporary files
- Monitor memory usage
- Kill orphaned processes

### Window State

- Persist window positions
- Restore last layout
- Lazy-load windows
- Cache frequently used data

## Error Handling

### Main Process Errors

```javascript
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  // Log to file, show dialog
});
```

### IPC Errors

All handlers return consistent error structure:

```javascript
{
  success: false,
  error: 'Error message'
}
```

### Python Process Errors

- Capture stderr
- Parse error messages
- Display user-friendly errors
- Log for debugging

## Testing Strategy

### Unit Tests

- Test IPC handlers independently
- Mock Python processes
- Validate data transformations

### Integration Tests

- Test full IPC flows
- Verify Python integration
- Check file operations

### End-to-End Tests

- Simulate user workflows
- Test window interactions
- Verify menu actions

## Deployment

### Development

```bash
npm run dev
# Loads from development server or local files
# DevTools enabled
# Source maps available
```

### Production

```bash
npm run build
# Creates optimized bundles
# Code signing (optional)
# Platform-specific installers
```

### Auto-Updates (Future)

```javascript
// Using electron-updater
autoUpdater.checkForUpdatesAndNotify();
```

## Future Enhancements

1. **Frontend Framework**: Integrate React/Vue for complex UI
2. **Monaco Editor**: Advanced code editing with IntelliSense
3. **Terminal Integration**: Embedded terminal (xterm.js)
4. **Git Integration**: Version control UI
5. **Extension System**: Plugin architecture
6. **Collaborative Editing**: Real-time collaboration
7. **Cloud Sync**: Project synchronization
8. **Performance Profiling**: Built-in profiler UI
9. **Auto-Updates**: Automatic version updates
10. **Crash Reporting**: Error tracking integration

## Resources

- [Electron Documentation](https://www.electronjs.org/docs)
- [Electron Security](https://www.electronjs.org/docs/tutorial/security)
- [IPC Communication](https://www.electronjs.org/docs/api/ipc-main)
- [electron-builder](https://www.electron.build/)
