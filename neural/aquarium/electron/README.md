# Neural Aquarium IDE - Electron Edition

Desktop IDE for Neural DSL built with Electron.

## Features

- **Project Management**: Create, open, and manage Neural DSL projects
- **DSL Editor**: Syntax-highlighted editor with validation
- **Multi-Backend Support**: TensorFlow, PyTorch, and ONNX
- **Integrated Debugger**: Real-time debugging with breakpoints
- **Visual Tools**: Model visualization and metrics tracking
- **Python Integration**: Direct Python backend communication

## Architecture

```
electron/
├── main.js                    # Main process entry point
├── preload.js                 # Preload script for IPC bridge
├── package.json               # Node.js dependencies
├── window/                    # Window management
│   └── window-manager.js
├── ipc/                       # IPC communication layer
│   ├── ipc-controller.js      # Main IPC controller
│   └── handlers/              # IPC request handlers
│       ├── project-handler.js
│       ├── file-handler.js
│       ├── dsl-handler.js
│       ├── execution-handler.js
│       ├── debugger-handler.js
│       ├── python-handler.js
│       ├── backend-handler.js
│       ├── config-handler.js
│       ├── window-handler.js
│       └── dialog-handler.js
├── menu/                      # Menu system
│   └── menu-builder.js
├── config/                    # Configuration management
│   ├── config-manager.js
│   └── project-schema.json
└── renderer/                  # Frontend (React/Vue)
    └── (to be implemented)
```

## Setup

### Install Dependencies

```bash
cd neural/aquarium/electron
npm install
```

### Development

```bash
npm run dev
```

### Build

```bash
# Build for current platform
npm run build

# Build for specific platforms
npm run build:win
npm run build:mac
npm run build:linux
```

## IPC Communication

The Electron app uses IPC (Inter-Process Communication) to communicate between the renderer (frontend) and main process (backend).

### Available APIs

All APIs are exposed through the `window.aquarium` object in the renderer:

```javascript
// Project management
await window.aquarium.project.create({ name: 'MyProject' });
await window.aquarium.project.open();
await window.aquarium.project.save(projectData);

// File operations
await window.aquarium.file.open();
await window.aquarium.file.save(filePath, content);

// DSL operations
await window.aquarium.dsl.parse(dslCode);
await window.aquarium.dsl.compile(dslCode, 'tensorflow', options);

// Execution
const { executionId } = await window.aquarium.execution.run(script, options);
window.aquarium.execution.onOutput((data) => console.log(data));

// Debugging
await window.aquarium.debugger.start(config);
await window.aquarium.debugger.setBreakpoint(file, line);
```

## Menu System

The IDE includes a comprehensive menu system:

- **File**: Project and file management
- **Edit**: Text editing operations
- **View**: UI layout controls
- **Run**: DSL compilation and execution
- **Debug**: Debugging tools
- **Help**: Documentation and support

### Keyboard Shortcuts

- `Ctrl/Cmd+N`: New File
- `Ctrl/Cmd+O`: Open File
- `Ctrl/Cmd+S`: Save File
- `Ctrl/Cmd+Shift+N`: New Project
- `Ctrl/Cmd+Shift+O`: Open Project
- `F5`: Run Training
- `F9`: Start Debugging
- `F10`: Step Over
- `F11`: Step Into

## Configuration

User preferences are stored using `electron-store` in:
- **Windows**: `%APPDATA%/neural-aquarium-ide/config.json`
- **macOS**: `~/Library/Application Support/neural-aquarium-ide/config.json`
- **Linux**: `~/.config/neural-aquarium-ide/config.json`

## Python Backend Integration

The Electron app communicates with the Python backend through:

1. **CLI Interface**: Spawns Python processes for DSL operations
2. **WebSocket**: Real-time communication for debugging and execution
3. **REST API**: HTTP requests for backend services

## Building Distributables

Electron Builder is configured to create installers for all platforms:

```bash
# Windows: NSIS installer and portable exe
npm run build:win

# macOS: DMG and zip
npm run build:mac

# Linux: AppImage, deb, and rpm
npm run build:linux
```

## Development Notes

- Main process runs Node.js with full system access
- Renderer process runs in sandboxed environment
- `preload.js` bridges the two with contextBridge
- Context isolation ensures security
- No `nodeIntegration` or `enableRemoteModule`

## License

MIT License - Same as Neural DSL package
