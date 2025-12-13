# Neural Aquarium IDE Setup Guide

## Prerequisites

- **Node.js**: Version 16 or higher
- **npm**: Comes with Node.js
- **Python**: Version 3.8 or higher (for backend)
- **Git**: For version control

## Installation

### 1. Install Node.js Dependencies

```bash
cd neural/aquarium/electron
npm install
```

This will install:
- Electron framework
- electron-builder (for creating distributables)
- electron-store (for configuration persistence)
- electron-window-state (for window state management)
- ws (WebSocket library)
- chokidar (file watching)

### 2. Verify Python Backend

Ensure the Python backend is set up:

```bash
cd ../../..
pip install -e ".[full]"
```

## Running the IDE

### Development Mode

```bash
npm run dev
```

This starts Electron with:
- Development tools enabled
- Hot reload support
- Connection to local development server (if running)

### Production Mode

```bash
npm start
```

Runs the IDE in production mode.

### Using Startup Scripts

**Windows:**
```cmd
start.bat
```

**macOS/Linux:**
```bash
chmod +x start.sh
./start.sh
```

## Building Distributables

### Build for Current Platform

```bash
npm run build
```

### Build for Specific Platforms

**Windows:**
```bash
npm run build:win
```

**macOS:**
```bash
npm run build:mac
```

**Linux:**
```bash
npm run build:linux
```

### Output

Built applications will be in the `dist/` directory:
- **Windows**: `.exe` installer and portable version
- **macOS**: `.dmg` disk image and `.zip` archive
- **Linux**: `.AppImage`, `.deb`, and `.rpm` packages

## Project Structure

```
electron/
├── main.js                    # Main process (Node.js)
├── preload.js                 # Preload script (IPC bridge)
├── package.json               # Dependencies and scripts
├── renderer/                  # Frontend
│   └── index.html            # Main UI
├── window/                    # Window management
│   └── window-manager.js
├── ipc/                       # IPC handlers
│   ├── ipc-controller.js
│   └── handlers/
├── menu/                      # Application menus
│   └── menu-builder.js
└── config/                    # Configuration
    ├── config-manager.js
    └── project-schema.json
```

## Configuration

The IDE stores user preferences in:

- **Windows**: `%APPDATA%/neural-aquarium-ide/`
- **macOS**: `~/Library/Application Support/neural-aquarium-ide/`
- **Linux**: `~/.config/neural-aquarium-ide/`

Configuration includes:
- Theme settings
- Editor preferences
- Recent projects
- Window state
- Python path
- Default backend

## Development Workflow

### 1. Frontend Development

The renderer process loads from `renderer/index.html`. For React/Vue integration:

```bash
# Install frontend framework
npm install react react-dom
# or
npm install vue

# Update main.js to load from dev server
# Development: http://localhost:3000
# Production: file://path/to/renderer/index.html
```

### 2. Backend Communication

Use the IPC API exposed via `window.aquarium`:

```javascript
// Parse DSL
const result = await window.aquarium.dsl.parse(code);

// Compile model
await window.aquarium.dsl.compile(code, 'tensorflow', options);

// Run training
const { executionId } = await window.aquarium.execution.run(script, options);

// Listen to output
window.aquarium.execution.onOutput((data) => {
    console.log(data);
});
```

### 3. Adding New Features

1. **Add IPC Handler**: Create handler in `ipc/handlers/`
2. **Register Handler**: Add to `ipc/ipc-controller.js`
3. **Expose API**: Add to `preload.js`
4. **Use in Renderer**: Call via `window.aquarium.*`

## Debugging

### Main Process

```bash
# Enable debug mode
npm run dev
```

Press `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Option+I` (macOS) to open DevTools.

### Renderer Process

DevTools automatically open in development mode.

### Logs

- Main process: Terminal output
- Renderer process: DevTools console
- Python backend: Check backend logs

## Troubleshooting

### Port Already in Use

If the Python backend port (5000) is in use:

```javascript
// Edit ipc/handlers/backend-handler.js
// Change the port number in connect() method
```

### Python Not Found

Set Python path in preferences:

```javascript
await window.aquarium.config.set('pythonPath', '/path/to/python');
```

### Build Errors

Clear cache and rebuild:

```bash
rm -rf node_modules dist
npm install
npm run build
```

### WebSocket Connection Failed

Ensure the Python backend server is running:

```bash
cd neural/aquarium/backend
python server.py
```

## Testing

### Unit Tests (Future)

```bash
npm test
```

### Manual Testing Checklist

- [ ] Create new project
- [ ] Open existing project
- [ ] Open file
- [ ] Save file
- [ ] Parse DSL code
- [ ] Compile model
- [ ] Run training
- [ ] Start debugger
- [ ] Set breakpoints
- [ ] View metrics

## Contributing

When contributing to the Electron IDE:

1. Follow the existing code structure
2. Add proper error handling
3. Update documentation
4. Test on multiple platforms
5. Ensure security best practices

## Security

- Context isolation enabled
- No `nodeIntegration`
- No `enableRemoteModule`
- Content Security Policy enforced
- IPC communication validated

## License

MIT License - Same as Neural DSL package
