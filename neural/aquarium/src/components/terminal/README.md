# Terminal Component

Integrated terminal panel with Xterm.js for shell access within Neural Aquarium IDE.

## Features

- **Full Terminal Emulation**: Powered by Xterm.js for authentic terminal experience
- **Multiple Shell Support**: Bash, Zsh, Sh, PowerShell, CMD
- **Command History**: Navigate through previous commands with arrow keys (↑/↓)
- **Autocomplete**: Tab completion for commands and file paths
- **Split Terminal**: Horizontal and vertical split views for multiple sessions
- **Neural CLI Integration**: Direct access to neural commands (compile, run, visualize, debug, hpo, automl)
- **Session Management**: Create, switch, and close multiple terminal sessions
- **Real-time Updates**: WebSocket-based communication for responsive terminal interaction

## Components

### Terminal
Main terminal component embedding Xterm.js with full terminal functionality.

**Props:**
- `sessionId: string` - Unique identifier for the terminal session
- `onClose?: () => void` - Callback when terminal is closed
- `isActive?: boolean` - Whether the terminal is currently active/visible

### TerminalPanel
Container component managing multiple terminal sessions with tabs and split views.

**Features:**
- Tab-based session management
- Single, horizontal, and vertical split modes
- Create new terminals
- Close terminals

### TerminalControls
Control bar for terminal operations.

**Features:**
- Connection status indicator
- Shell selector dropdown
- Search functionality
- Copy/paste operations
- Clear terminal
- Close button

## Usage

### Basic Usage

```tsx
import { TerminalPanel } from './components/terminal';

function App() {
  return (
    <div className="app">
      <TerminalPanel />
    </div>
  );
}
```

### Single Terminal

```tsx
import { Terminal } from './components/terminal';

function MyComponent() {
  return (
    <Terminal 
      sessionId="my-terminal-1" 
      onClose={() => console.log('Terminal closed')}
      isActive={true}
    />
  );
}
```

## Configuration

### Available Shells

The terminal supports the following shells:

- **bash** - Bourne Again Shell (default)
- **zsh** - Z Shell
- **sh** - Standard Shell
- **powershell** - Windows PowerShell (cross-platform with pwsh)
- **cmd** - Windows Command Prompt

### Neural CLI Commands

The terminal provides autocomplete for neural commands:

- `neural compile` - Compile DSL to backend code
- `neural run` - Execute neural model
- `neural visualize` - Visualize model architecture
- `neural debug` - Start debugging session
- `neural hpo` - Hyperparameter optimization
- `neural automl` - AutoML and Neural Architecture Search

## Backend Integration

### WebSocket Endpoint

The terminal connects to the backend via WebSocket:

```
ws://localhost:5000/terminal/{session_id}
```

### Message Types

**Client → Server:**
- `command` - Execute shell command
- `autocomplete` - Request autocomplete suggestions
- `change_shell` - Switch shell type

**Server → Client:**
- `output` - Command output
- `prompt` - Shell prompt
- `shell_change` - Shell changed confirmation
- `autocomplete` - Autocomplete suggestions

### Python Backend

The backend uses `TerminalHandler` class to manage terminal sessions:

```python
from terminal_handler import TerminalManager

terminal_manager = TerminalManager()
session = terminal_manager.create_session('session-1', shell='bash')
output = await session.execute_command('ls -la')
```

## Keyboard Shortcuts

- **Enter** - Execute command
- **Backspace/Delete** - Delete character
- **Tab** - Autocomplete
- **↑/↓** - Navigate command history
- **Ctrl+C** - Interrupt command
- **Ctrl+Shift+C** - Copy selection
- **Ctrl+Shift+V** - Paste from clipboard

## Styling

The terminal uses VS Code-inspired dark theme:

- Background: `#1e1e1e`
- Foreground: `#d4d4d4`
- Cursor: Block style with blink
- Font: Menlo, Monaco, "Courier New", monospace
- Font Size: 14px

Custom themes can be applied by modifying the `theme` object in `Terminal.tsx`.

## Accessibility

- Keyboard navigation support
- Screen reader compatible (via Xterm.js accessibility addon)
- High contrast color scheme
- Configurable font sizes

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

Requires WebSocket support.

## Dependencies

- `xterm` ^5.3.0 - Terminal emulator
- `xterm-addon-fit` ^0.8.0 - Fit terminal to container
- `xterm-addon-search` ^0.13.0 - Search functionality
- `xterm-addon-web-links` ^0.9.0 - Clickable URLs

## Performance

- Scrollback buffer: 10,000 lines
- Efficient rendering via Xterm.js canvas/WebGL
- Automatic resize on window changes
- Lazy loading for inactive terminals

## Troubleshooting

### Connection Failed
- Ensure backend server is running on port 5000
- Check WebSocket endpoint is accessible
- Verify CORS settings allow WebSocket connections

### Shell Not Available
- Ensure selected shell is installed on the system
- Check shell executable is in PATH
- Try falling back to default bash/sh

### Autocomplete Not Working
- Verify backend terminal handler is processing autocomplete requests
- Check file system permissions for path suggestions
- Ensure command history is being maintained

## Future Enhancements

- [ ] Terminal themes selector
- [ ] Font size adjustment
- [ ] Save/load terminal sessions
- [ ] Terminal replay/history
- [ ] SSH/remote terminal support
- [ ] File upload/download via terminal
- [ ] Terminal synchronization across split views
