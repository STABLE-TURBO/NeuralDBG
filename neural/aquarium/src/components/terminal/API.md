# Terminal Component API Reference

Complete API documentation for the Terminal component system.

## Components

### Terminal

Main terminal component with Xterm.js integration.

#### Props

```typescript
interface TerminalProps {
  sessionId: string;        // Required: Unique session identifier
  onClose?: () => void;     // Optional: Callback when terminal is closed
  isActive?: boolean;       // Optional: Whether terminal is active (default: true)
}
```

#### Example

```tsx
<Terminal 
  sessionId="my-terminal-1"
  onClose={() => console.log('Terminal closed')}
  isActive={true}
/>
```

### TerminalPanel

Container managing multiple terminal sessions with tabs and split views.

#### Props

None (self-contained state management)

#### Example

```tsx
<TerminalPanel />
```

#### Features

- Automatic session management
- Tab-based navigation
- Split view support (horizontal/vertical)
- Session creation and deletion

### TerminalControls

Control bar for terminal operations.

#### Props

```typescript
interface TerminalControlsProps {
  connected: boolean;                      // Connection status
  currentShell: string;                    // Current shell name
  onClear: () => void;                     // Clear terminal callback
  onShellChange: (shell: string) => void;  // Shell change callback
  onSearch: (term: string) => void;        // Search callback
  onCopy: () => void;                      // Copy callback
  onPaste: () => void;                     // Paste callback
  onClose?: () => void;                    // Optional close callback
}
```

#### Example

```tsx
<TerminalControls
  connected={true}
  currentShell="bash"
  onClear={handleClear}
  onShellChange={handleShellChange}
  onSearch={handleSearch}
  onCopy={handleCopy}
  onPaste={handlePaste}
  onClose={handleClose}
/>
```

## Types

### TerminalSession

```typescript
interface TerminalSession {
  id: string;           // Unique session ID
  name: string;         // Display name
  active: boolean;      // Active state
  shell?: string;       // Shell type (optional)
}
```

### TerminalMessage

```typescript
interface TerminalMessage {
  type: 'output' | 'prompt' | 'command' | 'shell_change' | 'autocomplete' | 'error';
  data?: string;              // Message data
  shell?: string;             // Shell name (for shell_change)
  suggestions?: string[];     // Autocomplete suggestions
}
```

### CommandHistoryEntry

```typescript
interface CommandHistoryEntry {
  command: string;      // Command text
  timestamp: number;    // Unix timestamp
  exitCode?: number;    // Optional exit code
}
```

### ShellConfig

```typescript
interface ShellConfig {
  name: string;         // Shell display name
  prompt: string;       // Shell prompt
  executable: string;   // Executable name
  args: string[];       // Command arguments
}
```

### TerminalTheme

```typescript
interface TerminalTheme {
  background: string;
  foreground: string;
  cursor: string;
  black: string;
  red: string;
  green: string;
  yellow: string;
  blue: string;
  magenta: string;
  cyan: string;
  white: string;
  brightBlack: string;
  brightRed: string;
  brightGreen: string;
  brightYellow: string;
  brightBlue: string;
  brightMagenta: string;
  brightCyan: string;
  brightWhite: string;
}
```

### TerminalConfig

```typescript
interface TerminalConfig {
  fontSize: number;
  fontFamily: string;
  cursorBlink: boolean;
  cursorStyle: 'block' | 'underline' | 'bar';
  scrollback: number;
  theme: TerminalTheme;
  defaultShell: string;
  websocketUrl: string;
}
```

## Configuration

### Available Shells

```typescript
const AVAILABLE_SHELLS: { [key: string]: ShellConfig } = {
  bash: { name: 'Bash', prompt: '$ ', executable: 'bash', args: ['-i'] },
  zsh: { name: 'Zsh', prompt: '% ', executable: 'zsh', args: ['-i'] },
  sh: { name: 'Sh', prompt: '$ ', executable: 'sh', args: ['-i'] },
  powershell: { name: 'PowerShell', prompt: 'PS> ', executable: 'powershell', args: ['-NoLogo', '-NoExit'] },
  cmd: { name: 'CMD', prompt: '> ', executable: 'cmd', args: ['/K'] },
};
```

### Default Configuration

```typescript
const DEFAULT_CONFIG: TerminalConfig = {
  fontSize: 14,
  fontFamily: 'Menlo, Monaco, "Courier New", monospace',
  cursorBlink: true,
  cursorStyle: 'block',
  scrollback: 10000,
  theme: DEFAULT_THEME,
  defaultShell: 'bash',
  websocketUrl: 'ws://localhost:5000/terminal',
};
```

### Load/Save Configuration

```typescript
// Load configuration from localStorage
function loadConfig(): TerminalConfig;

// Save configuration to localStorage
function saveConfig(config: Partial<TerminalConfig>): void;

// Reset to default configuration
function resetConfig(): void;
```

## Utility Functions

### Session Management

```typescript
// Generate unique session ID
function generateSessionId(): string;

// Example: 'terminal-1638360123456-a1b2c3'
```

### Command Processing

```typescript
// Format command string
function formatCommand(command: string): string;

// Parse command output
function parseCommandOutput(output: string): {
  text: string;
  exitCode?: number;
  error?: boolean;
};

// Check if command is neural CLI
function isNeuralCommand(command: string): boolean;

// Parse neural command
function parseNeuralCommand(command: string): {
  action: string;
  args: string[];
  flags: { [key: string]: string | boolean };
};
```

### ANSI Handling

```typescript
// Remove ANSI escape codes
function escapeAnsiCodes(text: string): string;

// Get color from ANSI code
function getAnsiColor(code: string): string;

// Highlight syntax
function highlightSyntax(text: string): string;
```

### Shell Utilities

```typescript
// Validate shell name
function validateShellName(shell: string): boolean;

// Get shell prompt
function getShellPrompt(shell: string): string;

// Sanitize command
function sanitizeCommand(command: string): string;
```

### Path Utilities

```typescript
// Check if path is absolute
function isPathAbsolute(path: string): boolean;

// Join path components
function joinPaths(...paths: string[]): string;

// Get basename
function basename(path: string): string;

// Get dirname
function dirname(path: string): string;

// Get file extension
function getFileExtension(filename: string): string;

// Extract file path from command
function extractFilePath(command: string): string | null;
```

### Clipboard Operations

```typescript
// Copy to clipboard
function copyToClipboard(text: string): Promise<void>;

// Read from clipboard
function readFromClipboard(): Promise<string>;
```

### File Operations

```typescript
// Download file
function downloadFile(content: string, filename: string): void;
```

### Formatting

```typescript
// Format bytes
function formatBytes(bytes: number): string;
// Example: 1024 → '1 KB'

// Format duration
function formatDuration(ms: number): string;
// Example: 65000 → '1m 5s'

// Get current timestamp
function getCurrentTimestamp(): string;

// Format timestamp
function formatTimestamp(timestamp: string): string;
```

### Command History Class

```typescript
class CommandHistory {
  constructor(maxSize?: number);
  
  add(command: string): void;
  previous(): string | null;
  next(): string | null;
  reset(): void;
  getAll(): string[];
  search(query: string): string[];
  clear(): void;
}
```

#### Usage

```typescript
const history = new CommandHistory(1000);

history.add('ls -la');
history.add('cd /tmp');

const prev = history.previous();  // 'cd /tmp'
const prev2 = history.previous(); // 'ls -la'

const results = history.search('ls'); // ['ls -la']
```

### Performance Utilities

```typescript
// Debounce function
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void;

// Throttle function
function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void;
```

#### Usage

```typescript
const debouncedSearch = debounce((term: string) => {
  console.log('Searching:', term);
}, 300);

const throttledResize = throttle(() => {
  console.log('Resizing...');
}, 100);
```

## WebSocket Protocol

### Client → Server Messages

#### Execute Command

```json
{
  "type": "command",
  "data": "ls -la"
}
```

#### Request Autocomplete

```json
{
  "type": "autocomplete",
  "data": "neural co"
}
```

#### Change Shell

```json
{
  "type": "change_shell",
  "shell": "zsh"
}
```

### Server → Client Messages

#### Command Output

```json
{
  "type": "output",
  "data": "file1.txt\nfile2.txt\n"
}
```

#### Shell Prompt

```json
{
  "type": "prompt",
  "data": "$ "
}
```

#### Shell Changed

```json
{
  "type": "shell_change",
  "shell": "zsh"
}
```

#### Autocomplete Suggestions

```json
{
  "type": "autocomplete",
  "suggestions": ["neural compile", "neural run"]
}
```

#### Error

```json
{
  "type": "error",
  "data": "Command not found"
}
```

## Backend API

### Python TerminalSession Class

```python
class TerminalSession:
    def __init__(self, session_id: str, shell: str = "bash")
    def start(self) -> bool
    async def execute_command(self, command: str) -> str
    def get_autocomplete_suggestions(self, partial_command: str) -> List[str]
    def change_shell(self, new_shell: str) -> bool
    def stop(self) -> None
```

### Python TerminalManager Class

```python
class TerminalManager:
    def __init__(self)
    def create_session(self, session_id: str, shell: str = "bash") -> TerminalSession
    def get_session(self, session_id: str) -> Optional[TerminalSession]
    def remove_session(self, session_id: str) -> None
    def cleanup_all(self) -> None
```

### FastAPI WebSocket Endpoint

```python
@app.websocket("/terminal/{session_id}")
async def terminal_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for terminal sessions."""
    # Implementation handles:
    # - Session creation/retrieval
    # - Command execution
    # - Autocomplete
    # - Shell changes
    pass
```

## Events

### Terminal Events

The Terminal component doesn't expose custom events, but you can listen to WebSocket events:

```typescript
const ws = new WebSocket('ws://localhost:5000/terminal/session-1');

ws.addEventListener('open', () => {
  console.log('Terminal connected');
});

ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
});

ws.addEventListener('close', () => {
  console.log('Terminal disconnected');
});

ws.addEventListener('error', (error) => {
  console.error('WebSocket error:', error);
});
```

## Error Handling

### Connection Errors

```typescript
// Terminal component automatically retries connection
// Check connection status via TerminalControls
<TerminalControls connected={connected} ... />
```

### Command Errors

Command errors are displayed in the terminal output with error formatting (red color).

### WebSocket Errors

WebSocket errors are logged to console and shown in terminal:
- Connection refused → "Failed to connect to terminal backend"
- Connection closed → "Disconnected from terminal backend"
- Automatic reconnection after 3 seconds

## Best Practices

1. **Session IDs**: Use unique, descriptive session IDs
2. **Memory Management**: Cleanup terminals on unmount
3. **Error Handling**: Handle WebSocket disconnections gracefully
4. **Performance**: Limit scrollback buffer for large outputs
5. **Security**: Never execute untrusted commands
6. **Accessibility**: Ensure keyboard navigation works
7. **Testing**: Mock WebSocket for unit tests

## Examples

See [example.tsx](./example.tsx) for complete usage examples.

## Testing

See [Terminal.test.tsx](./Terminal.test.tsx) for test examples.

## Version History

- **1.0.0**: Initial release
  - Basic terminal emulation
  - WebSocket integration
  - Command history
  - Autocomplete
  - Multiple shells
  - Split view support
