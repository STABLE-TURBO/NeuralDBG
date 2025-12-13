# Terminal Component Integration Guide

This guide shows how to integrate the Terminal component into Neural Aquarium IDE.

## Installation

### 1. Install Dependencies

```bash
npm install xterm xterm-addon-fit xterm-addon-search xterm-addon-web-links
```

Or add to package.json:

```json
{
  "dependencies": {
    "xterm": "^5.3.0",
    "xterm-addon-fit": "^0.8.0",
    "xterm-addon-search": "^0.13.0",
    "xterm-addon-web-links": "^0.9.0"
  }
}
```

### 2. Backend Setup

Install Python dependencies (if not already installed):

```bash
pip install fastapi uvicorn websockets
```

## Quick Start

### Add Terminal to Main App

```tsx
// src/App.tsx
import React, { useState } from 'react';
import { TerminalPanel } from './components/terminal';

function App() {
  const [showTerminal, setShowTerminal] = useState(false);

  return (
    <div className="app">
      <nav>
        <button onClick={() => setShowTerminal(!showTerminal)}>
          Toggle Terminal
        </button>
      </nav>
      
      <main className="workspace">
        <div className="editor-area">
          {/* Your editor component */}
        </div>
        
        {showTerminal && (
          <div className="terminal-area" style={{ height: '400px' }}>
            <TerminalPanel />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
```

### Backend Integration

Ensure the backend server includes the terminal WebSocket endpoint (already added to `backend/server.py`):

```python
from terminal_handler import TerminalManager

terminal_manager = TerminalManager()

@app.websocket("/terminal/{session_id}")
async def terminal_websocket(websocket: WebSocket, session_id: str):
    # WebSocket handler for terminal sessions
    # (Implementation in backend/server.py)
    pass
```

Start the backend server:

```bash
cd neural/aquarium/backend
python -m uvicorn server:app --host 0.0.0.0 --port 5000
```

## Integration Patterns

### Pattern 1: IDE Layout with Terminal

```tsx
import React, { useState } from 'react';
import { TerminalPanel } from './components/terminal';
import NeuralDSLMonacoEditor from './components/editor/NeuralDSLMonacoEditor';

export const IDELayout: React.FC = () => {
  const [code, setCode] = useState('');
  const [terminalHeight, setTerminalHeight] = useState(300);

  return (
    <div className="ide-layout" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Top toolbar */}
      <div className="toolbar" style={{ height: '50px', borderBottom: '1px solid #ccc' }}>
        <button onClick={() => console.log('Compile')}>Compile</button>
        <button onClick={() => console.log('Run')}>Run</button>
      </div>

      {/* Main editor area */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <NeuralDSLMonacoEditor value={code} onChange={setCode} />
      </div>

      {/* Resizable terminal */}
      <div
        style={{
          height: `${terminalHeight}px`,
          borderTop: '2px solid #555',
          resize: 'vertical',
          overflow: 'auto',
        }}
      >
        <TerminalPanel />
      </div>
    </div>
  );
};
```

### Pattern 2: Split View with Terminal

```tsx
import React, { useState } from 'react';
import { Terminal } from './components/terminal';

export const SplitViewLayout: React.FC = () => {
  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* Left: Code editor */}
      <div style={{ flex: 1, borderRight: '1px solid #ccc' }}>
        <h3>Neural DSL Editor</h3>
        <textarea style={{ width: '100%', height: '90%' }} />
      </div>

      {/* Right: Terminal */}
      <div style={{ flex: 1 }}>
        <Terminal sessionId="split-view-terminal" isActive={true} />
      </div>
    </div>
  );
};
```

### Pattern 3: Bottom Panel with Tabs

```tsx
import React, { useState } from 'react';
import { TerminalPanel } from './components/terminal';
import { Debugger } from './components/debugger';

export const TabbedBottomPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'terminal' | 'debugger' | 'output'>('terminal');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Main content area */}
      <div style={{ flex: 1 }}>
        <h2>Main Content</h2>
      </div>

      {/* Bottom panel with tabs */}
      <div style={{ height: '350px', borderTop: '1px solid #ccc' }}>
        <div style={{ display: 'flex', borderBottom: '1px solid #ccc' }}>
          <button
            onClick={() => setActiveTab('terminal')}
            style={{
              padding: '8px 16px',
              background: activeTab === 'terminal' ? '#1e1e1e' : '#2d2d30',
              color: '#fff',
              border: 'none',
            }}
          >
            Terminal
          </button>
          <button
            onClick={() => setActiveTab('debugger')}
            style={{
              padding: '8px 16px',
              background: activeTab === 'debugger' ? '#1e1e1e' : '#2d2d30',
              color: '#fff',
              border: 'none',
            }}
          >
            Debugger
          </button>
          <button
            onClick={() => setActiveTab('output')}
            style={{
              padding: '8px 16px',
              background: activeTab === 'output' ? '#1e1e1e' : '#2d2d30',
              color: '#fff',
              border: 'none',
            }}
          >
            Output
          </button>
        </div>

        <div style={{ height: 'calc(100% - 40px)' }}>
          {activeTab === 'terminal' && <TerminalPanel />}
          {activeTab === 'debugger' && <Debugger code="" onChange={() => {}} />}
          {activeTab === 'output' && <div>Output panel</div>}
        </div>
      </div>
    </div>
  );
};
```

### Pattern 4: Floating Terminal Window

```tsx
import React, { useState } from 'react';
import { Terminal } from './components/terminal';

export const FloatingTerminal: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({ x: 100, y: 100 });

  return (
    <>
      <button onClick={() => setIsOpen(true)}>Open Terminal</button>

      {isOpen && (
        <div
          style={{
            position: 'fixed',
            left: position.x,
            top: position.y,
            width: '600px',
            height: '400px',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            borderRadius: '8px',
            overflow: 'hidden',
          }}
        >
          <Terminal
            sessionId="floating-terminal"
            onClose={() => setIsOpen(false)}
            isActive={true}
          />
        </div>
      )}
    </>
  );
};
```

## Advanced Features

### Custom Shell Configuration

```tsx
import React, { useEffect, useRef } from 'react';
import { Terminal } from './components/terminal';

export const CustomShellTerminal: React.FC = () => {
  const terminalRef = useRef<any>(null);

  useEffect(() => {
    // Send shell change command after connection
    const timer = setTimeout(() => {
      // This would need to be exposed via ref or context
      // handleShellChange('zsh');
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  return <Terminal sessionId="custom-shell" isActive={true} />;
};
```

### Terminal Output Monitoring

```tsx
import React, { useState, useEffect } from 'react';
import { Terminal } from './components/terminal';

export const MonitoredTerminal: React.FC = () => {
  const [lastOutput, setLastOutput] = useState<string>('');

  // You would need to extend the Terminal component to expose output
  const handleOutputChange = (output: string) => {
    setLastOutput(output);
    
    // Check for specific patterns
    if (output.includes('error')) {
      console.error('Terminal error detected:', output);
    }
  };

  return (
    <div>
      <div style={{ padding: '10px', background: '#f0f0f0' }}>
        Last Output: {lastOutput.substring(0, 50)}...
      </div>
      <Terminal sessionId="monitored-terminal" isActive={true} />
    </div>
  );
};
```

### Execute Commands Programmatically

To execute commands programmatically, you would need to extend the Terminal component to expose a method:

```tsx
// Extended Terminal component (Terminal.tsx)
import React, { useImperativeHandle, forwardRef } from 'react';

export interface TerminalHandle {
  executeCommand: (command: string) => void;
  clear: () => void;
}

const Terminal = forwardRef<TerminalHandle, TerminalProps>((props, ref) => {
  // ... existing code ...

  useImperativeHandle(ref, () => ({
    executeCommand: (command: string) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'command',
          data: command,
        }));
      }
    },
    clear: handleClear,
  }));

  // ... rest of component ...
});

// Usage
const MyComponent: React.FC = () => {
  const terminalRef = useRef<TerminalHandle>(null);

  const runNeuralCompile = () => {
    terminalRef.current?.executeCommand('neural compile model.neural');
  };

  return (
    <div>
      <button onClick={runNeuralCompile}>Compile Model</button>
      <Terminal ref={terminalRef} sessionId="programmatic" isActive={true} />
    </div>
  );
};
```

## Styling Integration

### Match Your Theme

```css
/* Custom terminal theme */
.terminal-container {
  --terminal-background: #282c34;
  --terminal-foreground: #abb2bf;
  --terminal-cursor: #528bff;
}

.terminal-controls {
  background-color: var(--terminal-background);
}

.control-button {
  color: var(--terminal-foreground);
}
```

### Responsive Design

```css
@media (max-width: 768px) {
  .terminal-panel {
    height: 300px !important;
  }

  .terminal-split.horizontal {
    flex-direction: column;
  }

  .terminal-tab {
    min-width: 60px;
    padding: 6px 8px;
    font-size: 11px;
  }
}
```

## Context Integration

### Terminal Context Provider

```tsx
// TerminalContext.tsx
import React, { createContext, useContext, useState } from 'react';

interface TerminalContextType {
  terminals: string[];
  activeTerminal: string | null;
  createTerminal: () => string;
  closeTerminal: (id: string) => void;
  setActiveTerminal: (id: string) => void;
}

const TerminalContext = createContext<TerminalContextType | undefined>(undefined);

export const TerminalProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [terminals, setTerminals] = useState<string[]>(['terminal-1']);
  const [activeTerminal, setActiveTerminal] = useState<string | null>('terminal-1');

  const createTerminal = () => {
    const id = `terminal-${Date.now()}`;
    setTerminals((prev) => [...prev, id]);
    setActiveTerminal(id);
    return id;
  };

  const closeTerminal = (id: string) => {
    setTerminals((prev) => prev.filter((t) => t !== id));
    if (activeTerminal === id) {
      setActiveTerminal(terminals[0] || null);
    }
  };

  return (
    <TerminalContext.Provider
      value={{ terminals, activeTerminal, createTerminal, closeTerminal, setActiveTerminal }}
    >
      {children}
    </TerminalContext.Provider>
  );
};

export const useTerminal = () => {
  const context = useContext(TerminalContext);
  if (!context) throw new Error('useTerminal must be used within TerminalProvider');
  return context;
};
```

## Troubleshooting

### WebSocket Connection Issues

If terminals fail to connect:

1. Verify backend is running on port 5000
2. Check browser console for WebSocket errors
3. Ensure CORS is configured correctly
4. Test WebSocket endpoint directly: `wscat -c ws://localhost:5000/terminal/test`

### Performance Issues

For large outputs:

1. Limit scrollback buffer in Terminal.tsx
2. Use virtual scrolling for history
3. Implement output filtering/throttling

### Shell Not Starting

If shell processes fail to start:

1. Check shell is installed: `which bash`, `which zsh`, etc.
2. Verify shell permissions
3. Check backend logs for errors
4. Try default sh shell as fallback

## Best Practices

1. **Resource Management**: Always cleanup terminal sessions on unmount
2. **Error Handling**: Handle WebSocket disconnections gracefully
3. **Security**: Sanitize commands on backend, never execute untrusted input
4. **Performance**: Limit active terminals, use lazy loading
5. **Accessibility**: Ensure keyboard navigation works properly
6. **User Experience**: Provide clear feedback for connection status

## Next Steps

- Integrate with Neural CLI commands
- Add terminal themes selector
- Implement session persistence
- Add terminal replay functionality
- Create terminal automation macros
