# Terminal Component Quick Start

Get the terminal component up and running in 5 minutes.

## Step 1: Install Dependencies

```bash
cd neural/aquarium
npm install xterm xterm-addon-fit xterm-addon-search xterm-addon-web-links
```

## Step 2: Start Backend Server

```bash
cd neural/aquarium/backend
python -m uvicorn server:app --host 0.0.0.0 --port 5000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000
```

## Step 3: Add Terminal to Your App

```tsx
// src/App.tsx
import React from 'react';
import { TerminalPanel } from './components/terminal';

function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Your header/toolbar */}
      <header style={{ height: '60px', background: '#252526' }}>
        <h1>Neural Aquarium</h1>
      </header>

      {/* Main content area */}
      <main style={{ flex: 1, overflow: 'hidden' }}>
        {/* Your editor or other components */}
      </main>

      {/* Terminal panel at bottom */}
      <div style={{ height: '400px' }}>
        <TerminalPanel />
      </div>
    </div>
  );
}

export default App;
```

## Step 4: Run Your App

```bash
cd neural/aquarium
npm start
```

Open http://localhost:3000 and you should see the terminal panel at the bottom!

## Step 5: Try It Out

In the terminal, try these commands:

```bash
# Basic shell commands
ls
pwd
echo "Hello from terminal!"

# Neural CLI commands (autocomplete with Tab)
neural help
neural compile
neural run
```

## That's It! 🎉

You now have a fully functional terminal in your IDE.

## Next Steps

### Customize Terminal Theme

```tsx
import { loadConfig, saveConfig } from './components/terminal/config';

// Change theme
saveConfig({ theme: 'monokai' });
```

### Add to Different Layout

```tsx
// Split view layout
<div style={{ display: 'flex', height: '100vh' }}>
  <div style={{ flex: 1 }}>
    {/* Your code editor */}
  </div>
  <div style={{ flex: 1 }}>
    <Terminal sessionId="side-terminal" isActive={true} />
  </div>
</div>
```

### Execute Commands Programmatically

```tsx
// Send commands via WebSocket
const ws = new WebSocket('ws://localhost:5000/terminal/my-session');
ws.onopen = () => {
  ws.send(JSON.stringify({ type: 'command', data: 'neural compile model.neural' }));
};
```

## Troubleshooting

### "Connection failed"
- Ensure backend is running on port 5000
- Check backend logs for errors
- Verify WebSocket URL in config

### "Shell not found"
- Ensure shell (bash, zsh, etc.) is installed
- Check shell is in PATH
- Try default 'sh' shell

### Terminal not visible
- Check container has height set
- Verify no CSS conflicts
- Check console for errors

## Common Issues

**Q: Terminal is too small**
```tsx
<div style={{ height: '600px' }}>  {/* Increase height */}
  <TerminalPanel />
</div>
```

**Q: Can't copy/paste**
```tsx
// Use Ctrl+Shift+C and Ctrl+Shift+V
// Or use the toolbar buttons
```

**Q: Commands not working**
```bash
# Check backend logs
cd neural/aquarium/backend
python -m uvicorn server:app --log-level debug
```

## Resources

- [README.md](./README.md) - Full documentation
- [INTEGRATION.md](./INTEGRATION.md) - Integration patterns
- [API.md](./API.md) - API reference
- [example.tsx](./example.tsx) - Usage examples

## Support

For issues or questions:
1. Check documentation
2. Review examples
3. Inspect browser console
4. Check backend logs
5. Open GitHub issue

Happy coding! 🚀
