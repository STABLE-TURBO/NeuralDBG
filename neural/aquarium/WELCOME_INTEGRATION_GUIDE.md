# Welcome Screen Integration Guide

This guide shows how to integrate the welcome screen components into your Neural Aquarium application.

## Quick Integration

### Step 1: Import Components

Add to your `App.tsx` or main component:

```tsx
import React, { useState } from 'react';
import { WelcomeScreen, InteractiveTutorial } from './components/welcome';
```

### Step 2: Add State Management

```tsx
function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [showTutorial, setShowTutorial] = useState(false);
  const [currentDSL, setCurrentDSL] = useState('');
  const [appliedDSL, setAppliedDSL] = useState('');
```

### Step 3: Add Handler Functions

```tsx
  const handleLoadTemplate = (dslCode: string) => {
    setCurrentDSL(dslCode);
    setAppliedDSL(dslCode);
    setShowWelcome(false);
  };

  const handleStartTutorial = () => {
    setShowWelcome(false);
    setShowTutorial(true);
  };

  const handleCloseTutorial = () => {
    setShowTutorial(false);
  };
```

### Step 4: Add Components to JSX

```tsx
  return (
    <div className="App">
      {/* Welcome Screen */}
      {showWelcome && (
        <WelcomeScreen
          onClose={() => setShowWelcome(false)}
          onLoadTemplate={handleLoadTemplate}
          onStartTutorial={handleStartTutorial}
        />
      )}
      
      {/* Interactive Tutorial */}
      {showTutorial && (
        <InteractiveTutorial
          onComplete={handleCloseTutorial}
          onSkip={handleCloseTutorial}
        />
      )}
      
      {/* Your existing app content */}
      <header className="App-header">
        <h1>Neural Aquarium</h1>
      </header>
      
      {/* ... rest of your app ... */}
    </div>
  );
}
```

## Complete Example

Here's a complete example showing integration with the existing Neural Aquarium app:

```tsx
import React, { useState } from 'react';
import AIAssistantSidebar from './components/ai/AIAssistantSidebar';
import { WelcomeScreen, InteractiveTutorial } from './components/welcome';
import './App.css';

function App() {
  // Welcome screen state
  const [showWelcome, setShowWelcome] = useState(true);
  const [showTutorial, setShowTutorial] = useState(false);
  
  // DSL state
  const [currentDSL, setCurrentDSL] = useState<string>('');
  const [appliedDSL, setAppliedDSL] = useState<string>('');

  // Welcome screen handlers
  const handleLoadTemplate = (dslCode: string) => {
    setCurrentDSL(dslCode);
    setAppliedDSL(dslCode);
    setShowWelcome(false);
  };

  const handleStartTutorial = () => {
    setShowWelcome(false);
    setShowTutorial(true);
  };

  // AI Assistant handlers
  const handleDSLGenerated = (dslCode: string) => {
    setCurrentDSL(dslCode);
    console.log('DSL Generated:', dslCode);
  };

  const handleDSLApplied = (dslCode: string) => {
    setAppliedDSL(dslCode);
    console.log('DSL Applied:', dslCode);
  };

  return (
    <div className="App">
      {/* Welcome Screen - shown on first load */}
      {showWelcome && (
        <WelcomeScreen
          onClose={() => setShowWelcome(false)}
          onLoadTemplate={handleLoadTemplate}
          onStartTutorial={handleStartTutorial}
        />
      )}
      
      {/* Interactive Tutorial - shown after welcome if user clicks "Start Tutorial" */}
      {showTutorial && (
        <InteractiveTutorial
          onComplete={() => setShowTutorial(false)}
          onSkip={() => setShowTutorial(false)}
        />
      )}
      
      {/* Main App Header */}
      <header className="App-header">
        <h1>Neural Aquarium</h1>
        <p>AI-Powered Neural DSL Builder</p>
        <button 
          className="help-button" 
          onClick={() => setShowWelcome(true)}
          title="Show welcome screen"
        >
          ❓ Help
        </button>
      </header>

      {/* Main Content Area */}
      <main className="App-main">
        <div className="content-area">
          <div className="model-workspace">
            <h2>Model Workspace</h2>
            {appliedDSL ? (
              <div className="applied-model">
                <h3>Applied Model</h3>
                <pre className="model-code">{appliedDSL}</pre>
              </div>
            ) : (
              <div className="placeholder">
                <p>Use the AI Assistant or Welcome Screen to create a model.</p>
                <p>Your DSL code will appear here once applied.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* AI Assistant Sidebar */}
      <AIAssistantSidebar
        onDSLGenerated={handleDSLGenerated}
        onDSLApplied={handleDSLApplied}
      />
    </div>
  );
}

export default App;
```

## Advanced Integration Options

### Option 1: Conditional Welcome (First Time Only)

Use localStorage to show welcome only on first visit:

```tsx
const [showWelcome, setShowWelcome] = useState(() => {
  const hasSeenWelcome = localStorage.getItem('hasSeenWelcome');
  return !hasSeenWelcome;
});

const handleCloseWelcome = () => {
  localStorage.setItem('hasSeenWelcome', 'true');
  setShowWelcome(false);
};
```

### Option 2: Reopen Welcome Button

Add a button to reopen the welcome screen:

```tsx
<button 
  className="welcome-button"
  onClick={() => setShowWelcome(true)}
  aria-label="Show welcome screen"
>
  🎓 Tutorial
</button>
```

### Option 3: Direct Tutorial Access

Add a button to jump directly to the tutorial:

```tsx
<button 
  className="tutorial-button"
  onClick={() => setShowTutorial(true)}
  aria-label="Start interactive tutorial"
>
  📚 Start Tutorial
</button>
```

### Option 4: Context Menu Integration

Add welcome screen access to a help menu:

```tsx
<div className="help-menu">
  <button onClick={() => setShowWelcome(true)}>
    Quick Start & Templates
  </button>
  <button onClick={() => setShowTutorial(true)}>
    Interactive Tutorial
  </button>
</div>
```

## Styling Integration

Add these styles to your `App.css`:

```css
/* Welcome button in header */
.help-button {
  position: absolute;
  top: 20px;
  right: 20px;
  background: transparent;
  border: 2px solid #61dafb;
  color: #61dafb;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.help-button:hover {
  background: #61dafb;
  color: #000;
  transform: translateY(-2px);
}

/* Tutorial button styling */
.tutorial-button,
.welcome-button {
  padding: 10px 20px;
  background: #61dafb;
  color: #000;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  margin: 10px;
}

.tutorial-button:hover,
.welcome-button:hover {
  background: #4fa8c5;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(97, 218, 251, 0.3);
}
```

## Backend Setup

Ensure your backend server is running with the examples and docs endpoints:

### FastAPI (backend/server.py)

The endpoints are already added:
```python
# Examples API
GET /api/examples/list
GET /api/examples/load?path={path}

# Documentation API  
GET /api/docs/{doc_path:path}
```

### Start Backend

```bash
# From neural/aquarium/backend/
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Or:

```bash
cd neural/aquarium/backend
python server.py
```

## Testing the Integration

1. **Start Backend Server**:
   ```bash
   cd neural/aquarium/backend
   python server.py
   ```

2. **Start Frontend**:
   ```bash
   cd neural/aquarium
   npm start
   ```

3. **Test Checklist**:
   - [ ] Welcome screen appears on page load
   - [ ] Can navigate between all tabs
   - [ ] Templates load into the workspace
   - [ ] Examples load from backend
   - [ ] Tutorial progresses through steps
   - [ ] Close button works
   - [ ] Help button reopens welcome
   - [ ] No console errors

## Troubleshooting

### Welcome Screen Doesn't Appear

Check that:
- State is initialized correctly: `useState(true)`
- Component is imported: `import { WelcomeScreen } from './components/welcome'`
- Component is in JSX before other content

### Templates Don't Load

Verify:
- `handleLoadTemplate` function is connected
- `setCurrentDSL` and `setAppliedDSL` update correctly
- Check browser console for errors

### Examples Don't Load

Check:
- Backend server is running on port 8000
- API endpoint returns data: `http://localhost:8000/api/examples/list`
- CORS is configured in backend
- Network tab in browser DevTools shows successful request

### Tutorial Doesn't Highlight Elements

Ensure:
- Target elements exist in DOM with correct class names
- CSS for `.tutorial-highlight` is loaded
- No z-index conflicts with other elements

### Styling Issues

Verify:
- All CSS files are imported
- No conflicting styles from parent components
- Dark theme colors are applied

## Production Deployment

### Disable Welcome Screen After First Visit

```tsx
useEffect(() => {
  // Check if user has visited before
  const hasVisited = localStorage.getItem('hasVisitedAquarium');
  if (hasVisited) {
    setShowWelcome(false);
  }
}, []);

const handleCloseWelcome = () => {
  localStorage.setItem('hasVisitedAquarium', 'true');
  setShowWelcome(false);
};
```

### Add Analytics

```tsx
const handleLoadTemplate = (dslCode: string) => {
  // Track template usage
  if (window.gtag) {
    window.gtag('event', 'load_template', {
      template_type: dslCode.match(/network (\w+)/)?.[1]
    });
  }
  
  setCurrentDSL(dslCode);
  setShowWelcome(false);
};
```

### Optimize Bundle Size

The welcome components are already optimized:
- No heavy dependencies
- CSS is modular
- Components lazy-load as needed

## Support

For issues or questions:
1. Check `WELCOME_SCREEN_IMPLEMENTATION.md` for detailed info
2. Review `src/components/welcome/README.md` for component docs
3. Examine example integration in this guide
4. Check browser console for errors
5. Verify backend endpoints are accessible

## Summary

The welcome screen integration requires:
1. Import components
2. Add state management
3. Create handler functions
4. Add components to JSX
5. Start backend server
6. Test all features

With these steps, your Neural Aquarium IDE will have a complete onboarding experience for new users!
