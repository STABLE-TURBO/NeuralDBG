#!/usr/bin/env python3
"""
Create integration examples for the Plugin Marketplace
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
MARKETPLACE_DIR = BASE_DIR / "src" / "components" / "marketplace"

MARKETPLACE_DIR.mkdir(parents=True, exist_ok=True)

integration_example = '''/**
 * Example of integrating PluginMarketplace into the main Neural Aquarium app
 * 
 * This file demonstrates how to add the plugin marketplace to your application.
 */

import React, { useState } from 'react';
import PluginMarketplace from './PluginMarketplace';

/**
 * Example 1: As a modal/overlay
 */
export const PluginMarketplaceModal: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button onClick={() => setIsOpen(true)}>
        🔌 Plugins
      </button>

      {isOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '24px',
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '16px',
            width: '100%',
            maxWidth: '1400px',
            height: '90vh',
            overflow: 'auto',
            position: 'relative',
          }}>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                position: 'absolute',
                top: '16px',
                right: '16px',
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                zIndex: 1001,
              }}
            >
              ×
            </button>
            <PluginMarketplace />
          </div>
        </div>
      )}
    </>
  );
};

/**
 * Example 2: Add to existing App.tsx
 */
export const ExampleAppIntegration = `
// In your neural/aquarium/src/App.tsx:

import React, { useState } from 'react';
import AIAssistantSidebar from './components/ai/AIAssistantSidebar';
import PluginMarketplace from './components/marketplace/PluginMarketplace';
import './App.css';

function App() {
  const [currentDSL, setCurrentDSL] = useState<string>('');
  const [appliedDSL, setAppliedDSL] = useState<string>('');
  const [showPlugins, setShowPlugins] = useState(false);

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
      <header className="App-header">
        <h1>Neural Aquarium</h1>
        <p>AI-Powered Neural DSL Builder</p>
        <nav style={{ marginTop: '16px' }}>
          <button 
            onClick={() => setShowPlugins(!showPlugins)}
            style={{
              padding: '8px 16px',
              fontSize: '14px',
              cursor: 'pointer',
              backgroundColor: showPlugins ? '#0066cc' : '#f5f5f5',
              color: showPlugins ? 'white' : '#333',
              border: 'none',
              borderRadius: '4px',
            }}
          >
            🔌 {showPlugins ? 'Hide' : 'Show'} Plugins
          </button>
        </nav>
      </header>

      <main className="App-main">
        {showPlugins ? (
          <PluginMarketplace />
        ) : (
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
                  <p>Use the AI Assistant to create a neural network model.</p>
                  <p>Your generated DSL code will appear here once applied.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <AIAssistantSidebar
        onDSLGenerated={handleDSLGenerated}
        onDSLApplied={handleDSLApplied}
      />
    </div>
  );
}

export default App;
`;

export default PluginMarketplaceModal;
'''

with open(MARKETPLACE_DIR / "IntegrationExample.tsx", 'w', encoding='utf-8') as f:
    f.write(integration_example)

print(f"✅ Created: {MARKETPLACE_DIR / 'IntegrationExample.tsx'}")
print("\n💡 Integration examples created!")
print("   See IntegrationExample.tsx for usage examples")
