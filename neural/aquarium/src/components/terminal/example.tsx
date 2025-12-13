import React, { useState } from 'react';
import { TerminalPanel, Terminal } from './index';

export const TerminalPanelExample: React.FC = () => {
  return (
    <div style={{ height: '600px', width: '100%' }}>
      <h2>Terminal Panel Example</h2>
      <p>Full-featured terminal with tabs and split views</p>
      <TerminalPanel />
    </div>
  );
};

export const SingleTerminalExample: React.FC = () => {
  const [showTerminal, setShowTerminal] = useState(true);

  return (
    <div style={{ height: '500px', width: '100%' }}>
      <h2>Single Terminal Example</h2>
      <button onClick={() => setShowTerminal(!showTerminal)}>
        {showTerminal ? 'Hide' : 'Show'} Terminal
      </button>
      {showTerminal && (
        <Terminal
          sessionId="example-terminal"
          onClose={() => setShowTerminal(false)}
          isActive={true}
        />
      )}
    </div>
  );
};

export const EmbeddedTerminalExample: React.FC = () => {
  return (
    <div style={{ display: 'flex', height: '600px', gap: '10px' }}>
      <div style={{ flex: 1, border: '1px solid #ccc' }}>
        <h3>Code Editor</h3>
        <textarea
          style={{ width: '100%', height: '90%' }}
          placeholder="Write your Neural DSL code here..."
        />
      </div>
      <div style={{ flex: 1, border: '1px solid #ccc' }}>
        <h3>Terminal</h3>
        <div style={{ height: '90%' }}>
          <Terminal sessionId="embedded-terminal" isActive={true} />
        </div>
      </div>
    </div>
  );
};

export const TerminalWorkspaceExample: React.FC = () => {
  const [layout, setLayout] = useState<'editor' | 'terminal' | 'both'>('both');

  return (
    <div style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '10px', borderBottom: '1px solid #ccc' }}>
        <h2>Workspace Layout Example</h2>
        <div>
          <button onClick={() => setLayout('editor')}>Editor Only</button>
          <button onClick={() => setLayout('terminal')}>Terminal Only</button>
          <button onClick={() => setLayout('both')}>Split View</button>
        </div>
      </div>
      <div style={{ flex: 1, display: 'flex' }}>
        {(layout === 'editor' || layout === 'both') && (
          <div style={{ flex: layout === 'both' ? 1 : 2, borderRight: '1px solid #ccc' }}>
            <h3>Editor</h3>
            <pre style={{ padding: '10px' }}>
              {`Network MNIST_CNN {
  Input(shape=(28, 28, 1))
  Conv2D(filters=32, kernel_size=3, activation=relu)
  MaxPooling2D(pool_size=2)
  Conv2D(filters=64, kernel_size=3, activation=relu)
  MaxPooling2D(pool_size=2)
  Flatten()
  Dense(units=128, activation=relu)
  Dropout(rate=0.5)
  Output(units=10, activation=softmax)
}`}
            </pre>
          </div>
        )}
        {(layout === 'terminal' || layout === 'both') && (
          <div style={{ flex: 1 }}>
            <TerminalPanel />
          </div>
        )}
      </div>
    </div>
  );
};

const TerminalExamples: React.FC = () => {
  const [activeExample, setActiveExample] = useState<string>('panel');

  const examples = {
    panel: <TerminalPanelExample />,
    single: <SingleTerminalExample />,
    embedded: <EmbeddedTerminalExample />,
    workspace: <TerminalWorkspaceExample />,
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Terminal Component Examples</h1>
      <div style={{ marginBottom: '20px' }}>
        <button onClick={() => setActiveExample('panel')}>Terminal Panel</button>
        <button onClick={() => setActiveExample('single')}>Single Terminal</button>
        <button onClick={() => setActiveExample('embedded')}>Embedded Terminal</button>
        <button onClick={() => setActiveExample('workspace')}>Workspace Layout</button>
      </div>
      {examples[activeExample as keyof typeof examples]}
    </div>
  );
};

export default TerminalExamples;
