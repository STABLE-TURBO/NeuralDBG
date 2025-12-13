import React, { useState, useCallback } from 'react';
import Terminal from './Terminal';
import './TerminalPanel.css';
import { TerminalSession } from './types';

const TerminalPanel: React.FC = () => {
  const [sessions, setSessions] = useState<TerminalSession[]>([
    { id: 'terminal-1', name: 'Terminal 1', active: true },
  ]);
  const [activeSessionId, setActiveSessionId] = useState<string>('terminal-1');
  const [splitMode, setSplitMode] = useState<'single' | 'horizontal' | 'vertical'>('single');

  const createNewSession = useCallback(() => {
    const newId = `terminal-${Date.now()}`;
    const newSession: TerminalSession = {
      id: newId,
      name: `Terminal ${sessions.length + 1}`,
      active: true,
    };
    
    setSessions((prev) => [...prev, newSession]);
    setActiveSessionId(newId);
  }, [sessions.length]);

  const closeSession = useCallback((sessionId: string) => {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== sessionId);
      if (filtered.length === 0) {
        const defaultSession: TerminalSession = {
          id: 'terminal-default',
          name: 'Terminal 1',
          active: true,
        };
        setActiveSessionId(defaultSession.id);
        return [defaultSession];
      }
      
      if (sessionId === activeSessionId && filtered.length > 0) {
        setActiveSessionId(filtered[0].id);
      }
      
      return filtered;
    });
  }, [activeSessionId]);

  const switchSession = useCallback((sessionId: string) => {
    setActiveSessionId(sessionId);
  }, []);

  const toggleSplitMode = useCallback((mode: 'single' | 'horizontal' | 'vertical') => {
    setSplitMode(mode);
  }, []);

  const renderTerminals = () => {
    if (splitMode === 'single') {
      const activeSession = sessions.find((s) => s.id === activeSessionId);
      return activeSession ? (
        <Terminal
          key={activeSession.id}
          sessionId={activeSession.id}
          onClose={sessions.length > 1 ? () => closeSession(activeSession.id) : undefined}
          isActive={true}
        />
      ) : null;
    }

    if (splitMode === 'horizontal') {
      const visibleSessions = sessions.slice(0, 2);
      return (
        <div className="terminal-split horizontal">
          {visibleSessions.map((session, index) => (
            <div key={session.id} className="terminal-pane">
              <Terminal
                sessionId={session.id}
                onClose={sessions.length > 1 ? () => closeSession(session.id) : undefined}
                isActive={session.id === activeSessionId}
              />
            </div>
          ))}
        </div>
      );
    }

    if (splitMode === 'vertical') {
      const visibleSessions = sessions.slice(0, 2);
      return (
        <div className="terminal-split vertical">
          {visibleSessions.map((session, index) => (
            <div key={session.id} className="terminal-pane">
              <Terminal
                sessionId={session.id}
                onClose={sessions.length > 1 ? () => closeSession(session.id) : undefined}
                isActive={session.id === activeSessionId}
              />
            </div>
          ))}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="terminal-panel">
      <div className="terminal-header">
        <div className="terminal-tabs">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`terminal-tab ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => switchSession(session.id)}
            >
              <span className="tab-name">{session.name}</span>
              {sessions.length > 1 && (
                <button
                  className="tab-close"
                  onClick={(e) => {
                    e.stopPropagation();
                    closeSession(session.id);
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          ))}
          <button className="new-terminal-button" onClick={createNewSession} title="New Terminal">
            +
          </button>
        </div>

        <div className="terminal-actions">
          <div className="split-controls">
            <button
              className={`split-button ${splitMode === 'single' ? 'active' : ''}`}
              onClick={() => toggleSplitMode('single')}
              title="Single Panel"
            >
              ▭
            </button>
            <button
              className={`split-button ${splitMode === 'horizontal' ? 'active' : ''}`}
              onClick={() => toggleSplitMode('horizontal')}
              title="Split Horizontal"
              disabled={sessions.length < 2}
            >
              ▬
            </button>
            <button
              className={`split-button ${splitMode === 'vertical' ? 'active' : ''}`}
              onClick={() => toggleSplitMode('vertical')}
              title="Split Vertical"
              disabled={sessions.length < 2}
            >
              ▌▌
            </button>
          </div>
        </div>
      </div>

      <div className="terminal-content">
        {renderTerminals()}
      </div>
    </div>
  );
};

export default TerminalPanel;
