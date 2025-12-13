import React, { useState } from 'react';
import './TerminalControls.css';

interface TerminalControlsProps {
  connected: boolean;
  currentShell: string;
  onClear: () => void;
  onShellChange: (shell: string) => void;
  onSearch: (term: string) => void;
  onCopy: () => void;
  onPaste: () => void;
  onClose?: () => void;
}

const TerminalControls: React.FC<TerminalControlsProps> = ({
  connected,
  currentShell,
  onClear,
  onShellChange,
  onSearch,
  onCopy,
  onPaste,
  onClose,
}) => {
  const [showShellMenu, setShowShellMenu] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [showSearch, setShowSearch] = useState(false);

  const shells = ['bash', 'zsh', 'sh', 'powershell', 'cmd'];

  const handleShellSelect = (shell: string) => {
    onShellChange(shell);
    setShowShellMenu(false);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchTerm) {
      onSearch(searchTerm);
    }
  };

  return (
    <div className="terminal-controls">
      <div className="terminal-controls-left">
        <div className="connection-status">
          <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
          <span className="status-text">
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        <div className="shell-selector">
          <button
            className="shell-button"
            onClick={() => setShowShellMenu(!showShellMenu)}
          >
            <span className="shell-icon">$</span>
            {currentShell}
            <span className="dropdown-arrow">▼</span>
          </button>
          {showShellMenu && (
            <div className="shell-menu">
              {shells.map((shell) => (
                <button
                  key={shell}
                  className={`shell-option ${shell === currentShell ? 'active' : ''}`}
                  onClick={() => handleShellSelect(shell)}
                >
                  {shell}
                  {shell === currentShell && <span className="checkmark">✓</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="terminal-controls-right">
        {showSearch && (
          <form onSubmit={handleSearchSubmit} className="search-form">
            <input
              type="text"
              className="search-input"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              autoFocus
            />
            <button type="submit" className="search-submit">
              🔍
            </button>
          </form>
        )}

        <button
          className="control-button"
          onClick={() => setShowSearch(!showSearch)}
          title="Search"
        >
          🔍
        </button>

        <button
          className="control-button"
          onClick={onCopy}
          title="Copy (Ctrl+Shift+C)"
        >
          📋
        </button>

        <button
          className="control-button"
          onClick={onPaste}
          title="Paste (Ctrl+Shift+V)"
        >
          📄
        </button>

        <button
          className="control-button"
          onClick={onClear}
          title="Clear Terminal"
        >
          🗑️
        </button>

        {onClose && (
          <button
            className="control-button close-button"
            onClick={onClose}
            title="Close Terminal"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

export default TerminalControls;
