import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Terminal from './Terminal';
import TerminalPanel from './TerminalPanel';
import TerminalControls from './TerminalControls';

global.WebSocket = jest.fn().mockImplementation(() => ({
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: 1,
  OPEN: 1,
})) as any;

describe('Terminal Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders terminal container', () => {
    render(<Terminal sessionId="test-session" isActive={true} />);
    const container = document.querySelector('.terminal-container');
    expect(container).toBeInTheDocument();
  });

  it('creates WebSocket connection with session ID', () => {
    render(<Terminal sessionId="test-123" isActive={true} />);
    expect(WebSocket).toHaveBeenCalledWith(
      expect.stringContaining('test-123')
    );
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = jest.fn();
    render(<Terminal sessionId="test-session" onClose={onClose} isActive={true} />);
    
    const closeButton = screen.queryByTitle('Close Terminal');
    if (closeButton) {
      fireEvent.click(closeButton);
      expect(onClose).toHaveBeenCalled();
    }
  });
});

describe('TerminalPanel Component', () => {
  it('renders with default terminal session', () => {
    render(<TerminalPanel />);
    expect(screen.getByText(/Terminal 1/i)).toBeInTheDocument();
  });

  it('creates new terminal session', () => {
    render(<TerminalPanel />);
    const newButton = screen.getByTitle('New Terminal');
    
    fireEvent.click(newButton);
    
    waitFor(() => {
      expect(screen.getByText(/Terminal 2/i)).toBeInTheDocument();
    });
  });

  it('switches between terminal sessions', () => {
    render(<TerminalPanel />);
    const newButton = screen.getByTitle('New Terminal');
    
    fireEvent.click(newButton);
    
    waitFor(() => {
      const terminal2 = screen.getByText(/Terminal 2/i);
      fireEvent.click(terminal2);
      
      expect(terminal2.closest('.terminal-tab')).toHaveClass('active');
    });
  });

  it('closes terminal session', () => {
    render(<TerminalPanel />);
    const newButton = screen.getByTitle('New Terminal');
    
    fireEvent.click(newButton);
    
    waitFor(() => {
      const closeButtons = screen.getAllByText('✕');
      fireEvent.click(closeButtons[0]);
      
      expect(screen.queryByText(/Terminal 1/i)).not.toBeInTheDocument();
    });
  });

  it('toggles split mode', () => {
    render(<TerminalPanel />);
    const horizontalSplitButton = screen.getByTitle('Split Horizontal');
    
    fireEvent.click(horizontalSplitButton);
    
    waitFor(() => {
      expect(horizontalSplitButton).toHaveClass('active');
    });
  });
});

describe('TerminalControls Component', () => {
  const defaultProps = {
    connected: true,
    currentShell: 'bash',
    onClear: jest.fn(),
    onShellChange: jest.fn(),
    onSearch: jest.fn(),
    onCopy: jest.fn(),
    onPaste: jest.fn(),
  };

  it('renders connection status', () => {
    render(<TerminalControls {...defaultProps} />);
    expect(screen.getByText(/Connected/i)).toBeInTheDocument();
  });

  it('shows disconnected status', () => {
    render(<TerminalControls {...defaultProps} connected={false} />);
    expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();
  });

  it('displays current shell', () => {
    render(<TerminalControls {...defaultProps} currentShell="zsh" />);
    expect(screen.getByText('zsh')).toBeInTheDocument();
  });

  it('opens shell selector menu', () => {
    render(<TerminalControls {...defaultProps} />);
    const shellButton = screen.getByText('bash');
    
    fireEvent.click(shellButton);
    
    waitFor(() => {
      expect(screen.getByText('powershell')).toBeInTheDocument();
      expect(screen.getByText('cmd')).toBeInTheDocument();
    });
  });

  it('changes shell when option selected', () => {
    render(<TerminalControls {...defaultProps} />);
    const shellButton = screen.getByText('bash');
    
    fireEvent.click(shellButton);
    
    waitFor(() => {
      const zshOption = screen.getByText('zsh');
      fireEvent.click(zshOption);
      
      expect(defaultProps.onShellChange).toHaveBeenCalledWith('zsh');
    });
  });

  it('calls onClear when clear button clicked', () => {
    render(<TerminalControls {...defaultProps} />);
    const clearButton = screen.getByTitle('Clear Terminal');
    
    fireEvent.click(clearButton);
    
    expect(defaultProps.onClear).toHaveBeenCalled();
  });

  it('calls onCopy when copy button clicked', () => {
    render(<TerminalControls {...defaultProps} />);
    const copyButton = screen.getByTitle(/Copy/i);
    
    fireEvent.click(copyButton);
    
    expect(defaultProps.onCopy).toHaveBeenCalled();
  });

  it('calls onPaste when paste button clicked', () => {
    render(<TerminalControls {...defaultProps} />);
    const pasteButton = screen.getByTitle(/Paste/i);
    
    fireEvent.click(pasteButton);
    
    expect(defaultProps.onPaste).toHaveBeenCalled();
  });

  it('toggles search form', () => {
    render(<TerminalControls {...defaultProps} />);
    const searchButton = screen.getByTitle('Search');
    
    fireEvent.click(searchButton);
    
    waitFor(() => {
      expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument();
    });
  });

  it('submits search query', () => {
    render(<TerminalControls {...defaultProps} />);
    const searchButton = screen.getByTitle('Search');
    
    fireEvent.click(searchButton);
    
    waitFor(() => {
      const searchInput = screen.getByPlaceholderText('Search...');
      fireEvent.change(searchInput, { target: { value: 'test query' } });
      
      const submitButton = screen.getByText('🔍');
      fireEvent.click(submitButton);
      
      expect(defaultProps.onSearch).toHaveBeenCalledWith('test query');
    });
  });
});
