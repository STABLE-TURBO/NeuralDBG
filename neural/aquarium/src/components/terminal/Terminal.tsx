import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Terminal as XTerm } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { WebLinksAddon } from 'xterm-addon-web-links';
import { SearchAddon } from 'xterm-addon-search';
import 'xterm/css/xterm.css';
import './Terminal.css';
import TerminalControls from './TerminalControls';
import { TerminalSession } from './types';

interface TerminalProps {
  sessionId: string;
  onClose?: () => void;
  isActive?: boolean;
}

const Terminal: React.FC<TerminalProps> = ({ sessionId, onClose, isActive = true }) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<XTerm | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [currentShell, setCurrentShell] = useState<string>('bash');
  const commandHistoryRef = useRef<string[]>([]);
  const historyIndexRef = useRef<number>(-1);
  const currentLineRef = useRef<string>('');

  useEffect(() => {
    if (!terminalRef.current || xtermRef.current) return;

    const term = new XTerm({
      cursorBlink: true,
      cursorStyle: 'block',
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: '#1e1e1e',
        foreground: '#d4d4d4',
        cursor: '#d4d4d4',
        black: '#000000',
        red: '#cd3131',
        green: '#0dbc79',
        yellow: '#e5e510',
        blue: '#2472c8',
        magenta: '#bc3fbc',
        cyan: '#11a8cd',
        white: '#e5e5e5',
        brightBlack: '#666666',
        brightRed: '#f14c4c',
        brightGreen: '#23d18b',
        brightYellow: '#f5f543',
        brightBlue: '#3b8eea',
        brightMagenta: '#d670d6',
        brightCyan: '#29b8db',
        brightWhite: '#e5e5e5',
      },
      rows: 30,
      cols: 100,
      scrollback: 10000,
      allowTransparency: false,
      convertEol: true,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();
    const searchAddon = new SearchAddon();

    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);
    term.loadAddon(searchAddon);

    term.open(terminalRef.current);
    fitAddon.fit();

    xtermRef.current = term;
    fitAddonRef.current = fitAddon;

    connectToBackend(term);

    const handleResize = () => {
      if (fitAddon && isActive) {
        setTimeout(() => fitAddon.fit(), 100);
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (wsRef.current) {
        wsRef.current.close();
      }
      term.dispose();
    };
  }, [sessionId]);

  useEffect(() => {
    if (fitAddonRef.current && isActive) {
      setTimeout(() => fitAddonRef.current?.fit(), 100);
    }
  }, [isActive]);

  const connectToBackend = (term: XTerm) => {
    try {
      const ws = new WebSocket(`ws://localhost:5000/terminal/${sessionId}`);

      ws.onopen = () => {
        setConnected(true);
        term.writeln('\x1b[1;32mConnected to terminal backend\x1b[0m');
        term.writeln('');
        term.write('$ ');
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'output') {
          term.write(data.data);
        } else if (data.type === 'prompt') {
          term.write(data.data);
        } else if (data.type === 'shell_change') {
          setCurrentShell(data.shell);
          term.writeln(`\x1b[1;36mShell changed to: ${data.shell}\x1b[0m`);
        } else if (data.type === 'autocomplete') {
          handleAutocomplete(term, data.suggestions);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        term.writeln('\x1b[1;31mConnection error\x1b[0m');
      };

      ws.onclose = () => {
        setConnected(false);
        term.writeln('\x1b[1;31mDisconnected from terminal backend\x1b[0m');
        setTimeout(() => connectToBackend(term), 3000);
      };

      let currentLine = '';
      
      term.onData((data) => {
        if (ws.readyState !== WebSocket.OPEN) return;

        const code = data.charCodeAt(0);

        if (code === 13) {
          term.write('\r\n');
          
          if (currentLine.trim()) {
            commandHistoryRef.current.push(currentLine);
            historyIndexRef.current = commandHistoryRef.current.length;
            
            ws.send(JSON.stringify({
              type: 'command',
              data: currentLine,
            }));
          } else {
            term.write('$ ');
          }
          
          currentLine = '';
          currentLineRef.current = '';
        } else if (code === 127) {
          if (currentLine.length > 0) {
            currentLine = currentLine.slice(0, -1);
            currentLineRef.current = currentLine;
            term.write('\b \b');
          }
        } else if (code === 9) {
          ws.send(JSON.stringify({
            type: 'autocomplete',
            data: currentLine,
          }));
        } else if (code === 27) {
          const seq = data.substring(1);
          if (seq === '[A') {
            navigateHistory(term, 'up');
          } else if (seq === '[B') {
            navigateHistory(term);
          }
        } else if (code >= 32) {
          currentLine += data;
          currentLineRef.current = currentLine;
          term.write(data);
        }
      });

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to connect to terminal backend:', error);
      term.writeln('\x1b[1;31mFailed to connect to terminal backend\x1b[0m');
    }
  };

  const navigateHistory = (term: XTerm, direction: 'up' | 'down' = 'down') => {
    const history = commandHistoryRef.current;
    
    if (direction === 'up' && historyIndexRef.current > 0) {
      historyIndexRef.current--;
    } else if (direction === 'down' && historyIndexRef.current < history.length - 1) {
      historyIndexRef.current++;
    } else if (direction === 'down' && historyIndexRef.current === history.length - 1) {
      historyIndexRef.current = history.length;
      clearCurrentLine(term);
      return;
    } else {
      return;
    }

    const command = history[historyIndexRef.current];
    clearCurrentLine(term);
    term.write(command);
    currentLineRef.current = command;
  };

  const clearCurrentLine = (term: XTerm) => {
    const currentLine = currentLineRef.current;
    for (let i = 0; i < currentLine.length; i++) {
      term.write('\b \b');
    }
    currentLineRef.current = '';
  };

  const handleAutocomplete = (term: XTerm, suggestions: string[]) => {
    if (suggestions.length === 0) return;

    if (suggestions.length === 1) {
      const completion = suggestions[0];
      const currentLine = currentLineRef.current;
      const parts = currentLine.split(' ');
      const lastPart = parts[parts.length - 1];
      const toAdd = completion.substring(lastPart.length);
      
      term.write(toAdd);
      currentLineRef.current += toAdd;
    } else {
      term.write('\r\n');
      term.writeln(suggestions.join('  '));
      term.write('$ ' + currentLineRef.current);
    }
  };

  const handleClear = useCallback(() => {
    if (xtermRef.current) {
      xtermRef.current.clear();
      xtermRef.current.write('$ ');
    }
  }, []);

  const handleShellChange = useCallback((shell: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'change_shell',
        shell,
      }));
    }
  }, []);

  const handleSearch = useCallback((term: string) => {
    if (xtermRef.current && term) {
      const searchAddon = new SearchAddon();
      xtermRef.current.loadAddon(searchAddon);
      searchAddon.findNext(term);
    }
  }, []);

  const handleCopy = useCallback(() => {
    if (xtermRef.current) {
      const selection = xtermRef.current.getSelection();
      if (selection) {
        navigator.clipboard.writeText(selection);
      }
    }
  }, []);

  const handlePaste = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (xtermRef.current && wsRef.current) {
        xtermRef.current.write(text);
        currentLineRef.current += text;
      }
    } catch (error) {
      console.error('Failed to paste:', error);
    }
  }, []);

  return (
    <div className="terminal-container">
      <TerminalControls
        connected={connected}
        currentShell={currentShell}
        onClear={handleClear}
        onShellChange={handleShellChange}
        onSearch={handleSearch}
        onCopy={handleCopy}
        onPaste={handlePaste}
        onClose={onClose}
      />
      <div ref={terminalRef} className="terminal-wrapper" />
    </div>
  );
};

export default Terminal;
