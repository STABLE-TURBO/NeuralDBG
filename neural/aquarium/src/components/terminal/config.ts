export interface TerminalTheme {
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

export const DEFAULT_THEME: TerminalTheme = {
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
};

export const SOLARIZED_DARK: TerminalTheme = {
  background: '#002b36',
  foreground: '#839496',
  cursor: '#839496',
  black: '#073642',
  red: '#dc322f',
  green: '#859900',
  yellow: '#b58900',
  blue: '#268bd2',
  magenta: '#d33682',
  cyan: '#2aa198',
  white: '#eee8d5',
  brightBlack: '#002b36',
  brightRed: '#cb4b16',
  brightGreen: '#586e75',
  brightYellow: '#657b83',
  brightBlue: '#839496',
  brightMagenta: '#6c71c4',
  brightCyan: '#93a1a1',
  brightWhite: '#fdf6e3',
};

export const MONOKAI: TerminalTheme = {
  background: '#272822',
  foreground: '#f8f8f2',
  cursor: '#f8f8f0',
  black: '#272822',
  red: '#f92672',
  green: '#a6e22e',
  yellow: '#f4bf75',
  blue: '#66d9ef',
  magenta: '#ae81ff',
  cyan: '#a1efe4',
  white: '#f8f8f2',
  brightBlack: '#75715e',
  brightRed: '#f92672',
  brightGreen: '#a6e22e',
  brightYellow: '#f4bf75',
  brightBlue: '#66d9ef',
  brightMagenta: '#ae81ff',
  brightCyan: '#a1efe4',
  brightWhite: '#f9f8f5',
};

export const DRACULA: TerminalTheme = {
  background: '#282a36',
  foreground: '#f8f8f2',
  cursor: '#f8f8f2',
  black: '#21222c',
  red: '#ff5555',
  green: '#50fa7b',
  yellow: '#f1fa8c',
  blue: '#bd93f9',
  magenta: '#ff79c6',
  cyan: '#8be9fd',
  white: '#f8f8f2',
  brightBlack: '#6272a4',
  brightRed: '#ff6e6e',
  brightGreen: '#69ff94',
  brightYellow: '#ffffa5',
  brightBlue: '#d6acff',
  brightMagenta: '#ff92df',
  brightCyan: '#a4ffff',
  brightWhite: '#ffffff',
};

export interface TerminalConfig {
  fontSize: number;
  fontFamily: string;
  cursorBlink: boolean;
  cursorStyle: 'block' | 'underline' | 'bar';
  scrollback: number;
  theme: TerminalTheme;
  defaultShell: string;
  websocketUrl: string;
}

export const DEFAULT_CONFIG: TerminalConfig = {
  fontSize: 14,
  fontFamily: 'Menlo, Monaco, "Courier New", monospace',
  cursorBlink: true,
  cursorStyle: 'block',
  scrollback: 10000,
  theme: DEFAULT_THEME,
  defaultShell: 'bash',
  websocketUrl: 'ws://localhost:5000/terminal',
};

export const NEURAL_CLI_COMMANDS = [
  'neural',
  'neural compile',
  'neural run',
  'neural visualize',
  'neural debug',
  'neural hpo',
  'neural automl',
  'neural help',
  'neural version',
];

export const COMMON_COMMANDS = [
  'ls',
  'cd',
  'pwd',
  'cat',
  'echo',
  'grep',
  'find',
  'mkdir',
  'rm',
  'cp',
  'mv',
  'touch',
  'chmod',
  'python',
  'pip',
  'npm',
  'git',
  'node',
];

export const AUTOCOMPLETE_CONFIG = {
  maxSuggestions: 10,
  minPrefixLength: 1,
  caseSensitive: false,
  includeNeuralCommands: true,
  includeCommonCommands: true,
  includePathCompletion: true,
};

export const KEYBOARD_SHORTCUTS = {
  clear: 'Ctrl+L',
  copy: 'Ctrl+Shift+C',
  paste: 'Ctrl+Shift+V',
  newTerminal: 'Ctrl+Shift+T',
  closeTerminal: 'Ctrl+Shift+W',
  nextTerminal: 'Ctrl+Tab',
  previousTerminal: 'Ctrl+Shift+Tab',
  search: 'Ctrl+F',
  interrupt: 'Ctrl+C',
};

export const TERMINAL_THEMES = {
  default: DEFAULT_THEME,
  solarized: SOLARIZED_DARK,
  monokai: MONOKAI,
  dracula: DRACULA,
};

export const AVAILABLE_FONTS = [
  'Menlo, Monaco, "Courier New", monospace',
  '"Fira Code", monospace',
  '"Source Code Pro", monospace',
  '"JetBrains Mono", monospace',
  '"Cascadia Code", monospace',
  'Consolas, monospace',
];

export const FONT_SIZES = [10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24];

export function loadConfig(): TerminalConfig {
  const stored = localStorage.getItem('terminal-config');
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      return { ...DEFAULT_CONFIG, ...parsed };
    } catch (e) {
      console.error('Failed to load terminal config:', e);
    }
  }
  return DEFAULT_CONFIG;
}

export function saveConfig(config: Partial<TerminalConfig>): void {
  try {
    const current = loadConfig();
    const updated = { ...current, ...config };
    localStorage.setItem('terminal-config', JSON.stringify(updated));
  } catch (e) {
    console.error('Failed to save terminal config:', e);
  }
}

export function resetConfig(): void {
  localStorage.removeItem('terminal-config');
}
