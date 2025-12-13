export interface TerminalSession {
  id: string;
  name: string;
  active: boolean;
  shell?: string;
}

export interface TerminalMessage {
  type: 'output' | 'prompt' | 'command' | 'shell_change' | 'autocomplete' | 'error';
  data?: string;
  shell?: string;
  suggestions?: string[];
}

export interface CommandHistoryEntry {
  command: string;
  timestamp: number;
  exitCode?: number;
}

export interface ShellConfig {
  name: string;
  prompt: string;
  executable: string;
  args: string[];
}

export const AVAILABLE_SHELLS: { [key: string]: ShellConfig } = {
  bash: {
    name: 'Bash',
    prompt: '$ ',
    executable: 'bash',
    args: ['-i'],
  },
  zsh: {
    name: 'Zsh',
    prompt: '% ',
    executable: 'zsh',
    args: ['-i'],
  },
  sh: {
    name: 'Sh',
    prompt: '$ ',
    executable: 'sh',
    args: ['-i'],
  },
  powershell: {
    name: 'PowerShell',
    prompt: 'PS> ',
    executable: 'powershell',
    args: ['-NoLogo', '-NoExit'],
  },
  cmd: {
    name: 'CMD',
    prompt: '> ',
    executable: 'cmd',
    args: ['/K'],
  },
};
