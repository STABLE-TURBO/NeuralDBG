export function generateSessionId(): string {
  return `terminal-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}

export function formatCommand(command: string): string {
  return command.trim();
}

export function parseCommandOutput(output: string): {
  text: string;
  exitCode?: number;
  error?: boolean;
} {
  const lines = output.split('\n');
  let exitCode: number | undefined;
  let error = false;

  const lastLine = lines[lines.length - 1];
  const exitCodeMatch = lastLine.match(/exit code: (\d+)/i);
  
  if (exitCodeMatch) {
    exitCode = parseInt(exitCodeMatch[1], 10);
    error = exitCode !== 0;
  }

  if (output.toLowerCase().includes('error')) {
    error = true;
  }

  return {
    text: output,
    exitCode,
    error,
  };
}

export function escapeAnsiCodes(text: string): string {
  return text.replace(/\x1b\[[0-9;]*m/g, '');
}

export function getAnsiColor(code: string): string {
  const colorMap: { [key: string]: string } = {
    '30': '#000000',
    '31': '#cd3131',
    '32': '#0dbc79',
    '33': '#e5e510',
    '34': '#2472c8',
    '35': '#bc3fbc',
    '36': '#11a8cd',
    '37': '#e5e5e5',
    '90': '#666666',
    '91': '#f14c4c',
    '92': '#23d18b',
    '93': '#f5f543',
    '94': '#3b8eea',
    '95': '#d670d6',
    '96': '#29b8db',
    '97': '#e5e5e5',
  };

  return colorMap[code] || '#d4d4d4';
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  return `${hours}h ${minutes}m`;
}

export function isNeuralCommand(command: string): boolean {
  const cmd = command.trim().toLowerCase();
  return cmd.startsWith('neural ') || cmd === 'neural';
}

export function parseNeuralCommand(command: string): {
  action: string;
  args: string[];
  flags: { [key: string]: string | boolean };
} {
  const parts = command.trim().split(/\s+/);
  
  if (parts[0] !== 'neural') {
    return { action: '', args: [], flags: {} };
  }

  const action = parts[1] || '';
  const args: string[] = [];
  const flags: { [key: string]: string | boolean } = {};

  for (let i = 2; i < parts.length; i++) {
    const part = parts[i];
    
    if (part.startsWith('--')) {
      const flagName = part.substring(2);
      const nextPart = parts[i + 1];
      
      if (nextPart && !nextPart.startsWith('-')) {
        flags[flagName] = nextPart;
        i++;
      } else {
        flags[flagName] = true;
      }
    } else if (part.startsWith('-')) {
      flags[part.substring(1)] = true;
    } else {
      args.push(part);
    }
  }

  return { action, args, flags };
}

export function getNeuralCommandHelp(action: string): string {
  const helpMap: { [key: string]: string } = {
    compile: 'Compile Neural DSL to backend code\nUsage: neural compile <file.neural> [--backend tensorflow|pytorch|onnx]',
    run: 'Execute Neural DSL model\nUsage: neural run <file.neural> [--backend tensorflow|pytorch]',
    visualize: 'Visualize model architecture\nUsage: neural visualize <file.neural>',
    debug: 'Start debugging session\nUsage: neural debug <file.neural>',
    hpo: 'Run hyperparameter optimization\nUsage: neural hpo <file.neural> [--trials 100]',
    automl: 'Run AutoML and Neural Architecture Search\nUsage: neural automl <file.neural>',
    help: 'Show help information\nUsage: neural help [command]',
    version: 'Show version information\nUsage: neural version',
  };

  return helpMap[action] || 'Unknown command. Type "neural help" for available commands.';
}

export function highlightSyntax(text: string): string {
  let highlighted = text;

  const neuralKeywords = ['Network', 'Input', 'Conv2D', 'Dense', 'Dropout', 'Output', 'MaxPooling2D', 'Flatten'];
  neuralKeywords.forEach((keyword) => {
    const regex = new RegExp(`\\b${keyword}\\b`, 'g');
    highlighted = highlighted.replace(regex, `\x1b[1;34m${keyword}\x1b[0m`);
  });

  highlighted = highlighted.replace(/\b\d+\b/g, '\x1b[1;33m$&\x1b[0m');

  highlighted = highlighted.replace(/(['"])(.*?)\1/g, '\x1b[1;32m$&\x1b[0m');

  return highlighted;
}

export function validateShellName(shell: string): boolean {
  const validShells = ['bash', 'zsh', 'sh', 'powershell', 'cmd'];
  return validShells.includes(shell.toLowerCase());
}

export function getShellPrompt(shell: string): string {
  const prompts: { [key: string]: string } = {
    bash: '$ ',
    zsh: '% ',
    sh: '$ ',
    powershell: 'PS> ',
    cmd: '> ',
  };

  return prompts[shell.toLowerCase()] || '$ ';
}

export function sanitizeCommand(command: string): string {
  return command
    .replace(/[;&|`$(){}[\]<>]/g, '')
    .trim();
}

export function extractFilePath(command: string): string | null {
  const filePattern = /\b[\w\-./]+\.[\w]+\b/;
  const match = command.match(filePattern);
  return match ? match[0] : null;
}

export function isPathAbsolute(path: string): boolean {
  return path.startsWith('/') || /^[a-zA-Z]:/.test(path);
}

export function joinPaths(...paths: string[]): string {
  return paths
    .filter(Boolean)
    .join('/')
    .replace(/\/+/g, '/')
    .replace(/\/$/, '');
}

export function basename(path: string): string {
  return path.split('/').pop() || '';
}

export function dirname(path: string): string {
  const parts = path.split('/');
  parts.pop();
  return parts.join('/') || '/';
}

export function getFileExtension(filename: string): string {
  const parts = filename.split('.');
  return parts.length > 1 ? parts.pop() || '' : '';
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

export class CommandHistory {
  private history: string[] = [];
  private maxSize: number;
  private currentIndex: number = -1;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  add(command: string): void {
    if (!command.trim()) return;

    if (this.history.length > 0 && this.history[this.history.length - 1] === command) {
      return;
    }

    this.history.push(command);

    if (this.history.length > this.maxSize) {
      this.history.shift();
    }

    this.currentIndex = this.history.length;
  }

  previous(): string | null {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      return this.history[this.currentIndex];
    }
    return null;
  }

  next(): string | null {
    if (this.currentIndex < this.history.length - 1) {
      this.currentIndex++;
      return this.history[this.currentIndex];
    }
    this.currentIndex = this.history.length;
    return null;
  }

  reset(): void {
    this.currentIndex = this.history.length;
  }

  getAll(): string[] {
    return [...this.history];
  }

  search(query: string): string[] {
    return this.history.filter((cmd) =>
      cmd.toLowerCase().includes(query.toLowerCase())
    );
  }

  clear(): void {
    this.history = [];
    this.currentIndex = -1;
  }
}

export function copyToClipboard(text: string): Promise<void> {
  return navigator.clipboard.writeText(text);
}

export function readFromClipboard(): Promise<string> {
  return navigator.clipboard.readText();
}

export function downloadFile(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function getCurrentTimestamp(): string {
  return new Date().toISOString();
}

export function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString();
}
