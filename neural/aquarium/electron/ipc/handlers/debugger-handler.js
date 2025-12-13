const { spawn } = require('child_process');
const WebSocket = require('ws');
const path = require('path');

class DebuggerHandler {
  constructor() {
    this.debugSession = null;
    this.pythonPath = this.findPythonPath();
    this.breakpoints = new Map();
  }

  findPythonPath() {
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  async start(sender, config) {
    try {
      if (this.debugSession) {
        await this.stop();
      }

      const debuggerScript = path.join(
        __dirname,
        '..',
        '..',
        '..',
        'backend',
        'debugger.py'
      );

      const args = [
        debuggerScript,
        '--file',
        config.file,
        '--port',
        config.port || '5678',
      ];

      const process = spawn(this.pythonPath, args);

      this.debugSession = {
        process,
        config,
        ws: null,
        state: 'starting',
      };

      setTimeout(() => {
        this.connectWebSocket(sender, config.port || '5678');
      }, 1000);

      process.stdout.on('data', (data) => {
        sender.send('debugger:state-change', {
          type: 'output',
          data: data.toString(),
        });
      });

      process.stderr.on('data', (data) => {
        sender.send('debugger:state-change', {
          type: 'error',
          data: data.toString(),
        });
      });

      process.on('close', (code) => {
        this.debugSession = null;
        sender.send('debugger:state-change', {
          type: 'stopped',
          code,
        });
      });

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  connectWebSocket(sender, port) {
    try {
      const ws = new WebSocket(`ws://localhost:${port}`);

      ws.on('open', () => {
        this.debugSession.ws = ws;
        this.debugSession.state = 'running';
        sender.send('debugger:state-change', {
          type: 'connected',
        });
      });

      ws.on('message', (data) => {
        const message = JSON.parse(data.toString());
        sender.send('debugger:state-change', {
          type: 'message',
          data: message,
        });
      });

      ws.on('error', (error) => {
        sender.send('debugger:state-change', {
          type: 'error',
          data: error.message,
        });
      });

      ws.on('close', () => {
        if (this.debugSession) {
          this.debugSession.ws = null;
        }
      });
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }

  async stop() {
    if (!this.debugSession) {
      return { success: false, error: 'No active debug session' };
    }

    if (this.debugSession.ws) {
      this.debugSession.ws.close();
    }

    if (this.debugSession.process && !this.debugSession.process.killed) {
      this.debugSession.process.kill();
    }

    this.debugSession = null;
    this.breakpoints.clear();

    return { success: true };
  }

  async setBreakpoint(file, line) {
    this.breakpoints.set(`${file}:${line}`, { file, line });

    if (this.debugSession && this.debugSession.ws) {
      this.debugSession.ws.send(
        JSON.stringify({
          type: 'setBreakpoint',
          file,
          line,
        })
      );
    }

    return { success: true };
  }

  async removeBreakpoint(file, line) {
    this.breakpoints.delete(`${file}:${line}`);

    if (this.debugSession && this.debugSession.ws) {
      this.debugSession.ws.send(
        JSON.stringify({
          type: 'removeBreakpoint',
          file,
          line,
        })
      );
    }

    return { success: true };
  }

  async step(type) {
    if (!this.debugSession || !this.debugSession.ws) {
      return { success: false, error: 'No active debug session' };
    }

    this.debugSession.ws.send(
      JSON.stringify({
        type: 'step',
        stepType: type,
      })
    );

    return { success: true };
  }

  async continue() {
    if (!this.debugSession || !this.debugSession.ws) {
      return { success: false, error: 'No active debug session' };
    }

    this.debugSession.ws.send(
      JSON.stringify({
        type: 'continue',
      })
    );

    return { success: true };
  }

  async getVariables() {
    if (!this.debugSession || !this.debugSession.ws) {
      return { success: false, error: 'No active debug session' };
    }

    return new Promise((resolve) => {
      const handler = (data) => {
        const message = JSON.parse(data.toString());
        if (message.type === 'variables') {
          this.debugSession.ws.removeListener('message', handler);
          resolve({
            success: true,
            variables: message.data,
          });
        }
      };

      this.debugSession.ws.on('message', handler);

      this.debugSession.ws.send(
        JSON.stringify({
          type: 'getVariables',
        })
      );

      setTimeout(() => {
        this.debugSession.ws.removeListener('message', handler);
        resolve({
          success: false,
          error: 'Timeout waiting for variables',
        });
      }, 5000);
    });
  }
}

module.exports = DebuggerHandler;
