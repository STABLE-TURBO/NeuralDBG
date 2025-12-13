const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');

class BackendHandler {
  constructor() {
    this.ws = null;
    this.backendProcess = null;
    this.status = 'disconnected';
    this.pythonPath = this.findPythonPath();
  }

  findPythonPath() {
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  async connect(sender) {
    try {
      if (this.status === 'connected') {
        return { success: true, status: 'already_connected' };
      }

      await this.startBackendServer();

      await new Promise((resolve) => setTimeout(resolve, 2000));

      this.ws = new WebSocket('ws://localhost:5000/ws');

      this.ws.on('open', () => {
        this.status = 'connected';
        sender.send('backend:message', {
          type: 'status',
          status: 'connected',
        });
      });

      this.ws.on('message', (data) => {
        const message = JSON.parse(data.toString());
        sender.send('backend:message', message);
      });

      this.ws.on('error', (error) => {
        this.status = 'error';
        sender.send('backend:message', {
          type: 'error',
          error: error.message,
        });
      });

      this.ws.on('close', () => {
        this.status = 'disconnected';
        sender.send('backend:message', {
          type: 'status',
          status: 'disconnected',
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

  async startBackendServer() {
    const serverScript = path.join(__dirname, '..', '..', '..', 'backend', 'server.py');
    
    this.backendProcess = spawn(this.pythonPath, [serverScript]);

    this.backendProcess.stdout.on('data', (data) => {
      console.log('Backend:', data.toString());
    });

    this.backendProcess.stderr.on('data', (data) => {
      console.error('Backend error:', data.toString());
    });

    this.backendProcess.on('close', (code) => {
      console.log('Backend process exited with code', code);
      this.backendProcess = null;
    });
  }

  async disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.backendProcess && !this.backendProcess.killed) {
      this.backendProcess.kill();
      this.backendProcess = null;
    }

    this.status = 'disconnected';

    return { success: true };
  }

  async getStatus() {
    return {
      success: true,
      status: this.status,
    };
  }

  async send(message) {
    if (!this.ws || this.status !== 'connected') {
      return {
        success: false,
        error: 'Backend not connected',
      };
    }

    try {
      this.ws.send(JSON.stringify(message));
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }
}

module.exports = BackendHandler;
