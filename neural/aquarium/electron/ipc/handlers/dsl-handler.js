const { spawn } = require('child_process');
const path = require('path');

class DSLHandler {
  constructor() {
    this.pythonPath = this.findPythonPath();
  }

  findPythonPath() {
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  async parse(dslCode) {
    return this.executePythonCommand('parse', { code: dslCode });
  }

  async compile(dslCode, backend, options) {
    return this.executePythonCommand('compile', {
      code: dslCode,
      backend,
      options,
    });
  }

  async validate(dslCode) {
    return this.executePythonCommand('validate', { code: dslCode });
  }

  async getExamples() {
    return this.executePythonCommand('get-examples', {});
  }

  async executePythonCommand(command, data) {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, '..', '..', '..', 'backend', 'cli.py');
      const args = [scriptPath, command, JSON.stringify(data)];

      const process = spawn(this.pythonPath, args);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code !== 0) {
          resolve({
            success: false,
            error: stderr || `Process exited with code ${code}`,
          });
        } else {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (error) {
            resolve({
              success: false,
              error: 'Failed to parse Python response',
            });
          }
        }
      });

      process.on('error', (error) => {
        resolve({
          success: false,
          error: error.message,
        });
      });
    });
  }
}

module.exports = DSLHandler;
