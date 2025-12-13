const { spawn } = require('child_process');

class PythonHandler {
  constructor() {
    this.pythonPath = this.findPythonPath();
  }

  findPythonPath() {
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  async getVersion() {
    return this.executeCommand([this.pythonPath, '--version']);
  }

  async getPath() {
    return this.executeCommand([this.pythonPath, '-c', 'import sys; print(sys.executable)']);
  }

  async installPackage(packageName) {
    return this.executeCommand([this.pythonPath, '-m', 'pip', 'install', packageName]);
  }

  async listPackages() {
    return this.executeCommand([this.pythonPath, '-m', 'pip', 'list', '--format=json']);
  }

  async executeCommand(args) {
    return new Promise((resolve) => {
      const process = spawn(args[0], args.slice(1));

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
          resolve({
            success: true,
            output: stdout.trim(),
          });
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

module.exports = PythonHandler;
