const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

class ExecutionHandler {
  constructor() {
    this.executions = new Map();
    this.nextExecutionId = 1;
    this.pythonPath = this.findPythonPath();
  }

  findPythonPath() {
    return process.platform === 'win32' ? 'python' : 'python3';
  }

  async run(sender, script, options) {
    try {
      const executionId = this.nextExecutionId++;
      const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'neural-'));
      const scriptPath = path.join(tempDir, 'script.py');

      await fs.writeFile(scriptPath, script);

      const process = spawn(this.pythonPath, [scriptPath], {
        cwd: tempDir,
        env: { ...process.env, ...options.env },
      });

      const execution = {
        id: executionId,
        process,
        scriptPath,
        tempDir,
        status: 'running',
        startTime: Date.now(),
      };

      this.executions.set(executionId, execution);

      process.stdout.on('data', (data) => {
        const output = data.toString();
        sender.send('execution:output', {
          executionId,
          type: 'stdout',
          data: output,
        });

        this.parseMetrics(sender, executionId, output);
      });

      process.stderr.on('data', (data) => {
        sender.send('execution:output', {
          executionId,
          type: 'stderr',
          data: data.toString(),
        });
      });

      process.on('close', async (code) => {
        execution.status = code === 0 ? 'completed' : 'failed';
        execution.exitCode = code;
        execution.endTime = Date.now();

        sender.send('execution:output', {
          executionId,
          type: 'status',
          data: `Process exited with code ${code}`,
        });

        await this.cleanup(executionId);
      });

      process.on('error', (error) => {
        execution.status = 'error';
        sender.send('execution:output', {
          executionId,
          type: 'error',
          data: error.message,
        });
      });

      return {
        success: true,
        executionId,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async stop(executionId) {
    try {
      const execution = this.executions.get(executionId);
      if (!execution) {
        return { success: false, error: 'Execution not found' };
      }

      if (execution.process && !execution.process.killed) {
        execution.process.kill();
        execution.status = 'stopped';
      }

      await this.cleanup(executionId);

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async getStatus(executionId) {
    const execution = this.executions.get(executionId);
    if (!execution) {
      return { success: false, error: 'Execution not found' };
    }

    return {
      success: true,
      status: execution.status,
      startTime: execution.startTime,
      endTime: execution.endTime,
      exitCode: execution.exitCode,
    };
  }

  parseMetrics(sender, executionId, output) {
    const epochRegex = /Epoch (\d+)\/(\d+).*loss: ([\d.]+).*accuracy: ([\d.]+)/;
    const match = output.match(epochRegex);

    if (match) {
      sender.send('execution:metrics', {
        executionId,
        epoch: parseInt(match[1]),
        totalEpochs: parseInt(match[2]),
        loss: parseFloat(match[3]),
        accuracy: parseFloat(match[4]),
      });
    }
  }

  async cleanup(executionId) {
    const execution = this.executions.get(executionId);
    if (!execution) return;

    try {
      await fs.rm(execution.tempDir, { recursive: true, force: true });
    } catch (error) {
      console.error('Cleanup error:', error);
    }

    setTimeout(() => {
      this.executions.delete(executionId);
    }, 60000);
  }
}

module.exports = ExecutionHandler;
