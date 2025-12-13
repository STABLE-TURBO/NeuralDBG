const { ipcMain, dialog, shell } = require('electron');
const ProjectManager = require('./handlers/project-handler');
const FileManager = require('./handlers/file-handler');
const DSLHandler = require('./handlers/dsl-handler');
const ExecutionHandler = require('./handlers/execution-handler');
const DebuggerHandler = require('./handlers/debugger-handler');
const PythonHandler = require('./handlers/python-handler');
const BackendHandler = require('./handlers/backend-handler');
const ConfigHandler = require('./handlers/config-handler');
const WindowHandler = require('./handlers/window-handler');
const DialogHandler = require('./handlers/dialog-handler');

class IPCController {
  constructor(configManager) {
    this.configManager = configManager;
    this.projectManager = new ProjectManager(configManager);
    this.fileManager = new FileManager();
    this.dslHandler = new DSLHandler();
    this.executionHandler = new ExecutionHandler();
    this.debuggerHandler = new DebuggerHandler();
    this.pythonHandler = new PythonHandler();
    this.backendHandler = new BackendHandler();
    this.configHandler = new ConfigHandler(configManager);
    this.windowHandler = new WindowHandler();
    this.dialogHandler = new DialogHandler();
  }

  registerHandlers() {
    this.registerProjectHandlers();
    this.registerFileHandlers();
    this.registerDSLHandlers();
    this.registerExecutionHandlers();
    this.registerDebuggerHandlers();
    this.registerPythonHandlers();
    this.registerBackendHandlers();
    this.registerConfigHandlers();
    this.registerWindowHandlers();
    this.registerDialogHandlers();
    this.registerShellHandlers();
  }

  registerProjectHandlers() {
    ipcMain.handle('project:create', async (event, projectData) => {
      return await this.projectManager.createProject(projectData);
    });

    ipcMain.handle('project:open', async (event) => {
      return await this.projectManager.openProject();
    });

    ipcMain.handle('project:save', async (event, projectData) => {
      return await this.projectManager.saveProject(projectData);
    });

    ipcMain.handle('project:close', async (event) => {
      return await this.projectManager.closeProject();
    });

    ipcMain.handle('project:get-recent', async (event) => {
      return await this.projectManager.getRecentProjects();
    });
  }

  registerFileHandlers() {
    ipcMain.handle('file:open', async (event) => {
      return await this.fileManager.openFile();
    });

    ipcMain.handle('file:save', async (event, filePath, content) => {
      return await this.fileManager.saveFile(filePath, content);
    });

    ipcMain.handle('file:save-as', async (event, content) => {
      return await this.fileManager.saveFileAs(content);
    });

    ipcMain.handle('file:read', async (event, filePath) => {
      return await this.fileManager.readFile(filePath);
    });
  }

  registerDSLHandlers() {
    ipcMain.handle('dsl:parse', async (event, dslCode) => {
      return await this.dslHandler.parse(dslCode);
    });

    ipcMain.handle('dsl:compile', async (event, dslCode, backend, options) => {
      return await this.dslHandler.compile(dslCode, backend, options);
    });

    ipcMain.handle('dsl:validate', async (event, dslCode) => {
      return await this.dslHandler.validate(dslCode);
    });

    ipcMain.handle('dsl:get-examples', async (event) => {
      return await this.dslHandler.getExamples();
    });
  }

  registerExecutionHandlers() {
    ipcMain.handle('execution:run', async (event, script, options) => {
      return await this.executionHandler.run(event.sender, script, options);
    });

    ipcMain.handle('execution:stop', async (event, executionId) => {
      return await this.executionHandler.stop(executionId);
    });

    ipcMain.handle('execution:get-status', async (event, executionId) => {
      return await this.executionHandler.getStatus(executionId);
    });
  }

  registerDebuggerHandlers() {
    ipcMain.handle('debugger:start', async (event, config) => {
      return await this.debuggerHandler.start(event.sender, config);
    });

    ipcMain.handle('debugger:stop', async (event) => {
      return await this.debuggerHandler.stop();
    });

    ipcMain.handle('debugger:set-breakpoint', async (event, file, line) => {
      return await this.debuggerHandler.setBreakpoint(file, line);
    });

    ipcMain.handle('debugger:remove-breakpoint', async (event, file, line) => {
      return await this.debuggerHandler.removeBreakpoint(file, line);
    });

    ipcMain.handle('debugger:step', async (event, type) => {
      return await this.debuggerHandler.step(type);
    });

    ipcMain.handle('debugger:continue', async (event) => {
      return await this.debuggerHandler.continue();
    });

    ipcMain.handle('debugger:get-variables', async (event) => {
      return await this.debuggerHandler.getVariables();
    });
  }

  registerPythonHandlers() {
    ipcMain.handle('python:get-version', async (event) => {
      return await this.pythonHandler.getVersion();
    });

    ipcMain.handle('python:get-path', async (event) => {
      return await this.pythonHandler.getPath();
    });

    ipcMain.handle('python:install', async (event, packageName) => {
      return await this.pythonHandler.installPackage(packageName);
    });

    ipcMain.handle('python:list-packages', async (event) => {
      return await this.pythonHandler.listPackages();
    });
  }

  registerBackendHandlers() {
    ipcMain.handle('backend:connect', async (event) => {
      return await this.backendHandler.connect(event.sender);
    });

    ipcMain.handle('backend:disconnect', async (event) => {
      return await this.backendHandler.disconnect();
    });

    ipcMain.handle('backend:get-status', async (event) => {
      return await this.backendHandler.getStatus();
    });

    ipcMain.handle('backend:send', async (event, message) => {
      return await this.backendHandler.send(message);
    });
  }

  registerConfigHandlers() {
    ipcMain.handle('config:get', async (event, key) => {
      return await this.configHandler.get(key);
    });

    ipcMain.handle('config:set', async (event, key, value) => {
      return await this.configHandler.set(key, value);
    });

    ipcMain.handle('config:get-all', async (event) => {
      return await this.configHandler.getAll();
    });

    ipcMain.handle('config:reset', async (event) => {
      return await this.configHandler.reset();
    });
  }

  registerWindowHandlers() {
    ipcMain.handle('window:minimize', async (event) => {
      return this.windowHandler.minimize(event.sender);
    });

    ipcMain.handle('window:maximize', async (event) => {
      return this.windowHandler.maximize(event.sender);
    });

    ipcMain.handle('window:close', async (event) => {
      return this.windowHandler.close(event.sender);
    });

    ipcMain.handle('window:toggle-fullscreen', async (event) => {
      return this.windowHandler.toggleFullScreen(event.sender);
    });
  }

  registerDialogHandlers() {
    ipcMain.handle('dialog:show-error', async (event, title, message) => {
      return await this.dialogHandler.showError(title, message);
    });

    ipcMain.handle('dialog:show-info', async (event, title, message) => {
      return await this.dialogHandler.showInfo(title, message);
    });

    ipcMain.handle('dialog:show-warning', async (event, title, message) => {
      return await this.dialogHandler.showWarning(title, message);
    });

    ipcMain.handle('dialog:show-confirm', async (event, title, message) => {
      return await this.dialogHandler.showConfirm(title, message);
    });
  }

  registerShellHandlers() {
    ipcMain.handle('shell:open-external', async (event, url) => {
      return await shell.openExternal(url);
    });

    ipcMain.handle('shell:open-path', async (event, path) => {
      return await shell.openPath(path);
    });
  }
}

module.exports = IPCController;
