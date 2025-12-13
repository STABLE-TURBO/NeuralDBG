const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const WindowManager = require('./window/window-manager');
const IPCController = require('./ipc/ipc-controller');
const MenuBuilder = require('./menu/menu-builder');
const ConfigManager = require('./config/config-manager');

class AquariumIDE {
  constructor() {
    this.windowManager = null;
    this.ipcController = null;
    this.menuBuilder = null;
    this.configManager = null;
    this.mainWindow = null;
  }

  async initialize() {
    this.configManager = new ConfigManager();
    await this.configManager.initialize();

    this.windowManager = new WindowManager(this.configManager);
    this.ipcController = new IPCController(this.configManager);
    this.menuBuilder = new MenuBuilder(this.configManager);

    this.setupAppEventHandlers();
    this.setupIPCHandlers();
  }

  setupAppEventHandlers() {
    app.on('ready', async () => {
      await this.createMainWindow();
    });

    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        app.quit();
      }
    });

    app.on('activate', async () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        await this.createMainWindow();
      }
    });

    app.on('before-quit', () => {
      this.configManager.save();
    });
  }

  setupIPCHandlers() {
    this.ipcController.registerHandlers();
  }

  async createMainWindow() {
    this.mainWindow = await this.windowManager.createMainWindow();
    
    const menu = this.menuBuilder.buildMenu(this.mainWindow);
    this.mainWindow.setMenu(menu);

    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });

    if (process.env.NODE_ENV === 'development') {
      this.mainWindow.webContents.openDevTools();
    }

    this.loadApplication();
  }

  loadApplication() {
    const isDev = process.env.NODE_ENV === 'development';
    
    if (isDev) {
      this.mainWindow.loadURL('http://localhost:3000');
    } else {
      this.mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
    }
  }
}

const aquarium = new AquariumIDE();

app.whenReady().then(() => {
  aquarium.initialize();
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
});

process.on('unhandledRejection', (error) => {
  console.error('Unhandled rejection:', error);
});
