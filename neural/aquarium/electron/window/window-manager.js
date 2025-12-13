const { BrowserWindow } = require('electron');
const path = require('path');
const windowStateKeeper = require('electron-window-state');

class WindowManager {
  constructor(configManager) {
    this.configManager = configManager;
    this.windows = new Map();
  }

  async createMainWindow() {
    const mainWindowState = windowStateKeeper({
      defaultWidth: 1400,
      defaultHeight: 900,
      file: 'main-window-state.json',
    });

    const window = new BrowserWindow({
      x: mainWindowState.x,
      y: mainWindowState.y,
      width: mainWindowState.width,
      height: mainWindowState.height,
      minWidth: 1024,
      minHeight: 768,
      title: 'Neural Aquarium IDE',
      backgroundColor: '#1e1e1e',
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, '..', 'preload.js'),
      },
      show: false,
    });

    mainWindowState.manage(window);

    window.once('ready-to-show', () => {
      window.show();
    });

    this.windows.set('main', window);
    return window;
  }

  createDebuggerWindow(mainWindow) {
    const window = new BrowserWindow({
      width: 1200,
      height: 800,
      minWidth: 800,
      minHeight: 600,
      title: 'Neural Debugger',
      parent: mainWindow,
      backgroundColor: '#1e1e1e',
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, '..', 'preload.js'),
      },
    });

    this.windows.set('debugger', window);
    return window;
  }

  createVisualizerWindow(mainWindow) {
    const window = new BrowserWindow({
      width: 1000,
      height: 700,
      minWidth: 600,
      minHeight: 500,
      title: 'Model Visualizer',
      parent: mainWindow,
      backgroundColor: '#1e1e1e',
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, '..', 'preload.js'),
      },
    });

    this.windows.set('visualizer', window);
    return window;
  }

  createPreferencesWindow(mainWindow) {
    const window = new BrowserWindow({
      width: 700,
      height: 600,
      minWidth: 600,
      minHeight: 500,
      title: 'Preferences',
      parent: mainWindow,
      modal: true,
      backgroundColor: '#1e1e1e',
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        enableRemoteModule: false,
        preload: path.join(__dirname, '..', 'preload.js'),
      },
    });

    this.windows.set('preferences', window);
    return window;
  }

  getWindow(name) {
    return this.windows.get(name);
  }

  closeWindow(name) {
    const window = this.windows.get(name);
    if (window && !window.isDestroyed()) {
      window.close();
    }
    this.windows.delete(name);
  }

  closeAllWindows() {
    this.windows.forEach((window, name) => {
      if (!window.isDestroyed()) {
        window.close();
      }
    });
    this.windows.clear();
  }
}

module.exports = WindowManager;
