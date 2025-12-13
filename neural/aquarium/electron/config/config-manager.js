const Store = require('electron-store');

class ConfigManager {
  constructor() {
    this.store = null;
    this.defaults = {
      theme: 'dark',
      fontSize: 14,
      fontFamily: 'Consolas, Monaco, monospace',
      autoSave: true,
      autoSaveInterval: 30000,
      pythonPath: '',
      backend: 'tensorflow',
      recentProjects: [],
      recentFiles: [],
      windowState: {
        width: 1400,
        height: 900,
        maximized: false,
      },
      editor: {
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'on',
        minimap: true,
        lineNumbers: true,
        renderWhitespace: 'selection',
      },
      debugger: {
        port: 5678,
        autoStart: false,
      },
      execution: {
        clearConsoleOnRun: true,
        showTimestamp: true,
      },
    };
  }

  async initialize() {
    this.store = new Store({
      name: 'aquarium-config',
      defaults: this.defaults,
    });
  }

  get(key) {
    return this.store.get(key);
  }

  set(key, value) {
    this.store.set(key, value);
  }

  getAll() {
    return this.store.store;
  }

  reset() {
    this.store.clear();
    this.store.store = this.defaults;
  }

  save() {
  }
}

module.exports = ConfigManager;
