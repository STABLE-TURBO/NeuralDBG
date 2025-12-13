const { contextBridge, ipcRenderer } = require('electron');

const API = {
  project: {
    create: (projectData) => ipcRenderer.invoke('project:create', projectData),
    open: () => ipcRenderer.invoke('project:open'),
    save: (projectData) => ipcRenderer.invoke('project:save', projectData),
    close: () => ipcRenderer.invoke('project:close'),
    getRecent: () => ipcRenderer.invoke('project:get-recent'),
  },

  file: {
    open: () => ipcRenderer.invoke('file:open'),
    save: (filePath, content) => ipcRenderer.invoke('file:save', filePath, content),
    saveAs: (content) => ipcRenderer.invoke('file:save-as', content),
    read: (filePath) => ipcRenderer.invoke('file:read', filePath),
  },

  dsl: {
    parse: (dslCode) => ipcRenderer.invoke('dsl:parse', dslCode),
    compile: (dslCode, backend, options) => ipcRenderer.invoke('dsl:compile', dslCode, backend, options),
    validate: (dslCode) => ipcRenderer.invoke('dsl:validate', dslCode),
    getExamples: () => ipcRenderer.invoke('dsl:get-examples'),
  },

  execution: {
    run: (script, options) => ipcRenderer.invoke('execution:run', script, options),
    stop: (executionId) => ipcRenderer.invoke('execution:stop', executionId),
    getStatus: (executionId) => ipcRenderer.invoke('execution:get-status', executionId),
    onOutput: (callback) => {
      ipcRenderer.on('execution:output', (event, data) => callback(data));
      return () => ipcRenderer.removeListener('execution:output', callback);
    },
    onMetrics: (callback) => {
      ipcRenderer.on('execution:metrics', (event, data) => callback(data));
      return () => ipcRenderer.removeListener('execution:metrics', callback);
    },
  },

  debugger: {
    start: (config) => ipcRenderer.invoke('debugger:start', config),
    stop: () => ipcRenderer.invoke('debugger:stop'),
    setBreakpoint: (file, line) => ipcRenderer.invoke('debugger:set-breakpoint', file, line),
    removeBreakpoint: (file, line) => ipcRenderer.invoke('debugger:remove-breakpoint', file, line),
    step: (type) => ipcRenderer.invoke('debugger:step', type),
    continue: () => ipcRenderer.invoke('debugger:continue'),
    getVariables: () => ipcRenderer.invoke('debugger:get-variables'),
    onStateChange: (callback) => {
      ipcRenderer.on('debugger:state-change', (event, data) => callback(data));
      return () => ipcRenderer.removeListener('debugger:state-change', callback);
    },
  },

  python: {
    getVersion: () => ipcRenderer.invoke('python:get-version'),
    getPath: () => ipcRenderer.invoke('python:get-path'),
    install: (packageName) => ipcRenderer.invoke('python:install', packageName),
    listPackages: () => ipcRenderer.invoke('python:list-packages'),
  },

  backend: {
    connect: () => ipcRenderer.invoke('backend:connect'),
    disconnect: () => ipcRenderer.invoke('backend:disconnect'),
    getStatus: () => ipcRenderer.invoke('backend:get-status'),
    send: (message) => ipcRenderer.invoke('backend:send', message),
    onMessage: (callback) => {
      ipcRenderer.on('backend:message', (event, data) => callback(data));
      return () => ipcRenderer.removeListener('backend:message', callback);
    },
  },

  config: {
    get: (key) => ipcRenderer.invoke('config:get', key),
    set: (key, value) => ipcRenderer.invoke('config:set', key, value),
    getAll: () => ipcRenderer.invoke('config:get-all'),
    reset: () => ipcRenderer.invoke('config:reset'),
  },

  window: {
    minimize: () => ipcRenderer.invoke('window:minimize'),
    maximize: () => ipcRenderer.invoke('window:maximize'),
    close: () => ipcRenderer.invoke('window:close'),
    toggleFullScreen: () => ipcRenderer.invoke('window:toggle-fullscreen'),
  },

  dialog: {
    showError: (title, message) => ipcRenderer.invoke('dialog:show-error', title, message),
    showInfo: (title, message) => ipcRenderer.invoke('dialog:show-info', title, message),
    showWarning: (title, message) => ipcRenderer.invoke('dialog:show-warning', title, message),
    showConfirm: (title, message) => ipcRenderer.invoke('dialog:show-confirm', title, message),
  },

  shell: {
    openExternal: (url) => ipcRenderer.invoke('shell:open-external', url),
    openPath: (path) => ipcRenderer.invoke('shell:open-path', path),
  },
};

contextBridge.exposeInMainWorld('aquarium', API);
