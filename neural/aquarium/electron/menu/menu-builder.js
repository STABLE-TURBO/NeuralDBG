const { Menu, shell, dialog, app } = require('electron');

class MenuBuilder {
  constructor(configManager) {
    this.configManager = configManager;
  }

  buildMenu(mainWindow) {
    const template = [
      this.buildFileMenu(mainWindow),
      this.buildEditMenu(mainWindow),
      this.buildViewMenu(mainWindow),
      this.buildRunMenu(mainWindow),
      this.buildDebugMenu(mainWindow),
      this.buildHelpMenu(mainWindow),
    ];

    if (process.platform === 'darwin') {
      template.unshift(this.buildMacAppMenu());
    }

    return Menu.buildFromTemplate(template);
  }

  buildMacAppMenu() {
    return {
      label: app.name,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        {
          label: 'Preferences',
          accelerator: 'Cmd+,',
          click: () => this.openPreferences(),
        },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    };
  }

  buildFileMenu(mainWindow) {
    return {
      label: 'File',
      submenu: [
        {
          label: 'New Project',
          accelerator: 'CmdOrCtrl+Shift+N',
          click: () => {
            mainWindow.webContents.send('menu:new-project');
          },
        },
        {
          label: 'Open Project',
          accelerator: 'CmdOrCtrl+Shift+O',
          click: () => {
            mainWindow.webContents.send('menu:open-project');
          },
        },
        {
          label: 'Recent Projects',
          submenu: this.buildRecentProjectsMenu(mainWindow),
        },
        { type: 'separator' },
        {
          label: 'New File',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu:new-file');
          },
        },
        {
          label: 'Open File',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            mainWindow.webContents.send('menu:open-file');
          },
        },
        {
          label: 'Save',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            mainWindow.webContents.send('menu:save-file');
          },
        },
        {
          label: 'Save As',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => {
            mainWindow.webContents.send('menu:save-file-as');
          },
        },
        { type: 'separator' },
        {
          label: 'Close Project',
          accelerator: 'CmdOrCtrl+Shift+W',
          click: () => {
            mainWindow.webContents.send('menu:close-project');
          },
        },
        { type: 'separator' },
        process.platform === 'darwin' ? { role: 'close' } : { role: 'quit' },
      ],
    };
  }

  buildEditMenu(mainWindow) {
    return {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' },
        { type: 'separator' },
        {
          label: 'Find',
          accelerator: 'CmdOrCtrl+F',
          click: () => {
            mainWindow.webContents.send('menu:find');
          },
        },
        {
          label: 'Replace',
          accelerator: 'CmdOrCtrl+H',
          click: () => {
            mainWindow.webContents.send('menu:replace');
          },
        },
        { type: 'separator' },
        {
          label: 'Preferences',
          accelerator: process.platform === 'darwin' ? 'Cmd+,' : 'CmdOrCtrl+,',
          click: () => this.openPreferences(),
        },
      ],
    };
  }

  buildViewMenu(mainWindow) {
    return {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
        { type: 'separator' },
        {
          label: 'Toggle Sidebar',
          accelerator: 'CmdOrCtrl+B',
          click: () => {
            mainWindow.webContents.send('menu:toggle-sidebar');
          },
        },
        {
          label: 'Toggle Terminal',
          accelerator: 'CmdOrCtrl+`',
          click: () => {
            mainWindow.webContents.send('menu:toggle-terminal');
          },
        },
        {
          label: 'Toggle Output',
          accelerator: 'CmdOrCtrl+Shift+U',
          click: () => {
            mainWindow.webContents.send('menu:toggle-output');
          },
        },
      ],
    };
  }

  buildRunMenu(mainWindow) {
    return {
      label: 'Run',
      submenu: [
        {
          label: 'Parse DSL',
          accelerator: 'CmdOrCtrl+Shift+P',
          click: () => {
            mainWindow.webContents.send('menu:parse-dsl');
          },
        },
        {
          label: 'Compile Model',
          accelerator: 'CmdOrCtrl+Shift+C',
          click: () => {
            mainWindow.webContents.send('menu:compile-model');
          },
        },
        {
          label: 'Run Training',
          accelerator: 'F5',
          click: () => {
            mainWindow.webContents.send('menu:run-training');
          },
        },
        {
          label: 'Stop Training',
          accelerator: 'Shift+F5',
          click: () => {
            mainWindow.webContents.send('menu:stop-training');
          },
        },
        { type: 'separator' },
        {
          label: 'Visualize Model',
          accelerator: 'CmdOrCtrl+Shift+V',
          click: () => {
            mainWindow.webContents.send('menu:visualize-model');
          },
        },
        {
          label: 'View Metrics',
          accelerator: 'CmdOrCtrl+Shift+M',
          click: () => {
            mainWindow.webContents.send('menu:view-metrics');
          },
        },
      ],
    };
  }

  buildDebugMenu(mainWindow) {
    return {
      label: 'Debug',
      submenu: [
        {
          label: 'Start Debugging',
          accelerator: 'F9',
          click: () => {
            mainWindow.webContents.send('menu:start-debugging');
          },
        },
        {
          label: 'Stop Debugging',
          accelerator: 'Shift+F9',
          click: () => {
            mainWindow.webContents.send('menu:stop-debugging');
          },
        },
        { type: 'separator' },
        {
          label: 'Toggle Breakpoint',
          accelerator: 'F8',
          click: () => {
            mainWindow.webContents.send('menu:toggle-breakpoint');
          },
        },
        {
          label: 'Step Over',
          accelerator: 'F10',
          click: () => {
            mainWindow.webContents.send('menu:step-over');
          },
        },
        {
          label: 'Step Into',
          accelerator: 'F11',
          click: () => {
            mainWindow.webContents.send('menu:step-into');
          },
        },
        {
          label: 'Step Out',
          accelerator: 'Shift+F11',
          click: () => {
            mainWindow.webContents.send('menu:step-out');
          },
        },
        {
          label: 'Continue',
          accelerator: 'F12',
          click: () => {
            mainWindow.webContents.send('menu:continue');
          },
        },
        { type: 'separator' },
        {
          label: 'Inspect Variables',
          accelerator: 'CmdOrCtrl+Shift+I',
          click: () => {
            mainWindow.webContents.send('menu:inspect-variables');
          },
        },
      ],
    };
  }

  buildHelpMenu(mainWindow) {
    return {
      label: 'Help',
      submenu: [
        {
          label: 'Documentation',
          click: async () => {
            await shell.openExternal('https://neuraldsl.readthedocs.io');
          },
        },
        {
          label: 'Quick Start Guide',
          click: () => {
            mainWindow.webContents.send('menu:quick-start');
          },
        },
        {
          label: 'Examples',
          click: () => {
            mainWindow.webContents.send('menu:examples');
          },
        },
        { type: 'separator' },
        {
          label: 'Report Issue',
          click: async () => {
            await shell.openExternal('https://github.com/neuraldsl/neural/issues');
          },
        },
        {
          label: 'Release Notes',
          click: () => {
            mainWindow.webContents.send('menu:release-notes');
          },
        },
        { type: 'separator' },
        {
          label: 'About',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About Neural Aquarium IDE',
              message: 'Neural Aquarium IDE',
              detail: 'Version 0.3.0\n\nA visual development environment for Neural DSL.\n\n© 2024 Neural DSL Team',
            });
          },
        },
      ],
    };
  }

  buildRecentProjectsMenu(mainWindow) {
    const recentProjects = this.configManager.get('recentProjects') || [];

    if (recentProjects.length === 0) {
      return [
        {
          label: 'No Recent Projects',
          enabled: false,
        },
      ];
    }

    const projectItems = recentProjects.slice(0, 10).map((projectPath) => ({
      label: projectPath,
      click: () => {
        mainWindow.webContents.send('menu:open-recent-project', projectPath);
      },
    }));

    return [
      ...projectItems,
      { type: 'separator' },
      {
        label: 'Clear Recent Projects',
        click: () => {
          this.configManager.set('recentProjects', []);
          mainWindow.setMenu(this.buildMenu(mainWindow));
        },
      },
    ];
  }

  openPreferences() {
    console.log('Opening preferences...');
  }
}

module.exports = MenuBuilder;
