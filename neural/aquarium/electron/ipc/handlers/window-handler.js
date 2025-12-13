const { BrowserWindow } = require('electron');

class WindowHandler {
  getWindowFromWebContents(webContents) {
    return BrowserWindow.fromWebContents(webContents);
  }

  minimize(webContents) {
    const window = this.getWindowFromWebContents(webContents);
    if (window) {
      window.minimize();
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  maximize(webContents) {
    const window = this.getWindowFromWebContents(webContents);
    if (window) {
      if (window.isMaximized()) {
        window.unmaximize();
      } else {
        window.maximize();
      }
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  close(webContents) {
    const window = this.getWindowFromWebContents(webContents);
    if (window) {
      window.close();
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }

  toggleFullScreen(webContents) {
    const window = this.getWindowFromWebContents(webContents);
    if (window) {
      window.setFullScreen(!window.isFullScreen());
      return { success: true };
    }
    return { success: false, error: 'Window not found' };
  }
}

module.exports = WindowHandler;
