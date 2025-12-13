const { dialog, BrowserWindow } = require('electron');

class DialogHandler {
  async showError(title, message) {
    return await dialog.showMessageBox({
      type: 'error',
      title,
      message,
      buttons: ['OK'],
    });
  }

  async showInfo(title, message) {
    return await dialog.showMessageBox({
      type: 'info',
      title,
      message,
      buttons: ['OK'],
    });
  }

  async showWarning(title, message) {
    return await dialog.showMessageBox({
      type: 'warning',
      title,
      message,
      buttons: ['OK'],
    });
  }

  async showConfirm(title, message) {
    const result = await dialog.showMessageBox({
      type: 'question',
      title,
      message,
      buttons: ['Yes', 'No'],
      defaultId: 0,
      cancelId: 1,
    });

    return {
      success: true,
      confirmed: result.response === 0,
    };
  }
}

module.exports = DialogHandler;
