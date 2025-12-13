const { dialog } = require('electron');
const fs = require('fs').promises;

class FileManager {
  constructor() {
    this.currentFile = null;
  }

  async openFile() {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Open Neural DSL File',
        filters: [
          { name: 'Neural DSL', extensions: ['ndsl', 'neural'] },
          { name: 'Python', extensions: ['py'] },
          { name: 'All Files', extensions: ['*'] },
        ],
        properties: ['openFile'],
      });

      if (result.canceled) {
        return { success: false, error: 'File opening canceled' };
      }

      const filePath = result.filePaths[0];
      const content = await fs.readFile(filePath, 'utf-8');

      this.currentFile = filePath;

      return {
        success: true,
        filePath,
        content,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async saveFile(filePath, content) {
    try {
      await fs.writeFile(filePath, content, 'utf-8');
      this.currentFile = filePath;

      return {
        success: true,
        filePath,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async saveFileAs(content) {
    try {
      const result = await dialog.showSaveDialog({
        title: 'Save Neural DSL File',
        filters: [
          { name: 'Neural DSL', extensions: ['ndsl', 'neural'] },
          { name: 'Python', extensions: ['py'] },
          { name: 'All Files', extensions: ['*'] },
        ],
      });

      if (result.canceled) {
        return { success: false, error: 'File saving canceled' };
      }

      const filePath = result.filePath;
      await fs.writeFile(filePath, content, 'utf-8');

      this.currentFile = filePath;

      return {
        success: true,
        filePath,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async readFile(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf-8');

      return {
        success: true,
        content,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }
}

module.exports = FileManager;
