const { dialog } = require('electron');
const fs = require('fs').promises;
const path = require('path');

class ProjectManager {
  constructor(configManager) {
    this.configManager = configManager;
    this.currentProject = null;
  }

  async createProject(projectData) {
    try {
      const result = await dialog.showSaveDialog({
        title: 'Create Neural DSL Project',
        defaultPath: projectData.name || 'neural-project',
        properties: ['createDirectory'],
      });

      if (result.canceled) {
        return { success: false, error: 'Project creation canceled' };
      }

      const projectPath = result.filePath;
      const projectConfig = {
        name: projectData.name || path.basename(projectPath),
        version: '1.0.0',
        description: projectData.description || '',
        backend: projectData.backend || 'tensorflow',
        pythonVersion: projectData.pythonVersion || '3.9',
        createdAt: new Date().toISOString(),
        files: [],
      };

      await fs.mkdir(projectPath, { recursive: true });
      await fs.mkdir(path.join(projectPath, 'models'), { recursive: true });
      await fs.mkdir(path.join(projectPath, 'scripts'), { recursive: true });
      await fs.mkdir(path.join(projectPath, 'data'), { recursive: true });
      
      const configPath = path.join(projectPath, 'neural-project.json');
      await fs.writeFile(configPath, JSON.stringify(projectConfig, null, 2));

      this.currentProject = {
        path: projectPath,
        config: projectConfig,
      };

      this.addToRecentProjects(projectPath);

      return {
        success: true,
        project: this.currentProject,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async openProject() {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Open Neural DSL Project',
        properties: ['openDirectory'],
      });

      if (result.canceled) {
        return { success: false, error: 'Project opening canceled' };
      }

      const projectPath = result.filePaths[0];
      const configPath = path.join(projectPath, 'neural-project.json');

      let projectConfig;
      try {
        const configData = await fs.readFile(configPath, 'utf-8');
        projectConfig = JSON.parse(configData);
      } catch (error) {
        return {
          success: false,
          error: 'Invalid Neural DSL project: neural-project.json not found',
        };
      }

      this.currentProject = {
        path: projectPath,
        config: projectConfig,
      };

      this.addToRecentProjects(projectPath);

      return {
        success: true,
        project: this.currentProject,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async saveProject(projectData) {
    try {
      if (!this.currentProject) {
        return { success: false, error: 'No project currently open' };
      }

      const updatedConfig = {
        ...this.currentProject.config,
        ...projectData,
        updatedAt: new Date().toISOString(),
      };

      const configPath = path.join(this.currentProject.path, 'neural-project.json');
      await fs.writeFile(configPath, JSON.stringify(updatedConfig, null, 2));

      this.currentProject.config = updatedConfig;

      return {
        success: true,
        project: this.currentProject,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async closeProject() {
    this.currentProject = null;
    return { success: true };
  }

  async getRecentProjects() {
    const recentProjects = this.configManager.get('recentProjects') || [];
    return { success: true, projects: recentProjects };
  }

  addToRecentProjects(projectPath) {
    const recentProjects = this.configManager.get('recentProjects') || [];
    const filtered = recentProjects.filter(p => p !== projectPath);
    filtered.unshift(projectPath);
    this.configManager.set('recentProjects', filtered.slice(0, 10));
  }
}

module.exports = ProjectManager;
