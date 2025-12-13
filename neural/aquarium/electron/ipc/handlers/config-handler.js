class ConfigHandler {
  constructor(configManager) {
    this.configManager = configManager;
  }

  async get(key) {
    try {
      const value = this.configManager.get(key);
      return {
        success: true,
        value,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async set(key, value) {
    try {
      this.configManager.set(key, value);
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async getAll() {
    try {
      const config = this.configManager.getAll();
      return {
        success: true,
        config,
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }

  async reset() {
    try {
      this.configManager.reset();
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.message,
      };
    }
  }
}

module.exports = ConfigHandler;
