/**
 * Plugin service for interacting with the plugin API
 */

import axios from 'axios';
import type {
  Plugin,
  PluginMetadata,
  PanelConfig,
  ThemeDefinition,
  Command,
  PluginAPI,
} from '../types/plugins';

const API_BASE = '/api/plugins';

export class PluginService implements PluginAPI {
  async listPlugins(): Promise<Plugin[]> {
    const response = await axios.get(`${API_BASE}/list`);
    return response.data.plugins;
  }

  async listEnabledPlugins(): Promise<Plugin[]> {
    const response = await axios.get(`${API_BASE}/enabled`);
    return response.data.plugins;
  }

  async getPluginDetails(pluginId: string): Promise<PluginMetadata> {
    const response = await axios.get(`${API_BASE}/details/${pluginId}`);
    return response.data;
  }

  async enablePlugin(pluginId: string): Promise<{ success: boolean; message?: string }> {
    const response = await axios.post(`${API_BASE}/enable`, { plugin_id: pluginId });
    return response.data;
  }

  async disablePlugin(pluginId: string): Promise<{ success: boolean; message?: string }> {
    const response = await axios.post(`${API_BASE}/disable`, { plugin_id: pluginId });
    return response.data;
  }

  async installPlugin(
    source: 'npm' | 'pypi',
    pluginName: string,
    version?: string
  ): Promise<{ success: boolean; message?: string }> {
    const response = await axios.post(`${API_BASE}/install`, {
      source,
      plugin_name: pluginName,
      version,
    });
    return response.data;
  }

  async uninstallPlugin(pluginId: string): Promise<{ success: boolean; message?: string }> {
    const response = await axios.post(`${API_BASE}/uninstall`, { plugin_id: pluginId });
    return response.data;
  }

  async searchPlugins(query: string): Promise<Plugin[]> {
    const response = await axios.get(`${API_BASE}/search`, { params: { q: query } });
    return response.data.plugins;
  }

  async getPanels(): Promise<PanelConfig[]> {
    const response = await axios.get(`${API_BASE}/panels`);
    return response.data.panels;
  }

  async getThemes(): Promise<ThemeDefinition[]> {
    const response = await axios.get(`${API_BASE}/themes`);
    return response.data.themes;
  }

  async getCommands(): Promise<Command[]> {
    const response = await axios.get(`${API_BASE}/commands`);
    return response.data.commands;
  }

  async executeCommand(commandId: string, args: any): Promise<any> {
    const response = await axios.post(`${API_BASE}/command/execute`, {
      command_id: commandId,
      args,
    });
    return response.data.result;
  }
}

export const pluginService = new PluginService();
