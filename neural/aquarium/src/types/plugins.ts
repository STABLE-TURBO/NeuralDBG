/**
 * TypeScript type definitions for Neural Aquarium Plugin System
 */

export type PluginCapability =
  | 'panel'
  | 'theme'
  | 'command'
  | 'visualization'
  | 'integration'
  | 'language_support'
  | 'code_completion'
  | 'linter'
  | 'formatter';

export interface PluginMetadata {
  id: string;
  name: string;
  version: string;
  author: string;
  description: string;
  capabilities: PluginCapability[];
  homepage?: string;
  repository?: string;
  keywords: string[];
  license: string;
  dependencies: string[];
  python_dependencies: string[];
  npm_dependencies: Record<string, string>;
  min_aquarium_version: string;
  icon?: string;
  rating: number;
  downloads: number;
}

export interface Plugin extends PluginMetadata {
  enabled: boolean;
  installed: boolean;
}

export interface PanelConfig {
  id: string;
  name: string;
  component: string;
  config: {
    title: string;
    position: 'left' | 'right' | 'top' | 'bottom';
    width?: number;
    height?: number;
    resizable?: boolean;
    closeable?: boolean;
    icon?: string;
  };
}

export interface ThemeDefinition {
  id: string;
  name: string;
  definition: {
    name: string;
    type: 'dark' | 'light';
    colors: Record<string, string>;
    fonts: Record<string, string>;
    spacing: Record<string, string>;
    borderRadius?: Record<string, string>;
    shadows?: Record<string, string>;
  };
  editor_theme?: {
    base: string;
    inherit: boolean;
    rules: Array<{
      token: string;
      foreground: string;
      fontStyle?: string;
    }>;
    colors: Record<string, string>;
  };
}

export interface Command {
  id: string;
  name: string;
  description: string;
  category?: string;
  keybinding?: string;
  icon?: string;
  when?: string;
}

export interface VisualizationData {
  type: string;
  component: string;
  data: any;
  options?: any;
}

export interface PluginConfig {
  [key: string]: any;
}

export interface ConfigSchema {
  [property: string]: {
    type: 'string' | 'number' | 'boolean' | 'array' | 'object';
    description: string;
    default?: any;
    required?: boolean;
    sensitive?: boolean;
    enum?: any[];
    minimum?: number;
    maximum?: number;
    pattern?: string;
  };
}

export interface PluginAPI {
  listPlugins: () => Promise<Plugin[]>;
  listEnabledPlugins: () => Promise<Plugin[]>;
  getPluginDetails: (pluginId: string) => Promise<PluginMetadata>;
  enablePlugin: (pluginId: string) => Promise<{ success: boolean; message?: string }>;
  disablePlugin: (pluginId: string) => Promise<{ success: boolean; message?: string }>;
  installPlugin: (source: 'npm' | 'pypi', pluginName: string, version?: string) => Promise<{ success: boolean; message?: string }>;
  uninstallPlugin: (pluginId: string) => Promise<{ success: boolean; message?: string }>;
  searchPlugins: (query: string) => Promise<Plugin[]>;
  getPanels: () => Promise<PanelConfig[]>;
  getThemes: () => Promise<ThemeDefinition[]>;
  getCommands: () => Promise<Command[]>;
  executeCommand: (commandId: string, args: any) => Promise<any>;
}
