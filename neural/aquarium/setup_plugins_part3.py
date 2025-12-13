#!/usr/bin/env python3
"""
Part 3: Marketplace UI and API endpoints
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MARKETPLACE_DIR = BASE_DIR / "src" / "components" / "marketplace"
BACKEND_DIR = BASE_DIR / "backend"

files = {}

# ============================================================================
# MARKETPLACE UI COMPONENTS (React/TypeScript)
# ============================================================================

files[MARKETPLACE_DIR / "PluginMarketplace.tsx"] = '''import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './PluginMarketplace.css';

interface PluginMetadata {
  id: string;
  name: string;
  version: string;
  author: string;
  description: string;
  capabilities: string[];
  homepage?: string;
  repository?: string;
  keywords: string[];
  license: string;
  icon?: string;
  rating: number;
  downloads: number;
}

interface Plugin extends PluginMetadata {
  enabled: boolean;
  installed: boolean;
}

const PluginMarketplace: React.FC = () => {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [filteredPlugins, setFilteredPlugins] = useState<Plugin[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCapability, setFilterCapability] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('rating');
  const [loading, setLoading] = useState(true);
  const [selectedPlugin, setSelectedPlugin] = useState<Plugin | null>(null);

  useEffect(() => {
    fetchPlugins();
  }, []);

  useEffect(() => {
    filterAndSortPlugins();
  }, [plugins, searchQuery, filterCapability, sortBy]);

  const fetchPlugins = async () => {
    try {
      const response = await axios.get('/api/plugins/list');
      setPlugins(response.data.plugins);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching plugins:', error);
      setLoading(false);
    }
  };

  const filterAndSortPlugins = () => {
    let filtered = [...plugins];

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(plugin =>
        plugin.name.toLowerCase().includes(query) ||
        plugin.description.toLowerCase().includes(query) ||
        plugin.keywords.some(kw => kw.toLowerCase().includes(query))
      );
    }

    if (filterCapability !== 'all') {
      filtered = filtered.filter(plugin =>
        plugin.capabilities.includes(filterCapability)
      );
    }

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'rating':
          return b.rating - a.rating;
        case 'downloads':
          return b.downloads - a.downloads;
        case 'name':
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    setFilteredPlugins(filtered);
  };

  const installPlugin = async (pluginId: string) => {
    try {
      await axios.post('/api/plugins/install', { plugin_id: pluginId });
      fetchPlugins();
    } catch (error) {
      console.error('Error installing plugin:', error);
    }
  };

  const uninstallPlugin = async (pluginId: string) => {
    try {
      await axios.post('/api/plugins/uninstall', { plugin_id: pluginId });
      fetchPlugins();
    } catch (error) {
      console.error('Error uninstalling plugin:', error);
    }
  };

  const enablePlugin = async (pluginId: string) => {
    try {
      await axios.post('/api/plugins/enable', { plugin_id: pluginId });
      fetchPlugins();
    } catch (error) {
      console.error('Error enabling plugin:', error);
    }
  };

  const disablePlugin = async (pluginId: string) => {
    try {
      await axios.post('/api/plugins/disable', { plugin_id: pluginId });
      fetchPlugins();
    } catch (error) {
      console.error('Error disabling plugin:', error);
    }
  };

  const renderStars = (rating: number) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;

    for (let i = 0; i < fullStars; i++) {
      stars.push(<span key={`full-${i}`} className="star full">★</span>);
    }
    if (hasHalfStar) {
      stars.push(<span key="half" className="star half">★</span>);
    }
    for (let i = stars.length; i < 5; i++) {
      stars.push(<span key={`empty-${i}`} className="star empty">☆</span>);
    }

    return <div className="rating">{stars} <span className="rating-text">({rating.toFixed(1)})</span></div>;
  };

  if (loading) {
    return <div className="plugin-marketplace loading">Loading plugins...</div>;
  }

  return (
    <div className="plugin-marketplace">
      <div className="marketplace-header">
        <h1>Plugin Marketplace</h1>
        <p>Extend Neural Aquarium with custom panels, themes, and integrations</p>
      </div>

      <div className="marketplace-controls">
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search plugins..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="filters">
          <select
            value={filterCapability}
            onChange={(e) => setFilterCapability(e.target.value)}
          >
            <option value="all">All Capabilities</option>
            <option value="panel">Panels</option>
            <option value="theme">Themes</option>
            <option value="command">Commands</option>
            <option value="visualization">Visualizations</option>
            <option value="integration">Integrations</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          >
            <option value="rating">Highest Rated</option>
            <option value="downloads">Most Downloads</option>
            <option value="name">Name (A-Z)</option>
          </select>
        </div>
      </div>

      <div className="plugin-grid">
        {filteredPlugins.map(plugin => (
          <div key={plugin.id} className="plugin-card">
            <div className="plugin-header">
              <div className="plugin-icon">{plugin.icon || '🧩'}</div>
              <div className="plugin-title">
                <h3>{plugin.name}</h3>
                <span className="plugin-version">v{plugin.version}</span>
              </div>
            </div>

            <div className="plugin-author">by {plugin.author}</div>

            <p className="plugin-description">{plugin.description}</p>

            <div className="plugin-capabilities">
              {plugin.capabilities.map(cap => (
                <span key={cap} className="capability-badge">{cap}</span>
              ))}
            </div>

            <div className="plugin-stats">
              {renderStars(plugin.rating)}
              <div className="downloads">↓ {plugin.downloads.toLocaleString()}</div>
            </div>

            <div className="plugin-actions">
              {plugin.installed ? (
                <>
                  {plugin.enabled ? (
                    <button
                      className="btn btn-secondary"
                      onClick={() => disablePlugin(plugin.id)}
                    >
                      Disable
                    </button>
                  ) : (
                    <button
                      className="btn btn-primary"
                      onClick={() => enablePlugin(plugin.id)}
                    >
                      Enable
                    </button>
                  )}
                  <button
                    className="btn btn-danger"
                    onClick={() => uninstallPlugin(plugin.id)}
                  >
                    Uninstall
                  </button>
                </>
              ) : (
                <button
                  className="btn btn-success"
                  onClick={() => installPlugin(plugin.id)}
                >
                  Install
                </button>
              )}
              <button
                className="btn btn-info"
                onClick={() => setSelectedPlugin(plugin)}
              >
                Details
              </button>
            </div>
          </div>
        ))}
      </div>

      {selectedPlugin && (
        <div className="plugin-modal-overlay" onClick={() => setSelectedPlugin(null)}>
          <div className="plugin-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedPlugin(null)}>×</button>
            
            <div className="modal-header">
              <div className="plugin-icon large">{selectedPlugin.icon || '🧩'}</div>
              <div>
                <h2>{selectedPlugin.name}</h2>
                <p className="plugin-author">by {selectedPlugin.author}</p>
              </div>
            </div>

            <div className="modal-content">
              <p className="plugin-description">{selectedPlugin.description}</p>

              <div className="plugin-meta">
                <div className="meta-item">
                  <strong>Version:</strong> {selectedPlugin.version}
                </div>
                <div className="meta-item">
                  <strong>License:</strong> {selectedPlugin.license}
                </div>
                {selectedPlugin.homepage && (
                  <div className="meta-item">
                    <strong>Homepage:</strong>{' '}
                    <a href={selectedPlugin.homepage} target="_blank" rel="noopener noreferrer">
                      {selectedPlugin.homepage}
                    </a>
                  </div>
                )}
                {selectedPlugin.repository && (
                  <div className="meta-item">
                    <strong>Repository:</strong>{' '}
                    <a href={selectedPlugin.repository} target="_blank" rel="noopener noreferrer">
                      {selectedPlugin.repository}
                    </a>
                  </div>
                )}
              </div>

              <div className="plugin-capabilities">
                <strong>Capabilities:</strong>
                {selectedPlugin.capabilities.map(cap => (
                  <span key={cap} className="capability-badge">{cap}</span>
                ))}
              </div>

              <div className="plugin-keywords">
                <strong>Keywords:</strong>
                {selectedPlugin.keywords.map(kw => (
                  <span key={kw} className="keyword-tag">{kw}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PluginMarketplace;
'''

files[MARKETPLACE_DIR / "PluginMarketplace.css"] = '''.plugin-marketplace {
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.marketplace-header {
  margin-bottom: 32px;
  text-align: center;
}

.marketplace-header h1 {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 8px;
  color: #1a1a1a;
}

.marketplace-header p {
  font-size: 16px;
  color: #666;
}

.marketplace-controls {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}

.search-bar {
  flex: 1;
  min-width: 300px;
}

.search-bar input {
  width: 100%;
  padding: 12px 16px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 8px;
  outline: none;
}

.search-bar input:focus {
  border-color: #0066cc;
  box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
}

.filters {
  display: flex;
  gap: 12px;
}

.filters select {
  padding: 12px 16px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: white;
  cursor: pointer;
  outline: none;
}

.filters select:hover {
  border-color: #0066cc;
}

.plugin-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 24px;
}

.plugin-card {
  background: white;
  border: 1px solid #e5e5e5;
  border-radius: 12px;
  padding: 20px;
  transition: all 0.2s;
  display: flex;
  flex-direction: column;
}

.plugin-card:hover {
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.plugin-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.plugin-icon {
  font-size: 32px;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  border-radius: 8px;
}

.plugin-icon.large {
  font-size: 48px;
  width: 72px;
  height: 72px;
}

.plugin-title {
  flex: 1;
}

.plugin-title h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #1a1a1a;
}

.plugin-version {
  font-size: 12px;
  color: #999;
  font-weight: 400;
}

.plugin-author {
  font-size: 13px;
  color: #666;
  margin-bottom: 12px;
}

.plugin-description {
  font-size: 14px;
  color: #4a4a4a;
  margin-bottom: 12px;
  line-height: 1.5;
  flex: 1;
}

.plugin-capabilities {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.capability-badge {
  background: #e3f2fd;
  color: #1976d2;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

.plugin-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-top: 12px;
  border-top: 1px solid #f0f0f0;
}

.rating {
  display: flex;
  align-items: center;
  gap: 4px;
}

.star {
  font-size: 16px;
  color: #ddd;
}

.star.full {
  color: #ffa726;
}

.star.half {
  color: #ffb84d;
}

.rating-text {
  font-size: 13px;
  color: #666;
  margin-left: 4px;
}

.downloads {
  font-size: 13px;
  color: #666;
}

.plugin-actions {
  display: flex;
  gap: 8px;
}

.btn {
  flex: 1;
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 500;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: #0066cc;
  color: white;
}

.btn-primary:hover {
  background: #0052a3;
}

.btn-secondary {
  background: #f5f5f5;
  color: #333;
}

.btn-secondary:hover {
  background: #e0e0e0;
}

.btn-success {
  background: #4caf50;
  color: white;
}

.btn-success:hover {
  background: #388e3c;
}

.btn-danger {
  background: #f44336;
  color: white;
}

.btn-danger:hover {
  background: #d32f2f;
}

.btn-info {
  background: #2196f3;
  color: white;
}

.btn-info:hover {
  background: #1976d2;
}

.plugin-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 24px;
}

.plugin-modal {
  background: white;
  border-radius: 16px;
  max-width: 600px;
  width: 100%;
  max-height: 80vh;
  overflow-y: auto;
  padding: 32px;
  position: relative;
}

.modal-close {
  position: absolute;
  top: 16px;
  right: 16px;
  background: none;
  border: none;
  font-size: 32px;
  cursor: pointer;
  color: #999;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  transition: all 0.2s;
}

.modal-close:hover {
  background: #f5f5f5;
  color: #333;
}

.modal-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.modal-header h2 {
  margin: 0;
  font-size: 24px;
  color: #1a1a1a;
}

.modal-content {
  font-size: 14px;
  line-height: 1.6;
}

.plugin-meta {
  margin: 20px 0;
  padding: 16px;
  background: #f9f9f9;
  border-radius: 8px;
}

.meta-item {
  margin-bottom: 8px;
}

.meta-item:last-child {
  margin-bottom: 0;
}

.meta-item strong {
  color: #333;
  margin-right: 8px;
}

.meta-item a {
  color: #0066cc;
  text-decoration: none;
}

.meta-item a:hover {
  text-decoration: underline;
}

.plugin-keywords {
  margin-top: 16px;
}

.keyword-tag {
  display: inline-block;
  background: #f5f5f5;
  color: #666;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  margin: 4px 4px 4px 0;
}

.loading {
  text-align: center;
  padding: 48px;
  color: #999;
  font-size: 18px;
}
'''

files[MARKETPLACE_DIR / "index.ts"] = '''export { default as PluginMarketplace } from './PluginMarketplace';
'''

# ============================================================================
# BACKEND API FOR PLUGINS
# ============================================================================

files[BACKEND_DIR / "plugin_api.py"] = '''"""
Flask API endpoints for plugin management
"""

from flask import Blueprint, request, jsonify
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)

plugin_bp = Blueprint('plugins', __name__, url_prefix='/api/plugins')

plugin_manager = PluginManager()


@plugin_bp.route('/list', methods=['GET'])
def list_plugins():
    """List all available plugins."""
    try:
        plugins = plugin_manager.list_plugins()
        enabled_ids = set(plugin_manager.registry.list_enabled_plugins())
        
        plugin_data = []
        for metadata in plugins:
            plugin_dict = metadata.to_dict()
            plugin_dict['enabled'] = metadata.id in enabled_ids
            plugin_dict['installed'] = True
            plugin_data.append(plugin_dict)
        
        return jsonify({
            'plugins': plugin_data,
            'count': len(plugin_data)
        }), 200
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/enabled', methods=['GET'])
def list_enabled_plugins():
    """List enabled plugins."""
    try:
        plugins = plugin_manager.list_enabled_plugins()
        plugin_data = [p.to_dict() for p in plugins]
        
        return jsonify({
            'plugins': plugin_data,
            'count': len(plugin_data)
        }), 200
    except Exception as e:
        logger.error(f"Error listing enabled plugins: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/details/<plugin_id>', methods=['GET'])
def get_plugin_details(plugin_id: str):
    """Get detailed information about a plugin."""
    try:
        metadata = plugin_manager.get_plugin_metadata(plugin_id)
        if not metadata:
            return jsonify({'error': 'Plugin not found'}), 404
        
        plugin = plugin_manager.get_plugin(plugin_id)
        
        data = metadata.to_dict()
        if plugin:
            data['enabled'] = plugin.enabled
            data['config_schema'] = plugin.get_config_schema()
        
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Error getting plugin details: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/enable', methods=['POST'])
def enable_plugin():
    """Enable a plugin."""
    try:
        data = request.get_json()
        plugin_id = data.get('plugin_id')
        
        if not plugin_id:
            return jsonify({'error': 'Missing plugin_id'}), 400
        
        success = plugin_manager.enable_plugin(plugin_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Plugin {plugin_id} enabled'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to enable plugin'
            }), 500
    except Exception as e:
        logger.error(f"Error enabling plugin: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/disable', methods=['POST'])
def disable_plugin():
    """Disable a plugin."""
    try:
        data = request.get_json()
        plugin_id = data.get('plugin_id')
        
        if not plugin_id:
            return jsonify({'error': 'Missing plugin_id'}), 400
        
        success = plugin_manager.disable_plugin(plugin_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Plugin {plugin_id} disabled'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to disable plugin'
            }), 500
    except Exception as e:
        logger.error(f"Error disabling plugin: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/install', methods=['POST'])
def install_plugin():
    """Install a plugin from npm or PyPI."""
    try:
        data = request.get_json()
        source = data.get('source', 'npm')
        plugin_name = data.get('plugin_name')
        version = data.get('version')
        
        if not plugin_name:
            return jsonify({'error': 'Missing plugin_name'}), 400
        
        success = plugin_manager.install_plugin(source, plugin_name, version)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Plugin {plugin_name} installed'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to install plugin'
            }), 500
    except Exception as e:
        logger.error(f"Error installing plugin: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/uninstall', methods=['POST'])
def uninstall_plugin():
    """Uninstall a plugin."""
    try:
        data = request.get_json()
        plugin_id = data.get('plugin_id')
        
        if not plugin_id:
            return jsonify({'error': 'Missing plugin_id'}), 400
        
        success = plugin_manager.uninstall_plugin(plugin_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Plugin {plugin_id} uninstalled'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to uninstall plugin'
            }), 500
    except Exception as e:
        logger.error(f"Error uninstalling plugin: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/search', methods=['GET'])
def search_plugins():
    """Search plugins."""
    try:
        query = request.args.get('q', '')
        
        if not query:
            return jsonify({'error': 'Missing query parameter q'}), 400
        
        results = plugin_manager.search_plugins(query)
        plugin_data = [p.to_dict() for p in results]
        
        return jsonify({
            'plugins': plugin_data,
            'count': len(plugin_data),
            'query': query
        }), 200
    except Exception as e:
        logger.error(f"Error searching plugins: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/panels', methods=['GET'])
def get_panels():
    """Get all panel plugins."""
    try:
        panels = plugin_manager.get_panels()
        return jsonify({'panels': panels}), 200
    except Exception as e:
        logger.error(f"Error getting panels: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/themes', methods=['GET'])
def get_themes():
    """Get all theme plugins."""
    try:
        themes = plugin_manager.get_themes()
        return jsonify({'themes': themes}), 200
    except Exception as e:
        logger.error(f"Error getting themes: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/commands', methods=['GET'])
def get_commands():
    """Get all available commands from plugins."""
    try:
        commands = plugin_manager.get_commands()
        return jsonify({'commands': commands}), 200
    except Exception as e:
        logger.error(f"Error getting commands: {e}")
        return jsonify({'error': str(e)}), 500


@plugin_bp.route('/command/execute', methods=['POST'])
def execute_command():
    """Execute a plugin command."""
    try:
        data = request.get_json()
        command_id = data.get('command_id')
        args = data.get('args', {})
        
        if not command_id:
            return jsonify({'error': 'Missing command_id'}), 400
        
        result = plugin_manager.execute_command(command_id, args)
        
        return jsonify({
            'success': True,
            'result': result
        }), 200
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return jsonify({'error': str(e)}), 500
'''

# Write all files
for filepath, content in files.items():
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\\n✅ Marketplace UI and API endpoints created successfully!")
