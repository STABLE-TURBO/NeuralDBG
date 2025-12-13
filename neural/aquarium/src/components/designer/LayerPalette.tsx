import React, { useState } from 'react';
import { LAYER_CATEGORIES, getLayersByCategory } from '../../data/layerDefinitions';
import { LayerCategory } from '../../types';
import './LayerPalette.css';

interface LayerPaletteProps {
  onLayerSelect: (layerType: string, category: LayerCategory) => void;
}

const LayerPalette: React.FC<LayerPaletteProps> = ({ onLayerSelect }) => {
  const [selectedCategory, setSelectedCategory] = useState<LayerCategory>('Core');
  const [searchQuery, setSearchQuery] = useState('');

  const layers = getLayersByCategory(selectedCategory);
  const filteredLayers = layers.filter(layer =>
    layer.type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const onDragStart = (event: React.DragEvent, layerType: string, category: LayerCategory) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify({ layerType, category }));
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="layer-palette">
      <div className="palette-header">
        <h3>Layer Palette</h3>
        <input
          type="text"
          placeholder="Search layers..."
          className="search-input"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <div className="category-tabs">
        {LAYER_CATEGORIES.map(category => (
          <button
            key={category}
            className={`category-tab ${selectedCategory === category ? 'active' : ''}`}
            onClick={() => setSelectedCategory(category)}
          >
            {category}
          </button>
        ))}
      </div>

      <div className="layers-list">
        {filteredLayers.map(layer => (
          <div
            key={layer.type}
            className="layer-item"
            draggable
            onDragStart={(e) => onDragStart(e, layer.type, layer.category)}
            onClick={() => onLayerSelect(layer.type, layer.category)}
            style={{ borderLeftColor: layer.color }}
          >
            <div className="layer-item-icon" style={{ color: layer.color }}>
              {layer.icon}
            </div>
            <div className="layer-item-content">
              <div className="layer-item-name">{layer.type}</div>
              <div className="layer-item-description">{layer.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LayerPalette;
