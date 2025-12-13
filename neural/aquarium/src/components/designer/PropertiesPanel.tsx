import React, { useState, useEffect } from 'react';
import { Node } from 'reactflow';
import { LayerNodeData } from '../../types';
import { getLayerDefinition } from '../../data/layerDefinitions';
import './PropertiesPanel.css';

interface PropertiesPanelProps {
  selectedNode: Node<LayerNodeData> | null;
  onUpdateNode: (nodeId: string, updates: Partial<LayerNodeData>) => void;
}

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({ selectedNode, onUpdateNode }) => {
  const [parameters, setParameters] = useState<Record<string, any>>({});

  useEffect(() => {
    if (selectedNode) {
      setParameters(selectedNode.data.parameters || {});
    }
  }, [selectedNode]);

  if (!selectedNode) {
    return (
      <div className="properties-panel">
        <div className="panel-header">
          <h3>Properties</h3>
        </div>
        <div className="panel-body empty">
          <p>Select a layer to edit its properties</p>
        </div>
      </div>
    );
  }

  const layerDef = getLayerDefinition(selectedNode.data.layerType);

  const handleParameterChange = (key: string, value: any) => {
    const updatedParams = { ...parameters, [key]: value };
    setParameters(updatedParams);
    onUpdateNode(selectedNode.id, { parameters: updatedParams });
  };

  const parseValue = (value: string, currentValue: any): any => {
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (value === 'null') return null;
    if (value.startsWith('[') && value.endsWith(']')) {
      try {
        return JSON.parse(value);
      } catch {
        return value;
      }
    }
    if (!isNaN(Number(value)) && value !== '') {
      return Number(value);
    }
    return value;
  };

  const renderParameterInput = (key: string, value: any) => {
    const stringValue = Array.isArray(value) ? JSON.stringify(value) : String(value);

    if (typeof value === 'boolean') {
      return (
        <select
          value={value ? 'true' : 'false'}
          onChange={(e) => handleParameterChange(key, e.target.value === 'true')}
          className="param-input"
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      );
    }

    if (key === 'activation') {
      return (
        <select
          value={value}
          onChange={(e) => handleParameterChange(key, e.target.value)}
          className="param-input"
        >
          <option value="relu">relu</option>
          <option value="sigmoid">sigmoid</option>
          <option value="tanh">tanh</option>
          <option value="softmax">softmax</option>
          <option value="linear">linear</option>
          <option value="elu">elu</option>
          <option value="selu">selu</option>
        </select>
      );
    }

    if (key === 'padding') {
      return (
        <select
          value={value}
          onChange={(e) => handleParameterChange(key, e.target.value)}
          className="param-input"
        >
          <option value="valid">valid</option>
          <option value="same">same</option>
        </select>
      );
    }

    return (
      <input
        type="text"
        value={stringValue}
        onChange={(e) => handleParameterChange(key, parseValue(e.target.value, value))}
        className="param-input"
      />
    );
  };

  return (
    <div className="properties-panel">
      <div className="panel-header" style={{ backgroundColor: layerDef?.color || '#95E1D3' }}>
        <span className="panel-icon">{layerDef?.icon || '●'}</span>
        <h3>{selectedNode.data.layerType}</h3>
      </div>

      <div className="panel-body">
        <div className="layer-info">
          <div className="info-row">
            <span className="info-label">Category:</span>
            <span className="info-value">{selectedNode.data.category}</span>
          </div>
          {selectedNode.data.outputShape && (
            <div className="info-row">
              <span className="info-label">Output Shape:</span>
              <span className="info-value">{selectedNode.data.outputShape}</span>
            </div>
          )}
        </div>

        <div className="parameters-section">
          <h4>Parameters</h4>
          {Object.entries(parameters).length === 0 ? (
            <p className="no-params">No configurable parameters</p>
          ) : (
            <div className="parameters-list">
              {Object.entries(parameters).map(([key, value]) => (
                <div key={key} className="parameter-row">
                  <label className="param-label">{key}</label>
                  {renderParameterInput(key, value)}
                </div>
              ))}
            </div>
          )}
        </div>

        {layerDef && (
          <div className="layer-description">
            <h4>Description</h4>
            <p>{layerDef.description}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PropertiesPanel;
