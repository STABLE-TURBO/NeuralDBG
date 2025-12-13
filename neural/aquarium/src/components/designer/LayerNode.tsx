import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { LayerNodeData } from '../../types';
import { getLayerDefinition } from '../../data/layerDefinitions';
import './LayerNode.css';

const LayerNode: React.FC<NodeProps<LayerNodeData>> = ({ data, selected }) => {
  const layerDef = getLayerDefinition(data.layerType);
  
  return (
    <div className={`layer-node ${selected ? 'selected' : ''}`} 
         style={{ borderColor: layerDef?.color || '#95E1D3' }}>
      <Handle type="target" position={Position.Top} className="layer-handle" />
      
      <div className="layer-header" style={{ backgroundColor: layerDef?.color || '#95E1D3' }}>
        <span className="layer-icon">{layerDef?.icon || '●'}</span>
        <span className="layer-type">{data.layerType}</span>
      </div>
      
      <div className="layer-body">
        {Object.entries(data.parameters).length > 0 && (
          <div className="layer-params">
            {Object.entries(data.parameters).slice(0, 3).map(([key, value]) => (
              <div key={key} className="param-row">
                <span className="param-key">{key}:</span>
                <span className="param-value">
                  {Array.isArray(value) ? `[${value.join(', ')}]` : String(value)}
                </span>
              </div>
            ))}
            {Object.entries(data.parameters).length > 3 && (
              <div className="param-more">
                +{Object.entries(data.parameters).length - 3} more
              </div>
            )}
          </div>
        )}
        
        {data.outputShape && (
          <div className="layer-shape">
            <span className="shape-label">Output:</span>
            <span className="shape-value">{data.outputShape}</span>
          </div>
        )}
        
        {data.parameterCount !== undefined && (
          <div className="layer-params-count">
            <span className="params-count-label">Params:</span>
            <span className="params-count-value">{data.parameterCount.toLocaleString()}</span>
          </div>
        )}
      </div>
      
      <Handle type="source" position={Position.Bottom} className="layer-handle" />
    </div>
  );
};

export default LayerNode;
