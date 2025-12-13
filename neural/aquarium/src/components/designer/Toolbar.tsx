import React from 'react';
import './Toolbar.css';

interface ToolbarProps {
  onToggleCodeEditor: () => void;
  onAutoLayout: () => void;
  onClear: () => void;
  onExport: () => void;
  onImport: () => void;
  showCodeEditor: boolean;
  nodeCount: number;
  edgeCount: number;
}

const Toolbar: React.FC<ToolbarProps> = ({
  onToggleCodeEditor,
  onAutoLayout,
  onClear,
  onExport,
  onImport,
  showCodeEditor,
  nodeCount,
  edgeCount,
}) => {
  return (
    <div className="designer-toolbar">
      <button className="toolbar-btn" onClick={onToggleCodeEditor}>
        {showCodeEditor ? '📊 Show Designer' : '💻 Show Code'}
      </button>
      <button className="toolbar-btn" onClick={onAutoLayout}>
        🔄 Auto Layout
      </button>
      <button className="toolbar-btn" onClick={onExport}>
        💾 Export
      </button>
      <button className="toolbar-btn" onClick={onImport}>
        📂 Import
      </button>
      <button className="toolbar-btn danger" onClick={onClear}>
        🗑️ Clear
      </button>
      <div className="toolbar-spacer" />
      <div className="node-count">
        Layers: {nodeCount - 1} | Connections: {edgeCount}
      </div>
    </div>
  );
};

export default Toolbar;
