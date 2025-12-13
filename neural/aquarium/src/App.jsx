import React, { useState } from 'react';
import { ShapePropagationPanel } from './components/shapes';
import ShapePropagationPlotly from './components/shapes/ShapePropagationPlotly';
import './App.css';

function App() {
  const [viewMode, setViewMode] = useState('d3');

  return (
    <div className="App">
      <header className="App-header">
        <h1>Neural Aquarium - Shape Propagation Viewer</h1>
        <div className="view-toggle">
          <button 
            className={viewMode === 'd3' ? 'active' : ''} 
            onClick={() => setViewMode('d3')}
          >
            D3.js View
          </button>
          <button 
            className={viewMode === 'plotly' ? 'active' : ''} 
            onClick={() => setViewMode('plotly')}
          >
            Plotly View
          </button>
        </div>
      </header>
      <main className="App-main">
        {viewMode === 'd3' ? (
          <ShapePropagationPanel />
        ) : (
          <ShapePropagationPlotly />
        )}
      </main>
    </div>
  );
}

export default App;
