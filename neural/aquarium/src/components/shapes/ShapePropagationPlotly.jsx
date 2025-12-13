import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './ShapePropagationPanel.css';

const ShapePropagationPlotly = ({ modelData, shapePropagator }) => {
  const [shapeHistory, setShapeHistory] = useState([]);
  const [errors, setErrors] = useState([]);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(1000);

  const fetchShapeData = async () => {
    try {
      const response = await fetch('/api/shape-propagation');
      const data = await response.json();
      
      if (data.shape_history) {
        setShapeHistory(data.shape_history);
      }
      
      if (data.errors) {
        setErrors(data.errors);
      }
    } catch (error) {
      console.error('Error fetching shape data:', error);
    }
  };

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchShapeData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  useEffect(() => {
    fetchShapeData();
  }, []);

  const createFlowChart = () => {
    if (shapeHistory.length === 0) return null;

    const layers = shapeHistory.map((item, idx) => item.layer_name || item.layer);
    const xPositions = shapeHistory.map((_, idx) => idx);
    const yPositions = shapeHistory.map(() => 0);

    const nodeColors = shapeHistory.map(item => 
      item.error ? '#e74c3c' : (selectedLayer === item.layer ? '#2ecc71' : '#3498db')
    );

    const nodeTrace = {
      x: xPositions,
      y: yPositions,
      mode: 'markers+text',
      type: 'scatter',
      marker: {
        size: 30,
        color: nodeColors,
        line: {
          color: '#2c3e50',
          width: 2
        }
      },
      text: layers,
      textposition: 'top center',
      textfont: {
        size: 12,
        color: '#2c3e50'
      },
      hovertemplate: '<b>%{text}</b><br>' +
        'Input: %{customdata[0]}<br>' +
        'Output: %{customdata[1]}<br>' +
        'Parameters: %{customdata[2]}<br>' +
        'Memory: %{customdata[3]}<br>' +
        '<extra></extra>',
      customdata: shapeHistory.map(item => [
        formatShape(item.input_shape),
        formatShape(item.output_shape),
        formatNumber(item.parameters || 0),
        formatMemory(item.memory || 0)
      ])
    };

    const edgeTraces = [];
    for (let i = 0; i < xPositions.length - 1; i++) {
      const hasError = checkShapeMismatch(
        shapeHistory[i].output_shape,
        shapeHistory[i + 1].input_shape
      );
      
      edgeTraces.push({
        x: [xPositions[i], xPositions[i + 1]],
        y: [yPositions[i], yPositions[i + 1]],
        mode: 'lines',
        type: 'scatter',
        line: {
          color: hasError ? '#e74c3c' : '#3498db',
          width: 2,
          dash: hasError ? 'dash' : 'solid'
        },
        hoverinfo: 'skip',
        showlegend: false
      });
    }

    const layout = {
      title: 'Shape Flow Diagram',
      showlegend: false,
      hovermode: 'closest',
      xaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: true,
        title: 'Layer Sequence'
      },
      yaxis: {
        showgrid: false,
        zeroline: false,
        showticklabels: false
      },
      height: 400,
      margin: { t: 50, b: 50, l: 50, r: 50 },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white'
    };

    return { data: [nodeTrace, ...edgeTraces], layout };
  };

  const createMemoryChart = () => {
    if (shapeHistory.length === 0) return null;

    const layers = shapeHistory.map((item, idx) => item.layer_name || item.layer);
    const memory = shapeHistory.map(item => (item.memory || 0) / (1024 * 1024));
    const parameters = shapeHistory.map(item => (item.parameters || 0) / 1e6);

    const data = [
      {
        x: layers,
        y: memory,
        type: 'bar',
        name: 'Memory (MB)',
        marker: { color: '#3498db' },
        hovertemplate: '<b>%{x}</b><br>Memory: %{y:.2f} MB<extra></extra>'
      },
      {
        x: layers,
        y: parameters,
        type: 'bar',
        name: 'Parameters (M)',
        marker: { color: '#9b59b6' },
        yaxis: 'y2',
        hovertemplate: '<b>%{x}</b><br>Parameters: %{y:.2f}M<extra></extra>'
      }
    ];

    const layout = {
      title: 'Memory & Parameters by Layer',
      xaxis: { title: 'Layer' },
      yaxis: { title: 'Memory (MB)', side: 'left' },
      yaxis2: {
        title: 'Parameters (Millions)',
        overlaying: 'y',
        side: 'right'
      },
      height: 400,
      margin: { t: 50, b: 100, l: 60, r: 60 },
      barmode: 'group',
      plot_bgcolor: 'white',
      paper_bgcolor: 'white'
    };

    return { data, layout };
  };

  const createShapeEvolutionChart = () => {
    if (shapeHistory.length === 0) return null;

    const layers = shapeHistory.map((item, idx) => item.layer_name || item.layer);
    
    const shapeSizes = shapeHistory.map(item => {
      const shape = item.output_shape;
      if (!shape) return 0;
      const arr = Array.isArray(shape) ? shape : [shape];
      return arr.reduce((acc, val) => acc * (val || 1), 1);
    });

    const data = [
      {
        x: layers,
        y: shapeSizes,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Tensor Size',
        line: { color: '#e74c3c', width: 3 },
        marker: { size: 10, color: '#c0392b' },
        fill: 'tozeroy',
        fillcolor: 'rgba(231, 76, 60, 0.1)',
        hovertemplate: '<b>%{x}</b><br>Tensor Size: %{y:,}<extra></extra>'
      }
    ];

    const layout = {
      title: 'Tensor Size Evolution',
      xaxis: { title: 'Layer' },
      yaxis: { title: 'Tensor Elements', type: 'log' },
      height: 400,
      margin: { t: 50, b: 100, l: 60, r: 60 },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white'
    };

    return { data, layout };
  };

  const checkShapeMismatch = (outputShape, inputShape) => {
    if (!outputShape || !inputShape) return false;
    
    const out = Array.isArray(outputShape) ? outputShape : String(outputShape).split(',').map(s => parseInt(s.trim()));
    const inp = Array.isArray(inputShape) ? inputShape : String(inputShape).split(',').map(s => parseInt(s.trim()));
    
    if (out.length !== inp.length) return true;
    
    for (let i = 1; i < out.length; i++) {
      if (out[i] !== inp[i] && out[i] !== null && inp[i] !== null) {
        return true;
      }
    }
    
    return false;
  };

  const formatShape = (shape) => {
    if (!shape) return 'N/A';
    if (Array.isArray(shape)) {
      return `[${shape.map(s => s === null ? 'None' : s).join(', ')}]`;
    }
    return String(shape);
  };

  const formatNumber = (num) => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toFixed(0);
  };

  const formatMemory = (bytes) => {
    if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${bytes.toFixed(0)} B`;
  };

  const flowChart = createFlowChart();
  const memoryChart = createMemoryChart();
  const evolutionChart = createShapeEvolutionChart();

  return (
    <div className="shape-propagation-panel">
      <div className="panel-header">
        <h2>Real-Time Shape Propagation (Plotly)</h2>
        <div className="controls">
          <label>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <label>
            Interval (ms):
            <input
              type="number"
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
              min="100"
              max="5000"
              step="100"
            />
          </label>
          <button onClick={fetchShapeData} className="refresh-btn">
            Refresh Now
          </button>
        </div>
      </div>

      <div className="panel-content">
        {flowChart && (
          <div className="visualization-container">
            <Plot
              data={flowChart.data}
              layout={flowChart.layout}
              config={{ responsive: true, displayModeBar: true }}
              style={{ width: '100%', height: '400px' }}
            />
          </div>
        )}

        {memoryChart && (
          <div className="visualization-container">
            <Plot
              data={memoryChart.data}
              layout={memoryChart.layout}
              config={{ responsive: true, displayModeBar: true }}
              style={{ width: '100%', height: '400px' }}
            />
          </div>
        )}

        {evolutionChart && (
          <div className="visualization-container">
            <Plot
              data={evolutionChart.data}
              layout={evolutionChart.layout}
              config={{ responsive: true, displayModeBar: true }}
              style={{ width: '100%', height: '400px' }}
            />
          </div>
        )}

        {errors.length > 0 && (
          <div className="errors-container">
            <h3>Shape Mismatches & Errors</h3>
            <ul className="error-list">
              {errors.map((error, index) => (
                <li key={index} className="error-item">
                  <span className="error-icon">⚠</span>
                  <div className="error-content">
                    <strong>{error.layer || 'Unknown Layer'}:</strong>
                    <p>{error.message}</p>
                    {error.expected_shape && (
                      <p className="error-hint">
                        Expected: {formatShape(error.expected_shape)}, Got: {formatShape(error.actual_shape)}
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {shapeHistory.length > 0 && (
          <div className="shape-table-container">
            <h3>Layer-by-Layer Shape Propagation</h3>
            <table className="shape-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Layer</th>
                  <th>Input Shape</th>
                  <th>Output Shape</th>
                  <th>Parameters</th>
                  <th>Memory</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {shapeHistory.map((item, index) => (
                  <tr
                    key={index}
                    className={`${item.error ? 'error-row' : ''} ${selectedLayer === index ? 'selected-row' : ''}`}
                    onClick={() => setSelectedLayer(index)}
                  >
                    <td>{index + 1}</td>
                    <td>{item.layer_name || item.layer}</td>
                    <td>{formatShape(item.input_shape)}</td>
                    <td>{formatShape(item.output_shape)}</td>
                    <td>{item.parameters ? formatNumber(item.parameters) : '-'}</td>
                    <td>{item.memory ? formatMemory(item.memory) : '-'}</td>
                    <td>
                      {item.error ? (
                        <span className="status-error">❌ Error</span>
                      ) : (
                        <span className="status-ok">✓ OK</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default ShapePropagationPlotly;
