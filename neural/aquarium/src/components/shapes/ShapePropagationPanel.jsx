import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './ShapePropagationPanel.css';

const ShapePropagationPanel = ({ modelData, shapePropagator }) => {
  const [shapeHistory, setShapeHistory] = useState([]);
  const [errors, setErrors] = useState([]);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const svgRef = useRef(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(1000);

  // Fetch shape propagation data
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

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchShapeData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  // Initial fetch
  useEffect(() => {
    fetchShapeData();
  }, []);

  // D3 visualization effect
  useEffect(() => {
    if (shapeHistory.length > 0 && svgRef.current) {
      renderShapeFlow();
    }
  }, [shapeHistory, selectedLayer]);

  const renderShapeFlow = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 900;
    const height = 600;
    const margin = { top: 40, right: 40, bottom: 40, left: 40 };

    svg.attr('width', width).attr('height', height);

    // Create nodes from shape history
    const nodes = shapeHistory.map((item, index) => ({
      id: index,
      layer: item.layer_name || item.layer,
      inputShape: item.input_shape,
      outputShape: item.output_shape,
      x: margin.left + (index * (width - margin.left - margin.right) / Math.max(shapeHistory.length - 1, 1)),
      y: height / 2,
      hasError: item.error || false,
      errorMessage: item.error_message || null,
      parameters: item.parameters || null,
      flops: item.flops || 0,
      memory: item.memory || 0
    }));

    // Create links
    const links = [];
    for (let i = 0; i < nodes.length - 1; i++) {
      links.push({
        source: nodes[i],
        target: nodes[i + 1],
        shapeMismatch: checkShapeMismatch(nodes[i].outputShape, nodes[i + 1].inputShape)
      });
    }

    // Draw links
    const link = svg.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', d => `link ${d.shapeMismatch ? 'error' : ''}`)
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
      .attr('stroke', d => d.shapeMismatch ? '#e74c3c' : '#3498db')
      .attr('stroke-width', 2)
      .attr('marker-end', d => d.shapeMismatch ? 'url(#arrow-error)' : 'url(#arrow)');

    // Define arrow markers
    const defs = svg.append('defs');
    
    defs.append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#3498db');

    defs.append('marker')
      .attr('id', 'arrow-error')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#e74c3c');

    // Create node groups
    const node = svg.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', d => `node ${d.hasError ? 'error' : ''} ${selectedLayer === d.id ? 'selected' : ''}`)
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .on('click', (event, d) => setSelectedLayer(d.id))
      .on('mouseover', showTooltip)
      .on('mouseout', hideTooltip);

    // Draw node circles
    node.append('circle')
      .attr('r', 30)
      .attr('fill', d => d.hasError ? '#e74c3c' : (selectedLayer === d.id ? '#2ecc71' : '#3498db'))
      .attr('stroke', '#2c3e50')
      .attr('stroke-width', 2);

    // Add layer names
    node.append('text')
      .attr('dy', -40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#2c3e50')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text(d => d.layer);

    // Add input shape text
    node.append('text')
      .attr('dy', 50)
      .attr('text-anchor', 'middle')
      .attr('fill', '#7f8c8d')
      .attr('font-size', '10px')
      .text(d => `In: ${formatShape(d.inputShape)}`);

    // Add output shape text
    node.append('text')
      .attr('dy', 65)
      .attr('text-anchor', 'middle')
      .attr('fill', '#7f8c8d')
      .attr('font-size', '10px')
      .text(d => `Out: ${formatShape(d.outputShape)}`);

    // Add error indicator
    node.filter(d => d.hasError)
      .append('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '20px')
      .text('⚠');

    // Add title for accessibility
    node.append('title')
      .text(d => `${d.layer}\nInput: ${formatShape(d.inputShape)}\nOutput: ${formatShape(d.outputShape)}`);
  };

  const showTooltip = (event, d) => {
    const tooltip = d3.select('body').append('div')
      .attr('class', 'shape-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(44, 62, 80, 0.95)')
      .style('color', 'white')
      .style('padding', '12px')
      .style('border-radius', '6px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('z-index', '1000')
      .style('box-shadow', '0 4px 6px rgba(0,0,0,0.3)');

    let content = `<strong>${d.layer}</strong><br/>`;
    content += `<strong>Input Shape:</strong> ${formatShape(d.inputShape)}<br/>`;
    content += `<strong>Output Shape:</strong> ${formatShape(d.outputShape)}<br/>`;
    
    if (d.parameters) {
      content += `<strong>Parameters:</strong> ${formatNumber(d.parameters)}<br/>`;
    }
    
    if (d.flops > 0) {
      content += `<strong>FLOPs:</strong> ${formatNumber(d.flops)}<br/>`;
    }
    
    if (d.memory > 0) {
      content += `<strong>Memory:</strong> ${formatMemory(d.memory)}<br/>`;
    }
    
    if (d.hasError && d.errorMessage) {
      content += `<strong style="color: #e74c3c;">Error:</strong> ${d.errorMessage}`;
    }

    tooltip.html(content)
      .style('left', (event.pageX + 15) + 'px')
      .style('top', (event.pageY - 28) + 'px');

    // Store tooltip for removal
    d3.select(event.currentTarget).attr('data-tooltip', 'active');
  };

  const hideTooltip = (event) => {
    d3.selectAll('.shape-tooltip').remove();
  };

  const checkShapeMismatch = (outputShape, inputShape) => {
    if (!outputShape || !inputShape) return false;
    
    // Handle different shape formats
    const out = Array.isArray(outputShape) ? outputShape : outputShape.split(',').map(s => parseInt(s.trim()));
    const inp = Array.isArray(inputShape) ? inputShape : inputShape.split(',').map(s => parseInt(s.trim()));
    
    // Compare shapes (ignoring batch dimension - first element)
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

  const getSelectedLayerDetails = () => {
    if (selectedLayer === null || !shapeHistory[selectedLayer]) return null;
    return shapeHistory[selectedLayer];
  };

  const selectedDetails = getSelectedLayerDetails();

  return (
    <div className="shape-propagation-panel">
      <div className="panel-header">
        <h2>Real-Time Shape Propagation</h2>
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
        <div className="visualization-container">
          <svg ref={svgRef} className="shape-flow-svg"></svg>
        </div>

        {selectedDetails && (
          <div className="layer-details">
            <h3>Layer Details: {selectedDetails.layer_name || selectedDetails.layer}</h3>
            <div className="details-grid">
              <div className="detail-item">
                <strong>Input Shape:</strong>
                <span>{formatShape(selectedDetails.input_shape)}</span>
              </div>
              <div className="detail-item">
                <strong>Output Shape:</strong>
                <span>{formatShape(selectedDetails.output_shape)}</span>
              </div>
              {selectedDetails.parameters && (
                <div className="detail-item">
                  <strong>Parameters:</strong>
                  <span>{formatNumber(selectedDetails.parameters)}</span>
                </div>
              )}
              {selectedDetails.flops > 0 && (
                <div className="detail-item">
                  <strong>FLOPs:</strong>
                  <span>{formatNumber(selectedDetails.flops)}</span>
                </div>
              )}
              {selectedDetails.memory > 0 && (
                <div className="detail-item">
                  <strong>Memory Usage:</strong>
                  <span>{formatMemory(selectedDetails.memory)}</span>
                </div>
              )}
              {selectedDetails.transformation && (
                <div className="detail-item full-width">
                  <strong>Transformation:</strong>
                  <pre>{JSON.stringify(selectedDetails.transformation, null, 2)}</pre>
                </div>
              )}
            </div>
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

export default ShapePropagationPanel;
