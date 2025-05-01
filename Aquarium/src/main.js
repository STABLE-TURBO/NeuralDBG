const { invoke } = window.__TAURI__.core;

// Model state
let networkLayers = [];
let selectedLayer = null;

// UI elements
let navButtons;
let views;
let layerButtons;
let canvas;
let propertiesContent;
let shapeTable;
let codeEditor;

// Initialize the application
window.addEventListener("DOMContentLoaded", () => {
  // Get UI elements
  navButtons = document.querySelectorAll('.nav-button');
  views = document.querySelectorAll('.view');
  layerButtons = document.querySelectorAll('.layer-button');
  canvas = document.querySelector('.canvas');
  propertiesContent = document.querySelector('.properties-content');
  shapeTable = document.querySelector('#shape-tbody');
  codeEditor = document.querySelector('#code-editor');

  // Set up navigation
  navButtons.forEach(button => {
    button.addEventListener('click', () => {
      const viewId = button.dataset.view;

      // Update active button
      navButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      // Show selected view
      views.forEach(view => {
        view.classList.remove('active');
        if (view.id === `${viewId}-view`) {
          view.classList.add('active');
        }
      });

      // Update content based on view
      if (viewId === 'shape') {
        updateShapePropagator();
      } else if (viewId === 'code') {
        updateCodeEditor();
      }
    });
  });

  // Set up layer buttons
  layerButtons.forEach(button => {
    button.addEventListener('click', () => {
      addLayer(button.dataset.layer);
    });
  });

  // Greet the user
  invoke("greet", { name: "Neural Network Designer" }).then((message) => {
    console.log(message);
  });
});

// Add a new layer to the network
function addLayer(layerType) {
  const layer = {
    id: `layer-${Date.now()}`,
    type: layerType,
    name: getLayerName(layerType),
    params: getDefaultParams(layerType)
  };

  networkLayers.push(layer);
  renderLayers();
  selectLayer(layer.id);
  updateShapePropagator();
  updateCodeEditor();
}

// Get a human-readable name for a layer type
function getLayerName(layerType) {
  const names = {
    'input': 'Input Layer',
    'conv2d': 'Conv2D',
    'maxpool': 'MaxPooling2D',
    'flatten': 'Flatten',
    'dense': 'Dense',
    'output': 'Output'
  };
  return names[layerType] || layerType;
}

// Get default parameters for a layer type
function getDefaultParams(layerType) {
  switch (layerType) {
    case 'input':
      return { shape: [28, 28, 1] };
    case 'conv2d':
      return { filters: 32, kernel_size: [3, 3], padding: 'same', activation: 'relu' };
    case 'maxpool':
      return { pool_size: [2, 2] };
    case 'flatten':
      return {};
    case 'dense':
      return { units: 128, activation: 'relu' };
    case 'output':
      return { units: 10, activation: 'softmax' };
    default:
      return {};
  }
}

// Render all layers in the canvas
function renderLayers() {
  // Clear the canvas
  canvas.innerHTML = '';

  if (networkLayers.length === 0) {
    canvas.innerHTML = '<div class="placeholder">Drag and drop layers here to build your neural network</div>';
    return;
  }

  // Create a layer element for each layer
  networkLayers.forEach((layer, index) => {
    const layerEl = document.createElement('div');
    layerEl.className = `network-layer ${layer.type} ${selectedLayer && layer.id === selectedLayer.id ? 'selected' : ''}`;
    layerEl.dataset.id = layer.id;
    layerEl.style.top = `${50 + index * 80}px`;
    layerEl.style.left = `${100}px`;

    layerEl.innerHTML = `
      <div class="layer-header">${layer.name}</div>
      <div class="layer-body">${getLayerDescription(layer)}</div>
    `;

    layerEl.addEventListener('click', () => {
      selectLayer(layer.id);
    });

    canvas.appendChild(layerEl);

    // Add connection line if not the first layer
    if (index > 0) {
      const connectionEl = document.createElement('div');
      connectionEl.className = 'connection';
      connectionEl.style.top = `${90 + (index - 1) * 80}px`;
      connectionEl.style.left = `${100}px`;
      connectionEl.style.height = `${80}px`;
      canvas.appendChild(connectionEl);
    }
  });
}

// Get a short description of a layer
function getLayerDescription(layer) {
  switch (layer.type) {
    case 'input':
      return `Shape: ${JSON.stringify(layer.params.shape)}`;
    case 'conv2d':
      return `${layer.params.filters} filters, ${layer.params.kernel_size.join('×')}`;
    case 'maxpool':
      return `Pool size: ${layer.params.pool_size.join('×')}`;
    case 'flatten':
      return 'Flatten dimensions';
    case 'dense':
      return `${layer.params.units} units`;
    case 'output':
      return `${layer.params.units} units`;
    default:
      return '';
  }
}

// Select a layer and show its properties
function selectLayer(layerId) {
  selectedLayer = networkLayers.find(layer => layer.id === layerId);
  renderLayers();
  renderProperties();
}

// Render the properties panel for the selected layer
function renderProperties() {
  if (!selectedLayer) {
    propertiesContent.innerHTML = '<p>Select a layer to view and edit its properties</p>';
    return;
  }

  let html = `<h4>${selectedLayer.name} Properties</h4>`;

  // Create form elements for each parameter
  Object.entries(selectedLayer.params).forEach(([key, value]) => {
    html += `<div class="property">
      <label>${key}:</label>
      ${renderPropertyInput(key, value)}
    </div>`;
  });

  propertiesContent.innerHTML = html;

  // Add event listeners to inputs
  document.querySelectorAll('.property-input').forEach(input => {
    input.addEventListener('change', (e) => {
      updateLayerProperty(e.target.dataset.property, e.target.value);
    });
  });
}

// Render an input element for a property
function renderPropertyInput(key, value) {
  if (Array.isArray(value)) {
    return `<div class="array-input">
      ${value.map((val, idx) => `
        <input type="number" class="property-input" data-property="${key}[${idx}]" value="${val}" />
      `).join('')}
    </div>`;
  } else if (typeof value === 'number') {
    return `<input type="number" class="property-input" data-property="${key}" value="${value}" />`;
  } else {
    return `<input type="text" class="property-input" data-property="${key}" value="${value}" />`;
  }
}

// Update a layer property
function updateLayerProperty(property, value) {
  if (!selectedLayer) return;

  // Handle array properties
  if (property.includes('[')) {
    const [key, indexStr] = property.split('[');
    const index = parseInt(indexStr);
    selectedLayer.params[key][index] = parseInt(value);
  } else {
    // Handle simple properties
    if (typeof selectedLayer.params[property] === 'number') {
      selectedLayer.params[property] = parseInt(value);
    } else {
      selectedLayer.params[property] = value;
    }
  }

  renderLayers();
  updateShapePropagator();
  updateCodeEditor();
}

// Update the shape propagator view
function updateShapePropagator() {
  if (networkLayers.length === 0) {
    shapeTable.innerHTML = '<tr><td colspan="4">No layers added yet</td></tr>';
    return;
  }

  // Clear the table
  shapeTable.innerHTML = '';

  // Calculate shapes (simplified mock implementation)
  let currentShape = null;

  networkLayers.forEach(layer => {
    let inputShape = currentShape;
    let outputShape;
    let params = 0;

    if (layer.type === 'input') {
      outputShape = [null, ...layer.params.shape];
      params = 0;
    } else if (layer.type === 'conv2d') {
      const [_, h, w, c] = inputShape;
      outputShape = [null, h, w, layer.params.filters];
      params = layer.params.kernel_size[0] * layer.params.kernel_size[1] * c * layer.params.filters + layer.params.filters;
    } else if (layer.type === 'maxpool') {
      const [_, h, w, c] = inputShape;
      outputShape = [null, Math.floor(h/2), Math.floor(w/2), c];
      params = 0;
    } else if (layer.type === 'flatten') {
      const [_, h, w, c] = inputShape;
      outputShape = [null, h * w * c];
      params = 0;
    } else if (layer.type === 'dense' || layer.type === 'output') {
      const [_, features] = inputShape;
      outputShape = [null, layer.params.units];
      params = features * layer.params.units + layer.params.units;
    }

    currentShape = outputShape;

    // Add row to table
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${layer.name}</td>
      <td>${inputShape ? JSON.stringify(inputShape) : '-'}</td>
      <td>${JSON.stringify(outputShape)}</td>
      <td>${params.toLocaleString()}</td>
    `;
    shapeTable.appendChild(row);
  });
}

// Update the code editor with Neural DSL code
function updateCodeEditor() {
  if (networkLayers.length === 0) {
    codeEditor.value = '# Add layers to generate Neural DSL code';
    return;
  }

  let code = '# Neural DSL Model\n\n';

  networkLayers.forEach(layer => {
    code += generateLayerCode(layer) + '\n';
  });

  codeEditor.value = code;
}

// Generate Neural DSL code for a layer
function generateLayerCode(layer) {
  switch (layer.type) {
    case 'input':
      return `Input(shape=${JSON.stringify(layer.params.shape)})`;
    case 'conv2d':
      return `Conv2D(filters=${layer.params.filters}, kernel_size=${JSON.stringify(layer.params.kernel_size)}, padding="${layer.params.padding}", activation="${layer.params.activation}")`;
    case 'maxpool':
      return `MaxPooling2D(pool_size=${JSON.stringify(layer.params.pool_size)})`;
    case 'flatten':
      return 'Flatten()';
    case 'dense':
      return `Dense(units=${layer.params.units}, activation="${layer.params.activation}")`;
    case 'output':
      return `Output(units=${layer.params.units}, activation="${layer.params.activation}")`;
    default:
      return `# Unknown layer type: ${layer.type}`;
  }
}
