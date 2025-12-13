# Neural Aquarium - Developer Guide

## Architecture Overview

Neural Aquarium is a visual network designer built with React, TypeScript, and ReactFlow. It provides a drag-and-drop interface for building neural networks with real-time DSL code generation.

## Project Structure

```
neural/aquarium/
├── src/
│   ├── components/
│   │   └── designer/
│   │       ├── NetworkDesigner.tsx    # Main container component
│   │       ├── LayerNode.tsx          # Custom ReactFlow node
│   │       ├── LayerPalette.tsx       # Layer selection sidebar
│   │       ├── PropertiesPanel.tsx    # Parameter editor
│   │       ├── CodeEditor.tsx         # Monaco DSL editor
│   │       └── Toolbar.tsx            # Action buttons
│   ├── data/
│   │   └── layerDefinitions.ts        # Layer metadata
│   ├── types/
│   │   └── index.ts                   # TypeScript definitions
│   ├── utils/
│   │   ├── dslParser.ts              # DSL <-> Graph conversion
│   │   ├── connectionValidator.ts    # Connection rules
│   │   └── fileHandlers.ts           # Import/export utilities
│   ├── App.tsx                        # Root component
│   ├── main.tsx                       # Entry point
│   └── index.css                      # Global styles
├── examples/                          # Sample .neural files
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## Key Components

### NetworkDesigner

Main orchestrator component managing:
- ReactFlow state (nodes, edges)
- Selected node tracking
- DSL code synchronization
- User interactions (add, delete, update layers)

**State Management:**
```typescript
const [nodes, setNodes, onNodesChange] = useNodesState([]);
const [edges, setEdges, onEdgesChange] = useEdgesState([]);
const [selectedNode, setSelectedNode] = useState<Node<LayerNodeData> | null>(null);
const [dslCode, setDslCode] = useState('');
```

**Key Methods:**
- `onConnect`: Validates and creates connections
- `onDrop`: Handles drag-drop from palette
- `onLayerSelect`: Adds layer via click
- `onUpdateNode`: Updates node parameters
- `onCodeChange`: Parses DSL to visual

### LayerNode

Custom ReactFlow node component displaying:
- Layer type with icon and color
- Top 3 parameters
- Output shape
- Parameter count (optional)

**Props:**
```typescript
interface NodeProps<LayerNodeData> {
  data: LayerNodeData;
  selected: boolean;
}
```

### LayerPalette

Sidebar with categorized layer list:
- Category tabs (Convolutional, Pooling, etc.)
- Search filter
- Drag-and-drop support
- Click-to-add support

### PropertiesPanel

Parameter editor for selected node:
- Layer info (category, output shape)
- Editable parameters
- Type-specific inputs (number, boolean, select)
- Layer description

### CodeEditor

Monaco-based DSL editor:
- Syntax highlighting
- Real-time parsing
- Copy to clipboard
- Bi-directional sync with visual

## Data Flow

### Visual → DSL

1. User adds/edits layers in visual designer
2. `nodes` and `edges` state updates
3. `useEffect` triggers `updateDSLCode()`
4. `nodesToDSL()` converts graph to DSL
5. DSL displayed in code editor

### DSL → Visual

1. User edits code in Monaco editor
2. `onChange` calls `onCodeChange()`
3. `parseDSLToNodes()` parses DSL
4. Creates new nodes and edges
5. `propagateShapes()` calculates outputs
6. Visual canvas updates

## Connection Validation

Implemented in `connectionValidator.ts`:

**Rules:**
1. No cycles allowed
2. Single input per layer (except merge layers)
3. Shape compatibility
4. Layer-specific constraints

**Validation Function:**
```typescript
export function validateConnection(
  connection: Connection,
  nodes: Node<LayerNodeData>[],
  edges: Edge[]
): ConnectionValidationResult
```

**Returns:**
```typescript
{
  valid: boolean;
  message?: string;
}
```

## Shape Propagation

Automatic output shape calculation:

1. Start from Input node
2. Topological sort of graph
3. Calculate output for each layer:
   - Dense: `(None, units)`
   - Conv2D: Maintains spatial dims
   - Flatten: Collapses to 1D
   - Pooling: Reduces spatial dims
   - etc.

**Implementation:**
```typescript
export function propagateShapes(
  nodes: Node<LayerNodeData>[],
  edges: Edge[]
): Node<LayerNodeData>[]
```

## DSL Parser

### Parsing (DSL → Graph)

```typescript
export function parseDSLToNodes(dslCode: string): {
  nodes: Node<LayerNodeData>[];
  edges: Edge[];
}
```

**Algorithm:**
1. Split by lines
2. Extract input shape
3. Parse layer definitions
4. Create nodes with positions
5. Create edges in sequence

### Generation (Graph → DSL)

```typescript
export function nodesToDSL(
  nodes: Node<LayerNodeData>[],
  edges: Edge[]
): string
```

**Algorithm:**
1. Topological sort nodes
2. Format input shape
3. Generate layer lines with params
4. Add loss and optimizer
5. Wrap in `network {}` block

## Adding New Layer Types

1. **Add to `layerDefinitions.ts`:**
```typescript
MyNewLayer: {
  type: 'MyNewLayer',
  category: 'Core',
  defaultParams: { param1: 128, param2: 'relu' },
  color: '#95E1D3',
  icon: '●',
  description: 'My new layer description'
}
```

2. **Update shape propagation in `connectionValidator.ts`:**
```typescript
case 'MyNewLayer':
  return `(None, ${params.output_dim})`;
```

3. **Add validation rules (if needed):**
```typescript
if (targetType === 'MyNewLayer' && !someCondition) {
  return { valid: false, message: 'Error message' };
}
```

## Styling

### Theme
- Background: `#1a1a1a`
- Primary: `#00bfff`
- Panels: `#282828`
- Cards: `#2a2a2a`
- Borders: `#333`, `#444`

### Layer Colors
- Convolutional: `#FF6B6B`
- Pooling: `#4ECDC4`
- Core: `#95E1D3`
- Recurrent: `#FCBAD3`
- Attention: `#FFD93D`
- Normalization: `#F38181`
- Regularization: `#AA96DA`
- Activation: `#A8D8EA`

## Testing Locally

```bash
npm run dev
```

Open http://localhost:3000

**Test Cases:**
1. Drag layer from palette
2. Connect two layers
3. Edit layer parameters
4. Switch to code view
5. Edit DSL code
6. Validate invalid connections
7. Export/import .neural file
8. Auto layout
9. Clear canvas

## Performance Considerations

### React Flow Optimization
- Use `useNodesState` and `useEdgesState` hooks
- Memoize callbacks with `useCallback`
- Lazy update shape propagation with `setTimeout`

### Monaco Editor
- Lazy load with `@monaco-editor/react`
- Disable minimap for performance
- Use `onChange` debouncing (if needed)

### Large Networks
- ReactFlow handles 1000+ nodes well
- MiniMap optional for very large graphs
- Consider virtualization for 10,000+ nodes

## Common Issues

### 1. Connection not allowed
**Cause:** Validation rule prevents connection
**Fix:** Check `connectionValidator.ts` rules

### 2. Shape propagation incorrect
**Cause:** Missing or incorrect shape calculation
**Fix:** Update `calculateOutputShape()` in `connectionValidator.ts`

### 3. DSL parsing error
**Cause:** Unexpected DSL format
**Fix:** Update regex in `parseDSLToNodes()`

### 4. Node not rendering
**Cause:** Missing node type registration
**Fix:** Add to `nodeTypes` object in `NetworkDesigner.tsx`

## Future Enhancements

- [ ] Undo/redo functionality
- [ ] Multi-input/output nodes (Add, Concatenate)
- [ ] Node grouping/subgraphs
- [ ] Real-time shape inference from backend
- [ ] Integration with NeuralDbg dashboard
- [ ] Model templates gallery
- [ ] Collaborative editing
- [ ] Custom layer creation UI
- [ ] Parameter constraints validation
- [ ] Training configuration editor
- [ ] Export to other formats (ONNX, etc.)

## API Integration

To connect with Python backend:

```typescript
// In NetworkDesigner.tsx
const compileModel = async () => {
  const response = await fetch('http://localhost:8051/api/compile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dsl: dslCode })
  });
  const result = await response.json();
  // Handle result
};
```

## Contributing

1. Follow existing code style
2. Add TypeScript types for new features
3. Update this guide for significant changes
4. Test all features before committing
5. Add examples for new layer types
