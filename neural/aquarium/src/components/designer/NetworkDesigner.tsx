import React, { useState, useCallback, useEffect, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
  NodeTypes,
  OnConnect,
} from 'reactflow';
import 'reactflow/dist/style.css';

import LayerNode from './LayerNode';
import LayerPalette from './LayerPalette';
import PropertiesPanel from './PropertiesPanel';
import CodeEditor from './CodeEditor';
import Toolbar from './Toolbar';
import { LayerNodeData, LayerCategory } from '../../types';
import { getLayerDefinition } from '../../data/layerDefinitions';
import { validateConnection, propagateShapes } from '../../utils/connectionValidator';
import { parseDSLToNodes, nodesToDSL } from '../../utils/dslParser';
import { downloadDSLFile, readDSLFile } from '../../utils/fileHandlers';
import './NetworkDesigner.css';

const nodeTypes: NodeTypes = {
  layerNode: LayerNode,
};

const NetworkDesigner: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node<LayerNodeData> | null>(null);
  const [dslCode, setDslCode] = useState('');
  const [showCodeEditor, setShowCodeEditor] = useState(false);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
  const nodeIdCounter = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const initialNodes: Node<LayerNodeData>[] = [
      {
        id: 'input',
        type: 'layerNode',
        position: { x: 400, y: 50 },
        data: {
          label: 'Input',
          layerType: 'Input',
          category: 'Core',
          parameters: { shape: '(None, 28, 28, 1)' },
          outputShape: '(None, 28, 28, 1)',
        },
      },
    ];
    setNodes(initialNodes);
    updateDSLCode(initialNodes, []);
  }, []);

  useEffect(() => {
    updateDSLCode(nodes, edges);
  }, [nodes, edges]);

  const updateDSLCode = (currentNodes: Node<LayerNodeData>[], currentEdges: Edge[]) => {
    if (currentNodes.length > 0) {
      const code = nodesToDSL(currentNodes, currentEdges);
      setDslCode(code);
    }
  };

  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      const validation = validateConnection(connection, nodes, edges);
      if (validation.valid) {
        setEdges((eds) => addEdge({ ...connection, type: 'smoothstep' }, eds));
        
        setTimeout(() => {
          const updatedNodes = propagateShapes(nodes, [...edges, connection as Edge]);
          setNodes(updatedNodes);
        }, 0);
      } else {
        alert(validation.message || 'Invalid connection');
      }
    },
    [nodes, edges, setEdges, setNodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      const data = event.dataTransfer.getData('application/reactflow');
      
      if (!data || !reactFlowBounds || !reactFlowInstance) return;

      const { layerType, category } = JSON.parse(data);
      const layerDef = getLayerDefinition(layerType);

      if (!layerDef) return;

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node<LayerNodeData> = {
        id: `layer-${nodeIdCounter.current++}`,
        type: 'layerNode',
        position,
        data: {
          label: layerType,
          layerType: layerType,
          category: category,
          parameters: { ...layerDef.defaultParams },
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const onLayerSelect = useCallback(
    (layerType: string, category: LayerCategory) => {
      const layerDef = getLayerDefinition(layerType);
      if (!layerDef) return;

      const newNode: Node<LayerNodeData> = {
        id: `layer-${nodeIdCounter.current++}`,
        type: 'layerNode',
        position: { 
          x: Math.random() * 300 + 200, 
          y: Math.random() * 300 + 200 
        },
        data: {
          label: layerType,
          layerType: layerType,
          category: category,
          parameters: { ...layerDef.defaultParams },
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes]
  );

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node as Node<LayerNodeData>);
    },
    []
  );

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const onUpdateNode = useCallback(
    (nodeId: string, updates: Partial<LayerNodeData>) => {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                ...updates,
              },
            };
          }
          return node;
        })
      );

      setTimeout(() => {
        const updatedNodes = propagateShapes(nodes, edges);
        setNodes(updatedNodes);
      }, 0);
    },
    [nodes, edges, setNodes]
  );

  const onCodeChange = useCallback(
    (code: string) => {
      setDslCode(code);
      try {
        const { nodes: newNodes, edges: newEdges } = parseDSLToNodes(code);
        setNodes(newNodes);
        setEdges(newEdges);
        
        setTimeout(() => {
          const updatedNodes = propagateShapes(newNodes, newEdges);
          setNodes(updatedNodes);
        }, 0);
      } catch (error) {
        console.error('Error parsing DSL:', error);
      }
    },
    [setNodes, setEdges]
  );

  const clearCanvas = useCallback(() => {
    if (window.confirm('Clear all layers? This cannot be undone.')) {
      const inputNode: Node<LayerNodeData> = {
        id: 'input',
        type: 'layerNode',
        position: { x: 400, y: 50 },
        data: {
          label: 'Input',
          layerType: 'Input',
          category: 'Core',
          parameters: { shape: '(None, 28, 28, 1)' },
          outputShape: '(None, 28, 28, 1)',
        },
      };
      setNodes([inputNode]);
      setEdges([]);
      setSelectedNode(null);
    }
  }, [setNodes, setEdges]);

  const autoLayout = useCallback(() => {
    const sorted = [...nodes].sort((a, b) => {
      const aIncoming = edges.filter(e => e.target === a.id).length;
      const bIncoming = edges.filter(e => e.target === b.id).length;
      return aIncoming - bIncoming;
    });

    const layoutNodes = sorted.map((node, index) => ({
      ...node,
      position: { x: 400, y: 50 + index * 150 },
    }));

    setNodes(layoutNodes);
  }, [nodes, edges, setNodes]);

  const handleExport = useCallback(() => {
    const timestamp = new Date().toISOString().split('T')[0];
    downloadDSLFile(dslCode, `model-${timestamp}.neural`);
  }, [dslCode]);

  const handleImport = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        try {
          const content = await readDSLFile(file);
          onCodeChange(content);
          if (event.target) {
            event.target.value = '';
          }
        } catch (error) {
          alert('Failed to read file');
        }
      }
    },
    [onCodeChange]
  );

  return (
    <div className="network-designer">
      <input
        ref={fileInputRef}
        type="file"
        accept=".neural"
        style={{ display: 'none' }}
        onChange={handleFileSelect}
      />
      
      <LayerPalette onLayerSelect={onLayerSelect} />

      <div className="designer-main">
        <Toolbar
          onToggleCodeEditor={() => setShowCodeEditor(!showCodeEditor)}
          onAutoLayout={autoLayout}
          onClear={clearCanvas}
          onExport={handleExport}
          onImport={handleImport}
          showCodeEditor={showCodeEditor}
          nodeCount={nodes.length}
          edgeCount={edges.length}
        />

        {showCodeEditor ? (
          <CodeEditor code={dslCode} onChange={onCodeChange} />
        ) : (
          <div className="react-flow-wrapper" ref={reactFlowWrapper}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onInit={setReactFlowInstance}
              nodeTypes={nodeTypes}
              fitView
              attributionPosition="bottom-left"
            >
              <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#333" />
              <Controls />
              <MiniMap
                nodeColor={(node) => {
                  const layerDef = getLayerDefinition(node.data.layerType);
                  return layerDef?.color || '#95E1D3';
                }}
                maskColor="rgba(0, 0, 0, 0.7)"
              />
            </ReactFlow>
          </div>
        )}
      </div>

      <PropertiesPanel selectedNode={selectedNode} onUpdateNode={onUpdateNode} />
    </div>
  );
};

export default NetworkDesigner;
