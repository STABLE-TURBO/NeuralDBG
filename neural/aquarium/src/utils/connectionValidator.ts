import { Node, Edge, Connection } from 'reactflow';
import { LayerNodeData, ConnectionValidationResult } from '../types';

export function validateConnection(
  connection: Connection,
  nodes: Node<LayerNodeData>[],
  edges: Edge[]
): ConnectionValidationResult {
  if (!connection.source || !connection.target) {
    return { valid: false, message: 'Invalid connection' };
  }

  const sourceNode = nodes.find(n => n.id === connection.source);
  const targetNode = nodes.find(n => n.id === connection.target);

  if (!sourceNode || !targetNode) {
    return { valid: false, message: 'Node not found' };
  }

  if (wouldCreateCycle(connection, edges)) {
    return { valid: false, message: 'Cannot create cycles' };
  }

  const existingEdge = edges.find(
    e => e.source === connection.source && e.target === connection.target
  );
  if (existingEdge) {
    return { valid: false, message: 'Connection already exists' };
  }

  const targetHasInput = edges.some(e => e.target === connection.target);
  if (targetHasInput && targetNode.data.layerType !== 'Concatenate' && targetNode.data.layerType !== 'Add') {
    return { valid: false, message: 'Target already has an input connection' };
  }

  const compatibilityCheck = checkLayerCompatibility(sourceNode, targetNode);
  if (!compatibilityCheck.valid) {
    return compatibilityCheck;
  }

  return { valid: true };
}

function wouldCreateCycle(newConnection: Connection, edges: Edge[]): boolean {
  const adjacencyList = new Map<string, Set<string>>();
  
  edges.forEach(edge => {
    if (!adjacencyList.has(edge.source)) {
      adjacencyList.set(edge.source, new Set());
    }
    adjacencyList.get(edge.source)!.add(edge.target);
  });

  if (!adjacencyList.has(newConnection.source!)) {
    adjacencyList.set(newConnection.source!, new Set());
  }
  adjacencyList.get(newConnection.source!)!.add(newConnection.target!);

  const visited = new Set<string>();
  const recStack = new Set<string>();

  function hasCycle(node: string): boolean {
    if (!visited.has(node)) {
      visited.add(node);
      recStack.add(node);

      const neighbors = adjacencyList.get(node);
      if (neighbors) {
        for (const neighbor of neighbors) {
          if (!visited.has(neighbor) && hasCycle(neighbor)) {
            return true;
          } else if (recStack.has(neighbor)) {
            return true;
          }
        }
      }
    }
    recStack.delete(node);
    return false;
  }

  return hasCycle(newConnection.source!);
}

function checkLayerCompatibility(
  sourceNode: Node<LayerNodeData>,
  targetNode: Node<LayerNodeData>
): ConnectionValidationResult {
  const sourceType = sourceNode.data.layerType;
  const targetType = targetNode.data.layerType;

  if (sourceType === 'Input') {
    return { valid: true };
  }

  const flatteningLayers = ['Flatten', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D', 
                             'GlobalMaxPooling1D', 'GlobalAveragePooling1D'];
  const requiresSequence = ['LSTM', 'GRU', 'SimpleRNN'];
  const requires2D = ['Conv2D', 'MaxPooling2D', 'AveragePooling2D'];

  if (flatteningLayers.includes(sourceType) && requires2D.includes(targetType)) {
    return {
      valid: false,
      message: `Cannot connect ${sourceType} to ${targetType}: output is flattened`
    };
  }

  if (sourceType === 'Embedding' && requires2D.includes(targetType)) {
    return {
      valid: false,
      message: `Cannot connect Embedding to ${targetType}: incompatible dimensions`
    };
  }

  if (requiresSequence.includes(targetType) && 
      !targetNode.data.parameters.return_sequences && 
      requiresSequence.includes(sourceType)) {
    return {
      valid: false,
      message: `Source ${sourceType} must have return_sequences=True`
    };
  }

  return { valid: true };
}

export function propagateShapes(
  nodes: Node<LayerNodeData>[],
  edges: Edge[]
): Node<LayerNodeData>[] {
  const updatedNodes = [...nodes];
  const nodeMap = new Map(updatedNodes.map(n => [n.id, n]));
  
  const sorted = topologicalSort(updatedNodes, edges);
  
  for (const node of sorted) {
    if (node.data.layerType === 'Input') {
      nodeMap.get(node.id)!.data.outputShape = node.data.parameters.shape;
      continue;
    }

    const incomingEdges = edges.filter(e => e.target === node.id);
    if (incomingEdges.length === 0) continue;

    const sourceNode = nodeMap.get(incomingEdges[0].source);
    if (!sourceNode || !sourceNode.data.outputShape) continue;

    const inputShape = sourceNode.data.outputShape;
    const outputShape = calculateOutputShape(node.data.layerType, node.data.parameters, inputShape);
    
    nodeMap.get(node.id)!.data.outputShape = outputShape;
  }

  return Array.from(nodeMap.values());
}

function calculateOutputShape(layerType: string, params: Record<string, any>, inputShape: string): string {
  const parseShape = (shape: string): (number | null)[] => {
    const match = shape.match(/\(([^)]+)\)/);
    if (!match) return [null];
    return match[1].split(',').map(s => {
      const trimmed = s.trim();
      return trimmed === 'None' ? null : parseInt(trimmed);
    });
  };

  const dims = parseShape(inputShape);
  
  switch (layerType) {
    case 'Dense':
      return `(None, ${params.units || 128})`;
    
    case 'Flatten':
      if (dims.length > 2) {
        const product = dims.slice(1).reduce((acc: number, val) => acc * (val || 1), 1);
        return `(None, ${product})`;
      }
      return inputShape;
    
    case 'Conv2D':
      return inputShape;
    
    case 'MaxPooling2D':
    case 'AveragePooling2D':
      if (dims.length === 4) {
        const [batch, h, w, c] = dims;
        const poolSize = params.pool_size || [2, 2];
        const newH = h ? Math.floor(h / poolSize[0]) : null;
        const newW = w ? Math.floor(w / poolSize[1]) : null;
        return `(None, ${newH}, ${newW}, ${c})`;
      }
      return inputShape;
    
    case 'GlobalMaxPooling2D':
    case 'GlobalAveragePooling2D':
      if (dims.length === 4) {
        return `(None, ${dims[3]})`;
      }
      return inputShape;
    
    case 'Dropout':
    case 'BatchNormalization':
    case 'LayerNormalization':
    case 'ReLU':
    case 'Softmax':
      return inputShape;
    
    case 'LSTM':
    case 'GRU':
    case 'SimpleRNN':
      if (params.return_sequences) {
        return `(None, ${dims[1]}, ${params.units || 128})`;
      }
      return `(None, ${params.units || 128})`;
    
    case 'Embedding':
      return `(None, ${dims[1]}, ${params.output_dim || 64})`;
    
    case 'MultiHeadAttention':
      return inputShape;
    
    default:
      return inputShape;
  }
}

function topologicalSort(nodes: Node<LayerNodeData>[], edges: Edge[]): Node<LayerNodeData>[] {
  const adjacencyList = new Map<string, string[]>();
  const inDegree = new Map<string, number>();
  
  nodes.forEach(node => {
    adjacencyList.set(node.id, []);
    inDegree.set(node.id, 0);
  });
  
  edges.forEach(edge => {
    adjacencyList.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
  });
  
  const queue: string[] = [];
  nodes.forEach(node => {
    if (inDegree.get(node.id) === 0) {
      queue.push(node.id);
    }
  });
  
  const sorted: Node<LayerNodeData>[] = [];
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    const node = nodeMap.get(nodeId);
    if (node) {
      sorted.push(node);
    }
    
    adjacencyList.get(nodeId)?.forEach(neighbor => {
      const degree = inDegree.get(neighbor)! - 1;
      inDegree.set(neighbor, degree);
      if (degree === 0) {
        queue.push(neighbor);
      }
    });
  }
  
  return sorted;
}
