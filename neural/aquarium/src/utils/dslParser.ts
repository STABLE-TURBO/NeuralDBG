import { Node, Edge } from 'reactflow';
import { LayerNodeData } from '../types';
import { getLayerDefinition } from '../data/layerDefinitions';

export function parseDSLToNodes(dslCode: string): { nodes: Node<LayerNodeData>[]; edges: Edge[] } {
  const nodes: Node<LayerNodeData>[] = [];
  const edges: Edge[] = [];

  const lines = dslCode.split('\n').map(l => l.trim()).filter(l => l);
  
  let inputShape = '(None, 28, 28, 1)';
  let layerIndex = 0;
  let yPosition = 100;
  
  for (const line of lines) {
    if (line.startsWith('input:')) {
      inputShape = line.replace('input:', '').trim();
      
      nodes.push({
        id: 'input',
        type: 'layerNode',
        position: { x: 400, y: yPosition },
        data: {
          label: 'Input',
          layerType: 'Input',
          category: 'Core',
          parameters: { shape: inputShape },
          outputShape: inputShape
        }
      });
      yPosition += 120;
    } 
    else if (line.includes('(') && !line.startsWith('network') && !line.startsWith('loss') && !line.startsWith('optimizer')) {
      const match = line.match(/(\w+)\(([^)]*)\)/);
      if (match) {
        const layerType = match[1];
        const paramsStr = match[2];
        
        const params: Record<string, any> = {};
        if (paramsStr) {
          const paramPairs = paramsStr.split(',').map(p => p.trim());
          for (const pair of paramPairs) {
            const [key, value] = pair.split('=').map(s => s.trim());
            if (key && value) {
              let parsedValue: any = value.replace(/"/g, '');
              if (parsedValue === 'true') parsedValue = true;
              else if (parsedValue === 'false') parsedValue = false;
              else if (parsedValue === 'null') parsedValue = null;
              else if (!isNaN(Number(parsedValue))) parsedValue = Number(parsedValue);
              params[key] = parsedValue;
            }
          }
        }
        
        const layerDef = getLayerDefinition(layerType);
        const nodeId = `layer-${layerIndex}`;
        
        nodes.push({
          id: nodeId,
          type: 'layerNode',
          position: { x: 400, y: yPosition },
          data: {
            label: layerType,
            layerType: layerType,
            category: layerDef?.category || 'Core',
            parameters: params
          }
        });
        
        const sourceId = layerIndex === 0 ? 'input' : `layer-${layerIndex - 1}`;
        edges.push({
          id: `edge-${sourceId}-${nodeId}`,
          source: sourceId,
          target: nodeId,
          type: 'smoothstep'
        });
        
        layerIndex++;
        yPosition += 120;
      }
    }
  }

  return { nodes, edges };
}

export function nodesToDSL(nodes: Node<LayerNodeData>[], edges: Edge[]): string {
  const sortedNodes = topologicalSort(nodes, edges);
  
  const inputNode = sortedNodes.find(n => n.data.layerType === 'Input');
  const layerNodes = sortedNodes.filter(n => n.data.layerType !== 'Input');
  
  let dsl = 'network MyModel {\n';
  
  if (inputNode) {
    dsl += `    input: ${inputNode.data.parameters.shape || '(None, 28, 28, 1)'}\n`;
  }
  
  dsl += '    layers:\n';
  
  for (const node of layerNodes) {
    const params = node.data.parameters;
    const paramStrs: string[] = [];
    
    for (const [key, value] of Object.entries(params)) {
      if (value !== null && value !== undefined) {
        if (typeof value === 'string') {
          paramStrs.push(`${key}="${value}"`);
        } else if (Array.isArray(value)) {
          paramStrs.push(`${key}=(${value.join(', ')})`);
        } else {
          paramStrs.push(`${key}=${value}`);
        }
      }
    }
    
    const paramStr = paramStrs.length > 0 ? paramStrs.join(', ') : '';
    dsl += `        ${node.data.layerType}(${paramStr})\n`;
  }
  
  dsl += '    loss: "categorical_crossentropy"\n';
  dsl += '    optimizer: "Adam"\n';
  dsl += '}\n';
  
  return dsl;
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
