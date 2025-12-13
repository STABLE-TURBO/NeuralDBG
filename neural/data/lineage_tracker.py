from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class LineageNode:
    """
    Represents a node in the lineage graph.
    
    Attributes:
        node_id: Unique identifier
        node_type: properties type (data, model, etc.)
        name: Human-readable name
        metadata: Additional metadata
    """
    def __init__(
        self,
        node_id: str,
        node_type: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineageNode:
        obj = cls.__new__(cls)
        obj.node_id = data["node_id"]
        obj.node_type = data["node_type"]
        obj.name = data["name"]
        obj.metadata = data.get("metadata", {})
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        return obj


class LineageEdge:
    """
    Represents an edge in the lineage graph.
    
    Attributes:
        source_id: Node ID of source
        target_id: Node ID of target
        edge_type: Type of relationship
        metadata: Additional metadata
    """
    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineageEdge:
        obj = cls.__new__(cls)
        obj.source_id = data["source_id"]
        obj.target_id = data["target_id"]
        obj.edge_type = data["edge_type"]
        obj.metadata = data.get("metadata", {})
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        return obj


class LineageGraph:
    """
    Represents a full lineage graph.
    
    Attributes:
        name: Graph name
        nodes: Dictionary of nodes
        edges: List of edges
    """
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
        self.created_at = datetime.now().isoformat()

    def add_node(self, node: LineageNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: LineageEdge):
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("Source or target node not found")
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[LineageNode]:
        return self.nodes.get(node_id)

    def get_successors(self, node_id: str) -> List[LineageNode]:
        successors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                if edge.target_id in self.nodes:
                    successors.append(self.nodes[edge.target_id])
        return successors

    def get_predecessors(self, node_id: str) -> List[LineageNode]:
        predecessors = []
        for edge in self.edges:
            if edge.target_id == node_id:
                if edge.source_id in self.nodes:
                    predecessors.append(self.nodes[edge.source_id])
        return predecessors

    def get_path(self, start_id: str, end_id: str) -> Optional[List[LineageNode]]:
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        visited: Set[str] = set()
        queue: List[tuple[str, List[str]]] = [(start_id, [start_id])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == end_id:
                return [self.nodes[nid] for nid in path]
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for successor in self.get_successors(current_id):
                if successor.node_id not in visited:
                    queue.append((successor.node_id, path + [successor.node_id]))
        
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LineageGraph:
        obj = cls.__new__(cls)
        obj.name = data["name"]
        obj.nodes = {
            k: LineageNode.from_dict(v) for k, v in data.get("nodes", {}).items()
        }
        obj.edges = [LineageEdge.from_dict(e) for e in data.get("edges", [])]
        obj.created_at = data.get("created_at", datetime.now().isoformat())
        return obj


class LineageTracker:
    """
    Tracks data and model lineage.
    
    Manages creation and persistence of lineage graphs.
    """
    def __init__(self, base_dir: Union[str, Path] = ".neural_data"):
        self.base_dir = Path(base_dir)
        self.lineage_dir = self.base_dir / "lineage"
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        self.graphs: Dict[str, LineageGraph] = {}
        self._load_graphs()

    def _load_graphs(self):
        for graph_file in self.lineage_dir.glob("*.json"):
            with open(graph_file, "r") as f:
                data = json.load(f)
                graph = LineageGraph.from_dict(data)
                self.graphs[graph.name] = graph

    def _save_graph(self, graph: LineageGraph):
        graph_file = self.lineage_dir / f"{graph.name}.json"
        with open(graph_file, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)

    def create_graph(self, name: str) -> LineageGraph:
        if name in self.graphs:
            raise ValueError(f"Graph already exists: {name}")
        
        graph = LineageGraph(name)
        self.graphs[name] = graph
        self._save_graph(graph)
        return graph

    def get_graph(self, name: str) -> Optional[LineageGraph]:
        return self.graphs.get(name)

    def list_graphs(self) -> List[str]:
        return list(self.graphs.keys())

    def delete_graph(self, name: str) -> bool:
        if name not in self.graphs:
            return False
        
        graph_file = self.lineage_dir / f"{name}.json"
        if graph_file.exists():
            graph_file.unlink()
        
        del self.graphs[name]
        return True

    def add_data_node(
        self,
        graph_name: str,
        node_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        graph = self.get_graph(graph_name)
        if not graph:
            graph = self.create_graph(graph_name)
        
        node = LineageNode(node_id, "data", name, metadata)
        graph.add_node(node)
        self._save_graph(graph)
        return node

    def add_model_node(
        self,
        graph_name: str,
        node_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        graph = self.get_graph(graph_name)
        if not graph:
            graph = self.create_graph(graph_name)
        
        node = LineageNode(node_id, "model", name, metadata)
        graph.add_node(node)
        self._save_graph(graph)
        return node

    def add_prediction_node(
        self,
        graph_name: str,
        node_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        graph = self.get_graph(graph_name)
        if not graph:
            graph = self.create_graph(graph_name)
        
        node = LineageNode(node_id, "prediction", name, metadata)
        graph.add_node(node)
        self._save_graph(graph)
        return node

    def add_preprocessing_node(
        self,
        graph_name: str,
        node_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        graph = self.get_graph(graph_name)
        if not graph:
            graph = self.create_graph(graph_name)
        
        node = LineageNode(node_id, "preprocessing", name, metadata)
        graph.add_node(node)
        self._save_graph(graph)
        return node

    def add_edge(
        self,
        graph_name: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageEdge:
        graph = self.get_graph(graph_name)
        if not graph:
            raise ValueError(f"Graph not found: {graph_name}")
        
        edge = LineageEdge(source_id, target_id, edge_type, metadata)
        graph.add_edge(edge)
        self._save_graph(graph)
        return edge

    def trace_lineage(
        self, graph_name: str, node_id: str, direction: str = "forward"
    ) -> List[LineageNode]:
        graph = self.get_graph(graph_name)
        if not graph:
            return []
        
        visited: Set[str] = set()
        result: List[LineageNode] = []
        queue: List[str] = [node_id]
        
        while queue:
            current_id = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            node = graph.get_node(current_id)
            if node:
                result.append(node)
            
            if direction == "forward":
                successors = graph.get_successors(current_id)
                queue.extend(s.node_id for s in successors)
            else:
                predecessors = graph.get_predecessors(current_id)
                queue.extend(p.node_id for p in predecessors)
        
        return result

    def get_full_lineage(
        self, graph_name: str, node_id: str
    ) -> Dict[str, List[LineageNode]]:
        forward = self.trace_lineage(graph_name, node_id, "forward")
        backward = self.trace_lineage(graph_name, node_id, "backward")
        
        return {
            "upstream": backward,
            "downstream": forward,
        }

    def visualize_lineage(
        self, graph_name: str, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        graph = self.get_graph(graph_name)
        if not graph:
            raise ValueError(f"Graph not found: {graph_name}")
        
        try:
            import graphviz
            
            dot = graphviz.Digraph(comment=graph.name)
            dot.attr(rankdir='LR')
            
            node_colors = {
                "data": "lightblue",
                "preprocessing": "lightyellow",
                "model": "lightgreen",
                "prediction": "lightcoral",
            }
            
            for node in graph.nodes.values():
                color = node_colors.get(node.node_type, "lightgray")
                dot.node(node.node_id, node.name, fillcolor=color, style="filled")
            
            for edge in graph.edges:
                dot.edge(edge.source_id, edge.target_id, label=edge.edge_type)
            
            if output_path:
                output_path = Path(output_path)
                dot.render(output_path.stem, directory=output_path.parent, format='png', cleanup=True)
                return str(output_path.with_suffix('.png'))
            else:
                default_output = self.lineage_dir / f"{graph_name}_lineage"
                dot.render(default_output, format='png', cleanup=True)
                return str(default_output.with_suffix('.png'))
        
        except ImportError:
            dot_content = f"digraph {graph.name} {{\n"
            dot_content += "  rankdir=LR;\n"
            
            for node in graph.nodes.values():
                dot_content += f'  "{node.node_id}" [label="{node.name}"];\n'
            
            for edge in graph.edges:
                dot_content += f'  "{edge.source_id}" -> "{edge.target_id}" [label="{edge.edge_type}"];\n'
            
            dot_content += "}\n"
            
            if output_path:
                output_path = Path(output_path).with_suffix('.dot')
            else:
                output_path = self.lineage_dir / f"{graph_name}_lineage.dot"
            
            with open(output_path, "w") as f:
                f.write(dot_content)
            
            return str(output_path)

    def export_graph(self, name: str, output_path: Union[str, Path]) -> bool:
        graph = self.get_graph(name)
        if not graph:
            return False
        
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        
        return True

    def import_graph(self, input_path: Union[str, Path]) -> LineageGraph:
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Graph file not found: {input_path}")
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        graph = LineageGraph.from_dict(data)
        self.graphs[graph.name] = graph
        self._save_graph(graph)
        
        return graph
