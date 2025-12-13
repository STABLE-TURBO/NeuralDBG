from __future__ import annotations
from typing import Optional, List
from pathlib import Path
from enum import Enum


class FileNodeType(Enum):
    FILE = "file"
    DIRECTORY = "directory"


class FileNode:
    def __init__(
        self,
        name: str,
        path: Path,
        node_type: FileNodeType,
        parent: Optional[FileNode] = None
    ):
        self.name = name
        self.path = path
        self.node_type = node_type
        self.parent = parent
        self.children: List[FileNode] = []
        self.is_expanded = False
        self.is_neural_file = path.suffix == '.neural' if node_type == FileNodeType.FILE else False
        
    def add_child(self, child: FileNode) -> None:
        child.parent = self
        self.children.append(child)
        
    def remove_child(self, child: FileNode) -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            
    def get_depth(self) -> int:
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    def get_all_neural_files(self) -> List[FileNode]:
        neural_files = []
        if self.is_neural_file:
            neural_files.append(self)
        for child in self.children:
            neural_files.extend(child.get_all_neural_files())
        return neural_files
    
    def sort_children(self) -> None:
        self.children.sort(key=lambda x: (x.node_type == FileNodeType.FILE, x.name.lower()))
        
    def __repr__(self) -> str:
        return f"FileNode(name={self.name}, type={self.node_type.value})"
