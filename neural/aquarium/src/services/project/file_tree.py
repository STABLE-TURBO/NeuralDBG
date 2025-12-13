from __future__ import annotations
from typing import Optional, List, Callable
from pathlib import Path
from .file_node import FileNode, FileNodeType


class FileTree:
    def __init__(self, root_path: Optional[Path] = None):
        self.root: Optional[FileNode] = None
        self.selected_node: Optional[FileNode] = None
        self.on_selection_changed: Optional[Callable[[Optional[FileNode]], None]] = None
        
        if root_path:
            self.load_directory(root_path)
    
    def load_directory(self, path: Path, ignore_patterns: Optional[List[str]] = None) -> None:
        if ignore_patterns is None:
            ignore_patterns = [
                '__pycache__',
                '.git',
                '.venv',
                'venv',
                '.venv312',
                'node_modules',
                '.pytest_cache',
                '.mypy_cache',
                '*.pyc',
                '.DS_Store',
                'Thumbs.db',
            ]
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        
        self.root = self._build_tree(path, None, ignore_patterns)
        
    def _build_tree(
        self,
        path: Path,
        parent: Optional[FileNode],
        ignore_patterns: List[str]
    ) -> FileNode:
        node = FileNode(
            name=path.name if parent else path.as_posix(),
            path=path,
            node_type=FileNodeType.DIRECTORY if path.is_dir() else FileNodeType.FILE,
            parent=parent
        )
        
        if path.is_dir():
            try:
                for item in sorted(path.iterdir()):
                    if self._should_ignore(item, ignore_patterns):
                        continue
                    
                    child = self._build_tree(item, node, ignore_patterns)
                    node.add_child(child)
                    
                node.sort_children()
            except PermissionError:
                pass
        
        return node
    
    def _should_ignore(self, path: Path, ignore_patterns: List[str]) -> bool:
        name = path.name
        for pattern in ignore_patterns:
            if pattern.startswith('*'):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith('*'):
                if name.startswith(pattern[:-1]):
                    return True
            elif name == pattern:
                return True
        return False
    
    def refresh(self, ignore_patterns: Optional[List[str]] = None) -> None:
        if self.root:
            self.load_directory(self.root.path, ignore_patterns)
    
    def select_node(self, node: Optional[FileNode]) -> None:
        self.selected_node = node
        if self.on_selection_changed:
            self.on_selection_changed(node)
    
    def toggle_expand(self, node: FileNode) -> None:
        if node.node_type == FileNodeType.DIRECTORY:
            node.is_expanded = not node.is_expanded
    
    def find_node_by_path(self, path: Path) -> Optional[FileNode]:
        if not self.root:
            return None
        return self._find_node_recursive(self.root, path)
    
    def _find_node_recursive(self, node: FileNode, path: Path) -> Optional[FileNode]:
        if node.path == path:
            return node
        
        for child in node.children:
            result = self._find_node_recursive(child, path)
            if result:
                return result
        
        return None
    
    def get_all_neural_files(self) -> List[FileNode]:
        if not self.root:
            return []
        return self.root.get_all_neural_files()
    
    def add_file(self, parent_dir: FileNode, file_name: str) -> Optional[FileNode]:
        if parent_dir.node_type != FileNodeType.DIRECTORY:
            return None
        
        new_path = parent_dir.path / file_name
        new_node = FileNode(
            name=file_name,
            path=new_path,
            node_type=FileNodeType.FILE,
            parent=parent_dir
        )
        parent_dir.add_child(new_node)
        parent_dir.sort_children()
        return new_node
    
    def add_directory(self, parent_dir: FileNode, dir_name: str) -> Optional[FileNode]:
        if parent_dir.node_type != FileNodeType.DIRECTORY:
            return None
        
        new_path = parent_dir.path / dir_name
        new_node = FileNode(
            name=dir_name,
            path=new_path,
            node_type=FileNodeType.DIRECTORY,
            parent=parent_dir
        )
        parent_dir.add_child(new_node)
        parent_dir.sort_children()
        return new_node
    
    def remove_node(self, node: FileNode) -> bool:
        if node.parent:
            node.parent.remove_child(node)
            return True
        return False
    
    def rename_node(self, node: FileNode, new_name: str) -> bool:
        if not node.parent:
            return False
        
        new_path = node.path.parent / new_name
        node.name = new_name
        node.path = new_path
        node.is_neural_file = new_path.suffix == '.neural' if node.node_type == FileNodeType.FILE else False
        node.parent.sort_children()
        return True
