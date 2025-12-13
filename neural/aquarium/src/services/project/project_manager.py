from __future__ import annotations
from typing import Optional, List, Callable
from pathlib import Path
from .file_tree import FileTree
from .file_node import FileNode, FileNodeType
from .workspace_config import WorkspaceConfig
from .recent_projects import RecentProjectsManager
from .file_operations import FileOperations, FileOperationResult
from .tab_manager import TabManager, EditorTab
from .project_metadata import ProjectMetadata


class ProjectManager:
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.home() / ".aquarium"
        
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_project_path: Optional[Path] = None
        self.file_tree: Optional[FileTree] = None
        self.workspace_config: Optional[WorkspaceConfig] = None
        self.project_metadata: Optional[ProjectMetadata] = None
        
        self.recent_projects = RecentProjectsManager(config_dir)
        self.file_operations = FileOperations()
        self.tab_manager = TabManager()
        
        self.on_project_opened: Optional[Callable[[Path], None]] = None
        self.on_project_closed: Optional[Callable[[Optional[Path]], None]] = None
        
        self._setup_file_operation_callbacks()
        
    def _setup_file_operation_callbacks(self) -> None:
        self.file_operations.on_file_opened = self._on_file_opened
        self.file_operations.on_file_saved = self._on_file_saved
        self.file_operations.on_file_closed = self._on_file_closed
        self.file_operations.on_file_deleted = self._on_file_deleted
        self.file_operations.on_file_created = self._on_file_created
    
    def open_project(self, project_path: Path) -> bool:
        if not project_path.exists() or not project_path.is_dir():
            return False
        
        self.close_project()
        
        self.current_project_path = project_path
        self.file_tree = FileTree(project_path)
        self.workspace_config = WorkspaceConfig(project_path)
        self.project_metadata = ProjectMetadata(project_path)
        
        self.project_metadata.load()
        
        self.recent_projects.add_project(project_path)
        self.recent_projects.save()
        
        self._restore_open_files()
        
        if self.on_project_opened:
            self.on_project_opened(project_path)
        
        return True
    
    def close_project(self, save_state: bool = True) -> None:
        if not self.current_project_path:
            return
        
        if save_state and self.project_metadata:
            self._save_project_state()
            self.project_metadata.save()
        
        unsaved_tabs = self.tab_manager.get_modified_tabs()
        if unsaved_tabs:
            pass
        
        old_path = self.current_project_path
        
        self.tab_manager.close_all_tabs()
        self.current_project_path = None
        self.file_tree = None
        self.workspace_config = None
        self.project_metadata = None
        
        if self.on_project_closed:
            self.on_project_closed(old_path)
    
    def create_project(self, project_path: Path, project_name: Optional[str] = None) -> bool:
        if project_path.exists():
            return False
        
        try:
            project_path.mkdir(parents=True, exist_ok=False)
            
            example_file = project_path / "main.neural"
            with open(example_file, 'w', encoding='utf-8') as f:
                f.write(self._get_default_neural_content())
            
            metadata = ProjectMetadata(project_path)
            if project_name:
                metadata.data["name"] = project_name
            metadata.save()
            
            return self.open_project(project_path)
        except OSError:
            return False
    
    def _get_default_neural_content(self) -> str:
        return """model ExampleModel {
    input: [batch, 28, 28, 1]
    
    layer conv1: Conv2D {
        filters: 32
        kernel_size: 3
        activation: "relu"
    }
    
    layer pool1: MaxPool2D {
        pool_size: 2
    }
    
    layer flatten: Flatten {}
    
    layer dense1: Dense {
        units: 128
        activation: "relu"
    }
    
    layer output: Dense {
        units: 10
        activation: "softmax"
    }
}
"""
    
    def is_project_open(self) -> bool:
        return self.current_project_path is not None
    
    def get_current_project_path(self) -> Optional[Path]:
        return self.current_project_path
    
    def get_project_name(self) -> Optional[str]:
        if self.project_metadata:
            return self.project_metadata.data.get("name")
        return None
    
    def new_file(self, filename: str, directory: Optional[Path] = None) -> FileOperationResult:
        if not self.current_project_path:
            return FileOperationResult(False, "No project is open")
        
        if directory is None:
            directory = self.current_project_path
        
        return self.file_operations.new_file(directory, filename)
    
    def open_file(self, file_path: Path) -> FileOperationResult:
        result = self.file_operations.open_file(file_path)
        
        if result.success:
            success, content = self.file_operations.read_file(file_path)
            if success:
                self.tab_manager.open_tab(file_path, content, activate=True)
        
        return result
    
    def save_file(self, tab: Optional[EditorTab] = None) -> FileOperationResult:
        if tab is None:
            tab = self.tab_manager.get_active_tab()
        
        if not tab:
            return FileOperationResult(False, "No active file to save")
        
        result = self.file_operations.save_file(tab.file_path, tab.content)
        
        if result.success:
            self.tab_manager.mark_tab_as_saved(tab)
        
        return result
    
    def save_file_as(self, new_path: Path, tab: Optional[EditorTab] = None) -> FileOperationResult:
        if tab is None:
            tab = self.tab_manager.get_active_tab()
        
        if not tab:
            return FileOperationResult(False, "No active file to save")
        
        result = self.file_operations.save_as(tab.file_path, new_path, tab.content)
        
        if result.success:
            self.tab_manager.rename_tab(tab, new_path)
            self.tab_manager.mark_tab_as_saved(tab)
        
        return result
    
    def close_file(self, tab: Optional[EditorTab] = None) -> bool:
        if tab is None:
            tab = self.tab_manager.get_active_tab()
        
        if not tab:
            return False
        
        if tab.is_modified:
            pass
        
        self.file_operations.close_file(tab.file_path)
        return self.tab_manager.close_tab(tab)
    
    def delete_file(self, file_path: Path) -> FileOperationResult:
        self.tab_manager.close_tab_by_path(file_path)
        
        result = self.file_operations.delete_file(file_path)
        
        if result.success and self.file_tree:
            node = self.file_tree.find_node_by_path(file_path)
            if node:
                self.file_tree.remove_node(node)
        
        return result
    
    def rename_file(self, file_path: Path, new_name: str) -> FileOperationResult:
        result = self.file_operations.rename_file(file_path, new_name)
        
        if result.success and result.path:
            tab = self.tab_manager.find_tab_by_path(file_path)
            if tab:
                self.tab_manager.rename_tab(tab, result.path)
            
            if self.file_tree:
                node = self.file_tree.find_node_by_path(file_path)
                if node:
                    self.file_tree.rename_node(node, new_name)
        
        return result
    
    def get_all_neural_files(self) -> List[Path]:
        if not self.file_tree:
            return []
        
        nodes = self.file_tree.get_all_neural_files()
        return [node.path for node in nodes]
    
    def refresh_file_tree(self) -> None:
        if self.file_tree:
            self.file_tree.refresh()
    
    def get_recent_projects(self, count: int = 10) -> list:
        return self.recent_projects.get_recent(count)
    
    def _save_project_state(self) -> None:
        if not self.project_metadata:
            return
        
        open_files = [str(tab.file_path) for tab in self.tab_manager.get_all_tabs()]
        self.project_metadata.set_open_files(open_files)
        
        active_tab = self.tab_manager.get_active_tab()
        if active_tab:
            self.project_metadata.set_active_file(str(active_tab.file_path))
        else:
            self.project_metadata.set_active_file(None)
    
    def _restore_open_files(self) -> None:
        if not self.project_metadata:
            return
        
        open_files = self.project_metadata.get_open_files()
        active_file = self.project_metadata.get_active_file()
        
        for file_path_str in open_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                success, content = self.file_operations.read_file(file_path)
                if success:
                    activate = (str(file_path) == active_file)
                    self.tab_manager.open_tab(file_path, content, activate=activate)
    
    def _on_file_opened(self, file_path: Path) -> None:
        pass
    
    def _on_file_saved(self, file_path: Path) -> None:
        pass
    
    def _on_file_closed(self, file_path: Path) -> None:
        pass
    
    def _on_file_deleted(self, file_path: Path) -> None:
        pass
    
    def _on_file_created(self, file_path: Path) -> None:
        if self.file_tree and self.current_project_path:
            if file_path.is_relative_to(self.current_project_path):
                self.refresh_file_tree()
