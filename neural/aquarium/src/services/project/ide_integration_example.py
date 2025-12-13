"""
Example of how the project management system would integrate with an IDE frontend.

This demonstrates the complete flow of typical IDE operations using the project
management system as the backend.
"""

from pathlib import Path
from typing import Optional, List, Callable
from .project_manager import ProjectManager
from .file_node import FileNode
from .tab_manager import EditorTab


class IDEController:
    """
    Example IDE controller that uses the project management system.
    This would typically connect to a GUI framework (Qt, Electron, etc.)
    """
    
    def __init__(self):
        self.pm = ProjectManager()
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        self.pm.on_project_opened = self._on_project_opened
        self.pm.on_project_closed = self._on_project_closed
        
        if self.pm.file_tree:
            self.pm.file_tree.on_selection_changed = self._on_tree_selection_changed
        
        self.pm.tab_manager.on_tab_opened = self._on_tab_opened
        self.pm.tab_manager.on_tab_closed = self._on_tab_closed
        self.pm.tab_manager.on_tab_activated = self._on_tab_activated
        self.pm.tab_manager.on_tab_modified = self._on_tab_modified
        
        self.pm.file_operations.on_file_created = self._on_file_created
        self.pm.file_operations.on_file_deleted = self._on_file_deleted
    
    def create_new_project(self, path: Path, name: str) -> bool:
        """File > New Project"""
        return self.pm.create_project(path, name)
    
    def open_project(self, path: Path) -> bool:
        """File > Open Project"""
        return self.pm.open_project(path)
    
    def close_project(self) -> None:
        """File > Close Project"""
        if self.pm.tab_manager.has_unsaved_changes():
            if self._prompt_save_changes():
                modified_tabs = self.pm.tab_manager.get_modified_tabs()
                for tab in modified_tabs:
                    self.pm.save_file(tab)
        
        self.pm.close_project(save_state=True)
    
    def new_file(self, filename: Optional[str] = None) -> bool:
        """File > New File"""
        if not filename:
            filename = self._prompt_filename("New File", default="untitled.neural")
        
        if not filename:
            return False
        
        result = self.pm.new_file(filename)
        if result.success and result.path:
            self.pm.open_file(result.path)
            return True
        else:
            self._show_error(f"Failed to create file: {result.message}")
            return False
    
    def open_file(self, path: Optional[Path] = None) -> bool:
        """File > Open File"""
        if not path:
            path = self._prompt_file_selection("Open File")
        
        if not path:
            return False
        
        result = self.pm.open_file(path)
        if not result.success:
            self._show_error(f"Failed to open file: {result.message}")
            return False
        
        return True
    
    def save_file(self, tab: Optional[EditorTab] = None) -> bool:
        """File > Save"""
        result = self.pm.save_file(tab)
        if not result.success:
            self._show_error(f"Failed to save file: {result.message}")
            return False
        
        return True
    
    def save_file_as(self) -> bool:
        """File > Save As"""
        new_path = self._prompt_file_selection("Save As", save_mode=True)
        if not new_path:
            return False
        
        result = self.pm.save_file_as(new_path)
        if not result.success:
            self._show_error(f"Failed to save file: {result.message}")
            return False
        
        return True
    
    def save_all_files(self) -> None:
        """File > Save All"""
        modified_tabs = self.pm.tab_manager.get_modified_tabs()
        for tab in modified_tabs:
            self.save_file(tab)
    
    def close_file(self, tab: Optional[EditorTab] = None) -> bool:
        """File > Close"""
        if tab is None:
            tab = self.pm.tab_manager.get_active_tab()
        
        if not tab:
            return False
        
        if tab.is_modified:
            response = self._prompt_save_before_close(tab.title)
            if response == "save":
                self.save_file(tab)
            elif response == "cancel":
                return False
        
        return self.pm.close_file(tab)
    
    def close_all_files(self) -> None:
        """File > Close All"""
        self.pm.tab_manager.close_all_tabs()
    
    def tree_node_clicked(self, node: FileNode) -> None:
        """User clicked a node in the file tree"""
        if node.node_type.value == "file":
            self.open_file(node.path)
        else:
            if self.pm.file_tree:
                self.pm.file_tree.toggle_expand(node)
                self._update_tree_view()
    
    def tree_node_double_clicked(self, node: FileNode) -> None:
        """User double-clicked a node in the file tree"""
        if node.node_type.value == "file":
            self.open_file(node.path)
    
    def tree_node_context_menu(self, node: FileNode) -> List[str]:
        """Right-click context menu for tree nodes"""
        if node.node_type.value == "file":
            return [
                "Open",
                "Rename",
                "Delete",
                "Copy Path",
                "---",
                "Add Bookmark",
                "Add Breakpoint"
            ]
        else:
            return [
                "New File",
                "New Folder",
                "---",
                "Rename",
                "Delete",
                "---",
                "Refresh"
            ]
    
    def execute_tree_context_action(self, node: FileNode, action: str) -> None:
        """Execute a context menu action"""
        if action == "Open":
            self.open_file(node.path)
        elif action == "Rename":
            new_name = self._prompt_filename("Rename", default=node.name)
            if new_name:
                self.pm.rename_file(node.path, new_name)
        elif action == "Delete":
            if self._confirm_delete(node.name):
                self.pm.delete_file(node.path)
        elif action == "New File":
            filename = self._prompt_filename("New File", default="untitled.neural")
            if filename:
                result = self.pm.file_operations.new_file(node.path, filename)
                if result.success:
                    self.pm.refresh_file_tree()
        elif action == "Refresh":
            self.pm.refresh_file_tree()
    
    def tab_clicked(self, index: int) -> None:
        """User clicked a tab"""
        self.pm.tab_manager.activate_tab_by_index(index)
    
    def tab_close_clicked(self, tab: EditorTab) -> None:
        """User clicked the close button on a tab"""
        self.close_file(tab)
    
    def editor_content_changed(self, content: str) -> None:
        """Editor content was modified by user"""
        tab = self.pm.tab_manager.get_active_tab()
        if tab:
            self.pm.tab_manager.update_tab_content(tab, content)
    
    def get_recent_projects(self) -> List:
        """Get recent projects for File > Recent Projects menu"""
        return self.pm.get_recent_projects(10)
    
    def open_recent_project(self, index: int) -> None:
        """Open a project from recent projects list"""
        recent = self.pm.get_recent_projects(10)
        if 0 <= index < len(recent):
            self.open_project(recent[index].path)
    
    def show_project_settings(self) -> None:
        """Edit > Project Settings"""
        if not self.pm.workspace_config:
            self._show_error("No project is open")
            return
        
        settings = self._prompt_project_settings_dialog()
        if settings:
            for key, value in settings.items():
                if key == "backend":
                    self.pm.workspace_config.set_compiler_backend(value)
    
    def show_preferences(self) -> None:
        """Edit > Preferences"""
        if not self.pm.project_metadata:
            self._show_error("No project is open")
            return
        
        preferences = self._prompt_preferences_dialog()
        if preferences:
            for section, values in preferences.items():
                for key, value in values.items():
                    self.pm.project_metadata.set_setting(section, key, value=value)
            self.pm.project_metadata.save()
    
    def find_in_files(self, query: str) -> None:
        """Search > Find in Files"""
        if self.pm.project_metadata:
            self.pm.project_metadata.add_recent_search(query)
            self.pm.project_metadata.save()
        
        results = self._search_in_project(query)
        self._show_search_results(results)
    
    def toggle_bookmark(self) -> None:
        """Toggle bookmark at current cursor position"""
        tab = self.pm.tab_manager.get_active_tab()
        if not tab or not self.pm.project_metadata:
            return
        
        line, column = tab.cursor_position
        
        existing = self.pm.project_metadata.get_bookmarks(str(tab.file_path))
        is_bookmarked = any(b["line"] == line for b in existing)
        
        if is_bookmarked:
            self.pm.project_metadata.remove_bookmark(str(tab.file_path), line)
        else:
            description = self._prompt_bookmark_description()
            self.pm.project_metadata.add_bookmark(
                str(tab.file_path),
                line,
                description
            )
        
        self.pm.project_metadata.save()
        self._update_gutter()
    
    def go_to_next_tab(self) -> None:
        """Keyboard shortcut: Ctrl+Tab"""
        self.pm.tab_manager.activate_next_tab()
    
    def go_to_previous_tab(self) -> None:
        """Keyboard shortcut: Ctrl+Shift+Tab"""
        self.pm.tab_manager.activate_previous_tab()
    
    def _on_project_opened(self, path: Path) -> None:
        """Callback: Project opened"""
        print(f"[IDE] Project opened: {path}")
        self._update_tree_view()
        self._update_title_bar()
        self._update_status_bar(f"Project opened: {path.name}")
    
    def _on_project_closed(self, path: Optional[Path]) -> None:
        """Callback: Project closed"""
        print(f"[IDE] Project closed")
        self._clear_tree_view()
        self._clear_tabs()
        self._update_title_bar()
        self._update_status_bar("No project open")
    
    def _on_tree_selection_changed(self, node: Optional[FileNode]) -> None:
        """Callback: Tree selection changed"""
        if node:
            self._update_status_bar(f"Selected: {node.path}")
    
    def _on_tab_opened(self, tab: EditorTab) -> None:
        """Callback: Tab opened"""
        print(f"[IDE] Tab opened: {tab.title}")
        self._add_tab_to_ui(tab)
        self._update_status_bar(f"Opened: {tab.title}")
    
    def _on_tab_closed(self, tab: EditorTab) -> None:
        """Callback: Tab closed"""
        print(f"[IDE] Tab closed: {tab.title}")
        self._remove_tab_from_ui(tab)
    
    def _on_tab_activated(self, tab: EditorTab) -> None:
        """Callback: Tab activated"""
        print(f"[IDE] Tab activated: {tab.title}")
        self._switch_editor_to_tab(tab)
        self._update_status_bar(f"Editing: {tab.title}")
    
    def _on_tab_modified(self, tab: EditorTab) -> None:
        """Callback: Tab content modified"""
        self._update_tab_ui(tab)
    
    def _on_file_created(self, path: Path) -> None:
        """Callback: File created"""
        print(f"[IDE] File created: {path}")
        self.pm.refresh_file_tree()
    
    def _on_file_deleted(self, path: Path) -> None:
        """Callback: File deleted"""
        print(f"[IDE] File deleted: {path}")
    
    def _prompt_save_changes(self) -> bool:
        """Mock: Prompt user to save changes"""
        print("[UI] Prompt: Save unsaved changes?")
        return True
    
    def _prompt_filename(self, title: str, default: str = "") -> Optional[str]:
        """Mock: Prompt for filename"""
        print(f"[UI] Prompt: {title} (default: {default})")
        return default
    
    def _prompt_file_selection(self, title: str, save_mode: bool = False) -> Optional[Path]:
        """Mock: File selection dialog"""
        print(f"[UI] File Dialog: {title}")
        return None
    
    def _prompt_save_before_close(self, filename: str) -> str:
        """Mock: Save before close dialog"""
        print(f"[UI] Prompt: Save {filename} before closing?")
        return "save"
    
    def _show_error(self, message: str) -> None:
        """Mock: Show error dialog"""
        print(f"[UI] Error: {message}")
    
    def _confirm_delete(self, name: str) -> bool:
        """Mock: Confirm delete dialog"""
        print(f"[UI] Confirm: Delete {name}?")
        return True
    
    def _prompt_project_settings_dialog(self) -> Optional[dict]:
        """Mock: Project settings dialog"""
        print("[UI] Project Settings Dialog")
        return None
    
    def _prompt_preferences_dialog(self) -> Optional[dict]:
        """Mock: Preferences dialog"""
        print("[UI] Preferences Dialog")
        return None
    
    def _prompt_bookmark_description(self) -> str:
        """Mock: Bookmark description prompt"""
        print("[UI] Bookmark Description Prompt")
        return ""
    
    def _search_in_project(self, query: str) -> List:
        """Mock: Search in project files"""
        print(f"[Search] Searching for: {query}")
        return []
    
    def _show_search_results(self, results: List) -> None:
        """Mock: Show search results panel"""
        print(f"[UI] Search Results: {len(results)} matches")
    
    def _update_tree_view(self) -> None:
        """Mock: Update file tree UI"""
        print("[UI] Update tree view")
    
    def _clear_tree_view(self) -> None:
        """Mock: Clear file tree UI"""
        print("[UI] Clear tree view")
    
    def _clear_tabs(self) -> None:
        """Mock: Clear all tabs"""
        print("[UI] Clear tabs")
    
    def _update_title_bar(self) -> None:
        """Mock: Update window title"""
        project_name = self.pm.get_project_name() or "Aquarium IDE"
        print(f"[UI] Title: {project_name}")
    
    def _update_status_bar(self, message: str) -> None:
        """Mock: Update status bar"""
        print(f"[UI] Status: {message}")
    
    def _add_tab_to_ui(self, tab: EditorTab) -> None:
        """Mock: Add tab to UI"""
        print(f"[UI] Add tab: {tab.title}")
    
    def _remove_tab_from_ui(self, tab: EditorTab) -> None:
        """Mock: Remove tab from UI"""
        print(f"[UI] Remove tab: {tab.title}")
    
    def _switch_editor_to_tab(self, tab: EditorTab) -> None:
        """Mock: Switch editor content to tab"""
        print(f"[UI] Switch to tab: {tab.title}")
    
    def _update_tab_ui(self, tab: EditorTab) -> None:
        """Mock: Update tab appearance"""
        title = tab.get_display_title()
        print(f"[UI] Update tab: {title}")
    
    def _update_gutter(self) -> None:
        """Mock: Update editor gutter (bookmarks, breakpoints)"""
        print("[UI] Update gutter")


def demo_ide_workflow():
    """Demonstrate a complete IDE workflow"""
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        print("=" * 60)
        print("Aquarium IDE Integration Demo")
        print("=" * 60)
        
        # Initialize IDE
        ide = IDEController()
        
        # Create project
        print("\n1. Creating new project...")
        project_path = temp_dir / "demo_project"
        ide.create_new_project(project_path, "Demo Project")
        
        # Create files
        print("\n2. Creating files...")
        ide.new_file("cnn_model.neural")
        ide.new_file("rnn_model.neural")
        
        # Edit content
        print("\n3. Editing content...")
        tab = ide.pm.tab_manager.get_active_tab()
        if tab:
            ide.editor_content_changed("model RNN { }")
        
        # Save
        print("\n4. Saving...")
        ide.save_file()
        
        # Navigate tabs
        print("\n5. Navigating tabs...")
        ide.go_to_previous_tab()
        
        # Add bookmark
        print("\n6. Adding bookmark...")
        ide.toggle_bookmark()
        
        # Configure
        print("\n7. Showing settings...")
        ide.show_project_settings()
        
        # Close
        print("\n8. Closing project...")
        ide.close_project()
        
        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    demo_ide_workflow()
