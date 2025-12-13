from __future__ import annotations
from typing import List, Optional, Dict, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EditorTab:
    file_path: Path
    title: str
    content: str = ""
    is_modified: bool = False
    is_active: bool = False
    cursor_position: tuple[int, int] = (0, 0)
    scroll_position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_display_title(self) -> str:
        prefix = "● " if self.is_modified else ""
        return f"{prefix}{self.title}"


class TabManager:
    def __init__(self):
        self.tabs: List[EditorTab] = []
        self.active_tab_index: int = -1
        self.on_tab_opened: Optional[Callable[[EditorTab], None]] = None
        self.on_tab_closed: Optional[Callable[[EditorTab], None]] = None
        self.on_tab_activated: Optional[Callable[[EditorTab], None]] = None
        self.on_tab_modified: Optional[Callable[[EditorTab], None]] = None
        
    def open_tab(
        self,
        file_path: Path,
        content: str = "",
        activate: bool = True
    ) -> EditorTab:
        existing_tab = self.find_tab_by_path(file_path)
        if existing_tab:
            if activate:
                self.activate_tab(existing_tab)
            return existing_tab
        
        tab = EditorTab(
            file_path=file_path,
            title=file_path.name,
            content=content,
            is_active=False,
        )
        
        self.tabs.append(tab)
        
        if activate:
            self.activate_tab(tab)
        
        if self.on_tab_opened:
            self.on_tab_opened(tab)
        
        return tab
    
    def close_tab(self, tab: EditorTab) -> bool:
        if tab not in self.tabs:
            return False
        
        tab_index = self.tabs.index(tab)
        was_active = tab.is_active
        
        self.tabs.remove(tab)
        
        if was_active and self.tabs:
            new_active_index = min(tab_index, len(self.tabs) - 1)
            self.activate_tab_by_index(new_active_index)
        elif not self.tabs:
            self.active_tab_index = -1
        
        if self.on_tab_closed:
            self.on_tab_closed(tab)
        
        return True
    
    def close_tab_by_path(self, file_path: Path) -> bool:
        tab = self.find_tab_by_path(file_path)
        if tab:
            return self.close_tab(tab)
        return False
    
    def close_all_tabs(self) -> None:
        tabs_copy = self.tabs.copy()
        for tab in tabs_copy:
            self.close_tab(tab)
    
    def close_other_tabs(self, tab: EditorTab) -> None:
        tabs_copy = self.tabs.copy()
        for other_tab in tabs_copy:
            if other_tab != tab:
                self.close_tab(other_tab)
    
    def activate_tab(self, tab: EditorTab) -> bool:
        if tab not in self.tabs:
            return False
        
        for t in self.tabs:
            t.is_active = False
        
        tab.is_active = True
        self.active_tab_index = self.tabs.index(tab)
        
        if self.on_tab_activated:
            self.on_tab_activated(tab)
        
        return True
    
    def activate_tab_by_index(self, index: int) -> bool:
        if 0 <= index < len(self.tabs):
            return self.activate_tab(self.tabs[index])
        return False
    
    def activate_tab_by_path(self, file_path: Path) -> bool:
        tab = self.find_tab_by_path(file_path)
        if tab:
            return self.activate_tab(tab)
        return False
    
    def activate_next_tab(self) -> bool:
        if not self.tabs:
            return False
        
        next_index = (self.active_tab_index + 1) % len(self.tabs)
        return self.activate_tab_by_index(next_index)
    
    def activate_previous_tab(self) -> bool:
        if not self.tabs:
            return False
        
        prev_index = (self.active_tab_index - 1) % len(self.tabs)
        return self.activate_tab_by_index(prev_index)
    
    def get_active_tab(self) -> Optional[EditorTab]:
        if 0 <= self.active_tab_index < len(self.tabs):
            return self.tabs[self.active_tab_index]
        return None
    
    def find_tab_by_path(self, file_path: Path) -> Optional[EditorTab]:
        for tab in self.tabs:
            if tab.file_path == file_path:
                return tab
        return None
    
    def get_all_tabs(self) -> List[EditorTab]:
        return self.tabs.copy()
    
    def get_modified_tabs(self) -> List[EditorTab]:
        return [tab for tab in self.tabs if tab.is_modified]
    
    def update_tab_content(self, tab: EditorTab, content: str) -> None:
        if tab not in self.tabs:
            return
        
        if tab.content != content:
            tab.content = content
            tab.is_modified = True
            
            if self.on_tab_modified:
                self.on_tab_modified(tab)
    
    def mark_tab_as_saved(self, tab: EditorTab) -> None:
        if tab not in self.tabs:
            return
        
        tab.is_modified = False
    
    def update_tab_cursor_position(self, tab: EditorTab, line: int, column: int) -> None:
        if tab in self.tabs:
            tab.cursor_position = (line, column)
    
    def update_tab_scroll_position(self, tab: EditorTab, position: int) -> None:
        if tab in self.tabs:
            tab.scroll_position = position
    
    def rename_tab(self, tab: EditorTab, new_path: Path) -> None:
        if tab not in self.tabs:
            return
        
        tab.file_path = new_path
        tab.title = new_path.name
    
    def move_tab(self, from_index: int, to_index: int) -> bool:
        if not (0 <= from_index < len(self.tabs) and 0 <= to_index < len(self.tabs)):
            return False
        
        tab = self.tabs.pop(from_index)
        self.tabs.insert(to_index, tab)
        
        if self.active_tab_index == from_index:
            self.active_tab_index = to_index
        elif from_index < self.active_tab_index <= to_index:
            self.active_tab_index -= 1
        elif to_index <= self.active_tab_index < from_index:
            self.active_tab_index += 1
        
        return True
    
    def has_unsaved_changes(self) -> bool:
        return any(tab.is_modified for tab in self.tabs)
    
    def get_tab_count(self) -> int:
        return len(self.tabs)
    
    def set_tab_metadata(self, tab: EditorTab, key: str, value: Any) -> None:
        if tab in self.tabs:
            tab.metadata[key] = value
    
    def get_tab_metadata(self, tab: EditorTab, key: str, default: Any = None) -> Any:
        if tab in self.tabs:
            return tab.metadata.get(key, default)
        return default
