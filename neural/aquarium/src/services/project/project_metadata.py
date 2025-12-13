from __future__ import annotations
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime


class ProjectMetadata:
    METADATA_FILE = ".aquarium-project"
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.metadata_path = project_path / self.METADATA_FILE
        self.data: Dict[str, Any] = self._default_metadata()
        
    def _default_metadata(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "name": self.project_path.name,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "settings": {
                "editor": {
                    "font_size": 14,
                    "theme": "dark",
                    "tab_size": 4,
                    "auto_save": True,
                    "auto_save_delay": 1000,
                    "show_line_numbers": True,
                    "word_wrap": False,
                },
                "project": {
                    "default_backend": "tensorflow",
                    "auto_compile": False,
                    "show_hidden_files": False,
                },
            },
            "open_files": [],
            "active_file": None,
            "workspace": {
                "layout": "default",
                "sidebar_width": 250,
                "panel_height": 200,
            },
            "recent_searches": [],
            "bookmarks": [],
            "breakpoints": {},
        }
    
    def load(self) -> bool:
        if not self.metadata_path.exists():
            return False
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                self.data.update(loaded_data)
            return True
        except (json.JSONDecodeError, IOError):
            return False
    
    def save(self) -> bool:
        self.data["last_modified"] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            return True
        except IOError:
            return False
    
    def get_setting(self, *keys: str) -> Any:
        value = self.data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def set_setting(self, *keys: str, value: Any) -> None:
        if len(keys) < 2:
            return
        
        current = self.data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_open_files(self) -> List[str]:
        return self.data.get("open_files", [])
    
    def set_open_files(self, files: List[str]) -> None:
        self.data["open_files"] = files
    
    def get_active_file(self) -> Optional[str]:
        return self.data.get("active_file")
    
    def set_active_file(self, file_path: Optional[str]) -> None:
        self.data["active_file"] = file_path
    
    def add_bookmark(self, file_path: str, line: int, description: str = "") -> None:
        bookmarks = self.data.get("bookmarks", [])
        bookmarks.append({
            "file": file_path,
            "line": line,
            "description": description,
            "created_at": datetime.now().isoformat(),
        })
        self.data["bookmarks"] = bookmarks
    
    def remove_bookmark(self, file_path: str, line: int) -> None:
        bookmarks = self.data.get("bookmarks", [])
        self.data["bookmarks"] = [
            b for b in bookmarks
            if not (b["file"] == file_path and b["line"] == line)
        ]
    
    def get_bookmarks(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        bookmarks = self.data.get("bookmarks", [])
        if file_path:
            return [b for b in bookmarks if b["file"] == file_path]
        return bookmarks
    
    def set_breakpoints(self, file_path: str, lines: List[int]) -> None:
        breakpoints = self.data.get("breakpoints", {})
        breakpoints[file_path] = lines
        self.data["breakpoints"] = breakpoints
    
    def get_breakpoints(self, file_path: Optional[str] = None) -> Dict[str, List[int]]:
        breakpoints = self.data.get("breakpoints", {})
        if file_path:
            return {file_path: breakpoints.get(file_path, [])}
        return breakpoints
    
    def add_recent_search(self, query: str) -> None:
        recent_searches = self.data.get("recent_searches", [])
        if query in recent_searches:
            recent_searches.remove(query)
        recent_searches.insert(0, query)
        self.data["recent_searches"] = recent_searches[:20]
    
    def get_recent_searches(self) -> List[str]:
        return self.data.get("recent_searches", [])
