from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json


class RecentProject:
    def __init__(
        self,
        path: Path,
        name: str,
        last_opened: datetime,
        description: str = ""
    ):
        self.path = path
        self.name = name
        self.last_opened = last_opened
        self.description = description
        
    def to_dict(self) -> Dict[str, str]:
        return {
            "path": str(self.path),
            "name": self.name,
            "last_opened": self.last_opened.isoformat(),
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> RecentProject:
        return cls(
            path=Path(data["path"]),
            name=data["name"],
            last_opened=datetime.fromisoformat(data["last_opened"]),
            description=data.get("description", ""),
        )
    
    def exists(self) -> bool:
        return self.path.exists() and self.path.is_dir()


class RecentProjectsManager:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = config_dir / "recent_projects.json"
        self.max_recent = 20
        self.projects: List[RecentProject] = []
        self.load()
        
    def load(self) -> bool:
        if not self.config_file.exists():
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.projects = [
                    RecentProject.from_dict(proj)
                    for proj in data.get("projects", [])
                ]
                self.projects = [p for p in self.projects if p.exists()]
            return True
        except (json.JSONDecodeError, IOError, KeyError):
            return False
    
    def save(self) -> bool:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                data = {
                    "projects": [proj.to_dict() for proj in self.projects]
                }
                json.dump(data, f, indent=2)
            return True
        except IOError:
            return False
    
    def add_project(
        self,
        path: Path,
        name: Optional[str] = None,
        description: str = ""
    ) -> RecentProject:
        if name is None:
            name = path.name
        
        existing = self.find_by_path(path)
        if existing:
            existing.last_opened = datetime.now()
            existing.description = description or existing.description
            self._sort_by_last_opened()
            return existing
        
        project = RecentProject(
            path=path,
            name=name,
            last_opened=datetime.now(),
            description=description,
        )
        
        self.projects.insert(0, project)
        
        if len(self.projects) > self.max_recent:
            self.projects = self.projects[:self.max_recent]
        
        return project
    
    def remove_project(self, path: Path) -> bool:
        project = self.find_by_path(path)
        if project:
            self.projects.remove(project)
            return True
        return False
    
    def find_by_path(self, path: Path) -> Optional[RecentProject]:
        for project in self.projects:
            if project.path == path:
                return project
        return None
    
    def get_all(self) -> List[RecentProject]:
        return self.projects.copy()
    
    def get_recent(self, count: int = 10) -> List[RecentProject]:
        return self.projects[:count]
    
    def clear(self) -> None:
        self.projects.clear()
    
    def _sort_by_last_opened(self) -> None:
        self.projects.sort(key=lambda p: p.last_opened, reverse=True)
    
    def update_project_name(self, path: Path, name: str) -> bool:
        project = self.find_by_path(path)
        if project:
            project.name = name
            return True
        return False
    
    def update_project_description(self, path: Path, description: str) -> bool:
        project = self.find_by_path(path)
        if project:
            project.description = description
            return True
        return False
    
    def clean_invalid_projects(self) -> int:
        original_count = len(self.projects)
        self.projects = [p for p in self.projects if p.exists()]
        return original_count - len(self.projects)
