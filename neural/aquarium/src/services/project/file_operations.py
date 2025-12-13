from __future__ import annotations
from typing import Optional, Callable
from pathlib import Path
import shutil
from enum import Enum


class FileOperationType(Enum):
    NEW = "new"
    OPEN = "open"
    SAVE = "save"
    SAVE_AS = "save_as"
    CLOSE = "close"
    DELETE = "delete"
    RENAME = "rename"
    COPY = "copy"
    MOVE = "move"


class FileOperationResult:
    def __init__(self, success: bool, message: str = "", path: Optional[Path] = None):
        self.success = success
        self.message = message
        self.path = path


class FileOperations:
    def __init__(self):
        self.on_file_created: Optional[Callable[[Path], None]] = None
        self.on_file_opened: Optional[Callable[[Path], None]] = None
        self.on_file_saved: Optional[Callable[[Path], None]] = None
        self.on_file_closed: Optional[Callable[[Path], None]] = None
        self.on_file_deleted: Optional[Callable[[Path], None]] = None
        
    def new_file(
        self,
        directory: Path,
        filename: str,
        content: str = "",
        overwrite: bool = False
    ) -> FileOperationResult:
        try:
            file_path = directory / filename
            
            if file_path.exists() and not overwrite:
                return FileOperationResult(
                    success=False,
                    message=f"File already exists: {filename}",
                    path=file_path
                )
            
            directory.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if self.on_file_created:
                self.on_file_created(file_path)
            
            return FileOperationResult(
                success=True,
                message=f"File created: {filename}",
                path=file_path
            )
        except (IOError, OSError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to create file: {str(e)}"
            )
    
    def open_file(self, file_path: Path) -> FileOperationResult:
        try:
            if not file_path.exists():
                return FileOperationResult(
                    success=False,
                    message=f"File does not exist: {file_path}"
                )
            
            if not file_path.is_file():
                return FileOperationResult(
                    success=False,
                    message=f"Not a file: {file_path}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.on_file_opened:
                self.on_file_opened(file_path)
            
            return FileOperationResult(
                success=True,
                message="File opened successfully",
                path=file_path
            )
        except (IOError, UnicodeDecodeError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to open file: {str(e)}"
            )
    
    def save_file(self, file_path: Path, content: str) -> FileOperationResult:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if self.on_file_saved:
                self.on_file_saved(file_path)
            
            return FileOperationResult(
                success=True,
                message="File saved successfully",
                path=file_path
            )
        except IOError as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to save file: {str(e)}"
            )
    
    def save_as(
        self,
        old_path: Path,
        new_path: Path,
        content: str
    ) -> FileOperationResult:
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if self.on_file_saved:
                self.on_file_saved(new_path)
            
            return FileOperationResult(
                success=True,
                message=f"File saved as: {new_path.name}",
                path=new_path
            )
        except IOError as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to save file: {str(e)}"
            )
    
    def close_file(self, file_path: Path) -> FileOperationResult:
        if self.on_file_closed:
            self.on_file_closed(file_path)
        
        return FileOperationResult(
            success=True,
            message="File closed",
            path=file_path
        )
    
    def delete_file(self, file_path: Path) -> FileOperationResult:
        try:
            if not file_path.exists():
                return FileOperationResult(
                    success=False,
                    message=f"File does not exist: {file_path}"
                )
            
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            
            if self.on_file_deleted:
                self.on_file_deleted(file_path)
            
            return FileOperationResult(
                success=True,
                message="File deleted successfully",
                path=file_path
            )
        except (IOError, OSError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to delete file: {str(e)}"
            )
    
    def rename_file(self, old_path: Path, new_name: str) -> FileOperationResult:
        try:
            if not old_path.exists():
                return FileOperationResult(
                    success=False,
                    message=f"File does not exist: {old_path}"
                )
            
            new_path = old_path.parent / new_name
            
            if new_path.exists():
                return FileOperationResult(
                    success=False,
                    message=f"File already exists: {new_name}"
                )
            
            old_path.rename(new_path)
            
            return FileOperationResult(
                success=True,
                message=f"File renamed to: {new_name}",
                path=new_path
            )
        except (IOError, OSError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to rename file: {str(e)}"
            )
    
    def copy_file(self, source: Path, destination: Path) -> FileOperationResult:
        try:
            if not source.exists():
                return FileOperationResult(
                    success=False,
                    message=f"Source file does not exist: {source}"
                )
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            
            return FileOperationResult(
                success=True,
                message="File copied successfully",
                path=destination
            )
        except (IOError, OSError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to copy file: {str(e)}"
            )
    
    def move_file(self, source: Path, destination: Path) -> FileOperationResult:
        try:
            if not source.exists():
                return FileOperationResult(
                    success=False,
                    message=f"Source file does not exist: {source}"
                )
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source), str(destination))
            
            return FileOperationResult(
                success=True,
                message="File moved successfully",
                path=destination
            )
        except (IOError, OSError) as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to move file: {str(e)}"
            )
    
    def read_file(self, file_path: Path) -> tuple[bool, str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except (IOError, UnicodeDecodeError) as e:
            return False, str(e)
    
    def create_directory(self, path: Path) -> FileOperationResult:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return FileOperationResult(
                success=True,
                message="Directory created successfully",
                path=path
            )
        except OSError as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to create directory: {str(e)}"
            )
