from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import re


def is_valid_filename(filename: str) -> bool:
    if not filename or len(filename) > 255:
        return False
    
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    if re.search(invalid_chars, filename):
        return False
    
    reserved_names = [
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]
    
    name_without_ext = filename.rsplit('.', 1)[0].upper()
    if name_without_ext in reserved_names:
        return False
    
    if filename.endswith('.') or filename.endswith(' '):
        return False
    
    return True


def is_neural_file(file_path: Path) -> bool:
    return file_path.suffix == '.neural'


def get_relative_path(file_path: Path, base_path: Path) -> Optional[Path]:
    try:
        return file_path.relative_to(base_path)
    except ValueError:
        return None


def find_neural_files(directory: Path, recursive: bool = True) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    
    neural_files = []
    
    if recursive:
        for file_path in directory.rglob('*.neural'):
            neural_files.append(file_path)
    else:
        for file_path in directory.glob('*.neural'):
            neural_files.append(file_path)
    
    return sorted(neural_files)


def get_file_size_formatted(file_path: Path) -> str:
    if not file_path.exists() or not file_path.is_file():
        return "0 B"
    
    size = file_path.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} PB"


def get_project_statistics(project_path: Path) -> dict:
    if not project_path.exists() or not project_path.is_dir():
        return {}
    
    stats = {
        'total_files': 0,
        'neural_files': 0,
        'total_size': 0,
        'directories': 0,
    }
    
    for item in project_path.rglob('*'):
        if item.name.startswith('.'):
            continue
        
        if item.is_file():
            stats['total_files'] += 1
            stats['total_size'] += item.stat().st_size
            
            if is_neural_file(item):
                stats['neural_files'] += 1
        elif item.is_dir():
            stats['directories'] += 1
    
    return stats


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    filename = filename.strip('. ')
    
    if not filename:
        filename = 'untitled'
    
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 255 - len(ext) - 1 if ext else 255
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename


def ensure_neural_extension(filename: str) -> str:
    if not filename.endswith('.neural'):
        return filename + '.neural'
    return filename


def get_unique_filename(directory: Path, base_name: str) -> str:
    if not (directory / base_name).exists():
        return base_name
    
    name, ext = base_name.rsplit('.', 1) if '.' in base_name else (base_name, '')
    counter = 1
    
    while True:
        new_name = f"{name}_{counter}"
        if ext:
            new_name += f".{ext}"
        
        if not (directory / new_name).exists():
            return new_name
        
        counter += 1


def is_text_file(file_path: Path) -> bool:
    text_extensions = {
        '.txt', '.neural', '.py', '.js', '.ts', '.json', '.yaml', '.yml',
        '.md', '.rst', '.html', '.css', '.xml', '.csv', '.log', '.ini',
        '.toml', '.sh', '.bat', '.ps1', '.c', '.cpp', '.h', '.java',
    }
    
    return file_path.suffix.lower() in text_extensions


def count_lines(file_path: Path) -> int:
    if not file_path.exists() or not file_path.is_file():
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except (IOError, UnicodeDecodeError):
        return 0
