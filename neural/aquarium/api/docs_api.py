"""
API endpoints for serving documentation content.
"""

import os
from pathlib import Path
from flask import Blueprint, send_file, jsonify, request

docs_bp = Blueprint('docs', __name__, url_prefix='/api/docs')

DOCS_DIR = Path(__file__).parent.parent


def find_doc_file(doc_path):
    doc_path = doc_path.lstrip('/')
    
    search_paths = [
        DOCS_DIR / doc_path,
        DOCS_DIR / doc_path.upper(),
        DOCS_DIR.parent.parent / 'docs' / doc_path,
        DOCS_DIR.parent.parent / 'docs' / doc_path.upper(),
    ]
    
    for path in search_paths:
        if path.exists() and path.is_file():
            return path
    
    return None


@docs_bp.route('/<path:doc_path>', methods=['GET'])
def get_documentation(doc_path):
    try:
        file_path = find_doc_file(doc_path)
        
        if not file_path:
            return jsonify({
                'error': 'Documentation file not found',
                'path': doc_path
            }), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@docs_bp.route('/list', methods=['GET'])
def list_documentation():
    try:
        docs = []
        
        doc_dirs = [
            DOCS_DIR,
            DOCS_DIR.parent.parent / 'docs'
        ]
        
        for doc_dir in doc_dirs:
            if not doc_dir.exists():
                continue
            
            for doc_file in doc_dir.glob('*.md'):
                docs.append({
                    'name': doc_file.stem,
                    'path': str(doc_file.relative_to(DOCS_DIR)),
                    'size': doc_file.stat().st_size
                })
        
        return jsonify({
            'docs': docs,
            'count': len(docs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@docs_bp.route('/search', methods=['GET'])
def search_documentation():
    try:
        query = request.args.get('q', '').lower()
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        results = []
        
        doc_dirs = [
            DOCS_DIR,
            DOCS_DIR.parent.parent / 'docs'
        ]
        
        for doc_dir in doc_dirs:
            if not doc_dir.exists():
                continue
            
            for doc_file in doc_dir.glob('*.md'):
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if query in content.lower() or query in doc_file.stem.lower():
                        lines = content.split('\n')
                        matching_lines = [
                            (i + 1, line) for i, line in enumerate(lines)
                            if query in line.lower()
                        ]
                        
                        results.append({
                            'name': doc_file.stem,
                            'path': str(doc_file.relative_to(DOCS_DIR)),
                            'matches': len(matching_lines),
                            'preview': matching_lines[:3] if matching_lines else []
                        })
                except Exception as e:
                    print(f"Error searching {doc_file}: {e}")
                    continue
        
        return jsonify({
            'results': results,
            'count': len(results),
            'query': query
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
