"""
API endpoints for loading and managing example neural network models.
"""

import os
import json
from pathlib import Path
from flask import Blueprint, jsonify, request

examples_bp = Blueprint('examples', __name__, url_prefix='/api/examples')

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


def get_example_metadata():
    examples = []
    
    if not EXAMPLES_DIR.exists():
        return examples
    
    for example_file in EXAMPLES_DIR.glob('*.neural'):
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            name = example_file.stem.replace('_', ' ').title()
            
            category = 'General'
            description = f'Neural network model: {name}'
            tags = []
            complexity = 'Intermediate'
            
            if 'cnn' in example_file.stem.lower() or 'conv' in content.lower():
                category = 'Computer Vision'
                tags.extend(['cnn', 'computer-vision'])
            elif 'lstm' in content.lower() or 'rnn' in content.lower() or 'gru' in content.lower():
                category = 'NLP'
                tags.extend(['nlp', 'recurrent'])
            elif 'gan' in example_file.stem.lower() or 'vae' in example_file.stem.lower():
                category = 'Generative'
                tags.extend(['generative'])
            elif 'transformer' in content.lower():
                category = 'NLP'
                tags.extend(['transformer', 'attention'])
            
            if 'mnist' in example_file.stem.lower():
                description = 'Convolutional Neural Network for MNIST digit classification'
                tags.append('mnist')
                complexity = 'Beginner'
            elif 'text' in example_file.stem.lower():
                description = 'LSTM network for text classification and sentiment analysis'
                tags.append('text')
                complexity = 'Beginner'
            
            examples.append({
                'name': name,
                'path': str(example_file.relative_to(EXAMPLES_DIR.parent)),
                'description': description,
                'category': category,
                'tags': tags,
                'complexity': complexity
            })
        except Exception as e:
            print(f"Error processing {example_file}: {e}")
            continue
    
    return examples


@examples_bp.route('/list', methods=['GET'])
def list_examples():
    try:
        examples = get_example_metadata()
        return jsonify({
            'examples': examples,
            'count': len(examples)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'examples': []
        }), 500


@examples_bp.route('/load', methods=['GET'])
def load_example():
    try:
        example_path = request.args.get('path')
        if not example_path:
            return jsonify({'error': 'Path parameter is required'}), 400
        
        full_path = Path(__file__).parent.parent / example_path
        
        if not full_path.exists():
            return jsonify({'error': 'Example file not found'}), 404
        
        if not full_path.is_file() or not str(full_path).endswith('.neural'):
            return jsonify({'error': 'Invalid example file'}), 400
        
        with open(full_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        return jsonify({
            'code': code,
            'path': example_path,
            'name': full_path.stem
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@examples_bp.route('/categories', methods=['GET'])
def get_categories():
    try:
        examples = get_example_metadata()
        categories = list(set(ex['category'] for ex in examples))
        return jsonify({
            'categories': sorted(categories)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@examples_bp.route('/search', methods=['GET'])
def search_examples():
    try:
        query = request.args.get('q', '').lower()
        category = request.args.get('category')
        
        examples = get_example_metadata()
        
        if category:
            examples = [ex for ex in examples if ex['category'] == category]
        
        if query:
            examples = [
                ex for ex in examples
                if query in ex['name'].lower()
                or query in ex['description'].lower()
                or any(query in tag.lower() for tag in ex['tags'])
            ]
        
        return jsonify({
            'examples': examples,
            'count': len(examples),
            'query': query,
            'category': category
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
