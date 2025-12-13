from __future__ import annotations

import os
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from neural.visualization.aquarium_integration import AquariumVisualizationManager
from neural.visualization.aquarium_web_components import AquariumWebComponentRenderer


class AquariumVisualizationServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 8052):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        self.manager = AquariumVisualizationManager()
        self.renderer = AquariumWebComponentRenderer(self.manager)
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            if self.manager.current_gallery is None:
                return self._render_setup_page()
            return self.renderer.render_gallery_view()
        
        @self.app.route('/visualization/<viz_type>')
        def view_visualization(viz_type: str):
            return self.renderer.render_visualization_detail(viz_type)
        
        @self.app.route('/api/load-model', methods=['POST'])
        def load_model():
            data = request.get_json()
            dsl_code = data.get('dsl_code')
            
            if not dsl_code:
                return jsonify({'success': False, 'error': 'No DSL code provided'}), 400
            
            try:
                model_data = self.manager.load_model_from_dsl(dsl_code)
                return jsonify({
                    'success': True,
                    'model_data': model_data,
                    'message': 'Model loaded successfully'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/create-gallery', methods=['POST'])
        def create_gallery():
            data = request.get_json() or {}
            input_shape = data.get('input_shape')
            
            if input_shape and isinstance(input_shape, list):
                input_shape = tuple(input_shape)
            
            try:
                gallery = self.manager.create_gallery(input_shape)
                return jsonify({
                    'success': True,
                    'metadata': gallery.get_gallery_metadata(),
                    'visualizations': self.manager.get_visualization_list(),
                    'message': 'Gallery created successfully'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/visualization/<viz_type>')
        def get_visualization(viz_type: str):
            try:
                if self.manager.current_gallery is None:
                    return jsonify({'success': False, 'error': 'No gallery available'}), 404
                
                viz_data = self.manager.current_gallery.get_visualization(viz_type)
                if viz_data is None:
                    return jsonify({'success': False, 'error': f'Visualization {viz_type} not found'}), 404
                
                return jsonify({
                    'success': True,
                    'viz_type': viz_type,
                    'data': viz_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/export/<viz_type>/<format>')
        def export_visualization(viz_type: str, format: str):
            try:
                output_dir = 'output'
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = f"{output_dir}/{viz_type}.{format}"
                path = self.manager.export_visualization(viz_type, format, output_path)
                
                return jsonify({
                    'success': True,
                    'path': path,
                    'message': f'Exported {viz_type} to {path}'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/export-all/<format>')
        def export_all(format: str):
            try:
                output_dir = 'output'
                os.makedirs(output_dir, exist_ok=True)
                
                paths = self.manager.export_all_visualizations(format, output_dir)
                
                return jsonify({
                    'success': True,
                    'paths': paths,
                    'message': f'Exported all visualizations to {output_dir}'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/gallery-metadata')
        def get_gallery_metadata():
            try:
                metadata = self.manager.get_gallery_metadata()
                return jsonify({'success': True, 'metadata': metadata})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/gallery-json')
        def get_gallery_json():
            try:
                if self.manager.current_gallery is None:
                    return jsonify({'success': False, 'error': 'No gallery available'}), 404
                
                gallery_json = self.manager.current_gallery.to_json()
                return Response(gallery_json, mimetype='application/json')
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/visualization/<viz_type>/thumbnail')
        def get_thumbnail(viz_type: str):
            try:
                thumbnail_data = self.renderer.generate_thumbnail(viz_type)
                if thumbnail_data is None:
                    return jsonify({'success': False, 'error': 'Thumbnail generation failed'}), 500
                
                return jsonify({'success': True, 'thumbnail': thumbnail_data})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/download/<path:filepath>')
        def download_file(filepath: str):
            try:
                return send_file(filepath, as_attachment=True)
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 404
    
    def _render_setup_page(self) -> str:
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Visualization Gallery - Setup</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .setup-container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            max-width: 800px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #1e3c72;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        textarea {
            width: 100%;
            min-height: 300px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .example-link {
            display: block;
            margin-top: 15px;
            color: #667eea;
            text-decoration: none;
            font-size: 0.9em;
        }
        
        .example-link:hover {
            text-decoration: underline;
        }
        
        #status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        
        #status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            display: block;
        }
        
        #status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            display: block;
        }
    </style>
</head>
<body>
    <div class="setup-container">
        <h1>🎨 Neural Visualization Gallery</h1>
        <p class="subtitle">Load a Neural DSL model to create visualizations</p>
        
        <div class="form-group">
            <label for="dsl-code">Neural DSL Code:</label>
            <textarea id="dsl-code" placeholder="Enter your Neural DSL code here...">network TestNet {
    input: (None, 28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(128, "relu")
        Output(10, "softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}</textarea>
        </div>
        
        <button class="btn" onclick="loadAndCreateGallery()">
            Load Model & Create Gallery
        </button>
        
        <div id="status"></div>
    </div>
    
    <script>
        async function loadAndCreateGallery() {
            const dslCode = document.getElementById('dsl-code').value;
            const statusDiv = document.getElementById('status');
            
            try {
                statusDiv.className = '';
                statusDiv.style.display = 'none';
                
                const loadResponse = await fetch('/api/load-model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({dsl_code: dslCode})
                });
                
                const loadResult = await loadResponse.json();
                
                if (!loadResult.success) {
                    throw new Error(loadResult.error);
                }
                
                const galleryResponse = await fetch('/api/create-gallery', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})
                });
                
                const galleryResult = await galleryResponse.json();
                
                if (!galleryResult.success) {
                    throw new Error(galleryResult.error);
                }
                
                statusDiv.textContent = 'Gallery created successfully! Redirecting...';
                statusDiv.className = 'success';
                
                setTimeout(() => {
                    window.location.href = '/';
                }, 1000);
                
            } catch (error) {
                statusDiv.textContent = `Error: ${error.message}`;
                statusDiv.className = 'error';
            }
        }
    </script>
</body>
</html>
"""
    
    def run(self, debug: bool = False):
        print(f"🎨 Neural Visualization Gallery starting on http://{self.host}:{self.port}")
        print(f"📊 Access the gallery at: http://localhost:{self.port}/")
        
        ssl_context = None
        if self.security_config.ssl_enabled and self.security_config.ssl_cert_file and self.security_config.ssl_key_file:
            ssl_context = (self.security_config.ssl_cert_file, self.security_config.ssl_key_file)
        
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            ssl_context=ssl_context
        )


def start_server(host: str = '0.0.0.0', port: int = 8052, debug: bool = False):
    server = AquariumVisualizationServer(host=host, port=port)
    server.run(debug=debug)


if __name__ == '__main__':
    import sys
    
    port = 8052
    debug = False
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == '--debug':
        debug = True
    
    start_server(port=port, debug=debug)
