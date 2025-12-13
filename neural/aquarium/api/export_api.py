"""
API endpoints for model export and deployment.
Provides REST endpoints for exporting models to various formats and deploying them.
"""

from flask import Blueprint, jsonify, request
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural.code_generation.export import ModelExporter
from neural.mlops.deployment import DeploymentManager, DeploymentStrategy

export_bp = Blueprint('export', __name__, url_prefix='/api/export')
deployment_bp = Blueprint('deployment', __name__, url_prefix='/api/deployment')

deployment_manager = None


def get_deployment_manager():
    global deployment_manager
    if deployment_manager is None:
        deployment_manager = DeploymentManager('./deployments')
    return deployment_manager


@export_bp.route('/model', methods=['POST'])
def export_model():
    """Export a model to the specified format."""
    try:
        data = request.json
        
        if not data or 'model_data' not in data or 'options' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request. Requires model_data and options.'
            }), 400
        
        model_data = data['model_data']
        options = data['options']
        
        format_type = options.get('format', 'onnx')
        backend = options.get('backend', 'tensorflow')
        output_path = options.get('outputPath', './exported_model')
        optimize = options.get('optimize', True)
        
        exporter = ModelExporter(model_data, backend=backend)
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if format_type == 'onnx':
            opset_version = options.get('opsetVersion', 13)
            exported_path = exporter.export_onnx(
                output_path=f"{output_path}.onnx",
                opset_version=opset_version,
                optimize=optimize
            )
        
        elif format_type == 'tflite':
            quantization = options.get('quantization', {})
            quantize = quantization.get('enabled', False)
            quantization_type = quantization.get('type', 'none')
            
            exported_path = exporter.export_tflite(
                output_path=f"{output_path}.tflite",
                quantize=quantize,
                quantization_type=quantization_type if quantize else 'dynamic'
            )
        
        elif format_type == 'torchscript':
            exported_path = exporter.export_torchscript(
                output_path=f"{output_path}.pt",
                method='trace'
            )
        
        elif format_type == 'savedmodel':
            exported_path = exporter.export_savedmodel(
                output_path=output_path
            )
        
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported format: {format_type}'
            }), 400
        
        file_size = 0
        if os.path.exists(exported_path):
            if os.path.isfile(exported_path):
                file_size = os.path.getsize(exported_path)
            elif os.path.isdir(exported_path):
                file_size = sum(
                    f.stat().st_size for f in Path(exported_path).rglob('*') if f.is_file()
                )
        
        return jsonify({
            'success': True,
            'exportPath': exported_path,
            'format': format_type,
            'size': file_size,
            'message': f'Model successfully exported to {format_type.upper()} format'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@deployment_bp.route('/deploy', methods=['POST'])
def deploy_model():
    """Deploy an exported model."""
    try:
        data = request.json
        
        if not data or 'export_path' not in data or 'config' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request. Requires export_path and config.'
            }), 400
        
        export_path = data['export_path']
        config = data['config']
        
        if not os.path.exists(export_path):
            return jsonify({
                'success': False,
                'error': f'Export path does not exist: {export_path}'
            }), 400
        
        manager = get_deployment_manager()
        
        model_name = config.get('modelName', 'neural_model')
        version = config.get('version', '1.0')
        strategy_name = config.get('strategy', 'direct')
        
        strategy_map = {
            'direct': DeploymentStrategy.DIRECT,
            'blue_green': DeploymentStrategy.BLUE_GREEN,
            'canary': DeploymentStrategy.CANARY,
            'shadow': DeploymentStrategy.SHADOW,
            'rolling': DeploymentStrategy.ROLLING,
        }
        
        strategy = strategy_map.get(strategy_name, DeploymentStrategy.DIRECT)
        
        deployment = manager.create_deployment(
            model_name=model_name,
            model_version=version,
            strategy=strategy,
            created_by='aquarium_user',
            environment=config.get('target', 'production'),
            metadata={
                'export_path': export_path,
                'serving_platform': config.get('servingPlatform', 'unknown'),
                'resources': config.get('resources', {}),
                'networking': config.get('networking', {}),
            }
        )
        
        manager.start_deployment(deployment.deployment_id)
        manager.complete_deployment(deployment.deployment_id)
        
        port = config.get('networking', {}).get('port', 8080)
        endpoint = f"http://localhost:{port}/predictions/{model_name}"
        
        return jsonify({
            'success': True,
            'deploymentId': deployment.deployment_id,
            'endpoint': endpoint,
            'message': f'Model deployed successfully with {strategy_name} strategy'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@deployment_bp.route('/serving-config', methods=['POST'])
def generate_serving_config():
    """Generate serving configuration files."""
    try:
        data = request.json
        
        if not data or 'model_path' not in data or 'platform' not in data:
            return jsonify({
                'error': 'Invalid request. Requires model_path and platform.'
            }), 400
        
        model_path = data['model_path']
        platform = data['platform']
        model_name = data.get('model_name', 'model')
        config = data.get('config', {})
        
        output_dir = os.path.join(os.path.dirname(model_path), 'serving_config')
        os.makedirs(output_dir, exist_ok=True)
        
        exporter = ModelExporter({}, backend='tensorflow')
        
        if platform == 'torchserve':
            config_path, model_store = exporter.create_torchserve_config(
                model_path=model_path,
                model_name=model_name,
                output_dir=output_dir,
                batch_size=config.get('resources', {}).get('batchSize', 1),
                max_batch_delay=config.get('resources', {}).get('maxBatchDelay', 100)
            )
            
            scripts = exporter.generate_deployment_scripts(
                output_dir=output_dir,
                deployment_type='torchserve'
            )
            
            return jsonify({
                'configPath': config_path,
                'modelStorePath': model_store,
                'scripts': scripts,
                'instructions': [
                    'Install TorchServe: pip install torchserve torch-model-archiver',
                    f'Archive your model and place it in {model_store}',
                    'Run ./start_torchserve.sh to start the server',
                    f'Test inference with python test_inference.py',
                    'Monitor logs at logs/model_metrics.log'
                ]
            })
        
        elif platform == 'tfserving':
            config_path = exporter.create_tfserving_config(
                model_path=model_path,
                model_name=model_name,
                output_dir=output_dir,
                version=1
            )
            
            scripts = exporter.generate_deployment_scripts(
                output_dir=output_dir,
                deployment_type='tfserving'
            )
            
            return jsonify({
                'configPath': config_path,
                'scripts': scripts,
                'instructions': [
                    'Install Docker if not already installed',
                    'Ensure model files are in the correct directory structure',
                    'Run ./start_tfserving.sh to start TF Serving',
                    'Access REST API at http://localhost:8501',
                    'Test inference with python test_inference.py'
                ]
            })
        
        elif platform == 'onnxruntime':
            return jsonify({
                'configPath': model_path,
                'scripts': [],
                'instructions': [
                    'Install ONNX Runtime: pip install onnxruntime',
                    'Load model: session = ort.InferenceSession(model_path)',
                    'Run inference: session.run(None, {input_name: input_data})',
                    'Wrap with FastAPI or Flask for REST API serving',
                    'Deploy to cloud or edge devices as needed'
                ]
            })
        
        elif platform == 'triton':
            return jsonify({
                'configPath': output_dir,
                'scripts': [],
                'instructions': [
                    'Install NVIDIA Triton Inference Server',
                    'Create model repository with proper structure',
                    'Add model configuration file (config.pbtxt)',
                    'Start Triton: tritonserver --model-repository=/models',
                    'Access HTTP endpoint at localhost:8000'
                ]
            })
        
        else:
            return jsonify({
                'error': f'Unsupported platform: {platform}'
            }), 400
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@deployment_bp.route('/<deployment_id>/status', methods=['GET'])
def get_deployment_status(deployment_id):
    """Get status of a deployment."""
    try:
        manager = get_deployment_manager()
        deployment = manager.get_deployment(deployment_id)
        
        return jsonify({
            'deploymentId': deployment.deployment_id,
            'status': deployment.status.value,
            'modelName': deployment.model_name,
            'modelVersion': deployment.model_version,
            'createdAt': deployment.created_at,
            'startedAt': deployment.started_at,
            'completedAt': deployment.completed_at,
        })
    
    except FileNotFoundError:
        return jsonify({
            'error': f'Deployment {deployment_id} not found'
        }), 404
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@deployment_bp.route('/list', methods=['GET'])
def list_deployments():
    """List all deployments."""
    try:
        manager = get_deployment_manager()
        deployments = manager.list_deployments()
        
        return jsonify({
            'deployments': [
                {
                    'deploymentId': d.deployment_id,
                    'modelName': d.model_name,
                    'modelVersion': d.model_version,
                    'status': d.status.value,
                    'createdAt': d.created_at,
                }
                for d in deployments
            ]
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


def register_blueprints(app):
    """Register export and deployment blueprints with Flask app."""
    app.register_blueprint(export_bp)
    app.register_blueprint(deployment_bp)
