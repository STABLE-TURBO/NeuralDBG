"""
Model export functionality for various deployment targets.
Supports ONNX, TensorFlow Lite, TorchScript, and SavedModel formats.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from neural.exceptions import DependencyError

logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles model export to various deployment formats."""

    def __init__(self, model_data: Dict[str, Any], backend: str = 'tensorflow'):
        """
        Initialize the ModelExporter.
        
        Parameters
        ----------
        model_data : dict
            Model configuration from DSL parser
        backend : str
            Backend framework ('tensorflow' or 'pytorch')
        """
        self.model_data = model_data
        self.backend = backend

    def export_onnx(
        self,
        output_path: str = 'model.onnx',
        opset_version: int = 13,
        optimize: bool = True,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """
        Export model to ONNX format with optimization.
        
        Parameters
        ----------
        output_path : str
            Path to save the ONNX model
        opset_version : int
            ONNX opset version (default: 13)
        optimize : bool
            Whether to apply ONNX optimization passes
        dynamic_axes : dict, optional
            Dynamic axes configuration for variable input shapes
            
        Returns
        -------
        str
            Path to the exported model
        """
        if self.backend == 'tensorflow':
            return self._export_tensorflow_to_onnx(
                output_path, opset_version, optimize, dynamic_axes
            )
        elif self.backend == 'pytorch':
            return self._export_pytorch_to_onnx(
                output_path, opset_version, optimize, dynamic_axes
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _export_tensorflow_to_onnx(
        self,
        output_path: str,
        opset_version: int,
        optimize: bool,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]]
    ) -> str:
        """Export TensorFlow model to ONNX."""
        try:
            import tensorflow as tf
            import tf2onnx

            from neural.code_generation.code_generator import generate_code

            code = generate_code(self.model_data, 'tensorflow')

            temp_model_path = 'temp_tf_model'
            exec_globals = {}
            exec(code, exec_globals)
            model = exec_globals.get('model')

            if model is None:
                raise ValueError("Failed to extract model from generated code")

            input_signature = [
                tf.TensorSpec(
                    shape=(None,) + tuple(self.model_data['input']['shape']),
                    dtype=tf.float32,
                    name='input'
                )
            ]

            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=opset_version,
                output_path=output_path
            )

            if optimize:
                model_proto = self._optimize_onnx_model(model_proto)

            import onnx
            onnx.save(model_proto, output_path)

            logger.info(f"TensorFlow model exported to ONNX: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"Required dependency not found: {e}")
            raise DependencyError(
                dependency="tf2onnx",
                feature="TensorFlow to ONNX conversion",
                install_hint="pip install tf2onnx"
            )

    def _export_pytorch_to_onnx(
        self,
        output_path: str,
        opset_version: int,
        optimize: bool,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]]
    ) -> str:
        """Export PyTorch model to ONNX."""
        try:
            import torch

            from neural.code_generation.code_generator import generate_code

            code = generate_code(self.model_data, 'pytorch')

            exec_globals = {}
            exec(code, exec_globals)
            model = exec_globals.get('model')

            if model is None:
                raise ValueError("Failed to extract model from generated code")

            model.eval()

            input_shape = (1,) + tuple(self.model_data['input']['shape'])
            dummy_input = torch.randn(input_shape)

            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=optimize,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )

            if optimize:
                import onnx
                model_proto = onnx.load(output_path)
                model_proto = self._optimize_onnx_model(model_proto)
                onnx.save(model_proto, output_path)

            logger.info(f"PyTorch model exported to ONNX: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"Required dependency not found: {e}")
            raise ImportError(
                "PyTorch is required for PyTorch to ONNX conversion. "
                "Install with: pip install torch"
            )

    def _optimize_onnx_model(self, model_proto):
        """Apply optimization passes to ONNX model."""
        try:
            from onnx import optimizer

            passes = [
                'eliminate_identity',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_bn_into_conv',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ]

            optimized_model = optimizer.optimize(model_proto, passes)
            logger.info("ONNX model optimized successfully")
            return optimized_model

        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}, returning original model")
            return model_proto

    def export_tflite(
        self,
        output_path: str = 'model.tflite',
        quantize: bool = False,
        quantization_type: str = 'int8',
        representative_dataset: Optional[callable] = None
    ) -> str:
        """
        Export model to TensorFlow Lite format.
        
        Parameters
        ----------
        output_path : str
            Path to save the TFLite model
        quantize : bool
            Whether to apply quantization
        quantization_type : str
            Type of quantization ('int8', 'float16', 'dynamic')
        representative_dataset : callable, optional
            Generator function for representative dataset (required for full int8 quantization)
            
        Returns
        -------
        str
            Path to the exported model
        """
        try:
            import tensorflow as tf

            from neural.code_generation.code_generator import generate_code

            code = generate_code(self.model_data, 'tensorflow')

            exec_globals = {}
            exec(code, exec_globals)
            model = exec_globals.get('model')

            if model is None:
                raise ValueError("Failed to extract model from generated code")

            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            if quantize:
                if quantization_type == 'int8':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    if representative_dataset:
                        converter.representative_dataset = representative_dataset
                        converter.target_spec.supported_ops = [
                            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                        ]
                        converter.inference_input_type = tf.int8
                        converter.inference_output_type = tf.int8
                elif quantization_type == 'float16':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                elif quantization_type == 'dynamic':
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_model = converter.convert()

            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            logger.info(f"TensorFlow Lite model exported: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"TensorFlow is required for TFLite export: {e}")
            raise ImportError(
                "TensorFlow is required for TFLite conversion. "
                "Install with: pip install tensorflow"
            )

    def export_torchscript(
        self,
        output_path: str = 'model.pt',
        method: str = 'trace'
    ) -> str:
        """
        Export PyTorch model to TorchScript format.
        
        Parameters
        ----------
        output_path : str
            Path to save the TorchScript model
        method : str
            Export method ('trace' or 'script')
            
        Returns
        -------
        str
            Path to the exported model
        """
        try:
            import torch

            from neural.code_generation.code_generator import generate_code

            code = generate_code(self.model_data, 'pytorch')

            exec_globals = {}
            exec(code, exec_globals)
            model = exec_globals.get('model')

            if model is None:
                raise ValueError("Failed to extract model from generated code")

            model.eval()

            input_shape = (1,) + tuple(self.model_data['input']['shape'])
            dummy_input = torch.randn(input_shape)

            if method == 'trace':
                traced_model = torch.jit.trace(model, dummy_input)
            elif method == 'script':
                traced_model = torch.jit.script(model)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")

            traced_model.save(output_path)

            logger.info(f"TorchScript model exported: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"PyTorch is required for TorchScript export: {e}")
            raise ImportError(
                "PyTorch is required for TorchScript conversion. "
                "Install with: pip install torch"
            )

    def export_savedmodel(
        self,
        output_path: str = 'saved_model'
    ) -> str:
        """
        Export TensorFlow model to SavedModel format for TF Serving.
        
        Parameters
        ----------
        output_path : str
            Directory path to save the SavedModel
            
        Returns
        -------
        str
            Path to the exported model directory
        """
        try:
            import tensorflow as tf

            from neural.code_generation.code_generator import generate_code

            code = generate_code(self.model_data, 'tensorflow')

            exec_globals = {}
            exec(code, exec_globals)
            model = exec_globals.get('model')

            if model is None:
                raise ValueError("Failed to extract model from generated code")

            tf.saved_model.save(model, output_path)

            logger.info(f"SavedModel exported: {output_path}")
            return output_path

        except ImportError as e:
            logger.error(f"TensorFlow is required for SavedModel export: {e}")
            raise ImportError(
                "TensorFlow is required for SavedModel conversion. "
                "Install with: pip install tensorflow"
            )

    def create_torchserve_config(
        self,
        model_path: str,
        model_name: str,
        output_dir: str = '.',
        handler: str = 'image_classifier',
        batch_size: int = 1,
        max_batch_delay: int = 100
    ) -> Tuple[str, str]:
        """
        Create configuration files for TorchServe deployment.
        
        Parameters
        ----------
        model_path : str
            Path to the TorchScript model
        model_name : str
            Name for the deployed model
        output_dir : str
            Output directory for config files
        handler : str
            Handler type ('image_classifier', 'text_classifier', 'object_detector', 'default')
        batch_size : int
            Batch size for inference
        max_batch_delay : int
            Maximum batch delay in milliseconds
            
        Returns
        -------
        tuple
            Paths to config.properties and model-store directory
        """
        os.makedirs(output_dir, exist_ok=True)

        config_path = os.path.join(output_dir, 'config.properties')
        config_content = f"""inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=100
model_store={os.path.join(output_dir, 'model-store')}
model_snapshot={{"name":"startup.cfg","modelCount":1,"models":{{"{model_name}":{{"1.0":{{"defaultVersion":true,"marName":"{model_name}.mar","minWorkers":1,"maxWorkers":4,"batchSize":{batch_size},"maxBatchDelay":{max_batch_delay},"responseTimeout":120}}}}}}}}
"""

        with open(config_path, 'w') as f:
            f.write(config_content)

        model_store_dir = os.path.join(output_dir, 'model-store')
        os.makedirs(model_store_dir, exist_ok=True)

        logger.info(f"TorchServe config created: {config_path}")
        return config_path, model_store_dir

    def create_tfserving_config(
        self,
        model_path: str,
        model_name: str,
        output_dir: str = '.',
        version: int = 1
    ) -> str:
        """
        Create configuration for TensorFlow Serving deployment.
        
        Parameters
        ----------
        model_path : str
            Path to the SavedModel directory
        model_name : str
            Name for the deployed model
        output_dir : str
            Output directory for config
        version : int
            Model version number
            
        Returns
        -------
        str
            Path to the config file
        """
        os.makedirs(output_dir, exist_ok=True)

        model_config_list = {
            "model_config_list": [
                {
                    "name": model_name,
                    "base_path": os.path.abspath(os.path.join(output_dir, model_name)),
                    "model_platform": "tensorflow",
                    "model_version_policy": {
                        "specific": {
                            "versions": [version]
                        }
                    }
                }
            ]
        }

        versioned_model_dir = os.path.join(output_dir, model_name, str(version))
        os.makedirs(versioned_model_dir, exist_ok=True)

        import shutil
        if os.path.exists(model_path):
            if os.path.isdir(model_path):
                shutil.copytree(model_path, versioned_model_dir, dirs_exist_ok=True)
            else:
                shutil.copy(model_path, versioned_model_dir)

        config_path = os.path.join(output_dir, 'models.config')
        with open(config_path, 'w') as f:
            json.dump(model_config_list, f, indent=2)

        logger.info(f"TensorFlow Serving config created: {config_path}")
        return config_path

    def generate_deployment_scripts(
        self,
        output_dir: str = '.',
        deployment_type: str = 'torchserve'
    ) -> List[str]:
        """
        Generate deployment scripts for the selected serving platform.
        
        Parameters
        ----------
        output_dir : str
            Output directory for scripts
        deployment_type : str
            Type of deployment ('torchserve' or 'tfserving')
            
        Returns
        -------
        list
            Paths to generated scripts
        """
        os.makedirs(output_dir, exist_ok=True)
        scripts = []

        if deployment_type == 'torchserve':
            scripts.extend(self._generate_torchserve_scripts(output_dir))
        elif deployment_type == 'tfserving':
            scripts.extend(self._generate_tfserving_scripts(output_dir))
        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")

        return scripts

    def _generate_torchserve_scripts(self, output_dir: str) -> List[str]:
        """Generate TorchServe deployment scripts."""
        scripts = []

        start_script = os.path.join(output_dir, 'start_torchserve.sh')
        with open(start_script, 'w') as f:
            f.write("""#!/bin/bash
# Start TorchServe with the model

torchserve --start \\
  --model-store model-store \\
  --models all \\
  --ts-config config.properties

echo "TorchServe started. Access at http://localhost:8080"
""")
        os.chmod(start_script, 0o755)
        scripts.append(start_script)

        stop_script = os.path.join(output_dir, 'stop_torchserve.sh')
        with open(stop_script, 'w') as f:
            f.write("""#!/bin/bash
# Stop TorchServe

torchserve --stop

echo "TorchServe stopped"
""")
        os.chmod(stop_script, 0o755)
        scripts.append(stop_script)

        test_script = os.path.join(output_dir, 'test_inference.py')
        with open(test_script, 'w') as f:
            f.write("""import requests
import json
import numpy as np

def test_inference(url="http://localhost:8080/predictions/model", data=None):
    \"\"\"Test inference endpoint.\"\"\"
    if data is None:
        # Generate sample data
        data = np.random.randn(1, 28, 28, 1).tolist()
    
    response = requests.post(url, json={"data": data})
    
    if response.status_code == 200:
        logger.info("Inference successful!")
        logger.info("Response: %s", json.dumps(response.json(), indent=2))
    else:
        logger.error("Inference failed with status code: %d", response.status_code)
        logger.error("Response: %s", response.text)

if __name__ == "__main__":
    test_inference()
""")
        scripts.append(test_script)

        logger.info(f"TorchServe deployment scripts generated in {output_dir}")
        return scripts

    def _generate_tfserving_scripts(self, output_dir: str) -> List[str]:
        """Generate TensorFlow Serving deployment scripts."""
        scripts = []

        docker_compose = os.path.join(output_dir, 'docker-compose.yml')
        with open(docker_compose, 'w') as f:
            f.write("""version: '3'
services:
  tensorflow-serving:
    image: tensorflow/serving:latest
    ports:
      - "8501:8501"
      - "8500:8500"
    volumes:
      - ./models:/models
    environment:
      - MODEL_NAME=model
    command: ["--model_config_file=/models/models.config"]
""")
        scripts.append(docker_compose)

        start_script = os.path.join(output_dir, 'start_tfserving.sh')
        with open(start_script, 'w') as f:
            f.write("""#!/bin/bash
# Start TensorFlow Serving with Docker

docker-compose up -d

echo "TensorFlow Serving started. REST API at http://localhost:8501"
echo "gRPC API at localhost:8500"
""")
        os.chmod(start_script, 0o755)
        scripts.append(start_script)

        stop_script = os.path.join(output_dir, 'stop_tfserving.sh')
        with open(stop_script, 'w') as f:
            f.write("""#!/bin/bash
# Stop TensorFlow Serving

docker-compose down

echo "TensorFlow Serving stopped"
""")
        os.chmod(stop_script, 0o755)
        scripts.append(stop_script)

        test_script = os.path.join(output_dir, 'test_inference.py')
        with open(test_script, 'w') as f:
            f.write("""import requests
import json
import numpy as np

def test_inference(url="http://localhost:8501/v1/models/model:predict", data=None):
    \"\"\"Test TensorFlow Serving REST API.\"\"\"
    if data is None:
        # Generate sample data
        data = np.random.randn(1, 28, 28, 1).tolist()
    
    payload = {
        "instances": data
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        logger.info("Inference successful!")
        logger.info("Response: %s", json.dumps(response.json(), indent=2))
    else:
        logger.error("Inference failed with status code: %d", response.status_code)
        logger.error("Response: %s", response.text)

if __name__ == "__main__":
    test_inference()
""")
        scripts.append(test_script)

        logger.info(f"TensorFlow Serving deployment scripts generated in {output_dir}")
        return scripts
