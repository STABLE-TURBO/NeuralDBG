import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional


class ExecutionManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.output_queue = Queue()
        self.metrics_queue = Queue()
        self.execution_thread: Optional[threading.Thread] = None
        self.is_running = False
        
    def compile_model(
        self,
        model_data: Dict[str, Any],
        backend: str,
        output_path: Optional[str] = None,
        auto_flatten: bool = False
    ) -> Dict[str, Any]:
        from neural.code_generation.code_generator import generate_code
        
        try:
            generated_code = generate_code(
                model_data,
                backend,
                auto_flatten_output=auto_flatten
            )
            
            if output_path is None:
                temp_dir = Path.home() / ".neural" / "aquarium" / "compiled"
                temp_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(temp_dir / f"model_{backend}_{int(time.time())}.py")
            
            with open(output_path, "w") as f:
                f.write(generated_code)
            
            return {
                "success": True,
                "script_path": output_path,
                "code_size": len(generated_code),
                "backend": backend
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_script(
        self,
        script_path: str,
        env_vars: Optional[Dict[str, str]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        if self.is_running:
            return False
        
        self.is_running = True
        self.execution_thread = threading.Thread(
            target=self._execute_script,
            args=(script_path, env_vars, callback),
            daemon=True
        )
        self.execution_thread.start()
        return True
    
    def _execute_script(
        self,
        script_path: str,
        env_vars: Optional[Dict[str, str]],
        callback: Optional[Callable[[str], None]]
    ):
        try:
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            for line in self.process.stdout:
                self.output_queue.put(line)
                if callback:
                    callback(line)
                
                self._parse_metrics(line)
            
            self.process.wait()
            
            exit_code = self.process.returncode
            if exit_code == 0:
                self.output_queue.put("\n✓ Execution completed successfully!\n")
            else:
                self.output_queue.put(f"\n✗ Execution failed with exit code {exit_code}\n")
                
        except Exception as e:
            self.output_queue.put(f"\n✗ Error: {str(e)}\n")
        finally:
            self.is_running = False
            self.process = None
    
    def _parse_metrics(self, line: str):
        try:
            if "loss:" in line.lower() or "accuracy:" in line.lower():
                metrics = {}
                
                if "epoch" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower().startswith("loss"):
                            try:
                                metrics["loss"] = float(parts[i].split(":")[-1].strip())
                            except (ValueError, IndexError):
                                pass
                        elif part.lower().startswith("acc"):
                            try:
                                metrics["accuracy"] = float(parts[i].split(":")[-1].strip())
                            except (ValueError, IndexError):
                                pass
                        elif part.lower().startswith("val_loss"):
                            try:
                                metrics["val_loss"] = float(parts[i].split(":")[-1].strip())
                            except (ValueError, IndexError):
                                pass
                        elif part.lower().startswith("val_acc"):
                            try:
                                metrics["val_accuracy"] = float(parts[i].split(":")[-1].strip())
                            except (ValueError, IndexError):
                                pass
                
                if metrics:
                    self.metrics_queue.put(metrics)
                    
        except Exception:
            pass
    
    def stop_execution(self) -> bool:
        if self.process and self.is_running:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.is_running = False
                return True
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.is_running = False
                return True
            except Exception:
                return False
        return False
    
    def get_output_lines(self, max_lines: int = 100) -> List[str]:
        lines = []
        try:
            while not self.output_queue.empty() and len(lines) < max_lines:
                lines.append(self.output_queue.get_nowait())
        except Empty:
            pass
        return lines
    
    def get_metrics(self) -> List[Dict[str, float]]:
        metrics_list = []
        try:
            while not self.metrics_queue.empty():
                metrics_list.append(self.metrics_queue.get_nowait())
        except Empty:
            pass
        return metrics_list
    
    def is_executing(self) -> bool:
        return self.is_running
    
    def export_script(
        self,
        source_path: str,
        destination_path: str,
        include_config: bool = True
    ) -> Dict[str, Any]:
        try:
            source = Path(source_path)
            destination = Path(destination_path)
            
            if not source.exists():
                return {"success": False, "error": "Source file not found"}
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            with open(source, "r") as src:
                code = src.read()
            
            with open(destination, "w") as dst:
                dst.write(code)
            
            if include_config:
                config_path = destination.with_suffix(".json")
                config = {
                    "source": str(source),
                    "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "neural_version": self._get_neural_version()
                }
                with open(config_path, "w") as cfg:
                    json.dump(config, cfg, indent=2)
            
            return {
                "success": True,
                "destination": str(destination),
                "size": destination.stat().st_size
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_neural_version(self) -> str:
        try:
            from neural.cli.version import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def open_in_editor(self, file_path: str) -> bool:
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            
            return True
            
        except Exception:
            return False


class DatasetManager:
    BUILTIN_DATASETS = {
        "MNIST": {
            "type": "image",
            "shape": (28, 28, 1),
            "classes": 10,
            "loader": "tensorflow.keras.datasets.mnist"
        },
        "CIFAR10": {
            "type": "image",
            "shape": (32, 32, 3),
            "classes": 10,
            "loader": "tensorflow.keras.datasets.cifar10"
        },
        "CIFAR100": {
            "type": "image",
            "shape": (32, 32, 3),
            "classes": 100,
            "loader": "tensorflow.keras.datasets.cifar100"
        },
        "ImageNet": {
            "type": "image",
            "shape": (224, 224, 3),
            "classes": 1000,
            "loader": "custom"
        }
    }
    
    @staticmethod
    def get_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
        return DatasetManager.BUILTIN_DATASETS.get(dataset_name)
    
    @staticmethod
    def list_datasets() -> List[str]:
        return list(DatasetManager.BUILTIN_DATASETS.keys())
    
    @staticmethod
    def validate_custom_dataset(path: str) -> Dict[str, Any]:
        path_obj = Path(path)
        
        if not path_obj.exists():
            return {"valid": False, "error": "Path does not exist"}
        
        if not path_obj.is_dir():
            return {"valid": False, "error": "Path is not a directory"}
        
        return {
            "valid": True,
            "path": str(path_obj),
            "type": "custom"
        }
