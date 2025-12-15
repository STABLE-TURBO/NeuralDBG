from __future__ import annotations

from typing import Any, Dict, Union

from neural.exceptions import DependencyError

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

try:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    SummaryWriter = None

from neural.code_generation.code_generator import generate_code

class TensorBoardLogger:
    def __init__(self, log_dir: str = "runs/neural") -> None:
        if not HAS_TORCH:
            raise DependencyError(
                dependency="torch",
                feature="TensorBoard logging",
                install_hint="pip install torch tensorboard"
            )
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_model(self, model: Union['tf.keras.Model', 'torch.nn.Module'], step: int) -> None:
        if HAS_TENSORFLOW and isinstance(model, tf.keras.Model):
            model.summary(print_fn=lambda x: self.writer.add_text("model_summary", x))
        elif HAS_TORCH and isinstance(model, torch.nn.Module):
            self.writer.add_graph(model, torch.randn(1, *model.input_shape))

class ShapePropagator:
    def __init__(self, debug: bool = False) -> None:
        self.tensorboard_logger = TensorBoardLogger()

    def propagate(self, input_shape: tuple, layer: Dict[str, Any], framework: str) -> tuple:
        resources = self.monitor_resources()
        self.tensorboard_logger.log_metrics({
            "cpu_usage": resources["cpu_usage"],
            "memory_usage": resources["memory_usage"],
            "gpu_memory": resources["gpu_memory"],
            "io_usage": resources["io_usage"],
            "execution_time": trace_entry["execution_time"]
        }, self.current_layer)
        if self.current_layer == 0:
            self.tensorboard_logger.log_model(self.model, 0)
