from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from neural.shape_propagation.shape_propagator import ShapePropagator


class BaseCodeGenerator(ABC):
    def __init__(self, model_data: Dict[str, Any], best_params: Optional[Dict[str, Any]] = None, auto_flatten_output: bool = False):
        self.model_data = model_data
        self.best_params = best_params or {}
        self.auto_flatten_output = auto_flatten_output
        self.propagator = ShapePropagator(debug=False)
        self.current_input_shape = (None,) + tuple(model_data['input']['shape'])
        self.indent = "    "

    @abstractmethod
    def generate(self) -> str:
        pass

    @abstractmethod
    def generate_layer(self, layer_type: str, params: Dict[str, Any]) -> str:
        pass

    def expand_layers(self):
        expanded_layers = []
        for layer in self.model_data.get('layers', []):
            if not isinstance(layer, dict) or 'type' not in layer:
                raise ValueError(f"Invalid layer format: {layer}")
            multiply = layer.get('multiply', 1)
            if not isinstance(multiply, int) or multiply < 1:
                raise ValueError(f"Invalid 'multiply' value: {multiply}")
            layer_copy = layer.copy()
            if 'multiply' in layer_copy:
                del layer_copy['multiply']
            for _ in range(multiply):
                expanded_layers.append(layer_copy.copy())
        return expanded_layers
