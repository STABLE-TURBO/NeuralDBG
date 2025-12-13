import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np


class LayerProfiler:
    def __init__(self):
        self.layer_stats = defaultdict(lambda: {
            'execution_times': [],
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'mean_time': 0.0,
            'std_time': 0.0,
        })
        self.current_layer = None
        self.start_time = None
        self.execution_history = []
        self.cumulative_time = 0.0

    def start_layer(self, layer_name: str):
        self.current_layer = layer_name
        self.start_time = time.perf_counter()

    def end_layer(self, layer_name: str, metadata: Optional[Dict[str, Any]] = None):
        if self.current_layer != layer_name:
            raise ValueError(f"Layer mismatch: expected {self.current_layer}, got {layer_name}")
        
        execution_time = time.perf_counter() - self.start_time
        self.cumulative_time += execution_time
        
        stats = self.layer_stats[layer_name]
        stats['execution_times'].append(execution_time)
        stats['call_count'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        execution_times = stats['execution_times']
        stats['mean_time'] = np.mean(execution_times)
        stats['std_time'] = np.std(execution_times)
        
        execution_record = {
            'layer': layer_name,
            'execution_time': execution_time,
            'timestamp': time.time(),
            'cumulative_time': self.cumulative_time,
        }
        
        if metadata:
            execution_record.update(metadata)
        
        self.execution_history.append(execution_record)
        
        self.current_layer = None
        self.start_time = None

    def get_layer_stats(self, layer_name: str) -> Dict[str, Any]:
        return dict(self.layer_stats.get(layer_name, {}))

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {layer: dict(stats) for layer, stats in self.layer_stats.items()}

    def get_slowest_layers(self, top_n: int = 5) -> List[tuple]:
        sorted_layers = sorted(
            self.layer_stats.items(),
            key=lambda x: x[1]['mean_time'],
            reverse=True
        )
        return [(name, stats['mean_time']) for name, stats in sorted_layers[:top_n]]

    def get_most_variable_layers(self, top_n: int = 5) -> List[tuple]:
        sorted_layers = sorted(
            self.layer_stats.items(),
            key=lambda x: x[1]['std_time'],
            reverse=True
        )
        return [(name, stats['std_time']) for name, stats in sorted_layers[:top_n]]

    def get_execution_timeline(self) -> List[Dict[str, Any]]:
        return self.execution_history.copy()

    def get_summary(self) -> Dict[str, Any]:
        total_layers = len(self.layer_stats)
        total_time = sum(stats['total_time'] for stats in self.layer_stats.values())
        total_calls = sum(stats['call_count'] for stats in self.layer_stats.values())
        
        return {
            'total_layers': total_layers,
            'total_time': total_time,
            'total_calls': total_calls,
            'average_time_per_call': total_time / total_calls if total_calls > 0 else 0,
            'slowest_layers': self.get_slowest_layers(),
            'most_variable_layers': self.get_most_variable_layers(),
        }

    def reset(self):
        self.layer_stats.clear()
        self.execution_history.clear()
        self.cumulative_time = 0.0
        self.current_layer = None
        self.start_time = None

    def export_to_dict(self) -> Dict[str, Any]:
        return {
            'layer_stats': {
                layer: {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                       for k, v in stats.items()}
                for layer, stats in self.layer_stats.items()
            },
            'execution_history': self.execution_history,
            'summary': self.get_summary(),
        }
