import gc
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import psutil
import numpy as np


class MemoryProfiler:
    def __init__(self):
        self.memory_snapshots = []
        self.layer_memory = defaultdict(list)
        self.baseline_memory = None
        self.peak_memory = 0
        
        try:
            import torch
            self.torch_available = torch.cuda.is_available()
        except ImportError:
            self.torch_available = False

    def start_profiling(self):
        gc.collect()
        if self.torch_available:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        self.baseline_memory = self._get_current_memory()

    def take_snapshot(self, layer_name: str, metadata: Optional[Dict] = None):
        memory_info = self._get_current_memory()
        
        snapshot = {
            'layer': layer_name,
            'timestamp': time.time(),
            'cpu_memory_mb': memory_info['cpu_memory_mb'],
            'memory_delta_mb': memory_info['cpu_memory_mb'] - (self.baseline_memory['cpu_memory_mb'] if self.baseline_memory else 0),
        }
        
        if self.torch_available:
            snapshot.update({
                'gpu_memory_mb': memory_info.get('gpu_memory_mb', 0),
                'gpu_reserved_mb': memory_info.get('gpu_reserved_mb', 0),
                'gpu_peak_mb': memory_info.get('gpu_peak_mb', 0),
            })
        
        if metadata:
            snapshot.update(metadata)
        
        self.memory_snapshots.append(snapshot)
        self.layer_memory[layer_name].append(snapshot)
        
        total_memory = snapshot['cpu_memory_mb']
        if self.torch_available:
            total_memory += snapshot.get('gpu_memory_mb', 0)
        self.peak_memory = max(self.peak_memory, total_memory)
        
        return snapshot

    def _get_current_memory(self) -> Dict[str, float]:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            'cpu_memory_mb': memory_info.rss / (1024 ** 2),
            'cpu_memory_percent': process.memory_percent(),
        }
        
        if self.torch_available:
            import torch
            result.update({
                'gpu_memory_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'gpu_reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
                'gpu_peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            })
        
        return result

    def get_memory_growth(self) -> List[Dict[str, Any]]:
        if not self.memory_snapshots:
            return []
        
        baseline = self.memory_snapshots[0]['cpu_memory_mb']
        growth = []
        
        for snapshot in self.memory_snapshots:
            growth.append({
                'layer': snapshot['layer'],
                'timestamp': snapshot['timestamp'],
                'memory_growth_mb': snapshot['cpu_memory_mb'] - baseline,
            })
        
        return growth

    def get_layer_memory_stats(self, layer_name: str) -> Dict[str, Any]:
        snapshots = self.layer_memory.get(layer_name, [])
        if not snapshots:
            return {}
        
        memory_values = [s['cpu_memory_mb'] for s in snapshots]
        
        stats = {
            'layer': layer_name,
            'call_count': len(snapshots),
            'mean_memory_mb': np.mean(memory_values),
            'std_memory_mb': np.std(memory_values),
            'min_memory_mb': np.min(memory_values),
            'max_memory_mb': np.max(memory_values),
        }
        
        if self.torch_available and 'gpu_memory_mb' in snapshots[0]:
            gpu_memory = [s['gpu_memory_mb'] for s in snapshots]
            stats.update({
                'mean_gpu_memory_mb': np.mean(gpu_memory),
                'max_gpu_memory_mb': np.max(gpu_memory),
            })
        
        return stats

    def get_summary(self) -> Dict[str, Any]:
        if not self.memory_snapshots:
            return {}
        
        summary = {
            'baseline_memory_mb': self.baseline_memory['cpu_memory_mb'] if self.baseline_memory else 0,
            'peak_memory_mb': self.peak_memory,
            'total_snapshots': len(self.memory_snapshots),
            'memory_growth': self.get_memory_growth(),
        }
        
        if self.torch_available:
            gpu_snapshots = [s for s in self.memory_snapshots if 'gpu_memory_mb' in s]
            if gpu_snapshots:
                summary['peak_gpu_memory_mb'] = max(s['gpu_memory_mb'] for s in gpu_snapshots)
        
        return summary

    def reset(self):
        self.memory_snapshots.clear()
        self.layer_memory.clear()
        self.baseline_memory = None
        self.peak_memory = 0


class MemoryLeakDetector:
    def __init__(self, threshold_mb: float = 10.0, window_size: int = 10):
        self.threshold_mb = threshold_mb
        self.window_size = window_size
        self.memory_history = []
        self.potential_leaks = []

    def add_measurement(self, layer_name: str, memory_mb: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        
        self.memory_history.append({
            'layer': layer_name,
            'memory_mb': memory_mb,
            'timestamp': timestamp,
        })
        
        if len(self.memory_history) >= self.window_size:
            self._check_for_leaks()

    def _check_for_leaks(self):
        recent_memory = [m['memory_mb'] for m in self.memory_history[-self.window_size:]]
        
        if len(recent_memory) < 2:
            return
        
        memory_increase = recent_memory[-1] - recent_memory[0]
        
        if memory_increase > self.threshold_mb:
            leak_info = {
                'start_memory_mb': recent_memory[0],
                'end_memory_mb': recent_memory[-1],
                'increase_mb': memory_increase,
                'window_size': self.window_size,
                'layers_involved': [m['layer'] for m in self.memory_history[-self.window_size:]],
                'timestamp': time.time(),
            }
            
            slope = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            leak_info['growth_rate_mb_per_step'] = slope
            
            self.potential_leaks.append(leak_info)

    def get_leaks(self) -> List[Dict[str, Any]]:
        return self.potential_leaks.copy()

    def has_leaks(self) -> bool:
        return len(self.potential_leaks) > 0

    def get_summary(self) -> Dict[str, Any]:
        if not self.potential_leaks:
            return {
                'has_leaks': False,
                'leak_count': 0,
            }
        
        return {
            'has_leaks': True,
            'leak_count': len(self.potential_leaks),
            'total_memory_leaked_mb': sum(leak['increase_mb'] for leak in self.potential_leaks),
            'average_growth_rate': np.mean([leak['growth_rate_mb_per_step'] for leak in self.potential_leaks]),
            'leaks': self.potential_leaks,
        }

    def reset(self):
        self.memory_history.clear()
        self.potential_leaks.clear()
