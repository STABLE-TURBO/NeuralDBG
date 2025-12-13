import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np


class GPUUtilizationProfiler:
    def __init__(self):
        self.utilization_history = []
        self.layer_gpu_stats = defaultdict(list)
        self.memory_timeline = []
        
        try:
            import torch
            self.torch_available = torch.cuda.is_available()
            if self.torch_available:
                self.device_count = torch.cuda.device_count()
                self.device_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
            else:
                self.device_count = 0
                self.device_names = []
        except ImportError:
            self.torch_available = False
            self.device_count = 0
            self.device_names = []

    def record_gpu_metrics(self, layer_name: str, device_id: int = 0):
        if not self.torch_available:
            return None
        
        import torch
        
        metrics = {
            'layer': layer_name,
            'device_id': device_id,
            'timestamp': time.time(),
            'memory_allocated_mb': torch.cuda.memory_allocated(device_id) / (1024 ** 2),
            'memory_reserved_mb': torch.cuda.memory_reserved(device_id) / (1024 ** 2),
            'memory_cached_mb': torch.cuda.memory_reserved(device_id) / (1024 ** 2),
        }
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics['gpu_utilization_percent'] = util.gpu
            metrics['memory_utilization_percent'] = util.memory
            
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics['temperature_celsius'] = temp
            
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            metrics['power_watts'] = power
            
            pynvml.nvmlShutdown()
        except Exception:
            metrics['gpu_utilization_percent'] = None
            metrics['memory_utilization_percent'] = None
        
        self.utilization_history.append(metrics)
        self.layer_gpu_stats[layer_name].append(metrics)
        
        return metrics

    def record_memory_snapshot(self, label: str = "", device_id: int = 0):
        if not self.torch_available:
            return None
        
        import torch
        
        snapshot = {
            'label': label,
            'device_id': device_id,
            'timestamp': time.time(),
            'memory_allocated_mb': torch.cuda.memory_allocated(device_id) / (1024 ** 2),
            'memory_reserved_mb': torch.cuda.memory_reserved(device_id) / (1024 ** 2),
            'max_memory_allocated_mb': torch.cuda.max_memory_allocated(device_id) / (1024 ** 2),
            'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device_id) / (1024 ** 2),
        }
        
        self.memory_timeline.append(snapshot)
        return snapshot

    def get_layer_gpu_stats(self, layer_name: str) -> Dict[str, Any]:
        stats_list = self.layer_gpu_stats.get(layer_name, [])
        if not stats_list:
            return {}
        
        memory_allocated = [s['memory_allocated_mb'] for s in stats_list]
        
        result = {
            'layer': layer_name,
            'call_count': len(stats_list),
            'mean_memory_allocated_mb': np.mean(memory_allocated),
            'max_memory_allocated_mb': np.max(memory_allocated),
            'min_memory_allocated_mb': np.min(memory_allocated),
        }
        
        gpu_utils = [s['gpu_utilization_percent'] for s in stats_list if s.get('gpu_utilization_percent') is not None]
        if gpu_utils:
            result['mean_gpu_utilization_percent'] = np.mean(gpu_utils)
            result['min_gpu_utilization_percent'] = np.min(gpu_utils)
            result['max_gpu_utilization_percent'] = np.max(gpu_utils)
        
        return result

    def get_utilization_summary(self) -> Dict[str, Any]:
        if not self.utilization_history:
            return {
                'torch_available': self.torch_available,
                'device_count': self.device_count,
                'measurements': 0,
            }
        
        gpu_utils = [m['gpu_utilization_percent'] for m in self.utilization_history 
                     if m.get('gpu_utilization_percent') is not None]
        
        memory_allocated = [m['memory_allocated_mb'] for m in self.utilization_history]
        
        summary = {
            'torch_available': self.torch_available,
            'device_count': self.device_count,
            'device_names': self.device_names,
            'measurements': len(self.utilization_history),
            'mean_memory_allocated_mb': np.mean(memory_allocated),
            'peak_memory_allocated_mb': np.max(memory_allocated),
        }
        
        if gpu_utils:
            summary.update({
                'mean_gpu_utilization_percent': np.mean(gpu_utils),
                'min_gpu_utilization_percent': np.min(gpu_utils),
                'max_gpu_utilization_percent': np.max(gpu_utils),
            })
            
            if np.mean(gpu_utils) < 50:
                summary['utilization_warning'] = 'Low GPU utilization detected'
        
        temps = [m['temperature_celsius'] for m in self.utilization_history 
                if m.get('temperature_celsius') is not None]
        if temps:
            summary['mean_temperature_celsius'] = np.mean(temps)
            summary['max_temperature_celsius'] = np.max(temps)
        
        powers = [m['power_watts'] for m in self.utilization_history 
                 if m.get('power_watts') is not None]
        if powers:
            summary['mean_power_watts'] = np.mean(powers)
            summary['max_power_watts'] = np.max(powers)
        
        return summary

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        return self.memory_timeline.copy()

    def detect_memory_inefficiency(self) -> Dict[str, Any]:
        if not self.memory_timeline:
            return {'has_issues': False}
        
        issues = []
        
        allocations = [s['memory_allocated_mb'] for s in self.memory_timeline]
        reservations = [s['memory_reserved_mb'] for s in self.memory_timeline]
        
        if len(allocations) > 1:
            allocation_variance = np.std(allocations) / np.mean(allocations) if np.mean(allocations) > 0 else 0
            
            if allocation_variance > 0.5:
                issues.append({
                    'type': 'high_variance',
                    'description': 'High variance in memory allocation',
                    'variance': allocation_variance,
                })
        
        for i, snapshot in enumerate(self.memory_timeline):
            allocated = snapshot['memory_allocated_mb']
            reserved = snapshot['memory_reserved_mb']
            
            if reserved > 0 and allocated / reserved < 0.5:
                issues.append({
                    'type': 'underutilization',
                    'snapshot_index': i,
                    'label': snapshot['label'],
                    'allocated_mb': allocated,
                    'reserved_mb': reserved,
                    'utilization_ratio': allocated / reserved,
                })
        
        max_allocated = max(s['max_memory_allocated_mb'] for s in self.memory_timeline)
        current_allocated = self.memory_timeline[-1]['memory_allocated_mb']
        
        if max_allocated > current_allocated * 2:
            issues.append({
                'type': 'peak_memory',
                'description': 'Peak memory usage significantly higher than current',
                'max_allocated_mb': max_allocated,
                'current_allocated_mb': current_allocated,
            })
        
        return {
            'has_issues': len(issues) > 0,
            'issue_count': len(issues),
            'issues': issues,
        }

    def get_recommendations(self) -> List[Dict[str, str]]:
        recommendations = []
        
        summary = self.get_utilization_summary()
        
        if summary.get('mean_gpu_utilization_percent', 100) < 50:
            recommendations.append({
                'category': 'utilization',
                'priority': 'high',
                'recommendation': 'Increase batch size or use mixed precision training to improve GPU utilization',
            })
        
        memory_issues = self.detect_memory_inefficiency()
        if memory_issues['has_issues']:
            for issue in memory_issues['issues']:
                if issue['type'] == 'underutilization':
                    recommendations.append({
                        'category': 'memory',
                        'priority': 'medium',
                        'recommendation': 'Consider clearing unused memory or reducing memory reservation',
                    })
                elif issue['type'] == 'peak_memory':
                    recommendations.append({
                        'category': 'memory',
                        'priority': 'high',
                        'recommendation': 'Optimize memory usage to reduce peak allocation',
                    })
        
        return recommendations

    def reset(self):
        self.utilization_history.clear()
        self.layer_gpu_stats.clear()
        self.memory_timeline.clear()
        
        if self.torch_available:
            import torch
            torch.cuda.reset_peak_memory_stats()
