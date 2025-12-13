import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BottleneckAnalyzer:
    def __init__(self, threshold_percentile: float = 90.0):
        self.threshold_percentile = threshold_percentile
        self.layer_data = defaultdict(lambda: {
            'execution_times': [],
            'memory_usage': [],
            'io_operations': [],
            'gpu_utilization': [],
        })
        self.bottlenecks = []
        self.recommendations = []

    def record_layer_metrics(self, layer_name: str, metrics: Dict[str, Any]):
        data = self.layer_data[layer_name]
        
        if 'execution_time' in metrics:
            data['execution_times'].append(metrics['execution_time'])
        
        if 'memory_usage' in metrics:
            data['memory_usage'].append(metrics['memory_usage'])
        
        if 'io_operations' in metrics:
            data['io_operations'].append(metrics['io_operations'])
        
        if 'gpu_utilization' in metrics:
            data['gpu_utilization'].append(metrics['gpu_utilization'])

    def analyze(self) -> Dict[str, Any]:
        self.bottlenecks.clear()
        self.recommendations.clear()
        
        all_execution_times = []
        for layer_name, data in self.layer_data.items():
            if data['execution_times']:
                all_execution_times.extend(data['execution_times'])
        
        if not all_execution_times:
            return {
                'bottlenecks': [],
                'recommendations': [],
                'summary': {'total_bottlenecks': 0},
            }
        
        threshold = np.percentile(all_execution_times, self.threshold_percentile)
        
        for layer_name, data in self.layer_data.items():
            if not data['execution_times']:
                continue
            
            mean_time = np.mean(data['execution_times'])
            
            if mean_time >= threshold:
                bottleneck = self._analyze_layer_bottleneck(layer_name, data, mean_time, threshold)
                self.bottlenecks.append(bottleneck)
                
                recommendations = self._generate_recommendations(layer_name, bottleneck)
                self.recommendations.extend(recommendations)
        
        self.bottlenecks.sort(key=lambda x: x['severity_score'], reverse=True)
        
        return {
            'bottlenecks': self.bottlenecks,
            'recommendations': self.recommendations,
            'summary': self._generate_summary(),
        }

    def _analyze_layer_bottleneck(self, layer_name: str, data: Dict, mean_time: float, threshold: float) -> Dict[str, Any]:
        bottleneck = {
            'layer': layer_name,
            'mean_execution_time': mean_time,
            'execution_time_std': np.std(data['execution_times']),
            'threshold': threshold,
            'overhead_percentage': ((mean_time - threshold) / threshold) * 100,
        }
        
        bottleneck_type = []
        severity_factors = []
        
        if mean_time > threshold * 1.5:
            bottleneck_type.append('compute')
            severity_factors.append(2.0)
        
        if data['memory_usage']:
            mean_memory = np.mean(data['memory_usage'])
            if mean_memory > 500:
                bottleneck_type.append('memory')
                severity_factors.append(1.5)
            bottleneck['mean_memory_usage'] = mean_memory
        
        if data['io_operations']:
            io_intensity = np.mean(data['io_operations'])
            if io_intensity > 100:
                bottleneck_type.append('io')
                severity_factors.append(1.3)
            bottleneck['io_intensity'] = io_intensity
        
        if data['gpu_utilization']:
            gpu_util = np.mean(data['gpu_utilization'])
            if gpu_util < 50:
                bottleneck_type.append('underutilization')
                severity_factors.append(1.2)
            bottleneck['gpu_utilization'] = gpu_util
        
        bottleneck['bottleneck_type'] = bottleneck_type if bottleneck_type else ['unknown']
        bottleneck['severity_score'] = mean_time * (np.mean(severity_factors) if severity_factors else 1.0)
        
        return bottleneck

    def _generate_recommendations(self, layer_name: str, bottleneck: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []
        
        for btype in bottleneck['bottleneck_type']:
            if btype == 'compute':
                recommendations.append({
                    'layer': layer_name,
                    'type': 'optimization',
                    'category': 'compute',
                    'priority': 'high',
                    'recommendation': f"Consider using optimized operations or reducing layer complexity for {layer_name}",
                    'details': "High compute time detected. Try kernel fusion, quantization, or pruning.",
                })
            
            elif btype == 'memory':
                recommendations.append({
                    'layer': layer_name,
                    'type': 'optimization',
                    'category': 'memory',
                    'priority': 'high',
                    'recommendation': f"Optimize memory usage in {layer_name}",
                    'details': "High memory usage. Consider gradient checkpointing or reducing batch size.",
                })
            
            elif btype == 'io':
                recommendations.append({
                    'layer': layer_name,
                    'type': 'optimization',
                    'category': 'io',
                    'priority': 'medium',
                    'recommendation': f"Reduce I/O operations in {layer_name}",
                    'details': "High I/O intensity. Consider data prefetching or caching.",
                })
            
            elif btype == 'underutilization':
                recommendations.append({
                    'layer': layer_name,
                    'type': 'optimization',
                    'category': 'gpu',
                    'priority': 'medium',
                    'recommendation': f"Improve GPU utilization for {layer_name}",
                    'details': "Low GPU utilization. Check for CPU bottlenecks or increase batch size.",
                })
        
        if bottleneck.get('execution_time_std', 0) / bottleneck.get('mean_execution_time', 1) > 0.3:
            recommendations.append({
                'layer': layer_name,
                'type': 'stability',
                'category': 'variance',
                'priority': 'low',
                'recommendation': f"High variance in execution time for {layer_name}",
                'details': "Execution time is inconsistent. Check for dynamic behavior or external factors.",
            })
        
        return recommendations

    def _generate_summary(self) -> Dict[str, Any]:
        if not self.bottlenecks:
            return {
                'total_bottlenecks': 0,
                'critical_bottlenecks': 0,
                'total_overhead_ms': 0,
            }
        
        total_overhead = sum(b['mean_execution_time'] - b['threshold'] for b in self.bottlenecks)
        critical_bottlenecks = sum(1 for b in self.bottlenecks if b['overhead_percentage'] > 50)
        
        bottleneck_types = defaultdict(int)
        for b in self.bottlenecks:
            for btype in b['bottleneck_type']:
                bottleneck_types[btype] += 1
        
        return {
            'total_bottlenecks': len(self.bottlenecks),
            'critical_bottlenecks': critical_bottlenecks,
            'total_overhead_ms': total_overhead * 1000,
            'bottleneck_types': dict(bottleneck_types),
            'top_bottlenecks': [
                {'layer': b['layer'], 'overhead_percentage': b['overhead_percentage']}
                for b in self.bottlenecks[:5]
            ],
        }

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        return self.bottlenecks.copy()

    def get_recommendations(self) -> List[Dict[str, Any]]:
        return self.recommendations.copy()

    def reset(self):
        self.layer_data.clear()
        self.bottlenecks.clear()
        self.recommendations.clear()
