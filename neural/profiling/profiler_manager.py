import json
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from .layer_profiler import LayerProfiler
from .memory_profiler import MemoryLeakDetector, MemoryProfiler
from .bottleneck_analyzer import BottleneckAnalyzer
from .comparative_profiler import ComparativeProfiler
from .distributed_profiler import DistributedTrainingProfiler
from .gpu_profiler import GPUUtilizationProfiler


class ProfilerManager:
    def __init__(self, enable_all: bool = True):
        self.layer_profiler = LayerProfiler() if enable_all else None
        self.memory_profiler = MemoryProfiler() if enable_all else None
        self.memory_leak_detector = MemoryLeakDetector() if enable_all else None
        self.bottleneck_analyzer = BottleneckAnalyzer() if enable_all else None
        self.comparative_profiler = ComparativeProfiler() if enable_all else None
        self.distributed_profiler = DistributedTrainingProfiler() if enable_all else None
        self.gpu_profiler = GPUUtilizationProfiler() if enable_all else None
        
        self.profiling_active = False
        self.start_timestamp = None
        self.end_timestamp = None

    def enable_profiler(self, profiler_name: str):
        profilers = {
            'layer': LayerProfiler,
            'memory': MemoryProfiler,
            'leak': MemoryLeakDetector,
            'bottleneck': BottleneckAnalyzer,
            'comparative': ComparativeProfiler,
            'distributed': DistributedTrainingProfiler,
            'gpu': GPUUtilizationProfiler,
        }
        
        if profiler_name not in profilers:
            raise ValueError(f"Unknown profiler: {profiler_name}")
        
        attr_name = f"{profiler_name}_profiler" if profiler_name != 'leak' else 'memory_leak_detector'
        setattr(self, attr_name, profilers[profiler_name]())

    def start_profiling(self):
        self.profiling_active = True
        self.start_timestamp = time.time()
        
        if self.memory_profiler:
            self.memory_profiler.start_profiling()

    def end_profiling(self):
        self.profiling_active = False
        self.end_timestamp = time.time()

    def profile_layer(self, layer_name: str, metadata: Optional[Dict[str, Any]] = None):
        if not self.profiling_active:
            return
        
        if self.layer_profiler:
            self.layer_profiler.start_layer(layer_name)
        
        class LayerContext:
            def __init__(self, manager, layer_name, metadata):
                self.manager = manager
                self.layer_name = layer_name
                self.metadata = metadata or {}
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.manager.layer_profiler:
                    self.manager.layer_profiler.end_layer(self.layer_name, self.metadata)
                
                if self.manager.memory_profiler:
                    snapshot = self.manager.memory_profiler.take_snapshot(self.layer_name, self.metadata)
                    if self.manager.memory_leak_detector:
                        self.manager.memory_leak_detector.add_measurement(
                            self.layer_name,
                            snapshot['cpu_memory_mb']
                        )
                
                if self.manager.gpu_profiler:
                    self.manager.gpu_profiler.record_gpu_metrics(self.layer_name)
                
                if self.manager.bottleneck_analyzer and self.manager.layer_profiler:
                    layer_stats = self.manager.layer_profiler.get_layer_stats(self.layer_name)
                    if layer_stats and 'mean_time' in layer_stats:
                        metrics = {
                            'execution_time': layer_stats['mean_time'],
                        }
                        if self.metadata:
                            metrics.update(self.metadata)
                        self.manager.bottleneck_analyzer.record_layer_metrics(
                            self.layer_name,
                            metrics
                        )
        
        return LayerContext(self, layer_name, metadata)

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        if not self.bottleneck_analyzer:
            return {'error': 'Bottleneck analyzer not enabled'}
        
        return self.bottleneck_analyzer.analyze()

    def compare_backends(self, backend_name: str):
        if not self.comparative_profiler:
            self.comparative_profiler = ComparativeProfiler()
        
        if self.layer_profiler:
            profile_data = self.layer_profiler.export_to_dict()
            self.comparative_profiler.add_backend_profile(backend_name, profile_data)

    def get_comparison_results(self, backends: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.comparative_profiler:
            return {'error': 'Comparative profiler not enabled'}
        
        return self.comparative_profiler.compare_backends(backends)

    def get_comprehensive_report(self) -> Dict[str, Any]:
        report = {
            'profiling_session': {
                'start_time': self.start_timestamp,
                'end_time': self.end_timestamp,
                'duration_seconds': (self.end_timestamp - self.start_timestamp) if self.end_timestamp else None,
            }
        }
        
        if self.layer_profiler:
            report['layer_profiling'] = self.layer_profiler.get_summary()
        
        if self.memory_profiler:
            report['memory_profiling'] = self.memory_profiler.get_summary()
        
        if self.memory_leak_detector:
            report['memory_leaks'] = self.memory_leak_detector.get_summary()
        
        if self.bottleneck_analyzer:
            bottleneck_analysis = self.bottleneck_analyzer.analyze()
            report['bottlenecks'] = bottleneck_analysis
        
        if self.gpu_profiler:
            report['gpu_utilization'] = self.gpu_profiler.get_utilization_summary()
            report['gpu_recommendations'] = self.gpu_profiler.get_recommendations()
        
        if self.distributed_profiler and self.distributed_profiler.node_profiles:
            report['distributed_training'] = self.distributed_profiler.get_summary()
        
        if self.comparative_profiler and self.comparative_profiler.backend_profiles:
            report['backend_comparison'] = self.comparative_profiler.export_comparison()
        
        return report

    def export_report(self, filepath: str):
        report = self.get_comprehensive_report()
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

    def get_dashboard_data(self) -> Dict[str, Any]:
        data = {
            'timestamp': time.time(),
        }
        
        if self.layer_profiler:
            data['execution_history'] = self.layer_profiler.get_execution_timeline()
            data['layer_stats'] = self.layer_profiler.get_all_stats()
        
        if self.memory_profiler:
            data['memory_snapshots'] = self.memory_profiler.memory_snapshots
            data['memory_growth'] = self.memory_profiler.get_memory_growth()
        
        if self.bottleneck_analyzer:
            data['bottlenecks'] = self.bottleneck_analyzer.get_bottlenecks()
            data['recommendations'] = self.bottleneck_analyzer.get_recommendations()
        
        if self.gpu_profiler:
            data['gpu_metrics'] = self.gpu_profiler.utilization_history
            data['gpu_summary'] = self.gpu_profiler.get_utilization_summary()
        
        if self.distributed_profiler and self.distributed_profiler.node_profiles:
            data['distributed_metrics'] = {
                'load_balance': self.distributed_profiler.analyze_load_balance(),
                'communication': self.distributed_profiler.analyze_communication_overhead(),
            }
        
        return data

    def reset_all(self):
        if self.layer_profiler:
            self.layer_profiler.reset()
        if self.memory_profiler:
            self.memory_profiler.reset()
        if self.memory_leak_detector:
            self.memory_leak_detector.reset()
        if self.bottleneck_analyzer:
            self.bottleneck_analyzer.reset()
        if self.comparative_profiler:
            self.comparative_profiler.reset()
        if self.distributed_profiler:
            self.distributed_profiler.reset()
        if self.gpu_profiler:
            self.gpu_profiler.reset()
        
        self.profiling_active = False
        self.start_timestamp = None
        self.end_timestamp = None
