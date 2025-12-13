import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np


class DistributedTrainingProfiler:
    def __init__(self):
        self.node_profiles = {}
        self.communication_logs = []
        self.synchronization_events = []
        self.load_balance_metrics = defaultdict(list)

    def add_node_profile(self, node_id: str, profile_data: Dict[str, Any]):
        self.node_profiles[node_id] = {
            'profile_data': profile_data,
            'timestamp': time.time(),
            'node_id': node_id,
        }

    def record_communication(self, source_node: str, dest_node: str, 
                           data_size_mb: float, duration_ms: float, 
                           operation_type: str = 'transfer'):
        comm_event = {
            'source': source_node,
            'destination': dest_node,
            'data_size_mb': data_size_mb,
            'duration_ms': duration_ms,
            'bandwidth_mbps': (data_size_mb / duration_ms) * 1000 if duration_ms > 0 else 0,
            'operation_type': operation_type,
            'timestamp': time.time(),
        }
        self.communication_logs.append(comm_event)

    def record_synchronization(self, node_ids: List[str], sync_type: str, 
                              duration_ms: float, barrier_wait_ms: Optional[float] = None):
        sync_event = {
            'nodes': node_ids,
            'sync_type': sync_type,
            'duration_ms': duration_ms,
            'barrier_wait_ms': barrier_wait_ms,
            'timestamp': time.time(),
        }
        self.synchronization_events.append(sync_event)

    def record_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        self.load_balance_metrics[node_id].append({
            'timestamp': time.time(),
            **metrics
        })

    def analyze_load_balance(self) -> Dict[str, Any]:
        if not self.node_profiles:
            return {'error': 'No node profiles available'}
        
        node_times = {}
        for node_id, data in self.node_profiles.items():
            profile = data['profile_data']
            if 'summary' in profile:
                node_times[node_id] = profile['summary'].get('total_time', 0)
        
        if not node_times:
            return {'error': 'No timing data available'}
        
        times = list(node_times.values())
        mean_time = np.mean(times)
        std_time = np.std(times)
        max_time = max(times)
        min_time = min(times)
        
        imbalance_factor = (max_time - min_time) / mean_time if mean_time > 0 else 0
        
        slowest_node = max(node_times, key=node_times.get)
        fastest_node = min(node_times, key=node_times.get)
        
        return {
            'node_times': node_times,
            'mean_time': mean_time,
            'std_time': std_time,
            'max_time': max_time,
            'min_time': min_time,
            'imbalance_factor': imbalance_factor,
            'slowest_node': slowest_node,
            'fastest_node': fastest_node,
            'is_balanced': imbalance_factor < 0.1,
        }

    def analyze_communication_overhead(self) -> Dict[str, Any]:
        if not self.communication_logs:
            return {'total_communication_time_ms': 0, 'event_count': 0}
        
        total_time = sum(event['duration_ms'] for event in self.communication_logs)
        total_data = sum(event['data_size_mb'] for event in self.communication_logs)
        
        bandwidths = [event['bandwidth_mbps'] for event in self.communication_logs]
        mean_bandwidth = np.mean(bandwidths)
        
        by_operation = defaultdict(lambda: {'count': 0, 'total_time_ms': 0, 'total_data_mb': 0})
        for event in self.communication_logs:
            op_type = event['operation_type']
            by_operation[op_type]['count'] += 1
            by_operation[op_type]['total_time_ms'] += event['duration_ms']
            by_operation[op_type]['total_data_mb'] += event['data_size_mb']
        
        return {
            'total_communication_time_ms': total_time,
            'total_data_transferred_mb': total_data,
            'event_count': len(self.communication_logs),
            'mean_bandwidth_mbps': mean_bandwidth,
            'by_operation_type': dict(by_operation),
        }

    def analyze_synchronization_overhead(self) -> Dict[str, Any]:
        if not self.synchronization_events:
            return {'total_sync_time_ms': 0, 'event_count': 0}
        
        total_sync_time = sum(event['duration_ms'] for event in self.synchronization_events)
        total_barrier_wait = sum(
            event['barrier_wait_ms'] for event in self.synchronization_events 
            if event['barrier_wait_ms'] is not None
        )
        
        by_sync_type = defaultdict(lambda: {'count': 0, 'total_time_ms': 0})
        for event in self.synchronization_events:
            sync_type = event['sync_type']
            by_sync_type[sync_type]['count'] += 1
            by_sync_type[sync_type]['total_time_ms'] += event['duration_ms']
        
        return {
            'total_sync_time_ms': total_sync_time,
            'total_barrier_wait_ms': total_barrier_wait,
            'event_count': len(self.synchronization_events),
            'by_sync_type': dict(by_sync_type),
        }

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        load_balance = self.analyze_load_balance()
        comm_overhead = self.analyze_communication_overhead()
        sync_overhead = self.analyze_synchronization_overhead()
        
        bottlenecks = []
        
        if load_balance.get('imbalance_factor', 0) > 0.2:
            bottlenecks.append({
                'type': 'load_imbalance',
                'severity': 'high' if load_balance['imbalance_factor'] > 0.5 else 'medium',
                'details': f"Load imbalance factor: {load_balance['imbalance_factor']:.2f}",
                'recommendation': f"Optimize workload distribution. Slowest node: {load_balance.get('slowest_node')}",
            })
        
        total_node_time = load_balance.get('mean_time', 1) * len(self.node_profiles)
        comm_percentage = (comm_overhead.get('total_communication_time_ms', 0) / 1000) / total_node_time * 100 if total_node_time > 0 else 0
        
        if comm_percentage > 20:
            bottlenecks.append({
                'type': 'communication_overhead',
                'severity': 'high' if comm_percentage > 40 else 'medium',
                'details': f"Communication overhead: {comm_percentage:.1f}% of total time",
                'recommendation': "Reduce communication frequency or increase computation per step",
            })
        
        sync_percentage = (sync_overhead.get('total_sync_time_ms', 0) / 1000) / total_node_time * 100 if total_node_time > 0 else 0
        
        if sync_percentage > 15:
            bottlenecks.append({
                'type': 'synchronization_overhead',
                'severity': 'high' if sync_percentage > 30 else 'medium',
                'details': f"Synchronization overhead: {sync_percentage:.1f}% of total time",
                'recommendation': "Consider asynchronous updates or gradient accumulation",
            })
        
        return {
            'bottlenecks': bottlenecks,
            'load_balance': load_balance,
            'communication_overhead': comm_overhead,
            'synchronization_overhead': sync_overhead,
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_nodes': len(self.node_profiles),
            'communication_events': len(self.communication_logs),
            'synchronization_events': len(self.synchronization_events),
            'load_balance': self.analyze_load_balance(),
            'communication_overhead': self.analyze_communication_overhead(),
            'synchronization_overhead': self.analyze_synchronization_overhead(),
            'bottleneck_analysis': self.get_bottleneck_analysis(),
        }

    def reset(self):
        self.node_profiles.clear()
        self.communication_logs.clear()
        self.synchronization_events.clear()
        self.load_balance_metrics.clear()
