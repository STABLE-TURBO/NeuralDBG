import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ComparativeProfiler:
    def __init__(self):
        self.backend_profiles = {}
        self.comparison_results = {}

    def add_backend_profile(self, backend_name: str, profile_data: Dict[str, Any]):
        self.backend_profiles[backend_name] = {
            'profile_data': profile_data,
            'timestamp': time.time(),
        }

    def compare_backends(self, backends: Optional[List[str]] = None) -> Dict[str, Any]:
        if backends is None:
            backends = list(self.backend_profiles.keys())
        
        if len(backends) < 2:
            return {
                'error': 'Need at least 2 backends to compare',
                'available_backends': list(self.backend_profiles.keys()),
            }
        
        comparison = {
            'backends': backends,
            'layer_comparisons': {},
            'summary': {},
        }
        
        all_layers = set()
        for backend in backends:
            if backend not in self.backend_profiles:
                continue
            profile = self.backend_profiles[backend]['profile_data']
            if 'layer_stats' in profile:
                all_layers.update(profile['layer_stats'].keys())
        
        for layer in all_layers:
            layer_comparison = self._compare_layer_across_backends(layer, backends)
            if layer_comparison:
                comparison['layer_comparisons'][layer] = layer_comparison
        
        comparison['summary'] = self._generate_comparison_summary(backends)
        
        self.comparison_results = comparison
        return comparison

    def _compare_layer_across_backends(self, layer_name: str, backends: List[str]) -> Dict[str, Any]:
        layer_data = {}
        
        for backend in backends:
            if backend not in self.backend_profiles:
                continue
            
            profile = self.backend_profiles[backend]['profile_data']
            if 'layer_stats' not in profile:
                continue
            
            layer_stats = profile['layer_stats'].get(layer_name, {})
            if not layer_stats:
                continue
            
            layer_data[backend] = {
                'mean_time': layer_stats.get('mean_time', 0),
                'std_time': layer_stats.get('std_time', 0),
                'total_time': layer_stats.get('total_time', 0),
                'call_count': layer_stats.get('call_count', 0),
            }
        
        if len(layer_data) < 2:
            return {}
        
        times = {backend: data['mean_time'] for backend, data in layer_data.items()}
        fastest_backend = min(times, key=times.get)
        slowest_backend = max(times, key=times.get)
        
        speedup = times[slowest_backend] / times[fastest_backend] if times[fastest_backend] > 0 else 1.0
        
        return {
            'layer': layer_name,
            'backend_data': layer_data,
            'fastest_backend': fastest_backend,
            'slowest_backend': slowest_backend,
            'speedup': speedup,
            'time_difference_ms': (times[slowest_backend] - times[fastest_backend]) * 1000,
        }

    def _generate_comparison_summary(self, backends: List[str]) -> Dict[str, Any]:
        backend_totals = {}
        
        for backend in backends:
            if backend not in self.backend_profiles:
                continue
            
            profile = self.backend_profiles[backend]['profile_data']
            
            if 'summary' in profile:
                total_time = profile['summary'].get('total_time', 0)
            elif 'layer_stats' in profile:
                total_time = sum(
                    stats.get('total_time', 0) 
                    for stats in profile['layer_stats'].values()
                )
            else:
                total_time = 0
            
            backend_totals[backend] = total_time
        
        if not backend_totals:
            return {}
        
        fastest_backend = min(backend_totals, key=backend_totals.get)
        slowest_backend = max(backend_totals, key=backend_totals.get)
        
        overall_speedup = (
            backend_totals[slowest_backend] / backend_totals[fastest_backend]
            if backend_totals[fastest_backend] > 0 else 1.0
        )
        
        wins_by_backend = defaultdict(int)
        if self.comparison_results and 'layer_comparisons' in self.comparison_results:
            for layer_comp in self.comparison_results['layer_comparisons'].values():
                fastest = layer_comp.get('fastest_backend')
                if fastest:
                    wins_by_backend[fastest] += 1
        
        return {
            'fastest_backend': fastest_backend,
            'slowest_backend': slowest_backend,
            'overall_speedup': overall_speedup,
            'total_times': backend_totals,
            'layer_wins': dict(wins_by_backend),
            'recommendation': self._generate_recommendation(wins_by_backend, backend_totals),
        }

    def _generate_recommendation(self, wins: Dict[str, int], totals: Dict[str, float]) -> str:
        if not wins or not totals:
            return "Insufficient data for recommendation"
        
        most_wins = max(wins, key=wins.get)
        fastest_overall = min(totals, key=totals.get)
        
        if most_wins == fastest_overall:
            return f"Use {most_wins} - fastest overall and wins most layer comparisons"
        else:
            return (
                f"Consider {fastest_overall} for overall speed or {most_wins} "
                f"for consistent layer performance"
            )

    def get_backend_ranking(self) -> List[Tuple[str, float]]:
        rankings = []
        
        for backend, data in self.backend_profiles.items():
            profile = data['profile_data']
            
            if 'summary' in profile:
                total_time = profile['summary'].get('total_time', float('inf'))
            elif 'layer_stats' in profile:
                total_time = sum(
                    stats.get('total_time', 0)
                    for stats in profile['layer_stats'].values()
                )
            else:
                total_time = float('inf')
            
            rankings.append((backend, total_time))
        
        rankings.sort(key=lambda x: x[1])
        return rankings

    def export_comparison(self) -> Dict[str, Any]:
        return {
            'backend_profiles': {
                backend: {
                    'timestamp': data['timestamp'],
                    'profile_summary': data['profile_data'].get('summary', {}),
                }
                for backend, data in self.backend_profiles.items()
            },
            'comparison_results': self.comparison_results,
            'backend_ranking': self.get_backend_ranking(),
        }

    def reset(self):
        self.backend_profiles.clear()
        self.comparison_results.clear()
