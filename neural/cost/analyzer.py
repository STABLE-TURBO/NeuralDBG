"""
Cost-performance tradeoff analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neural.cost.estimator import CloudProvider, CostEstimator


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    
    accuracy: float
    training_time_hours: float
    inference_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    memory_usage_gb: float = 0.0
    
    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted performance score."""
        if weights is None:
            weights = {
                'accuracy': 0.4,
                'speed': 0.3,
                'latency': 0.2,
                'memory': 0.1
            }
        
        speed_score = 1.0 / (1.0 + self.training_time_hours / 10.0)
        latency_score = 1.0 / (1.0 + self.inference_latency_ms / 100.0)
        memory_score = 1.0 / (1.0 + self.memory_usage_gb / 50.0)
        
        return (
            weights.get('accuracy', 0.4) * self.accuracy +
            weights.get('speed', 0.3) * speed_score +
            weights.get('latency', 0.2) * latency_score +
            weights.get('memory', 0.1) * memory_score
        )


@dataclass
class CostPerformancePoint:
    """Cost-performance data point."""
    
    configuration: Dict[str, Any]
    cost: float
    performance: PerformanceMetrics
    efficiency_score: float = 0.0
    
    def __post_init__(self):
        if self.cost > 0:
            self.efficiency_score = self.performance.accuracy / self.cost
        else:
            self.efficiency_score = 0.0


@dataclass
class CostPerformanceAnalysis:
    """Cost-performance analysis result."""
    
    points: List[CostPerformancePoint]
    pareto_frontier: List[CostPerformancePoint]
    best_value: CostPerformancePoint
    best_performance: CostPerformancePoint
    best_cost: CostPerformancePoint
    recommendations: Dict[str, Any] = field(default_factory=dict)


class CostAnalyzer:
    """Analyze cost-performance tradeoffs."""
    
    def __init__(self, cost_estimator: Optional[CostEstimator] = None):
        """
        Initialize cost analyzer.
        
        Parameters
        ----------
        cost_estimator : CostEstimator, optional
            Cost estimator instance
        """
        self.cost_estimator = cost_estimator or CostEstimator()
        self.analysis_history: List[CostPerformanceAnalysis] = []
        
    def analyze_tradeoffs(
        self,
        configurations: List[Dict[str, Any]],
        performance_metrics: List[PerformanceMetrics],
        costs: List[float]
    ) -> CostPerformanceAnalysis:
        """
        Analyze cost-performance tradeoffs.
        
        Parameters
        ----------
        configurations : list
            List of configuration dictionaries
        performance_metrics : list
            Performance metrics for each configuration
        costs : list
            Costs for each configuration
            
        Returns
        -------
        CostPerformanceAnalysis
            Analysis results
        """
        points = [
            CostPerformancePoint(
                configuration=config,
                cost=cost,
                performance=perf
            )
            for config, perf, cost in zip(configurations, performance_metrics, costs)
        ]
        
        pareto_frontier = self._compute_pareto_frontier(points)
        
        best_value = max(points, key=lambda p: p.efficiency_score)
        best_performance = max(points, key=lambda p: p.performance.accuracy)
        best_cost = min(points, key=lambda p: p.cost)
        
        recommendations = self._generate_recommendations(
            points,
            pareto_frontier,
            best_value
        )
        
        analysis = CostPerformanceAnalysis(
            points=points,
            pareto_frontier=pareto_frontier,
            best_value=best_value,
            best_performance=best_performance,
            best_cost=best_cost,
            recommendations=recommendations
        )
        
        self.analysis_history.append(analysis)
        
        return analysis
    
    def optimize_for_budget(
        self,
        budget: float,
        configurations: List[Dict[str, Any]],
        performance_metrics: List[PerformanceMetrics],
        costs: List[float]
    ) -> Optional[CostPerformancePoint]:
        """
        Find best configuration within budget.
        
        Parameters
        ----------
        budget : float
            Maximum budget
        configurations : list
            Available configurations
        performance_metrics : list
            Performance metrics
        costs : list
            Costs
            
        Returns
        -------
        CostPerformancePoint, optional
            Best configuration within budget
        """
        points = [
            CostPerformancePoint(
                configuration=config,
                cost=cost,
                performance=perf
            )
            for config, perf, cost in zip(configurations, performance_metrics, costs)
            if cost <= budget
        ]
        
        if not points:
            return None
        
        return max(points, key=lambda p: p.performance.accuracy)
    
    def optimize_for_performance(
        self,
        target_accuracy: float,
        configurations: List[Dict[str, Any]],
        performance_metrics: List[PerformanceMetrics],
        costs: List[float]
    ) -> Optional[CostPerformancePoint]:
        """
        Find cheapest configuration meeting performance target.
        
        Parameters
        ----------
        target_accuracy : float
            Target accuracy
        configurations : list
            Available configurations
        performance_metrics : list
            Performance metrics
        costs : list
            Costs
            
        Returns
        -------
        CostPerformancePoint, optional
            Cheapest configuration meeting target
        """
        points = [
            CostPerformancePoint(
                configuration=config,
                cost=cost,
                performance=perf
            )
            for config, perf, cost in zip(configurations, performance_metrics, costs)
            if perf.accuracy >= target_accuracy
        ]
        
        if not points:
            return None
        
        return min(points, key=lambda p: p.cost)
    
    def estimate_scaling_curve(
        self,
        base_config: Dict[str, Any],
        base_performance: PerformanceMetrics,
        base_cost: float,
        scale_factors: List[float]
    ) -> List[Tuple[float, float, float]]:
        """
        Estimate scaling curve (cost vs performance vs scale).
        
        Parameters
        ----------
        base_config : dict
            Base configuration
        base_performance : PerformanceMetrics
            Base performance
        base_cost : float
            Base cost
        scale_factors : list
            Scale factors to evaluate
            
        Returns
        -------
        list
            List of (scale_factor, estimated_cost, estimated_performance) tuples
        """
        results = []
        
        for scale in scale_factors:
            estimated_cost = base_cost * scale
            
            estimated_accuracy = min(
                1.0,
                base_performance.accuracy * (1 + 0.1 * np.log(scale))
            )
            
            results.append((scale, estimated_cost, estimated_accuracy))
        
        return results
    
    def compare_providers(
        self,
        workload: Dict[str, Any],
        training_hours: float,
        gpu_count: int = 1
    ) -> Dict[CloudProvider, Dict[str, Any]]:
        """
        Compare costs across cloud providers.
        
        Parameters
        ----------
        workload : dict
            Workload specification
        training_hours : float
            Expected training duration
        gpu_count : int
            Number of GPUs required
            
        Returns
        -------
        dict
            Comparison by provider
        """
        comparisons = {}
        
        for provider in CloudProvider:
            estimates = self.cost_estimator.compare_providers(
                gpu_count,
                training_hours
            )
            
            provider_estimates = [
                e for e in estimates if e.provider == provider
            ]
            
            if provider_estimates:
                best = min(provider_estimates, key=lambda e: e.total_spot_cost)
                
                comparisons[provider] = {
                    'cheapest_instance': best.instance_type.name,
                    'spot_cost': best.total_spot_cost,
                    'on_demand_cost': best.total_on_demand_cost,
                    'savings_percent': (
                        (best.total_on_demand_cost - best.total_spot_cost) /
                        best.total_on_demand_cost * 100
                    ),
                    'all_options': len(provider_estimates)
                }
        
        return comparisons
    
    def calculate_roi(
        self,
        training_cost: float,
        improvement_accuracy: float,
        business_value_per_point: float = 1000.0
    ) -> Dict[str, float]:
        """
        Calculate ROI for model improvement.
        
        Parameters
        ----------
        training_cost : float
            Cost of training
        improvement_accuracy : float
            Accuracy improvement (e.g., 0.05 for 5%)
        business_value_per_point : float
            Business value per accuracy point
            
        Returns
        -------
        dict
            ROI metrics
        """
        value_gained = improvement_accuracy * 100 * business_value_per_point
        roi_percent = ((value_gained - training_cost) / training_cost) * 100
        payback_period_days = training_cost / (value_gained / 365)
        
        return {
            'training_cost': training_cost,
            'value_gained': value_gained,
            'roi_percent': roi_percent,
            'payback_period_days': payback_period_days,
            'net_benefit': value_gained - training_cost,
        }
    
    def _compute_pareto_frontier(
        self,
        points: List[CostPerformancePoint]
    ) -> List[CostPerformancePoint]:
        """Compute Pareto frontier."""
        if not points:
            return []
        
        sorted_points = sorted(points, key=lambda p: p.cost)
        
        frontier = [sorted_points[0]]
        
        for point in sorted_points[1:]:
            if point.performance.accuracy > frontier[-1].performance.accuracy:
                frontier.append(point)
        
        return frontier
    
    def _generate_recommendations(
        self,
        points: List[CostPerformancePoint],
        pareto_frontier: List[CostPerformancePoint],
        best_value: CostPerformancePoint
    ) -> Dict[str, Any]:
        """Generate recommendations."""
        avg_cost = np.mean([p.cost for p in points])
        avg_accuracy = np.mean([p.performance.accuracy for p in points])
        
        recommendations = {
            'summary': {
                'total_configurations': len(points),
                'pareto_optimal_configurations': len(pareto_frontier),
                'avg_cost': avg_cost,
                'avg_accuracy': avg_accuracy,
            },
            'best_value': {
                'configuration': best_value.configuration,
                'cost': best_value.cost,
                'accuracy': best_value.performance.accuracy,
                'efficiency_score': best_value.efficiency_score,
            },
            'insights': []
        }
        
        if best_value in pareto_frontier:
            recommendations['insights'].append(
                "Best value configuration is Pareto optimal"
            )
        
        cost_variance = np.std([p.cost for p in points])
        if cost_variance > avg_cost * 0.5:
            recommendations['insights'].append(
                "High cost variance detected - consider standardizing instances"
            )
        
        accuracy_variance = np.std([p.performance.accuracy for p in points])
        if accuracy_variance > 0.1:
            recommendations['insights'].append(
                "High performance variance - some configurations significantly outperform others"
            )
        
        return recommendations
