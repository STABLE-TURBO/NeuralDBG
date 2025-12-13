"""
Automatic resource right-sizing recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neural.cost.estimator import CloudProvider, CostEstimator, InstanceType


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float = 0.0
    gpu_memory_utilization: float = 0.0
    network_bandwidth_mbps: float = 0.0
    disk_iops: float = 0.0
    samples: int = 0
    
    def average_with(self, other: ResourceMetrics) -> ResourceMetrics:
        """Average metrics with another sample."""
        total_samples = self.samples + other.samples
        
        return ResourceMetrics(
            cpu_utilization=(
                self.cpu_utilization * self.samples + 
                other.cpu_utilization * other.samples
            ) / total_samples,
            memory_utilization=(
                self.memory_utilization * self.samples + 
                other.memory_utilization * other.samples
            ) / total_samples,
            gpu_utilization=(
                self.gpu_utilization * self.samples + 
                other.gpu_utilization * other.samples
            ) / total_samples,
            gpu_memory_utilization=(
                self.gpu_memory_utilization * self.samples + 
                other.gpu_memory_utilization * other.samples
            ) / total_samples,
            network_bandwidth_mbps=(
                self.network_bandwidth_mbps * self.samples + 
                other.network_bandwidth_mbps * other.samples
            ) / total_samples,
            disk_iops=(
                self.disk_iops * self.samples + 
                other.disk_iops * other.samples
            ) / total_samples,
            samples=total_samples
        )


@dataclass
class ResourceRecommendation:
    """Resource optimization recommendation."""
    
    current_instance: InstanceType
    recommended_instance: InstanceType
    current_cost_per_hour: float
    recommended_cost_per_hour: float
    estimated_savings_percent: float
    reason: str
    metrics: ResourceMetrics
    confidence_score: float = 0.0
    warnings: List[str] = field(default_factory=list)


class ResourceOptimizer:
    """Provide resource right-sizing recommendations."""
    
    def __init__(self, cost_estimator: Optional[CostEstimator] = None):
        """
        Initialize resource optimizer.
        
        Parameters
        ----------
        cost_estimator : CostEstimator, optional
            Cost estimator instance
        """
        self.cost_estimator = cost_estimator or CostEstimator()
        self.utilization_history: List[ResourceMetrics] = []
        
    def analyze_utilization(
        self,
        current_instance: InstanceType,
        metrics: ResourceMetrics
    ) -> ResourceRecommendation:
        """
        Analyze resource utilization and provide recommendation.
        
        Parameters
        ----------
        current_instance : InstanceType
            Current instance type
        metrics : ResourceMetrics
            Resource utilization metrics
            
        Returns
        -------
        ResourceRecommendation
            Right-sizing recommendation
        """
        self.utilization_history.append(metrics)
        
        avg_metrics = self._calculate_average_metrics()
        
        overprovisioned = self._detect_overprovisioning(avg_metrics)
        underprovisioned = self._detect_underprovisioning(avg_metrics)
        
        if overprovisioned:
            return self._recommend_downsize(current_instance, avg_metrics)
        elif underprovisioned:
            return self._recommend_upsize(current_instance, avg_metrics)
        else:
            return self._recommend_current(current_instance, avg_metrics)
    
    def batch_analyze(
        self,
        instances: List[InstanceType],
        metrics_list: List[ResourceMetrics]
    ) -> List[ResourceRecommendation]:
        """
        Analyze multiple instances.
        
        Parameters
        ----------
        instances : list
            List of instance types
        metrics_list : list
            List of metrics for each instance
            
        Returns
        -------
        list
            Recommendations for each instance
        """
        recommendations = []
        
        for instance, metrics in zip(instances, metrics_list):
            recommendation = self.analyze_utilization(instance, metrics)
            recommendations.append(recommendation)
        
        return recommendations
    
    def predict_optimal_instance(
        self,
        provider: CloudProvider,
        workload_profile: Dict[str, Any]
    ) -> InstanceType:
        """
        Predict optimal instance for workload.
        
        Parameters
        ----------
        provider : CloudProvider
            Cloud provider
        workload_profile : dict
            Workload characteristics (batch_size, model_size, etc.)
            
        Returns
        -------
        InstanceType
            Recommended instance type
        """
        required_memory_gb = workload_profile.get('model_memory_gb', 10)
        requires_gpu = workload_profile.get('requires_gpu', True)
        gpu_count = workload_profile.get('gpu_count', 1)
        
        instances = self.cost_estimator.instance_types.get(provider, [])
        
        suitable_instances = []
        for instance in instances:
            if requires_gpu and instance.gpu_count < gpu_count:
                continue
            
            if instance.memory_gb < required_memory_gb:
                continue
            
            suitable_instances.append(instance)
        
        if not suitable_instances:
            raise ValueError("No suitable instances found for workload")
        
        suitable_instances.sort(key=lambda x: x.price_per_hour)
        
        return suitable_instances[0]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get optimization summary.
        
        Returns
        -------
        dict
            Summary of optimization opportunities
        """
        if not self.utilization_history:
            return {
                'status': 'no_data',
                'message': 'No utilization data available'
            }
        
        avg_metrics = self._calculate_average_metrics()
        
        return {
            'avg_cpu_utilization': avg_metrics.cpu_utilization,
            'avg_memory_utilization': avg_metrics.memory_utilization,
            'avg_gpu_utilization': avg_metrics.gpu_utilization,
            'avg_gpu_memory_utilization': avg_metrics.gpu_memory_utilization,
            'samples_collected': len(self.utilization_history),
            'overprovisioned': self._detect_overprovisioning(avg_metrics),
            'underprovisioned': self._detect_underprovisioning(avg_metrics),
        }
    
    def _calculate_average_metrics(self) -> ResourceMetrics:
        """Calculate average metrics from history."""
        if not self.utilization_history:
            return ResourceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        result = self.utilization_history[0]
        for metrics in self.utilization_history[1:]:
            result = result.average_with(metrics)
        
        return result
    
    def _detect_overprovisioning(self, metrics: ResourceMetrics) -> bool:
        """Detect if resources are overprovisioned."""
        thresholds = {
            'cpu': 0.3,
            'memory': 0.4,
            'gpu': 0.3,
        }
        
        checks = [
            metrics.cpu_utilization < thresholds['cpu'],
            metrics.memory_utilization < thresholds['memory'],
            metrics.gpu_utilization < thresholds['gpu'] if metrics.gpu_utilization > 0 else False,
        ]
        
        return sum(checks) >= 2
    
    def _detect_underprovisioning(self, metrics: ResourceMetrics) -> bool:
        """Detect if resources are underprovisioned."""
        thresholds = {
            'cpu': 0.9,
            'memory': 0.85,
            'gpu': 0.9,
        }
        
        checks = [
            metrics.cpu_utilization > thresholds['cpu'],
            metrics.memory_utilization > thresholds['memory'],
            metrics.gpu_utilization > thresholds['gpu'] if metrics.gpu_utilization > 0 else False,
        ]
        
        return any(checks)
    
    def _recommend_downsize(
        self,
        current_instance: InstanceType,
        metrics: ResourceMetrics
    ) -> ResourceRecommendation:
        """Recommend smaller instance."""
        instances = self.cost_estimator.instance_types.get(
            current_instance.provider, []
        )
        
        candidates = []
        for instance in instances:
            if instance.price_per_hour >= current_instance.price_per_hour:
                continue
            
            if instance.gpu_count < current_instance.gpu_count:
                continue
            
            if instance.gpu_type and current_instance.gpu_type:
                if instance.gpu_type != current_instance.gpu_type:
                    continue
            
            candidates.append(instance)
        
        if not candidates:
            return self._recommend_current(current_instance, metrics)
        
        candidates.sort(key=lambda x: x.price_per_hour, reverse=True)
        recommended = candidates[0]
        
        savings = (
            (current_instance.price_per_hour - recommended.price_per_hour) /
            current_instance.price_per_hour * 100
        )
        
        return ResourceRecommendation(
            current_instance=current_instance,
            recommended_instance=recommended,
            current_cost_per_hour=current_instance.price_per_hour,
            recommended_cost_per_hour=recommended.price_per_hour,
            estimated_savings_percent=savings,
            reason="Resources are underutilized, consider smaller instance",
            metrics=metrics,
            confidence_score=self._calculate_confidence(metrics),
            warnings=[]
        )
    
    def _recommend_upsize(
        self,
        current_instance: InstanceType,
        metrics: ResourceMetrics
    ) -> ResourceRecommendation:
        """Recommend larger instance."""
        instances = self.cost_estimator.instance_types.get(
            current_instance.provider, []
        )
        
        candidates = []
        for instance in instances:
            if instance.price_per_hour <= current_instance.price_per_hour:
                continue
            
            if instance.gpu_count < current_instance.gpu_count:
                continue
            
            candidates.append(instance)
        
        if not candidates:
            return self._recommend_current(current_instance, metrics)
        
        candidates.sort(key=lambda x: x.price_per_hour)
        recommended = candidates[0]
        
        cost_increase = (
            (recommended.price_per_hour - current_instance.price_per_hour) /
            current_instance.price_per_hour * 100
        )
        
        return ResourceRecommendation(
            current_instance=current_instance,
            recommended_instance=recommended,
            current_cost_per_hour=current_instance.price_per_hour,
            recommended_cost_per_hour=recommended.price_per_hour,
            estimated_savings_percent=-cost_increase,
            reason="Resources are constrained, consider larger instance",
            metrics=metrics,
            confidence_score=self._calculate_confidence(metrics),
            warnings=["This will increase costs but improve performance"]
        )
    
    def _recommend_current(
        self,
        current_instance: InstanceType,
        metrics: ResourceMetrics
    ) -> ResourceRecommendation:
        """Recommend keeping current instance."""
        return ResourceRecommendation(
            current_instance=current_instance,
            recommended_instance=current_instance,
            current_cost_per_hour=current_instance.price_per_hour,
            recommended_cost_per_hour=current_instance.price_per_hour,
            estimated_savings_percent=0.0,
            reason="Current instance is appropriately sized",
            metrics=metrics,
            confidence_score=self._calculate_confidence(metrics),
            warnings=[]
        )
    
    def _calculate_confidence(self, metrics: ResourceMetrics) -> float:
        """Calculate confidence score for recommendation."""
        if metrics.samples < 10:
            return 0.5
        elif metrics.samples < 50:
            return 0.7
        else:
            return 0.9
