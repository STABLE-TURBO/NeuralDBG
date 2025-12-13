"""
Training cost estimation for cloud providers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


@dataclass
class InstanceType:
    """Cloud instance type configuration."""
    
    name: str
    provider: CloudProvider
    cpu_count: int
    memory_gb: float
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    price_per_hour: float = 0.0
    spot_price_per_hour: float = 0.0
    performance_score: float = 1.0
    
    def __hash__(self):
        return hash((self.name, self.provider.value))


@dataclass
class CostEstimate:
    """Cost estimation result."""
    
    provider: CloudProvider
    instance_type: InstanceType
    estimated_hours: float
    on_demand_cost: float
    spot_cost: float
    potential_savings: float
    storage_cost: float = 0.0
    data_transfer_cost: float = 0.0
    total_on_demand_cost: float = 0.0
    total_spot_cost: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_on_demand_cost = (
            self.on_demand_cost + self.storage_cost + self.data_transfer_cost
        )
        self.total_spot_cost = (
            self.spot_cost + self.storage_cost + self.data_transfer_cost
        )
        self.potential_savings = self.total_on_demand_cost - self.total_spot_cost


class CostEstimator:
    """Estimate training costs for different cloud providers."""
    
    def __init__(self, pricing_data_path: Optional[str] = None):
        """
        Initialize cost estimator.
        
        Parameters
        ----------
        pricing_data_path : str, optional
            Path to custom pricing data file
        """
        self.instance_types: Dict[CloudProvider, List[InstanceType]] = {
            CloudProvider.AWS: self._get_aws_instances(),
            CloudProvider.GCP: self._get_gcp_instances(),
            CloudProvider.AZURE: self._get_azure_instances(),
        }
        
        if pricing_data_path:
            self._load_pricing_data(pricing_data_path)
    
    def estimate_cost(
        self,
        provider: CloudProvider,
        instance_name: str,
        training_hours: float,
        storage_gb: float = 100.0,
        data_transfer_gb: float = 10.0,
        use_spot: bool = False
    ) -> CostEstimate:
        """
        Estimate training cost.
        
        Parameters
        ----------
        provider : CloudProvider
            Cloud provider
        instance_name : str
            Instance type name
        training_hours : float
            Estimated training duration in hours
        storage_gb : float
            Storage required in GB
        data_transfer_gb : float
            Data transfer in GB
        use_spot : bool
            Whether to use spot/preemptible instances
            
        Returns
        -------
        CostEstimate
            Cost estimation result
        """
        instance = self._find_instance(provider, instance_name)
        
        if not instance:
            raise ValueError(f"Instance type {instance_name} not found for {provider.value}")
        
        compute_cost = instance.price_per_hour * training_hours
        spot_compute_cost = instance.spot_price_per_hour * training_hours
        
        storage_cost = self._calculate_storage_cost(provider, storage_gb, training_hours)
        transfer_cost = self._calculate_transfer_cost(provider, data_transfer_gb)
        
        return CostEstimate(
            provider=provider,
            instance_type=instance,
            estimated_hours=training_hours,
            on_demand_cost=compute_cost,
            spot_cost=spot_compute_cost,
            potential_savings=compute_cost - spot_compute_cost,
            storage_cost=storage_cost,
            data_transfer_cost=transfer_cost,
            breakdown={
                'compute_on_demand': compute_cost,
                'compute_spot': spot_compute_cost,
                'storage': storage_cost,
                'data_transfer': transfer_cost,
            }
        )
    
    def compare_providers(
        self,
        gpu_count: int,
        training_hours: float,
        gpu_type: Optional[str] = None
    ) -> List[CostEstimate]:
        """
        Compare costs across providers.
        
        Parameters
        ----------
        gpu_count : int
            Number of GPUs required
        training_hours : float
            Training duration in hours
        gpu_type : str, optional
            Preferred GPU type (V100, A100, etc.)
            
        Returns
        -------
        list
            Cost estimates sorted by total cost
        """
        estimates = []
        
        for provider in CloudProvider:
            instances = self._find_matching_instances(
                provider, gpu_count, gpu_type
            )
            
            for instance in instances:
                estimate = self.estimate_cost(
                    provider,
                    instance.name,
                    training_hours
                )
                estimates.append(estimate)
        
        return sorted(estimates, key=lambda x: x.total_spot_cost)
    
    def get_cheapest_option(
        self,
        gpu_count: int,
        training_hours: float,
        max_cost: Optional[float] = None
    ) -> Optional[CostEstimate]:
        """
        Find cheapest instance option.
        
        Parameters
        ----------
        gpu_count : int
            Number of GPUs required
        training_hours : float
            Training duration in hours
        max_cost : float, optional
            Maximum acceptable cost
            
        Returns
        -------
        CostEstimate, optional
            Cheapest option or None if no valid option found
        """
        estimates = self.compare_providers(gpu_count, training_hours)
        
        if not estimates:
            return None
        
        if max_cost is not None:
            estimates = [e for e in estimates if e.total_spot_cost <= max_cost]
        
        return estimates[0] if estimates else None
    
    def _find_instance(
        self,
        provider: CloudProvider,
        instance_name: str
    ) -> Optional[InstanceType]:
        """Find instance by name."""
        instances = self.instance_types.get(provider, [])
        for instance in instances:
            if instance.name == instance_name:
                return instance
        return None
    
    def _find_matching_instances(
        self,
        provider: CloudProvider,
        gpu_count: int,
        gpu_type: Optional[str] = None
    ) -> List[InstanceType]:
        """Find instances matching criteria."""
        instances = self.instance_types.get(provider, [])
        
        matching = []
        for instance in instances:
            if instance.gpu_count != gpu_count:
                continue
            
            if gpu_type and instance.gpu_type:
                if gpu_type.lower() not in instance.gpu_type.lower():
                    continue
            
            matching.append(instance)
        
        return matching
    
    def _calculate_storage_cost(
        self,
        provider: CloudProvider,
        storage_gb: float,
        hours: float
    ) -> float:
        """Calculate storage cost."""
        monthly_rates = {
            CloudProvider.AWS: 0.10,
            CloudProvider.GCP: 0.08,
            CloudProvider.AZURE: 0.09,
        }
        
        rate = monthly_rates.get(provider, 0.10)
        days = hours / 24.0
        return (storage_gb * rate * days) / 30.0
    
    def _calculate_transfer_cost(
        self,
        provider: CloudProvider,
        transfer_gb: float
    ) -> float:
        """Calculate data transfer cost."""
        rates_per_gb = {
            CloudProvider.AWS: 0.09,
            CloudProvider.GCP: 0.08,
            CloudProvider.AZURE: 0.087,
        }
        
        rate = rates_per_gb.get(provider, 0.09)
        
        if transfer_gb <= 10:
            return 0.0
        
        return (transfer_gb - 10) * rate
    
    def _get_aws_instances(self) -> List[InstanceType]:
        """Get AWS instance types with pricing."""
        return [
            InstanceType(
                name="p3.2xlarge",
                provider=CloudProvider.AWS,
                cpu_count=8,
                memory_gb=61,
                gpu_type="V100",
                gpu_count=1,
                price_per_hour=3.06,
                spot_price_per_hour=0.92,
                performance_score=1.0
            ),
            InstanceType(
                name="p3.8xlarge",
                provider=CloudProvider.AWS,
                cpu_count=32,
                memory_gb=244,
                gpu_type="V100",
                gpu_count=4,
                price_per_hour=12.24,
                spot_price_per_hour=3.67,
                performance_score=4.0
            ),
            InstanceType(
                name="p4d.24xlarge",
                provider=CloudProvider.AWS,
                cpu_count=96,
                memory_gb=1152,
                gpu_type="A100",
                gpu_count=8,
                price_per_hour=32.77,
                spot_price_per_hour=9.83,
                performance_score=10.0
            ),
            InstanceType(
                name="g4dn.xlarge",
                provider=CloudProvider.AWS,
                cpu_count=4,
                memory_gb=16,
                gpu_type="T4",
                gpu_count=1,
                price_per_hour=0.526,
                spot_price_per_hour=0.158,
                performance_score=0.5
            ),
            InstanceType(
                name="g5.xlarge",
                provider=CloudProvider.AWS,
                cpu_count=4,
                memory_gb=16,
                gpu_type="A10G",
                gpu_count=1,
                price_per_hour=1.006,
                spot_price_per_hour=0.302,
                performance_score=0.8
            ),
        ]
    
    def _get_gcp_instances(self) -> List[InstanceType]:
        """Get GCP instance types with pricing."""
        return [
            InstanceType(
                name="n1-standard-8-v100",
                provider=CloudProvider.GCP,
                cpu_count=8,
                memory_gb=30,
                gpu_type="V100",
                gpu_count=1,
                price_per_hour=2.48,
                spot_price_per_hour=0.74,
                performance_score=1.0
            ),
            InstanceType(
                name="n1-standard-32-v100-x4",
                provider=CloudProvider.GCP,
                cpu_count=32,
                memory_gb=120,
                gpu_type="V100",
                gpu_count=4,
                price_per_hour=9.92,
                spot_price_per_hour=2.98,
                performance_score=4.0
            ),
            InstanceType(
                name="a2-highgpu-1g",
                provider=CloudProvider.GCP,
                cpu_count=12,
                memory_gb=85,
                gpu_type="A100",
                gpu_count=1,
                price_per_hour=3.67,
                spot_price_per_hour=1.10,
                performance_score=1.3
            ),
            InstanceType(
                name="a2-highgpu-8g",
                provider=CloudProvider.GCP,
                cpu_count=96,
                memory_gb=680,
                gpu_type="A100",
                gpu_count=8,
                price_per_hour=29.39,
                spot_price_per_hour=8.82,
                performance_score=10.0
            ),
            InstanceType(
                name="n1-standard-4-t4",
                provider=CloudProvider.GCP,
                cpu_count=4,
                memory_gb=15,
                gpu_type="T4",
                gpu_count=1,
                price_per_hour=0.47,
                spot_price_per_hour=0.14,
                performance_score=0.5
            ),
        ]
    
    def _get_azure_instances(self) -> List[InstanceType]:
        """Get Azure instance types with pricing."""
        return [
            InstanceType(
                name="Standard_NC6s_v3",
                provider=CloudProvider.AZURE,
                cpu_count=6,
                memory_gb=112,
                gpu_type="V100",
                gpu_count=1,
                price_per_hour=3.06,
                spot_price_per_hour=0.92,
                performance_score=1.0
            ),
            InstanceType(
                name="Standard_NC24s_v3",
                provider=CloudProvider.AZURE,
                cpu_count=24,
                memory_gb=448,
                gpu_type="V100",
                gpu_count=4,
                price_per_hour=12.24,
                spot_price_per_hour=3.67,
                performance_score=4.0
            ),
            InstanceType(
                name="Standard_ND96asr_v4",
                provider=CloudProvider.AZURE,
                cpu_count=96,
                memory_gb=900,
                gpu_type="A100",
                gpu_count=8,
                price_per_hour=27.20,
                spot_price_per_hour=8.16,
                performance_score=10.0
            ),
            InstanceType(
                name="Standard_NC4as_T4_v3",
                provider=CloudProvider.AZURE,
                cpu_count=4,
                memory_gb=28,
                gpu_type="T4",
                gpu_count=1,
                price_per_hour=0.526,
                spot_price_per_hour=0.158,
                performance_score=0.5
            ),
        ]
    
    def _load_pricing_data(self, filepath: str):
        """Load custom pricing data from file."""
        path = Path(filepath)
        
        if not path.exists():
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for provider_name, instances in data.items():
            try:
                provider = CloudProvider(provider_name)
                
                instance_list = []
                for inst_data in instances:
                    instance = InstanceType(
                        name=inst_data['name'],
                        provider=provider,
                        cpu_count=inst_data['cpu_count'],
                        memory_gb=inst_data['memory_gb'],
                        gpu_type=inst_data.get('gpu_type'),
                        gpu_count=inst_data.get('gpu_count', 0),
                        price_per_hour=inst_data['price_per_hour'],
                        spot_price_per_hour=inst_data.get('spot_price_per_hour', 0.0),
                        performance_score=inst_data.get('performance_score', 1.0)
                    )
                    instance_list.append(instance)
                
                self.instance_types[provider] = instance_list
                
            except (ValueError, KeyError):
                continue
    
    def save_pricing_data(self, filepath: str):
        """Save current pricing data to file."""
        data = {}
        
        for provider, instances in self.instance_types.items():
            provider_data = []
            for inst in instances:
                provider_data.append({
                    'name': inst.name,
                    'cpu_count': inst.cpu_count,
                    'memory_gb': inst.memory_gb,
                    'gpu_type': inst.gpu_type,
                    'gpu_count': inst.gpu_count,
                    'price_per_hour': inst.price_per_hour,
                    'spot_price_per_hour': inst.spot_price_per_hour,
                    'performance_score': inst.performance_score,
                })
            data[provider.value] = provider_data
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
