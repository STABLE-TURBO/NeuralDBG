"""
Spot instance orchestration for cost-efficient training.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from neural.cost.estimator import CloudProvider, InstanceType


logger = logging.getLogger(__name__)


class SpotStrategy(Enum):
    """Spot instance management strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@dataclass
class SpotInstance:
    """Spot instance information."""
    
    instance_id: str
    provider: CloudProvider
    instance_type: InstanceType
    status: str
    launch_time: float
    termination_time: Optional[float] = None
    cost_saved: float = 0.0
    interruptions: int = 0
    
    @property
    def uptime_hours(self) -> float:
        """Calculate uptime in hours."""
        end_time = self.termination_time or time.time()
        return (end_time - self.launch_time) / 3600.0


@dataclass
class CheckpointConfig:
    """Checkpoint configuration for fault tolerance."""
    
    enabled: bool = True
    frequency_minutes: int = 15
    storage_path: str = "checkpoints"
    max_checkpoints: int = 5
    auto_resume: bool = True


class SpotInstanceOrchestrator:
    """Manage spot instances for cost-efficient training."""
    
    def __init__(
        self,
        provider: CloudProvider,
        strategy: SpotStrategy = SpotStrategy.BALANCED,
        checkpoint_config: Optional[CheckpointConfig] = None,
        max_retries: int = 3
    ):
        """
        Initialize spot orchestrator.
        
        Parameters
        ----------
        provider : CloudProvider
            Cloud provider to use
        strategy : SpotStrategy
            Spot management strategy
        checkpoint_config : CheckpointConfig, optional
            Checkpoint configuration
        max_retries : int
            Maximum retry attempts on interruption
        """
        self.provider = provider
        self.strategy = strategy
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.max_retries = max_retries
        
        self.active_instances: Dict[str, SpotInstance] = {}
        self.instance_history: List[SpotInstance] = []
        self.total_cost_saved = 0.0
        self.total_interruptions = 0
        
    def launch_spot_instance(
        self,
        instance_type: InstanceType,
        max_bid_price: Optional[float] = None,
        availability_zones: Optional[List[str]] = None
    ) -> Optional[SpotInstance]:
        """
        Launch a spot instance.
        
        Parameters
        ----------
        instance_type : InstanceType
            Instance type to launch
        max_bid_price : float, optional
            Maximum bid price per hour
        availability_zones : list, optional
            Preferred availability zones
            
        Returns
        -------
        SpotInstance, optional
            Launched instance or None if failed
        """
        if max_bid_price is None:
            max_bid_price = self._calculate_max_bid(instance_type)
        
        current_price = self._get_spot_price(instance_type, availability_zones)
        
        if current_price > max_bid_price:
            logger.warning(
                f"Current spot price ${current_price:.2f} exceeds max bid ${max_bid_price:.2f}"
            )
            return None
        
        instance_id = self._request_spot_instance(
            instance_type,
            max_bid_price,
            availability_zones
        )
        
        if not instance_id:
            return None
        
        instance = SpotInstance(
            instance_id=instance_id,
            provider=self.provider,
            instance_type=instance_type,
            status="running",
            launch_time=time.time()
        )
        
        self.active_instances[instance_id] = instance
        logger.info(f"Launched spot instance {instance_id} at ${current_price:.2f}/hr")
        
        return instance
    
    def monitor_instances(self) -> List[Dict[str, Any]]:
        """
        Monitor active spot instances.
        
        Returns
        -------
        list
            Status of all active instances
        """
        statuses = []
        
        for instance_id, instance in list(self.active_instances.items()):
            status = self._check_instance_status(instance_id)
            
            if status == "interrupted":
                self._handle_interruption(instance)
            
            statuses.append({
                'instance_id': instance_id,
                'status': status,
                'uptime_hours': instance.uptime_hours,
                'cost_saved': instance.cost_saved,
                'interruptions': instance.interruptions,
            })
        
        return statuses
    
    def handle_interruption(
        self,
        instance_id: str,
        resume_callback: Optional[Callable] = None
    ) -> bool:
        """
        Handle spot instance interruption.
        
        Parameters
        ----------
        instance_id : str
            Interrupted instance ID
        resume_callback : callable, optional
            Callback to resume training
            
        Returns
        -------
        bool
            True if successfully recovered, False otherwise
        """
        instance = self.active_instances.get(instance_id)
        
        if not instance:
            return False
        
        instance.interruptions += 1
        self.total_interruptions += 1
        
        logger.warning(
            f"Spot instance {instance_id} interrupted (attempt {instance.interruptions})"
        )
        
        if instance.interruptions >= self.max_retries:
            logger.error(f"Max retries exceeded for {instance_id}, falling back to on-demand")
            return self._fallback_to_on_demand(instance, resume_callback)
        
        if self.checkpoint_config.auto_resume:
            return self._resume_from_checkpoint(instance, resume_callback)
        
        return False
    
    def terminate_instance(self, instance_id: str):
        """Terminate a spot instance."""
        instance = self.active_instances.pop(instance_id, None)
        
        if instance:
            instance.termination_time = time.time()
            instance.status = "terminated"
            
            savings = self._calculate_savings(instance)
            instance.cost_saved = savings
            self.total_cost_saved += savings
            
            self.instance_history.append(instance)
            
            logger.info(
                f"Terminated instance {instance_id}, saved ${savings:.2f}"
            )
    
    def get_cost_savings(self) -> Dict[str, Any]:
        """
        Get cost savings summary.
        
        Returns
        -------
        dict
            Cost savings statistics
        """
        total_uptime = sum(
            inst.uptime_hours for inst in self.instance_history
        )
        
        avg_interruptions = 0.0
        if self.instance_history:
            avg_interruptions = sum(
                inst.interruptions for inst in self.instance_history
            ) / len(self.instance_history)
        
        return {
            'total_cost_saved': self.total_cost_saved,
            'total_instances': len(self.instance_history),
            'total_uptime_hours': total_uptime,
            'total_interruptions': self.total_interruptions,
            'avg_interruptions_per_instance': avg_interruptions,
            'strategy': self.strategy.value,
        }
    
    def _calculate_max_bid(self, instance_type: InstanceType) -> float:
        """Calculate maximum bid price based on strategy."""
        on_demand_price = instance_type.price_per_hour
        
        strategy_multipliers = {
            SpotStrategy.AGGRESSIVE: 0.5,
            SpotStrategy.BALANCED: 0.7,
            SpotStrategy.CONSERVATIVE: 0.9,
        }
        
        multiplier = strategy_multipliers.get(self.strategy, 0.7)
        return on_demand_price * multiplier
    
    def _get_spot_price(
        self,
        instance_type: InstanceType,
        availability_zones: Optional[List[str]] = None
    ) -> float:
        """Get current spot price (simulated)."""
        base_price = instance_type.spot_price_per_hour
        
        import random
        variance = random.uniform(0.8, 1.2)
        return base_price * variance
    
    def _request_spot_instance(
        self,
        instance_type: InstanceType,
        max_bid_price: float,
        availability_zones: Optional[List[str]] = None
    ) -> Optional[str]:
        """Request spot instance (simulated)."""
        import uuid
        instance_id = f"spot-{uuid.uuid4().hex[:8]}"
        return instance_id
    
    def _check_instance_status(self, instance_id: str) -> str:
        """Check instance status (simulated)."""
        instance = self.active_instances.get(instance_id)
        
        if not instance:
            return "unknown"
        
        import random
        if random.random() < 0.001:
            return "interrupted"
        
        return "running"
    
    def _handle_interruption(self, instance: SpotInstance):
        """Handle instance interruption."""
        self.handle_interruption(instance.instance_id)
    
    def _resume_from_checkpoint(
        self,
        instance: SpotInstance,
        resume_callback: Optional[Callable] = None
    ) -> bool:
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint for {instance.instance_id}")
        
        new_instance = self.launch_spot_instance(instance.instance_type)
        
        if new_instance:
            if resume_callback:
                resume_callback(new_instance)
            
            self.terminate_instance(instance.instance_id)
            return True
        
        return False
    
    def _fallback_to_on_demand(
        self,
        instance: SpotInstance,
        resume_callback: Optional[Callable] = None
    ) -> bool:
        """Fallback to on-demand instance."""
        logger.info(f"Falling back to on-demand for {instance.instance_id}")
        
        on_demand_id = self._launch_on_demand_instance(instance.instance_type)
        
        if on_demand_id:
            if resume_callback:
                resume_callback(on_demand_id)
            
            self.terminate_instance(instance.instance_id)
            return True
        
        return False
    
    def _launch_on_demand_instance(self, instance_type: InstanceType) -> Optional[str]:
        """Launch on-demand instance (simulated)."""
        import uuid
        instance_id = f"on-demand-{uuid.uuid4().hex[:8]}"
        return instance_id
    
    def _calculate_savings(self, instance: SpotInstance) -> float:
        """Calculate cost savings for instance."""
        on_demand_cost = instance.instance_type.price_per_hour * instance.uptime_hours
        spot_cost = instance.instance_type.spot_price_per_hour * instance.uptime_hours
        return on_demand_cost - spot_cost
