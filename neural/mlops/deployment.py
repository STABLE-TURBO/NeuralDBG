"""
Deployment Manager with Shadow Deployment and Automated Rollback.

Provides safe deployment strategies including shadow deployment for
risk-free testing and automated rollback on performance degradation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class DeploymentStrategy(Enum):
    """Deployment strategy type."""
    DIRECT = "direct"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    SHADOW = "shadow"
    ROLLING = "rolling"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a deployment."""
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    requests_per_second: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerformanceMetrics:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class HealthCheck:
    """Health check result."""
    timestamp: str
    status: HealthStatus
    metrics: PerformanceMetrics
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        data['metrics'] = self.metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HealthCheck:
        """Create from dictionary."""
        data = data.copy()
        data['status'] = HealthStatus(data['status'])
        data['metrics'] = PerformanceMetrics.from_dict(data['metrics'])
        return cls(**data)


@dataclass
class Deployment:
    """Deployment configuration and state."""
    deployment_id: str
    model_name: str
    model_version: str
    strategy: DeploymentStrategy
    status: DeploymentStatus
    created_at: str
    created_by: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    rolled_back_at: Optional[str] = None
    environment: str = "production"
    baseline_metrics: Optional[PerformanceMetrics] = None
    current_metrics: Optional[PerformanceMetrics] = None
    health_checks: List[HealthCheck] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['strategy'] = self.strategy.value
        data['status'] = self.status.value
        if self.baseline_metrics:
            data['baseline_metrics'] = self.baseline_metrics.to_dict()
        if self.current_metrics:
            data['current_metrics'] = self.current_metrics.to_dict()
        data['health_checks'] = [hc.to_dict() for hc in self.health_checks]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Deployment:
        """Create from dictionary."""
        data = data.copy()
        data['strategy'] = DeploymentStrategy(data['strategy'])
        data['status'] = DeploymentStatus(data['status'])
        if data.get('baseline_metrics'):
            data['baseline_metrics'] = PerformanceMetrics.from_dict(data['baseline_metrics'])
        if data.get('current_metrics'):
            data['current_metrics'] = PerformanceMetrics.from_dict(data['current_metrics'])
        data['health_checks'] = [
            HealthCheck.from_dict(hc) for hc in data.get('health_checks', [])
        ]
        return cls(**data)


@dataclass
class ShadowDeployment:
    """Shadow deployment configuration."""
    shadow_id: str
    primary_model: str
    shadow_model: str
    traffic_percentage: float
    started_at: str
    comparison_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShadowDeployment:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RollbackConfig:
    """Rollback configuration."""
    enabled: bool = True
    error_rate_threshold: float = 0.05
    latency_threshold_multiplier: float = 2.0
    min_requests_before_check: int = 100
    check_interval_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RollbackConfig:
        """Create from dictionary."""
        return cls(**data)


class RollbackManager:
    """Manages automated rollback on performance degradation."""
    
    def __init__(self, config: Optional[RollbackConfig] = None):
        self.config = config or RollbackConfig()
    
    def should_rollback(
        self,
        baseline: PerformanceMetrics,
        current: PerformanceMetrics,
        request_count: int
    ) -> tuple[bool, List[str]]:
        """
        Determine if deployment should be rolled back.
        
        Returns:
            Tuple of (should_rollback, reasons)
        """
        if not self.config.enabled:
            return False, []
        
        if request_count < self.config.min_requests_before_check:
            return False, []
        
        reasons = []
        
        if current.error_rate > self.config.error_rate_threshold:
            reasons.append(
                f"Error rate {current.error_rate:.2%} exceeds threshold "
                f"{self.config.error_rate_threshold:.2%}"
            )
        
        if baseline.error_rate > 0:
            error_rate_increase = (current.error_rate - baseline.error_rate) / baseline.error_rate
            if error_rate_increase > 1.0:
                reasons.append(
                    f"Error rate increased by {error_rate_increase:.2%} from baseline"
                )
        
        if baseline.latency_p95 > 0:
            latency_multiplier = current.latency_p95 / baseline.latency_p95
            if latency_multiplier > self.config.latency_threshold_multiplier:
                reasons.append(
                    f"P95 latency {current.latency_p95:.3f}s is {latency_multiplier:.2f}x "
                    f"baseline {baseline.latency_p95:.3f}s"
                )
        
        if baseline.latency_p99 > 0:
            latency_multiplier = current.latency_p99 / baseline.latency_p99
            if latency_multiplier > self.config.latency_threshold_multiplier:
                reasons.append(
                    f"P99 latency {current.latency_p99:.3f}s is {latency_multiplier:.2f}x "
                    f"baseline {baseline.latency_p99:.3f}s"
                )
        
        return len(reasons) > 0, reasons


class DeploymentManager:
    """
    Manages model deployments with shadow deployment and automated rollback.
    
    Example:
        manager = DeploymentManager("./deployments")
        
        # Direct deployment
        deployment = manager.create_deployment(
            model_name="fraud_detector",
            model_version="v2.0.0",
            strategy=DeploymentStrategy.DIRECT,
            created_by="devops@company.com"
        )
        
        # Shadow deployment (no production impact)
        shadow = manager.shadow_deploy(
            primary_model="fraud_detector:v1.0.0",
            shadow_model="fraud_detector:v2.0.0",
            traffic_percentage=100.0  # Copy 100% of traffic
        )
        
        # Compare shadow results
        comparison = manager.compare_shadow_deployment(shadow.shadow_id)
        
        # Deploy with automated rollback
        deployment = manager.create_deployment(
            model_name="fraud_detector",
            model_version="v2.0.0",
            strategy=DeploymentStrategy.CANARY,
            created_by="devops@company.com",
            rollback_config=RollbackConfig(
                error_rate_threshold=0.01,
                latency_threshold_multiplier=1.5
            )
        )
        
        # Monitor and auto-rollback if needed
        manager.check_deployment_health(deployment.deployment_id)
    """
    
    def __init__(
        self,
        storage_path: str = "./deployments",
        default_rollback_config: Optional[RollbackConfig] = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.deployments_path = self.storage_path / "deployments"
        self.shadows_path = self.storage_path / "shadows"
        
        self.deployments_path.mkdir(parents=True, exist_ok=True)
        self.shadows_path.mkdir(parents=True, exist_ok=True)
        
        self.rollback_manager = RollbackManager(default_rollback_config)
    
    def create_deployment(
        self,
        model_name: str,
        model_version: str,
        strategy: DeploymentStrategy,
        created_by: str,
        environment: str = "production",
        baseline_metrics: Optional[PerformanceMetrics] = None,
        rollback_config: Optional[RollbackConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Deployment:
        """Create a new deployment."""
        deployment_id = (
            f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{model_name}_{model_version}"
        )
        
        deployment = Deployment(
            deployment_id=deployment_id,
            model_name=model_name,
            model_version=model_version,
            strategy=strategy,
            status=DeploymentStatus.PENDING,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            environment=environment,
            baseline_metrics=baseline_metrics,
            metadata=metadata or {}
        )
        
        if rollback_config:
            deployment.metadata['rollback_config'] = rollback_config.to_dict()
        
        self._save_deployment(deployment)
        return deployment
    
    def start_deployment(self, deployment_id: str) -> None:
        """Start a deployment."""
        deployment = self.get_deployment(deployment_id)
        deployment.status = DeploymentStatus.IN_PROGRESS
        deployment.started_at = datetime.now().isoformat()
        self._save_deployment(deployment)
    
    def complete_deployment(self, deployment_id: str) -> None:
        """Mark deployment as complete."""
        deployment = self.get_deployment(deployment_id)
        deployment.status = DeploymentStatus.ACTIVE
        deployment.completed_at = datetime.now().isoformat()
        self._save_deployment(deployment)
    
    def fail_deployment(self, deployment_id: str, reason: str) -> None:
        """Mark deployment as failed."""
        deployment = self.get_deployment(deployment_id)
        deployment.status = DeploymentStatus.FAILED
        deployment.metadata['failure_reason'] = reason
        self._save_deployment(deployment)
    
    def rollback_deployment(
        self,
        deployment_id: str,
        reason: str,
        triggered_by: str = "automated"
    ) -> None:
        """Rollback a deployment."""
        deployment = self.get_deployment(deployment_id)
        deployment.status = DeploymentStatus.ROLLED_BACK
        deployment.rolled_back_at = datetime.now().isoformat()
        deployment.metadata['rollback_reason'] = reason
        deployment.metadata['rollback_triggered_by'] = triggered_by
        self._save_deployment(deployment)
    
    def update_metrics(
        self,
        deployment_id: str,
        metrics: PerformanceMetrics
    ) -> None:
        """Update deployment metrics."""
        deployment = self.get_deployment(deployment_id)
        deployment.current_metrics = metrics
        self._save_deployment(deployment)
    
    def add_health_check(
        self,
        deployment_id: str,
        status: HealthStatus,
        metrics: PerformanceMetrics,
        message: str = ""
    ) -> None:
        """Add a health check result."""
        deployment = self.get_deployment(deployment_id)
        
        health_check = HealthCheck(
            timestamp=datetime.now().isoformat(),
            status=status,
            metrics=metrics,
            message=message
        )
        
        deployment.health_checks.append(health_check)
        deployment.current_metrics = metrics
        
        self._save_deployment(deployment)
    
    def check_deployment_health(
        self,
        deployment_id: str,
        request_count: int
    ) -> tuple[bool, List[str]]:
        """
        Check deployment health and determine if rollback needed.
        
        Returns:
            Tuple of (needs_rollback, reasons)
        """
        deployment = self.get_deployment(deployment_id)
        
        if deployment.status != DeploymentStatus.ACTIVE:
            return False, []
        
        if not deployment.baseline_metrics or not deployment.current_metrics:
            return False, []
        
        rollback_config = None
        if 'rollback_config' in deployment.metadata:
            rollback_config = RollbackConfig.from_dict(
                deployment.metadata['rollback_config']
            )
        
        rollback_manager = RollbackManager(rollback_config)
        needs_rollback, reasons = rollback_manager.should_rollback(
            deployment.baseline_metrics,
            deployment.current_metrics,
            request_count
        )
        
        if needs_rollback:
            self.rollback_deployment(
                deployment_id,
                reason="; ".join(reasons),
                triggered_by="automated"
            )
        
        return needs_rollback, reasons
    
    def shadow_deploy(
        self,
        primary_model: str,
        shadow_model: str,
        traffic_percentage: float = 100.0
    ) -> ShadowDeployment:
        """
        Create a shadow deployment.
        
        Shadow deployments receive a copy of production traffic but
        don't affect responses sent to users.
        """
        shadow_id = (
            f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{shadow_model.replace(':', '_')}"
        )
        
        shadow = ShadowDeployment(
            shadow_id=shadow_id,
            primary_model=primary_model,
            shadow_model=shadow_model,
            traffic_percentage=traffic_percentage,
            started_at=datetime.now().isoformat()
        )
        
        self._save_shadow_deployment(shadow)
        return shadow
    
    def record_shadow_comparison(
        self,
        shadow_id: str,
        primary_result: Any,
        shadow_result: Any,
        primary_latency: float,
        shadow_latency: float,
        agreement: bool
    ) -> None:
        """Record comparison between primary and shadow models."""
        shadow = self.get_shadow_deployment(shadow_id)
        
        if 'comparisons' not in shadow.comparison_metrics:
            shadow.comparison_metrics['comparisons'] = []
        
        shadow.comparison_metrics['comparisons'].append({
            "timestamp": datetime.now().isoformat(),
            "primary_latency": primary_latency,
            "shadow_latency": shadow_latency,
            "agreement": agreement
        })
        
        self._save_shadow_deployment(shadow)
    
    def compare_shadow_deployment(self, shadow_id: str) -> Dict[str, Any]:
        """Analyze shadow deployment results."""
        shadow = self.get_shadow_deployment(shadow_id)
        
        comparisons = shadow.comparison_metrics.get('comparisons', [])
        
        if not comparisons:
            return {
                "shadow_id": shadow_id,
                "total_requests": 0,
                "agreement_rate": 0.0,
                "average_latency_diff": 0.0
            }
        
        total = len(comparisons)
        agreements = sum(1 for c in comparisons if c['agreement'])
        
        primary_latencies = [c['primary_latency'] for c in comparisons]
        shadow_latencies = [c['shadow_latency'] for c in comparisons]
        
        return {
            "shadow_id": shadow_id,
            "primary_model": shadow.primary_model,
            "shadow_model": shadow.shadow_model,
            "total_requests": total,
            "agreement_rate": agreements / total,
            "primary_latency": {
                "mean": np.mean(primary_latencies),
                "p50": np.percentile(primary_latencies, 50),
                "p95": np.percentile(primary_latencies, 95),
                "p99": np.percentile(primary_latencies, 99)
            },
            "shadow_latency": {
                "mean": np.mean(shadow_latencies),
                "p50": np.percentile(shadow_latencies, 50),
                "p95": np.percentile(shadow_latencies, 95),
                "p99": np.percentile(shadow_latencies, 99)
            },
            "average_latency_diff": np.mean(shadow_latencies) - np.mean(primary_latencies)
        }
    
    def get_deployment(self, deployment_id: str) -> Deployment:
        """Get deployment by ID."""
        deployment_file = self.deployments_path / f"{deployment_id}.json"
        if not deployment_file.exists():
            raise FileNotFoundError(f"Deployment {deployment_id} not found")
        
        with open(deployment_file, 'r') as f:
            data = json.load(f)
        return Deployment.from_dict(data)
    
    def get_shadow_deployment(self, shadow_id: str) -> ShadowDeployment:
        """Get shadow deployment by ID."""
        shadow_file = self.shadows_path / f"{shadow_id}.json"
        if not shadow_file.exists():
            raise FileNotFoundError(f"Shadow deployment {shadow_id} not found")
        
        with open(shadow_file, 'r') as f:
            data = json.load(f)
        return ShadowDeployment.from_dict(data)
    
    def list_deployments(
        self,
        status: Optional[DeploymentStatus] = None,
        environment: Optional[str] = None
    ) -> List[Deployment]:
        """List deployments with optional filtering."""
        deployments = []
        for deployment_file in self.deployments_path.glob("*.json"):
            with open(deployment_file, 'r') as f:
                data = json.load(f)
            deployment = Deployment.from_dict(data)
            
            if status and deployment.status != status:
                continue
            if environment and deployment.environment != environment:
                continue
            
            deployments.append(deployment)
        
        return sorted(deployments, key=lambda d: d.created_at, reverse=True)
    
    def _save_deployment(self, deployment: Deployment) -> None:
        """Save deployment to disk."""
        deployment_file = self.deployments_path / f"{deployment.deployment_id}.json"
        with open(deployment_file, 'w') as f:
            json.dump(deployment.to_dict(), f, indent=2)
    
    def _save_shadow_deployment(self, shadow: ShadowDeployment) -> None:
        """Save shadow deployment to disk."""
        shadow_file = self.shadows_path / f"{shadow.shadow_id}.json"
        with open(shadow_file, 'w') as f:
            json.dump(shadow.to_dict(), f, indent=2)
