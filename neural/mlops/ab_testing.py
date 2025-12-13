"""
A/B Testing Framework with Traffic Splitting.

Provides robust A/B testing capabilities for comparing model performance
with configurable traffic splitting and statistical analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

import numpy as np


class TestStatus(Enum):
    """A/B test status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategy."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    PERCENTAGE = "percentage"
    CANARY = "canary"


@dataclass
class ABTestMetrics:
    """Metrics for a variant in an A/B test."""
    variant_name: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.latency_count == 0:
            return 0.0
        return self.latency_sum / self.latency_count
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successes + self.failures
        if total == 0:
            return 0.0
        return self.successes / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ABTestMetrics:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ABTest:
    """A/B test configuration and results."""
    test_id: str
    name: str
    description: str
    control_variant: str
    treatment_variant: str
    traffic_split: float
    strategy: TrafficSplitStrategy
    status: TestStatus
    created_at: str
    created_by: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    control_metrics: ABTestMetrics = field(default_factory=lambda: ABTestMetrics("control"))
    treatment_metrics: ABTestMetrics = field(default_factory=lambda: ABTestMetrics("treatment"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['strategy'] = self.strategy.value
        data['status'] = self.status.value
        data['control_metrics'] = self.control_metrics.to_dict()
        data['treatment_metrics'] = self.treatment_metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ABTest:
        """Create from dictionary."""
        data = data.copy()
        data['strategy'] = TrafficSplitStrategy(data['strategy'])
        data['status'] = TestStatus(data['status'])
        data['control_metrics'] = ABTestMetrics.from_dict(data['control_metrics'])
        data['treatment_metrics'] = ABTestMetrics.from_dict(data['treatment_metrics'])
        return cls(**data)


class TrafficSplitter:
    """Handles traffic splitting for A/B tests."""
    
    def __init__(self, strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM):
        self.strategy = strategy
        self.random = random.Random()
    
    def should_use_treatment(
        self,
        traffic_split: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Determine if request should use treatment variant.
        
        Args:
            traffic_split: Percentage of traffic to treatment (0.0-1.0)
            user_id: Optional user identifier for hash-based splitting
            request_id: Optional request identifier
            
        Returns:
            True if request should use treatment variant
        """
        if self.strategy == TrafficSplitStrategy.RANDOM:
            return self.random.random() < traffic_split
        
        elif self.strategy == TrafficSplitStrategy.HASH_BASED:
            if user_id is None:
                user_id = request_id or str(self.random.random())
            
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return (hash_val % 100) < (traffic_split * 100)
        
        elif self.strategy == TrafficSplitStrategy.PERCENTAGE:
            return self.random.random() < traffic_split
        
        elif self.strategy == TrafficSplitStrategy.CANARY:
            return self.random.random() < traffic_split
        
        return False
    
    def get_variant(
        self,
        control: str,
        treatment: str,
        traffic_split: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> str:
        """Get variant name for request."""
        if self.should_use_treatment(traffic_split, user_id, request_id):
            return treatment
        return control


class StatisticalAnalyzer:
    """Performs statistical analysis on A/B test results."""
    
    @staticmethod
    def calculate_confidence_interval(
        successes: int,
        total: int,
        confidence_level: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for success rate."""
        if total == 0:
            return (0.0, 0.0)
        
        p = successes / total
        z = 1.96 if confidence_level == 0.95 else 2.576
        
        margin = z * np.sqrt((p * (1 - p)) / total)
        return (max(0, p - margin), min(1, p + margin))
    
    @staticmethod
    def is_statistically_significant(
        control_metrics: ABTestMetrics,
        treatment_metrics: ABTestMetrics,
        confidence_level: float = 0.95
    ) -> bool:
        """
        Determine if difference between variants is statistically significant.
        
        Uses two-proportion z-test.
        """
        control_total = control_metrics.successes + control_metrics.failures
        treatment_total = treatment_metrics.successes + treatment_metrics.failures
        
        if control_total == 0 or treatment_total == 0:
            return False
        
        p1 = control_metrics.successes / control_total
        p2 = treatment_metrics.successes / treatment_total
        
        p_pooled = (control_metrics.successes + treatment_metrics.successes) / (
            control_total + treatment_total
        )
        
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        if se == 0:
            return False
        
        z = (p2 - p1) / se
        z_critical = 1.96 if confidence_level == 0.95 else 2.576
        
        return abs(z) > z_critical
    
    @staticmethod
    def calculate_improvement(
        control_metrics: ABTestMetrics,
        treatment_metrics: ABTestMetrics
    ) -> Dict[str, float]:
        """Calculate improvement metrics."""
        control_rate = control_metrics.success_rate()
        treatment_rate = treatment_metrics.success_rate()
        
        improvement = {}
        if control_rate > 0:
            improvement['relative_improvement'] = (
                (treatment_rate - control_rate) / control_rate * 100
            )
        else:
            improvement['relative_improvement'] = 0.0
        
        improvement['absolute_improvement'] = (treatment_rate - control_rate) * 100
        
        control_latency = control_metrics.average_latency()
        treatment_latency = treatment_metrics.average_latency()
        
        if control_latency > 0:
            improvement['latency_improvement'] = (
                (control_latency - treatment_latency) / control_latency * 100
            )
        else:
            improvement['latency_improvement'] = 0.0
        
        return improvement


class ABTestManager:
    """
    Manages A/B testing experiments with traffic splitting.
    
    Example:
        manager = ABTestManager("./ab_tests")
        
        # Create a new A/B test
        test = manager.create_test(
            name="Model V2 Test",
            description="Testing new model version",
            control_variant="model_v1",
            treatment_variant="model_v2",
            traffic_split=0.1,  # 10% to treatment
            strategy=TrafficSplitStrategy.HASH_BASED,
            created_by="ml_engineer@company.com"
        )
        
        # Start the test
        manager.start_test(test.test_id)
        
        # Record results
        manager.record_request(
            test_id=test.test_id,
            variant="treatment",
            success=True,
            latency=0.045,
            custom_metrics={"accuracy": 0.96}
        )
        
        # Analyze results
        analysis = manager.analyze_test(test.test_id)
        if analysis['statistically_significant']:
            print(f"Treatment improved by {analysis['improvement']['relative_improvement']:.2f}%")
    """
    
    def __init__(self, storage_path: str = "./ab_tests"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.analyzer = StatisticalAnalyzer()
    
    def create_test(
        self,
        name: str,
        description: str,
        control_variant: str,
        treatment_variant: str,
        traffic_split: float,
        strategy: TrafficSplitStrategy,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ABTest:
        """Create a new A/B test."""
        if not 0 <= traffic_split <= 1:
            raise ValueError("traffic_split must be between 0 and 1")
        
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_')}"
        
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            control_variant=control_variant,
            treatment_variant=treatment_variant,
            traffic_split=traffic_split,
            strategy=strategy,
            status=TestStatus.DRAFT,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            metadata=metadata or {}
        )
        
        self._save_test(test)
        return test
    
    def start_test(self, test_id: str) -> None:
        """Start an A/B test."""
        test = self.get_test(test_id)
        test.status = TestStatus.RUNNING
        test.started_at = datetime.now().isoformat()
        self._save_test(test)
    
    def pause_test(self, test_id: str) -> None:
        """Pause an A/B test."""
        test = self.get_test(test_id)
        test.status = TestStatus.PAUSED
        self._save_test(test)
    
    def complete_test(self, test_id: str) -> None:
        """Complete an A/B test."""
        test = self.get_test(test_id)
        test.status = TestStatus.COMPLETED
        test.completed_at = datetime.now().isoformat()
        self._save_test(test)
    
    def cancel_test(self, test_id: str) -> None:
        """Cancel an A/B test."""
        test = self.get_test(test_id)
        test.status = TestStatus.CANCELLED
        self._save_test(test)
    
    def record_request(
        self,
        test_id: str,
        variant: str,
        success: bool,
        latency: float,
        custom_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Record a request result for a variant."""
        test = self.get_test(test_id)
        
        if variant == "control" or variant == test.control_variant:
            metrics = test.control_metrics
        elif variant == "treatment" or variant == test.treatment_variant:
            metrics = test.treatment_metrics
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        metrics.requests += 1
        if success:
            metrics.successes += 1
        else:
            metrics.failures += 1
        
        metrics.latency_sum += latency
        metrics.latency_count += 1
        
        if custom_metrics:
            for key, value in custom_metrics.items():
                if key not in metrics.custom_metrics:
                    metrics.custom_metrics[key] = []
                metrics.custom_metrics[key].append(value)
        
        self._save_test(test)
    
    def get_variant(
        self,
        test_id: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> str:
        """Get variant assignment for a request."""
        test = self.get_test(test_id)
        
        if test.status != TestStatus.RUNNING:
            return test.control_variant
        
        splitter = TrafficSplitter(test.strategy)
        return splitter.get_variant(
            control=test.control_variant,
            treatment=test.treatment_variant,
            traffic_split=test.traffic_split,
            user_id=user_id,
            request_id=request_id
        )
    
    def analyze_test(
        self,
        test_id: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Analyze A/B test results."""
        test = self.get_test(test_id)
        
        is_significant = self.analyzer.is_statistically_significant(
            test.control_metrics,
            test.treatment_metrics,
            confidence_level
        )
        
        improvement = self.analyzer.calculate_improvement(
            test.control_metrics,
            test.treatment_metrics
        )
        
        control_ci = self.analyzer.calculate_confidence_interval(
            test.control_metrics.successes,
            test.control_metrics.successes + test.control_metrics.failures,
            confidence_level
        )
        
        treatment_ci = self.analyzer.calculate_confidence_interval(
            test.treatment_metrics.successes,
            test.treatment_metrics.successes + test.treatment_metrics.failures,
            confidence_level
        )
        
        return {
            "test_id": test_id,
            "name": test.name,
            "status": test.status.value,
            "statistically_significant": is_significant,
            "confidence_level": confidence_level,
            "control": {
                "variant": test.control_variant,
                "requests": test.control_metrics.requests,
                "success_rate": test.control_metrics.success_rate(),
                "average_latency": test.control_metrics.average_latency(),
                "confidence_interval": control_ci
            },
            "treatment": {
                "variant": test.treatment_variant,
                "requests": test.treatment_metrics.requests,
                "success_rate": test.treatment_metrics.success_rate(),
                "average_latency": test.treatment_metrics.average_latency(),
                "confidence_interval": treatment_ci
            },
            "improvement": improvement
        }
    
    def get_test(self, test_id: str) -> ABTest:
        """Get A/B test by ID."""
        test_file = self.storage_path / f"{test_id}.json"
        if not test_file.exists():
            raise FileNotFoundError(f"Test {test_id} not found")
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        return ABTest.from_dict(data)
    
    def list_tests(self, status: Optional[TestStatus] = None) -> List[ABTest]:
        """List all A/B tests with optional status filter."""
        tests = []
        for test_file in self.storage_path.glob("*.json"):
            with open(test_file, 'r') as f:
                data = json.load(f)
            test = ABTest.from_dict(data)
            
            if status and test.status != status:
                continue
            
            tests.append(test)
        
        return sorted(tests, key=lambda t: t.created_at, reverse=True)
    
    def _save_test(self, test: ABTest) -> None:
        """Save A/B test to disk."""
        test_file = self.storage_path / f"{test.test_id}.json"
        with open(test_file, 'w') as f:
            json.dump(test.to_dict(), f, indent=2)
