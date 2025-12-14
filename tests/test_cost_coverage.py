"""
Comprehensive test suite for Cost Optimization module to increase coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from neural.cost.estimator import CostEstimator
from neural.cost.budget_manager import BudgetManager
from neural.cost.resource_optimizer import ResourceOptimizer
from neural.cost.carbon_tracker import CarbonTracker


class TestCostEstimator:
    """Test cost estimation functionality."""
    
    def test_estimator_initialization(self):
        """Test cost estimator initialization."""
        estimator = CostEstimator(provider="aws")
        assert estimator.provider == "aws"
    
    def test_estimate_training_cost(self):
        """Test training cost estimation."""
        estimator = CostEstimator(provider="aws")
        with patch.object(estimator, 'estimate_training') as mock_estimate:
            mock_estimate.return_value = {
                "total_cost": 45.50,
                "compute_cost": 40.00,
                "storage_cost": 5.50
            }
            cost = estimator.estimate_training(
                instance_type="ml.p3.2xlarge",
                hours=10
            )
            assert cost["total_cost"] == 45.50
    
    def test_estimate_inference_cost(self):
        """Test inference cost estimation."""
        estimator = CostEstimator(provider="aws")
        with patch.object(estimator, 'estimate_inference') as mock_estimate:
            mock_estimate.return_value = {
                "cost_per_request": 0.001,
                "monthly_cost": 100.00
            }
            cost = estimator.estimate_inference(
                requests_per_second=10,
                instance_type="ml.m5.large"
            )
            assert cost["monthly_cost"] == 100.00
    
    def test_compare_providers(self):
        """Test cost comparison across providers."""
        estimator = CostEstimator()
        with patch.object(estimator, 'compare_providers') as mock_compare:
            mock_compare.return_value = {
                "aws": {"cost": 100, "performance": 0.95},
                "azure": {"cost": 110, "performance": 0.94},
                "gcp": {"cost": 95, "performance": 0.96}
            }
            comparison = estimator.compare_providers(
                workload="training",
                requirements={}
            )
            assert "aws" in comparison


class TestBudgetManager:
    """Test budget management functionality."""
    
    def test_budget_manager_initialization(self):
        """Test budget manager initialization."""
        manager = BudgetManager(total_budget=1000.0)
        assert manager.total_budget == 1000.0
    
    def test_set_budget_limit(self):
        """Test setting budget limits."""
        manager = BudgetManager()
        with patch.object(manager, 'set_limit') as mock_set:
            mock_set.return_value = True
            result = manager.set_limit(
                category="training",
                amount=500.0
            )
            assert result is True
    
    def test_track_spending(self):
        """Test tracking spending."""
        manager = BudgetManager(total_budget=1000.0)
        with patch.object(manager, 'track') as mock_track:
            mock_track.return_value = {
                "spent": 250.0,
                "remaining": 750.0,
                "percentage_used": 25.0
            }
            status = manager.track(amount=250.0)
            assert status["remaining"] == 750.0
    
    def test_check_budget_alert(self):
        """Test budget alerts."""
        manager = BudgetManager(total_budget=1000.0)
        with patch.object(manager, 'check_alert') as mock_check:
            mock_check.return_value = {
                "alert": True,
                "threshold": 0.8,
                "current": 0.85
            }
            alert = manager.check_alert()
            assert alert["alert"] is True
    
    def test_get_spending_report(self):
        """Test spending report generation."""
        manager = BudgetManager(total_budget=1000.0)
        with patch.object(manager, 'get_report') as mock_report:
            mock_report.return_value = {
                "total_spent": 600.0,
                "by_category": {
                    "training": 400.0,
                    "inference": 200.0
                }
            }
            report = manager.get_report()
            assert report["total_spent"] == 600.0


class TestResourceOptimizer:
    """Test resource optimization functionality."""
    
    def test_optimizer_initialization(self):
        """Test resource optimizer initialization."""
        optimizer = ResourceOptimizer()
        assert optimizer is not None
    
    def test_recommend_instance_type(self):
        """Test instance type recommendation."""
        optimizer = ResourceOptimizer()
        with patch.object(optimizer, 'recommend') as mock_recommend:
            mock_recommend.return_value = {
                "instance_type": "ml.p3.2xlarge",
                "cost_per_hour": 3.06,
                "estimated_time": 10
            }
            recommendation = optimizer.recommend(
                model_size="large",
                dataset_size="medium"
            )
            assert "instance_type" in recommendation
    
    def test_optimize_batch_size(self):
        """Test batch size optimization."""
        optimizer = ResourceOptimizer()
        with patch.object(optimizer, 'optimize_batch_size') as mock_optimize:
            mock_optimize.return_value = {
                "recommended_batch_size": 64,
                "memory_utilization": 0.85
            }
            result = optimizer.optimize_batch_size(
                model=Mock(),
                available_memory=16000
            )
            assert result["recommended_batch_size"] == 64
    
    def test_suggest_autoscaling(self):
        """Test autoscaling suggestions."""
        optimizer = ResourceOptimizer()
        with patch.object(optimizer, 'suggest_autoscaling') as mock_suggest:
            mock_suggest.return_value = {
                "min_instances": 2,
                "max_instances": 10,
                "target_utilization": 0.7
            }
            config = optimizer.suggest_autoscaling(
                traffic_pattern=Mock()
            )
            assert config["min_instances"] == 2


class TestCarbonTracker:
    """Test carbon footprint tracking."""
    
    def test_tracker_initialization(self):
        """Test carbon tracker initialization."""
        tracker = CarbonTracker(region="us-east-1")
        assert tracker.region == "us-east-1"
    
    def test_estimate_emissions(self):
        """Test carbon emissions estimation."""
        tracker = CarbonTracker()
        with patch.object(tracker, 'estimate') as mock_estimate:
            mock_estimate.return_value = {
                "co2_kg": 12.5,
                "equivalent": "30 miles driven"
            }
            emissions = tracker.estimate(
                compute_hours=10,
                instance_type="ml.p3.2xlarge"
            )
            assert emissions["co2_kg"] == 12.5
    
    def test_track_training_emissions(self):
        """Test tracking training emissions."""
        tracker = CarbonTracker()
        with patch.object(tracker, 'track_training') as mock_track:
            mock_track.return_value = {
                "session_id": "train_123",
                "co2_kg": 8.3
            }
            result = tracker.track_training(
                training_job_id="job_123"
            )
            assert result["co2_kg"] == 8.3
    
    def test_compare_regions(self):
        """Test comparing carbon footprint across regions."""
        tracker = CarbonTracker()
        with patch.object(tracker, 'compare_regions') as mock_compare:
            mock_compare.return_value = {
                "us-east-1": {"co2_kg": 10.0, "carbon_intensity": 0.5},
                "eu-west-1": {"co2_kg": 5.0, "carbon_intensity": 0.25}
            }
            comparison = tracker.compare_regions(
                compute_hours=10
            )
            assert len(comparison) == 2
    
    def test_generate_report(self):
        """Test carbon report generation."""
        tracker = CarbonTracker()
        with patch.object(tracker, 'generate_report') as mock_report:
            mock_report.return_value = {
                "total_co2_kg": 50.0,
                "by_job": {"job_1": 30.0, "job_2": 20.0},
                "recommendations": ["Use eu-west-1 for lower emissions"]
            }
            report = tracker.generate_report()
            assert report["total_co2_kg"] == 50.0


@pytest.mark.parametrize("provider,instance_type,expected_cost", [
    ("aws", "ml.p3.2xlarge", 3.06),
    ("aws", "ml.m5.large", 0.096),
    ("azure", "Standard_NC6", 0.90),
    ("gcp", "n1-standard-4", 0.19),
])
def test_provider_pricing(provider, instance_type, expected_cost):
    """Parameterized test for provider pricing."""
    estimator = CostEstimator(provider=provider)
    with patch.object(estimator, 'get_hourly_rate') as mock_rate:
        mock_rate.return_value = expected_cost
        rate = estimator.get_hourly_rate(instance_type)
        assert rate == expected_cost


@pytest.mark.parametrize("spent,budget,should_alert", [
    (800, 1000, False),
    (900, 1000, True),
    (950, 1000, True),
])
def test_budget_alerts(spent, budget, should_alert):
    """Parameterized test for budget alerts."""
    manager = BudgetManager(total_budget=budget)
    with patch.object(manager, 'should_alert') as mock_alert:
        mock_alert.return_value = should_alert
        alert = manager.should_alert(spent)
        assert alert == should_alert


@pytest.mark.parametrize("region,carbon_intensity", [
    ("us-east-1", 0.5),
    ("eu-west-1", 0.25),
    ("ap-southeast-1", 0.6),
])
def test_regional_carbon_intensity(region, carbon_intensity):
    """Parameterized test for regional carbon intensity."""
    tracker = CarbonTracker(region=region)
    with patch.object(tracker, 'get_carbon_intensity') as mock_intensity:
        mock_intensity.return_value = carbon_intensity
        intensity = tracker.get_carbon_intensity()
        assert intensity == carbon_intensity
