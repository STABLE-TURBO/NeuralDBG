"""
Tests for cost optimization module.
"""

import pytest
from neural.cost import (
    CostEstimator,
    CloudProvider,
    SpotInstanceOrchestrator,
    SpotStrategy,
    ResourceOptimizer,
    ResourceMetrics,
    TrainingPredictor,
    CarbonTracker,
    CostAnalyzer,
    PerformanceMetrics,
    BudgetManager,
)


class TestCostEstimator:
    """Test cost estimation."""
    
    def test_estimate_cost(self):
        estimator = CostEstimator()
        
        estimate = estimator.estimate_cost(
            provider=CloudProvider.AWS,
            instance_name="p3.2xlarge",
            training_hours=10.0,
            storage_gb=100.0
        )
        
        assert estimate.provider == CloudProvider.AWS
        assert estimate.estimated_hours == 10.0
        assert estimate.on_demand_cost > 0
        assert estimate.spot_cost > 0
        assert estimate.potential_savings > 0
    
    def test_compare_providers(self):
        estimator = CostEstimator()
        
        estimates = estimator.compare_providers(
            gpu_count=1,
            training_hours=10.0
        )
        
        assert len(estimates) > 0
        assert all(e.total_spot_cost > 0 for e in estimates)
    
    def test_get_cheapest_option(self):
        estimator = CostEstimator()
        
        cheapest = estimator.get_cheapest_option(
            gpu_count=1,
            training_hours=10.0
        )
        
        assert cheapest is not None
        assert cheapest.total_spot_cost > 0


class TestSpotOrchestrator:
    """Test spot instance orchestration."""
    
    def test_launch_spot_instance(self):
        estimator = CostEstimator()
        instance_type = estimator.instance_types[CloudProvider.AWS][0]
        
        orchestrator = SpotInstanceOrchestrator(
            provider=CloudProvider.AWS,
            strategy=SpotStrategy.BALANCED
        )
        
        instance = orchestrator.launch_spot_instance(instance_type)
        
        assert instance is not None
        assert instance.instance_id.startswith("spot-")
    
    def test_get_cost_savings(self):
        orchestrator = SpotInstanceOrchestrator(
            provider=CloudProvider.AWS,
            strategy=SpotStrategy.BALANCED
        )
        
        savings = orchestrator.get_cost_savings()
        
        assert 'total_cost_saved' in savings
        assert 'total_instances' in savings


class TestResourceOptimizer:
    """Test resource optimization."""
    
    def test_analyze_utilization(self):
        estimator = CostEstimator()
        instance_type = estimator.instance_types[CloudProvider.AWS][0]
        
        optimizer = ResourceOptimizer(estimator)
        
        metrics = ResourceMetrics(
            cpu_utilization=0.45,
            memory_utilization=0.50,
            gpu_utilization=0.35,
            samples=100
        )
        
        recommendation = optimizer.analyze_utilization(instance_type, metrics)
        
        assert recommendation is not None
        assert recommendation.current_instance == instance_type
    
    def test_optimization_summary(self):
        optimizer = ResourceOptimizer()
        
        summary = optimizer.get_optimization_summary()
        
        assert 'status' in summary


class TestTrainingPredictor:
    """Test training prediction."""
    
    def test_predict_training_time(self):
        estimator = CostEstimator()
        instance_type = estimator.instance_types[CloudProvider.AWS][0]
        
        predictor = TrainingPredictor(estimator)
        
        model_params = {'total_params': 50_000_000}
        
        estimate = predictor.predict_training_time(
            model_params=model_params,
            dataset_size=100_000,
            batch_size=32,
            epochs=100,
            instance_type=instance_type
        )
        
        assert estimate.estimated_hours > 0
        assert estimate.total_cost > 0
        assert len(estimate.confidence_interval) == 2
    
    def test_optimize_batch_size(self):
        estimator = CostEstimator()
        instance_type = estimator.instance_types[CloudProvider.AWS][0]
        
        predictor = TrainingPredictor(estimator)
        
        model_params = {'total_params': 50_000_000}
        
        optimal = predictor.optimize_batch_size(
            model_params=model_params,
            dataset_size=100_000,
            instance_type=instance_type
        )
        
        assert 'optimal_batch_size' in optimal
        assert optimal['optimal_batch_size'] > 0


class TestCarbonTracker:
    """Test carbon tracking."""
    
    def test_track_training(self):
        estimator = CostEstimator()
        instance_type = estimator.instance_types[CloudProvider.AWS][0]
        
        tracker = CarbonTracker()
        
        report = tracker.track_training(
            job_id="test-001",
            provider=CloudProvider.AWS,
            region="us-west-2",
            instance_type=instance_type,
            duration_hours=10.0
        )
        
        assert report.carbon_kg_co2 > 0
        assert report.energy_kwh > 0
    
    def test_get_greenest_region(self):
        estimator = CostEstimator()
        instance_type = estimator.instance_types[CloudProvider.AWS][0]
        
        tracker = CarbonTracker()
        
        greenest = tracker.get_greenest_region(
            CloudProvider.AWS,
            instance_type
        )
        
        assert greenest is not None


class TestCostAnalyzer:
    """Test cost analysis."""
    
    def test_analyze_tradeoffs(self):
        analyzer = CostAnalyzer()
        
        configs = [
            {'instance': 'p3.2xlarge'},
            {'instance': 'p3.8xlarge'},
        ]
        
        metrics = [
            PerformanceMetrics(accuracy=0.85, training_time_hours=12.0),
            PerformanceMetrics(accuracy=0.87, training_time_hours=4.0),
        ]
        
        costs = [92.0, 367.0]
        
        analysis = analyzer.analyze_tradeoffs(configs, metrics, costs)
        
        assert len(analysis.points) == 2
        assert analysis.best_value is not None
    
    def test_calculate_roi(self):
        analyzer = CostAnalyzer()
        
        roi = analyzer.calculate_roi(
            training_cost=367.0,
            improvement_accuracy=0.02
        )
        
        assert 'roi_percent' in roi
        assert 'net_benefit' in roi


class TestBudgetManager:
    """Test budget management."""
    
    def test_create_budget(self):
        manager = BudgetManager(storage_path="test_budget_data")
        
        budget = manager.create_budget(
            name="Test-Budget",
            total_amount=1000.0,
            period_days=30
        )
        
        assert budget.name == "Test-Budget"
        assert budget.total_amount == 1000.0
    
    def test_record_expense(self):
        manager = BudgetManager(storage_path="test_budget_data")
        
        manager.create_budget(
            name="Test-Budget",
            total_amount=1000.0,
            period_days=30
        )
        
        manager.record_expense(
            budget_name="Test-Budget",
            amount=250.0,
            provider=CloudProvider.AWS,
            description="Test expense"
        )
        
        status = manager.get_budget_status("Test-Budget")
        
        assert status['spent_amount'] == 250.0
        assert status['remaining_amount'] == 750.0
    
    def test_get_spending_report(self):
        manager = BudgetManager(storage_path="test_budget_data")
        
        report = manager.get_spending_report(days=30)
        
        assert 'total_expenses' in report
        assert 'total_amount' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
