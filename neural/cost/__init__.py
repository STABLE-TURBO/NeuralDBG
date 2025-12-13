"""
Neural DSL Cost Optimization Module.

This module provides comprehensive cost optimization capabilities including:
- Training cost estimation for cloud providers (AWS/GCP/Azure)
- Spot instance orchestration
- Automatic resource right-sizing recommendations
- Training time prediction
- Carbon footprint tracking
- Cost-performance tradeoff analysis
- Cost dashboard and budget alerts

Classes
-------
CostEstimator
    Estimate training costs for different cloud providers
SpotInstanceOrchestrator
    Manage spot instances for cost-efficient training
ResourceOptimizer
    Provide resource right-sizing recommendations
TrainingPredictor
    Predict training time and costs
CarbonTracker
    Track carbon footprint of training jobs
CostAnalyzer
    Analyze cost-performance tradeoffs
CostDashboard
    Interactive dashboard for cost monitoring
BudgetManager
    Manage budgets and alerts
"""

from neural.cost.estimator import CostEstimator, CloudProvider, InstanceType
from neural.cost.spot_orchestrator import SpotInstanceOrchestrator, SpotStrategy
from neural.cost.resource_optimizer import ResourceOptimizer, ResourceRecommendation
from neural.cost.training_predictor import TrainingPredictor, TrainingEstimate
from neural.cost.carbon_tracker import CarbonTracker, CarbonReport
from neural.cost.analyzer import CostAnalyzer, CostPerformanceAnalysis
from neural.cost.budget_manager import BudgetManager, Budget, BudgetAlert
from neural.cost.dashboard import CostDashboard

__all__ = [
    'CostEstimator',
    'CloudProvider',
    'InstanceType',
    'SpotInstanceOrchestrator',
    'SpotStrategy',
    'ResourceOptimizer',
    'ResourceRecommendation',
    'TrainingPredictor',
    'TrainingEstimate',
    'CarbonTracker',
    'CarbonReport',
    'CostAnalyzer',
    'CostPerformanceAnalysis',
    'BudgetManager',
    'Budget',
    'BudgetAlert',
    'CostDashboard',
]
