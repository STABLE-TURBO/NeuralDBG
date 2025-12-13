# Cost Optimization Guide

Complete guide to cost optimization features in Neural DSL.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Features](#core-features)
4. [Quick Start](#quick-start)
5. [CLI Usage](#cli-usage)
6. [Python API](#python-api)
7. [Dashboard](#dashboard)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Overview

The Neural DSL cost optimization module provides comprehensive tools for managing and optimizing the costs of training neural networks in the cloud. It supports AWS, GCP, and Azure with features for cost estimation, spot instance management, resource optimization, carbon tracking, and budget management.

### Key Features

- **Multi-Cloud Support**: AWS, GCP, Azure
- **Cost Estimation**: Accurate pricing for on-demand and spot instances
- **Spot Orchestration**: Automatic spot instance management with fault tolerance
- **Resource Optimization**: Right-sizing recommendations based on utilization
- **Training Prediction**: Estimate training time and costs before running
- **Carbon Tracking**: Monitor environmental impact
- **Cost Analysis**: Analyze cost-performance tradeoffs
- **Budget Management**: Track spending and set alerts
- **Interactive Dashboard**: Web-based cost monitoring

## Installation

```bash
# Full installation with all features
pip install -e ".[full]"

# Or install specific features
pip install -e ".[dashboard,integrations]"
```

## Core Features

### 1. Cost Estimation

Estimate training costs across different cloud providers and instance types.

**Features:**
- On-demand and spot instance pricing
- Storage and data transfer costs
- Provider comparison
- Cost breakdown

**Use Cases:**
- Budget planning
- Instance selection
- Cost comparison

### 2. Spot Instance Orchestration

Manage spot instances with automatic interruption handling.

**Features:**
- Spot instance launching
- Interruption detection and recovery
- Checkpoint-based fault tolerance
- Cost savings tracking
- Multiple bidding strategies

**Use Cases:**
- Cost-efficient training
- Long-running jobs
- Fault-tolerant workloads

### 3. Resource Optimization

Get recommendations for right-sizing instances based on actual usage.

**Features:**
- CPU, memory, GPU utilization analysis
- Overprovisioning detection
- Underprovisioning alerts
- Cost savings recommendations

**Use Cases:**
- Resource efficiency
- Cost reduction
- Performance optimization

### 4. Training Time Prediction

Predict training duration and costs before starting.

**Features:**
- Time estimation based on model size and dataset
- Historical learning
- Batch size optimization
- Confidence intervals

**Use Cases:**
- Planning and scheduling
- Resource allocation
- Budget forecasting

### 5. Carbon Footprint Tracking

Monitor the environmental impact of training jobs.

**Features:**
- CO₂ emission tracking
- Energy consumption monitoring
- Regional carbon intensity
- Green region recommendations

**Use Cases:**
- Sustainability reporting
- Green computing
- Carbon offset planning

### 6. Cost-Performance Analysis

Analyze tradeoffs between cost and model performance.

**Features:**
- Pareto frontier analysis
- ROI calculation
- Budget optimization
- Performance optimization

**Use Cases:**
- Model selection
- Configuration optimization
- Business decision making

### 7. Budget Management

Track spending and manage budgets with alerts.

**Features:**
- Budget creation and tracking
- Expense recording
- Alert thresholds
- Spending reports
- Burn rate monitoring

**Use Cases:**
- Cost control
- Team budget management
- Financial reporting

### 8. Interactive Dashboard

Web-based dashboard for real-time cost monitoring.

**Features:**
- Real-time cost tracking
- Visual analytics
- Budget status
- Carbon footprint
- Multi-tab interface

**Use Cases:**
- Team visibility
- Executive reporting
- Cost monitoring

## Quick Start

### Basic Cost Estimation

```python
from neural.cost import CostEstimator, CloudProvider

estimator = CostEstimator()

# Estimate cost
estimate = estimator.estimate_cost(
    provider=CloudProvider.AWS,
    instance_name="p3.2xlarge",
    training_hours=10.0
)

print(f"Cost: ${estimate.total_spot_cost:.2f}")
print(f"Savings: ${estimate.potential_savings:.2f}")
```

### Compare Providers

```python
# Compare all providers
estimates = estimator.compare_providers(
    gpu_count=1,
    training_hours=10.0
)

for est in estimates[:3]:
    print(f"{est.provider.value}: ${est.total_spot_cost:.2f}")
```

### Track Carbon Footprint

```python
from neural.cost import CarbonTracker

tracker = CarbonTracker()

report = tracker.track_training(
    job_id="train-001",
    provider=CloudProvider.AWS,
    region="us-west-2",
    instance_type=instance_type,
    duration_hours=10.0
)

print(f"Carbon: {report.carbon_kg_co2:.2f} kg CO₂")
```

### Manage Budgets

```python
from neural.cost import BudgetManager

manager = BudgetManager()

# Create budget
budget = manager.create_budget(
    name="Q1-Training",
    total_amount=10000.0,
    period_days=90
)

# Record expense
manager.record_expense(
    budget_name="Q1-Training",
    amount=250.0,
    provider=CloudProvider.AWS
)

# Check status
status = manager.get_budget_status("Q1-Training")
print(f"Remaining: ${status['remaining_amount']:.2f}")
```

## CLI Usage

### Cost Estimation

```bash
# Estimate training cost
neural cost estimate \
  --provider aws \
  --instance p3.2xlarge \
  --hours 10.0 \
  --storage-gb 100.0 \
  --spot

# Compare providers
neural cost compare \
  --gpu-count 1 \
  --hours 10.0 \
  --max-cost 100.0
```

### Budget Management

```bash
# Create budget
neural cost create-budget \
  --name Q1-Training \
  --amount 10000.0 \
  --days 90 \
  --providers aws gcp

# Record expense
neural cost record-expense \
  --budget Q1-Training \
  --amount 250.0 \
  --provider aws \
  --description "Training run #1"

# Check status
neural cost budget-status --budget Q1-Training
```

### Carbon Tracking

```bash
# Track carbon footprint
neural cost track-carbon \
  --job-id train-001 \
  --provider aws \
  --region us-west-2 \
  --instance p3.2xlarge \
  --hours 10.0

# View summary
neural cost carbon-summary --days 30
```

### Dashboard

```bash
# Launch dashboard
neural cost dashboard --port 8052
```

## Python API

### Cost Estimator

```python
from neural.cost import CostEstimator, CloudProvider

estimator = CostEstimator()

# Estimate cost
estimate = estimator.estimate_cost(
    provider=CloudProvider.AWS,
    instance_name="p3.2xlarge",
    training_hours=10.0,
    storage_gb=100.0,
    data_transfer_gb=10.0,
    use_spot=True
)

# Compare providers
estimates = estimator.compare_providers(
    gpu_count=1,
    training_hours=10.0,
    gpu_type="V100"
)

# Find cheapest option
cheapest = estimator.get_cheapest_option(
    gpu_count=1,
    training_hours=10.0,
    max_cost=100.0
)

# Save/load pricing data
estimator.save_pricing_data("pricing.json")
```

### Spot Orchestrator

```python
from neural.cost import SpotInstanceOrchestrator, SpotStrategy

orchestrator = SpotInstanceOrchestrator(
    provider=CloudProvider.AWS,
    strategy=SpotStrategy.BALANCED,
    max_retries=3
)

# Launch spot instance
instance = orchestrator.launch_spot_instance(
    instance_type=instance_type,
    max_bid_price=1.0
)

# Monitor instances
statuses = orchestrator.monitor_instances()

# Handle interruption
orchestrator.handle_interruption(
    instance_id="spot-12345",
    resume_callback=lambda inst: print(f"Resumed on {inst}")
)

# Get savings
savings = orchestrator.get_cost_savings()
```

### Resource Optimizer

```python
from neural.cost import ResourceOptimizer, ResourceMetrics

optimizer = ResourceOptimizer()

# Analyze utilization
metrics = ResourceMetrics(
    cpu_utilization=0.45,
    memory_utilization=0.50,
    gpu_utilization=0.35,
    gpu_memory_utilization=0.40,
    samples=100
)

recommendation = optimizer.analyze_utilization(
    current_instance=instance_type,
    metrics=metrics
)

# Predict optimal instance
optimal = optimizer.predict_optimal_instance(
    provider=CloudProvider.AWS,
    workload_profile={
        'model_memory_gb': 16,
        'requires_gpu': True,
        'gpu_count': 1
    }
)
```

### Training Predictor

```python
from neural.cost import TrainingPredictor

predictor = TrainingPredictor()

# Predict training time
estimate = predictor.predict_training_time(
    model_params={'total_params': 50_000_000},
    dataset_size=100_000,
    batch_size=32,
    epochs=100,
    instance_type=instance_type
)

# Compare configurations
configs = [
    {'batch_size': 32, 'epochs': 100, 'instance_type': inst1},
    {'batch_size': 64, 'epochs': 100, 'instance_type': inst2},
]

estimates = predictor.compare_configurations(
    model_params={'total_params': 50_000_000},
    dataset_size=100_000,
    configs=configs
)

# Optimize batch size
optimal = predictor.optimize_batch_size(
    model_params={'total_params': 50_000_000},
    dataset_size=100_000,
    instance_type=instance_type
)
```

### Carbon Tracker

```python
from neural.cost import CarbonTracker

tracker = CarbonTracker()

# Track training
report = tracker.track_training(
    job_id="train-001",
    provider=CloudProvider.AWS,
    region="us-west-2",
    instance_type=instance_type,
    duration_hours=10.0
)

# Get total emissions
summary = tracker.get_total_emissions(time_period_days=30)

# Compare regions
comparisons = tracker.compare_regions(
    instance_type=instance_type,
    duration_hours=10.0
)

# Find greenest region
greenest = tracker.get_greenest_region(
    CloudProvider.AWS,
    instance_type
)

# Estimate offset cost
offset_cost = tracker.estimate_offset_cost(
    carbon_kg_co2=100.0,
    price_per_ton=15.0
)
```

### Cost Analyzer

```python
from neural.cost import CostAnalyzer, PerformanceMetrics

analyzer = CostAnalyzer()

# Analyze tradeoffs
analysis = analyzer.analyze_tradeoffs(
    configurations=configs,
    performance_metrics=metrics,
    costs=costs
)

# Optimize for budget
best = analyzer.optimize_for_budget(
    budget=500.0,
    configurations=configs,
    performance_metrics=metrics,
    costs=costs
)

# Optimize for performance
cheapest = analyzer.optimize_for_performance(
    target_accuracy=0.90,
    configurations=configs,
    performance_metrics=metrics,
    costs=costs
)

# Calculate ROI
roi = analyzer.calculate_roi(
    training_cost=367.0,
    improvement_accuracy=0.02,
    business_value_per_point=1000.0
)
```

### Budget Manager

```python
from neural.cost import BudgetManager

manager = BudgetManager()

# Create budget
budget = manager.create_budget(
    name="Q1-Training",
    total_amount=10000.0,
    period_days=90,
    providers=[CloudProvider.AWS, CloudProvider.GCP],
    alert_thresholds=[0.5, 0.8, 0.95]
)

# Record expense
manager.record_expense(
    budget_name="Q1-Training",
    amount=250.0,
    provider=CloudProvider.AWS,
    description="Training run #1"
)

# Get status
status = manager.get_budget_status("Q1-Training")

# Get spending report
report = manager.get_spending_report(days=30)

# Get alerts
alerts = manager.get_alerts(severity=AlertSeverity.CRITICAL)
```

## Dashboard

The cost dashboard provides a web-based interface for monitoring costs.

### Launching the Dashboard

```python
from neural.cost import CostDashboard

dashboard = CostDashboard(
    cost_estimator=estimator,
    budget_manager=manager,
    carbon_tracker=tracker,
    port=8052
)

dashboard.run(debug=False)
```

Or via CLI:

```bash
neural cost dashboard --port 8052
```

### Dashboard Features

- **Overview Tab**: Summary of costs, budgets, and carbon
- **Budgets Tab**: Detailed budget status and utilization
- **Estimation Tab**: Cost comparison and estimation tools
- **Carbon Tab**: Carbon footprint tracking and analysis
- **Analysis Tab**: Cost-performance analysis

## Best Practices

### 1. Use Spot Instances

Spot instances can save 60-70% compared to on-demand pricing.

```python
# Always use spot when possible
estimate = estimator.estimate_cost(
    provider=CloudProvider.AWS,
    instance_name="p3.2xlarge",
    training_hours=10.0,
    use_spot=True  # Enable spot pricing
)
```

### 2. Enable Checkpointing

Enable frequent checkpointing when using spot instances.

```python
from neural.cost.spot_orchestrator import CheckpointConfig

checkpoint_config = CheckpointConfig(
    enabled=True,
    frequency_minutes=15,  # Checkpoint every 15 minutes
    auto_resume=True
)

orchestrator = SpotInstanceOrchestrator(
    provider=CloudProvider.AWS,
    strategy=SpotStrategy.BALANCED,
    checkpoint_config=checkpoint_config
)
```

### 3. Monitor Utilization

Regularly check resource utilization for optimization opportunities.

```python
# Collect metrics during training
metrics = ResourceMetrics(
    cpu_utilization=0.45,
    memory_utilization=0.50,
    gpu_utilization=0.35,
    samples=100
)

# Get recommendations
recommendation = optimizer.analyze_utilization(
    current_instance=instance_type,
    metrics=metrics
)

if recommendation.estimated_savings_percent > 20:
    print(f"Switch to {recommendation.recommended_instance.name}")
```

### 4. Set Budget Alerts

Configure budget alerts to avoid overspending.

```python
# Create budget with multiple alert thresholds
budget = manager.create_budget(
    name="Monthly-Training",
    total_amount=5000.0,
    period_days=30,
    alert_thresholds=[0.5, 0.75, 0.9, 0.95]  # 50%, 75%, 90%, 95%
)

# Add custom alert callback
def alert_handler(alert):
    # Send to Slack, email, etc.
    print(f"ALERT: {alert.message}")

manager = BudgetManager(alert_callback=alert_handler)
```

### 5. Choose Green Regions

Use regions with lower carbon intensity when possible.

```python
# Find greenest region
greenest = tracker.get_greenest_region(
    CloudProvider.AWS,
    instance_type
)

print(f"Use region: {greenest}")

# Compare all regions
comparisons = tracker.compare_regions(
    instance_type=instance_type,
    duration_hours=10.0
)

# Use the greenest region that meets latency requirements
```

### 6. Predict Before Training

Always predict costs before starting long training runs.

```python
# Predict training time and cost
estimate = predictor.predict_training_time(
    model_params={'total_params': 100_000_000},
    dataset_size=500_000,
    batch_size=64,
    epochs=100,
    instance_type=instance_type
)

print(f"Estimated cost: ${estimate.total_cost:.2f}")
print(f"Estimated time: {estimate.estimated_hours:.1f} hours")

# Check if within budget
budget_status = manager.get_budget_status("Q1-Training")
if estimate.total_cost > budget_status['remaining_amount']:
    print("WARNING: Insufficient budget!")
```

### 7. Regular Cost Reviews

Review costs regularly to identify optimization opportunities.

```python
# Weekly cost review
report = manager.get_spending_report(days=7)
print(f"Last 7 days: ${report['total_amount']:.2f}")

# By provider
for provider, amount in report['by_provider'].items():
    print(f"  {provider}: ${amount:.2f}")

# Identify optimization opportunities
summary = optimizer.get_optimization_summary()
if summary['overprovisioned']:
    print("Action: Review resource sizing")
```

## Examples

See `neural/cost/examples.py` for comprehensive examples of all features.

Run examples:

```bash
python -m neural.cost.examples
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: https://github.com/Lemniscate-world/Neural
- Email: Lemniscate_zero@proton.me

## License

MIT License - See LICENSE file for details
