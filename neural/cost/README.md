# Neural DSL - Cost Optimization Module

Comprehensive cost optimization capabilities for neural network training in the cloud.

## Features

### 1. Cost Estimation (`estimator.py`)
- **Multi-Cloud Support**: AWS, GCP, Azure
- **Instance Pricing**: On-demand and spot instance pricing
- **Cost Breakdown**: Compute, storage, and data transfer costs
- **Provider Comparison**: Compare costs across providers
- **Cost Optimization**: Find cheapest options for requirements

### 2. Spot Instance Orchestration (`spot_orchestrator.py`)
- **Spot Instance Management**: Launch and monitor spot instances
- **Interruption Handling**: Automatic recovery from spot interruptions
- **Checkpoint Integration**: Fault-tolerant training with checkpoints
- **Cost Tracking**: Track savings from spot instances
- **Strategy Options**: Aggressive, balanced, or conservative bidding

### 3. Resource Right-Sizing (`resource_optimizer.py`)
- **Utilization Analysis**: Analyze CPU, memory, and GPU usage
- **Right-Sizing Recommendations**: Optimize instance size
- **Overprovisioning Detection**: Identify wasted resources
- **Underprovisioning Alerts**: Detect resource constraints
- **Batch Analysis**: Analyze multiple instances at once

### 4. Training Time Prediction (`training_predictor.py`)
- **Time Estimation**: Predict training duration
- **Cost Prediction**: Estimate total training costs
- **Historical Learning**: Improve predictions from past runs
- **Configuration Comparison**: Compare different setups
- **Batch Size Optimization**: Find optimal batch size

### 5. Carbon Footprint Tracking (`carbon_tracker.py`)
- **Emission Tracking**: Monitor CO₂ emissions
- **Regional Comparison**: Compare carbon intensity by region
- **Energy Consumption**: Track energy usage (kWh)
- **Equivalence Metrics**: Miles driven, trees needed
- **Green Region Finder**: Identify lowest-carbon regions

### 6. Cost-Performance Analysis (`analyzer.py`)
- **Tradeoff Analysis**: Analyze cost vs performance
- **Pareto Frontier**: Identify optimal configurations
- **Budget Optimization**: Find best option within budget
- **Performance Optimization**: Cheapest way to meet targets
- **ROI Calculator**: Calculate return on investment

### 7. Budget Management (`budget_manager.py`)
- **Budget Creation**: Set budgets with time periods
- **Expense Tracking**: Track all training expenses
- **Alert Thresholds**: Configurable budget alerts
- **Burn Rate Monitoring**: Track spending velocity
- **Spending Reports**: Comprehensive cost reports

### 8. Cost Dashboard (`dashboard.py`)
- **Interactive UI**: Web-based cost monitoring
- **Real-Time Updates**: Live cost tracking
- **Visualizations**: Charts and graphs
- **Multi-Tab Interface**: Overview, budgets, estimation, carbon, analysis
- **Alert Display**: Budget and cost alerts

## Installation

```bash
# Install with cost optimization features
pip install -e ".[full]"

# Or install specific dependencies
pip install -e ".[dashboard,integrations]"
```

## Quick Start

### Basic Cost Estimation

```python
from neural.cost import CostEstimator, CloudProvider

estimator = CostEstimator()

# Estimate cost for AWS p3.2xlarge, 10 hours
estimate = estimator.estimate_cost(
    provider=CloudProvider.AWS,
    instance_name="p3.2xlarge",
    training_hours=10.0,
    use_spot=True
)

print(f"On-demand cost: ${estimate.on_demand_cost:.2f}")
print(f"Spot cost: ${estimate.spot_cost:.2f}")
print(f"Savings: ${estimate.potential_savings:.2f}")

# Compare providers
estimates = estimator.compare_providers(
    gpu_count=1,
    training_hours=10.0
)

for est in estimates[:3]:
    print(f"{est.provider.value}: ${est.total_spot_cost:.2f}")
```

### Spot Instance Orchestration

```python
from neural.cost import SpotInstanceOrchestrator, SpotStrategy, CloudProvider
from neural.cost.estimator import InstanceType

orchestrator = SpotInstanceOrchestrator(
    provider=CloudProvider.AWS,
    strategy=SpotStrategy.BALANCED
)

# Launch spot instance
instance_type = estimator.instance_types[CloudProvider.AWS][0]
instance = orchestrator.launch_spot_instance(instance_type)

# Monitor instances
statuses = orchestrator.monitor_instances()

# Get cost savings
savings = orchestrator.get_cost_savings()
print(f"Total saved: ${savings['total_cost_saved']:.2f}")
```

### Resource Optimization

```python
from neural.cost import ResourceOptimizer, ResourceMetrics

optimizer = ResourceOptimizer()

# Analyze current utilization
metrics = ResourceMetrics(
    cpu_utilization=0.45,
    memory_utilization=0.50,
    gpu_utilization=0.35,
    gpu_memory_utilization=0.40
)

recommendation = optimizer.analyze_utilization(
    current_instance=instance_type,
    metrics=metrics
)

print(f"Recommendation: {recommendation.reason}")
print(f"Estimated savings: {recommendation.estimated_savings_percent:.1f}%")
```

### Training Prediction

```python
from neural.cost import TrainingPredictor

predictor = TrainingPredictor()

# Predict training time and cost
model_params = {'total_params': 50_000_000}
estimate = predictor.predict_training_time(
    model_params=model_params,
    dataset_size=100_000,
    batch_size=32,
    epochs=100,
    instance_type=instance_type
)

print(f"Estimated time: {estimate.estimated_hours:.1f} hours")
print(f"Estimated cost: ${estimate.total_cost:.2f}")
print(f"Confidence: {estimate.confidence_interval}")
```

### Carbon Tracking

```python
from neural.cost import CarbonTracker

tracker = CarbonTracker()

# Track training job
report = tracker.track_training(
    job_id="train-001",
    provider=CloudProvider.AWS,
    region="us-west-2",
    instance_type=instance_type,
    duration_hours=10.0
)

print(f"Carbon emissions: {report.carbon_kg_co2:.2f} kg CO₂")
print(f"Equivalent to {report.equivalent_miles_driven:.0f} miles driven")
print(f"Energy used: {report.energy_kwh:.2f} kWh")

# Find greenest region
greenest = tracker.get_greenest_region(CloudProvider.AWS, instance_type)
print(f"Greenest region: {greenest}")
```

### Budget Management

```python
from neural.cost import BudgetManager

manager = BudgetManager()

# Create budget
budget = manager.create_budget(
    name="Q1-Training",
    total_amount=10000.0,
    period_days=90,
    alert_thresholds=[0.5, 0.8, 0.95]
)

# Record expenses
manager.record_expense(
    budget_name="Q1-Training",
    amount=250.0,
    provider=CloudProvider.AWS,
    description="Model training run #1"
)

# Check status
status = manager.get_budget_status("Q1-Training")
print(f"Spent: ${status['spent_amount']:.2f}")
print(f"Remaining: ${status['remaining_amount']:.2f}")
print(f"Status: {status['status']}")
```

### Cost-Performance Analysis

```python
from neural.cost import CostAnalyzer, PerformanceMetrics

analyzer = CostAnalyzer()

# Analyze tradeoffs
configs = [
    {'instance': 'p3.2xlarge', 'batch_size': 32},
    {'instance': 'p3.8xlarge', 'batch_size': 128},
]

metrics = [
    PerformanceMetrics(accuracy=0.85, training_time_hours=12.0),
    PerformanceMetrics(accuracy=0.87, training_time_hours=4.0),
]

costs = [92.0, 367.0]

analysis = analyzer.analyze_tradeoffs(configs, metrics, costs)

print(f"Best value: {analysis.best_value.configuration}")
print(f"Efficiency: {analysis.best_value.efficiency_score:.4f}")
```

### Cost Dashboard

```python
from neural.cost import CostDashboard

# Create dashboard
dashboard = CostDashboard(
    cost_estimator=estimator,
    budget_manager=manager,
    carbon_tracker=tracker,
    port=8052
)

# Run dashboard
dashboard.run(debug=False)
# Visit http://localhost:8052
```

## Advanced Usage

### Custom Pricing Data

```python
# Save current pricing
estimator.save_pricing_data("pricing.json")

# Load custom pricing
custom_estimator = CostEstimator(pricing_data_path="pricing.json")
```

### Historical Training Data

```python
from neural.cost.training_predictor import TrainingHistory

# Add historical data
predictor.add_training_record(TrainingHistory(
    model_size=50_000_000,
    dataset_size=100_000,
    batch_size=32,
    epochs=100,
    actual_hours=8.5,
    instance_type="p3.2xlarge",
    gpu_count=1
))

# Fit predictor for better predictions
predictor.fit_from_history()
```

### Budget Alerts with Callbacks

```python
def alert_handler(alert):
    print(f"ALERT: {alert.message}")
    # Send to Slack, email, etc.

manager = BudgetManager(alert_callback=alert_handler)
```

### Multi-Region Carbon Comparison

```python
# Compare all regions
comparisons = tracker.compare_regions(
    instance_type=instance_type,
    duration_hours=10.0
)

for comp in comparisons[:5]:
    print(f"{comp['region']}: {comp['carbon_kg_co2']:.2f} kg CO₂")
```

## Architecture

```
neural/cost/
├── __init__.py           # Module exports
├── estimator.py          # Cost estimation engine
├── spot_orchestrator.py  # Spot instance management
├── resource_optimizer.py # Resource right-sizing
├── training_predictor.py # Training time prediction
├── carbon_tracker.py     # Carbon footprint tracking
├── analyzer.py           # Cost-performance analysis
├── budget_manager.py     # Budget management
├── dashboard.py          # Interactive dashboard
└── README.md            # This file
```

## Data Storage

The cost module stores data in these directories:
- `budget_data/` - Budget configurations and expenses
- `carbon_data/` - Carbon emission reports
- `monitoring_data/` - General monitoring data

## Integration with Neural DSL

```python
from neural.cost import CostEstimator, BudgetManager, CarbonTracker

# Integrate with training workflow
def train_with_cost_tracking(model, dataset, config):
    # Estimate costs
    estimator = CostEstimator()
    estimate = estimator.estimate_cost(...)
    
    # Check budget
    manager = BudgetManager()
    budget_ok = estimate.total_cost <= manager.budgets['training'].remaining_amount
    
    if not budget_ok:
        raise ValueError("Insufficient budget")
    
    # Train model
    start_time = time.time()
    train(model, dataset, config)
    duration = (time.time() - start_time) / 3600
    
    # Track costs and carbon
    manager.record_expense(...)
    tracker = CarbonTracker()
    tracker.track_training(...)
```

## API Reference

See docstrings in each module for detailed API documentation.

## Contributing

Contributions welcome! Areas for improvement:
- Additional cloud providers
- More accurate cost models
- Better prediction algorithms
- Enhanced dashboard features
- Integration with more ML platforms

## License

MIT License - see main LICENSE file
