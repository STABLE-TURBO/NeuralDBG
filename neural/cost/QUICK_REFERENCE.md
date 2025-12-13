# Cost Optimization - Quick Reference

## Installation

```bash
pip install -e ".[full]"  # All features
pip install -e ".[dashboard]"  # With dashboard
```

## Python API Quick Reference

### Cost Estimation

```python
from neural.cost import CostEstimator, CloudProvider

estimator = CostEstimator()

# Estimate single instance
estimate = estimator.estimate_cost(
    provider=CloudProvider.AWS,
    instance_name="p3.2xlarge",
    training_hours=10.0
)

# Compare providers
estimates = estimator.compare_providers(gpu_count=1, training_hours=10.0)

# Find cheapest
cheapest = estimator.get_cheapest_option(gpu_count=1, training_hours=10.0)
```

### Spot Instances

```python
from neural.cost import SpotInstanceOrchestrator, SpotStrategy

orchestrator = SpotInstanceOrchestrator(
    provider=CloudProvider.AWS,
    strategy=SpotStrategy.BALANCED
)

# Launch spot instance
instance = orchestrator.launch_spot_instance(instance_type)

# Monitor
statuses = orchestrator.monitor_instances()

# Get savings
savings = orchestrator.get_cost_savings()
```

### Resource Optimization

```python
from neural.cost import ResourceOptimizer, ResourceMetrics

optimizer = ResourceOptimizer()

metrics = ResourceMetrics(
    cpu_utilization=0.45,
    memory_utilization=0.50,
    gpu_utilization=0.35
)

recommendation = optimizer.analyze_utilization(instance_type, metrics)
```

### Training Prediction

```python
from neural.cost import TrainingPredictor

predictor = TrainingPredictor()

estimate = predictor.predict_training_time(
    model_params={'total_params': 50_000_000},
    dataset_size=100_000,
    batch_size=32,
    epochs=100,
    instance_type=instance_type
)
```

### Carbon Tracking

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

# Find greenest region
greenest = tracker.get_greenest_region(CloudProvider.AWS, instance_type)
```

### Budget Management

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
```

### Cost Analysis

```python
from neural.cost import CostAnalyzer, PerformanceMetrics

analyzer = CostAnalyzer()

analysis = analyzer.analyze_tradeoffs(
    configurations=configs,
    performance_metrics=metrics,
    costs=costs
)

roi = analyzer.calculate_roi(
    training_cost=367.0,
    improvement_accuracy=0.02
)
```

### Dashboard

```python
from neural.cost import CostDashboard

dashboard = CostDashboard(port=8052)
dashboard.run()
```

## CLI Quick Reference

### Cost Estimation

```bash
# Estimate cost
neural cost estimate --provider aws --instance p3.2xlarge --hours 10.0

# Compare providers
neural cost compare --gpu-count 1 --hours 10.0

# With max cost filter
neural cost compare --gpu-count 1 --hours 10.0 --max-cost 100.0
```

### Budget Management

```bash
# Create budget
neural cost create-budget --name Q1 --amount 10000.0 --days 90

# Record expense
neural cost record-expense --budget Q1 --amount 250.0 --provider aws --description "Training run"

# Check status
neural cost budget-status --budget Q1

# List all budgets
neural cost budget-status
```

### Carbon Tracking

```bash
# Track carbon
neural cost track-carbon \
  --job-id train-001 \
  --provider aws \
  --region us-west-2 \
  --instance p3.2xlarge \
  --hours 10.0

# View summary
neural cost carbon-summary --days 30
```

### Training Prediction

```bash
neural cost predict \
  --model-params 50000000 \
  --dataset-size 100000 \
  --batch-size 32 \
  --epochs 100 \
  --provider aws \
  --instance p3.2xlarge
```

### Dashboard

```bash
# Launch dashboard
neural cost dashboard --port 8052

# Visit http://localhost:8052
```

## Common Instance Types

### AWS
- `p3.2xlarge` - V100 x1 ($3.06/hr, spot $0.92/hr)
- `p3.8xlarge` - V100 x4 ($12.24/hr, spot $3.67/hr)
- `p4d.24xlarge` - A100 x8 ($32.77/hr, spot $9.83/hr)
- `g4dn.xlarge` - T4 x1 ($0.526/hr, spot $0.158/hr)

### GCP
- `n1-standard-8-v100` - V100 x1 ($2.48/hr, spot $0.74/hr)
- `a2-highgpu-1g` - A100 x1 ($3.67/hr, spot $1.10/hr)
- `a2-highgpu-8g` - A100 x8 ($29.39/hr, spot $8.82/hr)

### Azure
- `Standard_NC6s_v3` - V100 x1 ($3.06/hr, spot $0.92/hr)
- `Standard_ND96asr_v4` - A100 x8 ($27.20/hr, spot $8.16/hr)

## Spot Strategies

- `AGGRESSIVE` - 50% of on-demand price (max savings, higher interruption risk)
- `BALANCED` - 70% of on-demand price (good balance)
- `CONSERVATIVE` - 90% of on-demand price (lower risk, less savings)

## Budget Alert Thresholds

Default thresholds: `[0.5, 0.8, 0.95]` (50%, 80%, 95%)

Custom thresholds:
```python
budget = manager.create_budget(
    name="Budget",
    total_amount=10000.0,
    period_days=90,
    alert_thresholds=[0.5, 0.75, 0.9, 0.95]
)
```

## Carbon Regions (by intensity)

### Lowest Carbon
1. `northeurope` (Azure) - 217.2 g CO₂/kWh
2. `us-west-2` (AWS) - 285.7 g CO₂/kWh
3. `europe-west4` (GCP) - 308.9 g CO₂/kWh

### Highest Carbon
1. `ap-northeast-1` (AWS) - 464.7 g CO₂/kWh
2. `asia-southeast1` (GCP) - 431.1 g CO₂/kWh
3. `us-east-1` (AWS) - 415.8 g CO₂/kWh

## Dashboard Tabs

1. **Overview** - Cost summary, spending charts, budget status
2. **Budgets** - Detailed budget tracking and utilization
3. **Estimation** - Cost comparison and estimation tools
4. **Carbon** - Carbon footprint tracking and analysis
5. **Analysis** - Cost-performance tradeoff analysis

## File Locations

- Budget data: `budget_data/`
- Carbon reports: `carbon_data/`
- Custom pricing: Load/save with `estimator.save_pricing_data()`

## Common Workflows

### Before Training
1. Estimate costs
2. Check budget
3. Predict training time
4. Compare providers
5. Choose greenest region

### During Training
1. Monitor spot instances
2. Track utilization
3. Record expenses
4. Monitor budget status

### After Training
1. Track carbon footprint
2. Analyze cost-performance
3. Update predictions with actual data
4. Review resource utilization
5. Generate reports

## Best Practices

1. ✅ Always use spot instances when possible (60-70% savings)
2. ✅ Enable checkpointing with spot instances
3. ✅ Set budget alerts at multiple thresholds
4. ✅ Monitor resource utilization regularly
5. ✅ Choose green regions when latency permits
6. ✅ Predict costs before long training runs
7. ✅ Track all expenses for accurate reporting
8. ✅ Review optimization recommendations weekly

## Example Integration

```python
from neural.cost import (
    CostEstimator, BudgetManager, CarbonTracker, TrainingPredictor
)

# Setup
estimator = CostEstimator()
manager = BudgetManager()
tracker = CarbonTracker()
predictor = TrainingPredictor(estimator)

# Create budget
budget = manager.create_budget("Project-Alpha", 5000.0, 30)

# Predict costs
estimate = predictor.predict_training_time(
    model_params={'total_params': 100_000_000},
    dataset_size=500_000,
    batch_size=64,
    epochs=50,
    instance_type=instance_type
)

# Check budget
budget_status = manager.get_budget_status("Project-Alpha")
if estimate.total_cost > budget_status['remaining_amount']:
    print("WARNING: Insufficient budget!")

# Train model
# ... training code ...

# Track carbon
carbon_report = tracker.track_training(
    job_id="project-alpha-001",
    provider=CloudProvider.AWS,
    region="us-west-2",
    instance_type=instance_type,
    duration_hours=estimate.estimated_hours
)

# Record expense
manager.record_expense(
    budget_name="Project-Alpha",
    amount=estimate.total_cost,
    provider=CloudProvider.AWS
)
```

## Troubleshooting

### "Instance type not found"
- Check instance name spelling
- Verify provider matches instance type

### "Dashboard dependencies not available"
```bash
pip install neural-dsl[dashboard]
```

### "Budget not found"
- Verify budget name
- Check storage path

### Pricing seems outdated
```python
# Update pricing data
estimator.save_pricing_data("pricing.json")
# Edit pricing.json with latest prices
estimator = CostEstimator(pricing_data_path="pricing.json")
```

## Support

- GitHub: https://github.com/Lemniscate-world/Neural
- Issues: https://github.com/Lemniscate-world/Neural/issues
- Email: Lemniscate_zero@proton.me
