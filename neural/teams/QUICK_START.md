# Neural DSL Teams - Quick Start Guide

## Installation

```bash
# Install Neural DSL with teams support
pip install -e ".[teams]"

# Or install full package
pip install -e ".[full]"
```

## Basic Workflow

### 1. Create an Organization

```bash
neural teams org create "My Company" my-company \
  --owner-email owner@company.com \
  --billing-plan professional
```

### 2. Create a Team

```bash
neural teams create my-company "Data Science Team" \
  --description "Our main DS team"
```

### 3. Add Team Members

```bash
# Add an admin
neural teams add-member <team-id> admin@company.com --role admin

# Add developers
neural teams add-member <team-id> dev1@company.com --role developer
neural teams add-member <team-id> dev2@company.com --role developer

# Add viewers
neural teams add-member <team-id> viewer@company.com --role viewer
```

### 4. View Team Info

```bash
# List all teams
neural teams list my-company

# Show team details
neural teams show <team-id>
```

### 5. Monitor Usage

```bash
# View usage summary
neural teams analytics usage <team-id> --days 30

# Generate dashboard
neural teams analytics dashboard <team-id> --output dashboard.json
```

### 6. Check Billing

```bash
# View pricing plans
neural teams billing pricing

# Check subscription
neural teams billing subscription my-company

# Calculate usage costs
neural teams billing usage-cost \
  --compute-hours 150 \
  --storage-gb 50 \
  --api-calls 50000
```

## Python API Quick Start

```python
from neural.teams import (
    OrganizationManager,
    TeamManager,
    TeamModelRegistry,
    TeamExperimentTracker,
    BillingPlan,
    Role,
)

# Initialize managers
org_manager = OrganizationManager()
team_manager = TeamManager()

# Create organization
org = org_manager.create_organization(
    name="My Company",
    slug="my-company",
    owner_id="owner_123",
    billing_plan=BillingPlan.PROFESSIONAL,
)

# Create team
team = team_manager.create_team(
    name="Data Science Team",
    organization_id=org.org_id,
)

# Add user
user = team_manager.create_user(
    username="alice",
    email="alice@company.com",
    full_name="Alice Smith",
)

# Add to team
team_manager.add_member_to_team(
    team_id=team.team_id,
    user_id=user.user_id,
    role=Role.DEVELOPER,
)

# Register a model
registry = TeamModelRegistry()
model = registry.register_model(
    team=team,
    user_id=user.user_id,
    model_id="classifier_v1",
    name="Image Classifier",
    model_path="./model.h5",
    description="ResNet-50 classifier",
    framework="tensorflow",
    tags=["classification", "resnet"],
)

# Create experiment
tracker = TeamExperimentTracker()
exp = tracker.create_experiment(
    team=team,
    user_id=user.user_id,
    experiment_id="exp_001",
    name="Hyperparameter Tuning",
    tags=["tuning"],
)

# Log metrics
tracker.log_metrics(
    team=team,
    user_id=user.user_id,
    experiment_id="exp_001",
    metrics={"accuracy": 0.95, "loss": 0.12},
    step=100,
)
```

## Role Permissions

| Action | Admin | Developer | Viewer |
|--------|-------|-----------|--------|
| View models | ✓ | ✓ | ✓ |
| Create models | ✓ | ✓ | ✗ |
| Edit models | ✓ | ✓ | ✗ |
| Delete models | ✓ | ✗ | ✗ |
| View experiments | ✓ | ✓ | ✓ |
| Create experiments | ✓ | ✓ | ✗ |
| Edit experiments | ✓ | ✓ | ✗ |
| Delete experiments | ✓ | ✗ | ✗ |
| Manage team | ✓ | ✗ | ✗ |
| View billing | ✓ | ✗ | ✗ |
| Manage billing | ✓ | ✗ | ✗ |

## Billing Plans Comparison

| Feature | Free | Starter | Professional | Enterprise |
|---------|------|---------|--------------|------------|
| Price/month | $0 | $29 | $99 | $499 |
| Models | 10 | 50 | 200 | Unlimited |
| Experiments | 100 | 500 | 2,000 | Unlimited |
| Storage | 10 GB | 100 GB | 500 GB | Unlimited |
| Compute | 100 hrs | 1,000 hrs | 5,000 hrs | Unlimited |
| Team members | 5 | 10 | 50 | Unlimited |
| API calls/day | 10K | 100K | 1M | Unlimited |
| Support | Community | Priority | Priority | 24/7 Dedicated |

## Common Tasks

### Check Quota Usage

```python
team = team_manager.get_team(team_id)

print(f"Models: {team.quota.current_models}/{team.quota.max_models}")
print(f"Usage: {team.quota.usage_percentage('models')}%")
```

### List All Models in Team

```python
registry = TeamModelRegistry()
models = registry.list_models(
    team=team,
    user_id=user.user_id,
    tags=["classification"],
)

for model in models:
    print(f"{model['name']} ({model['framework']})")
```

### Track Usage

```python
from neural.teams import UsageAnalytics

analytics = UsageAnalytics()

# Log compute usage
analytics.log_compute_usage(
    team_id=team.team_id,
    user_id=user.user_id,
    duration_hours=5.5,
)

# Log storage usage
analytics.log_storage_usage(
    team_id=team.team_id,
    user_id=user.user_id,
    size_gb=12.5,
)

# Get summary
summary = analytics.get_usage_summary(team.team_id)
print(f"Total compute: {summary['compute_usage']['total_hours']} hours")
```

### Generate Analytics Dashboard

```python
from neural.teams import AnalyticsDashboard

dashboard = AnalyticsDashboard(analytics)
data = dashboard.generate_team_dashboard(team, days=30)

# Export to JSON
dashboard.export_dashboard_json(team, "dashboard.json", days=30)
```

## Support

- Documentation: `neural/teams/README.md`
- CLI Help: `neural teams --help`
- Tests: `tests/test_teams.py`

## Next Steps

1. Read the full documentation in `neural/teams/README.md`
2. Try the example scripts (when created)
3. Integrate with your existing Neural DSL workflows
4. Set up billing and quotas for your organization
5. Monitor usage and optimize costs
