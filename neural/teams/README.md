# Neural DSL - Team Management & Multi-Tenancy

A comprehensive multi-tenancy and team management system for Neural DSL, enabling organizations to collaborate, manage resources, and track usage in a SaaS environment.

## Features

### 🏢 Organization Management
- **Multi-tenant architecture** - Complete isolation between organizations
- **Flexible billing plans** - Free, Starter, Professional, Enterprise tiers
- **Organization-level settings** - Customizable configurations per org
- **Owner management** - Full control for organization owners

### 👥 Team Management
- **Multiple teams per organization** - Organize users into logical groups
- **Role-based access control (RBAC)** - Admin, Developer, Viewer roles
- **Team isolation** - Resources scoped to teams
- **Member management** - Easy invite/remove workflows

### 🔐 Role-Based Access Control

#### Roles & Permissions

**Admin Role:**
- Full access to all team resources
- Manage team members (invite/remove)
- Edit team settings and billing
- Delete models and experiments
- View analytics and quotas

**Developer Role:**
- Create and edit models
- Create and edit experiments
- View team resources
- View analytics
- Cannot manage team members or billing

**Viewer Role:**
- Read-only access to models
- Read-only access to experiments
- View team information
- View analytics
- Cannot create or modify resources

### 📊 Resource Quotas

Each billing plan includes:
- Maximum number of models
- Maximum number of experiments
- Storage capacity (GB)
- Compute hours per month
- Team member limit
- API call limits
- Concurrent training runs

### 🎯 Team Model Registry
- **Shared model storage** - All team members can access team models
- **Version tracking** - Track model versions and updates
- **Metadata & tags** - Organize models with tags and metadata
- **Download tracking** - Track model downloads and views
- **Permission checks** - Role-based access to model operations

### 🔬 Team Experiment Tracking
- **Shared experiments** - Collaborate on experiments within teams
- **Hyperparameter logging** - Track hyperparameters across experiments
- **Metrics tracking** - Log and visualize metrics over time
- **Artifact management** - Store and share experiment artifacts
- **Status tracking** - Monitor experiment lifecycle

### 📈 Usage Analytics
- **Real-time metrics** - Track compute, storage, and API usage
- **Time-series data** - Visualize usage trends over time
- **Member activity** - Monitor individual team member contributions
- **Cost tracking** - Calculate usage-based costs
- **Dashboard generation** - Comprehensive analytics dashboards

### 💳 Billing Integration
- **Subscription management** - Create and manage subscriptions
- **Invoice generation** - Automated invoice creation
- **Payment methods** - Support for multiple payment methods
- **Usage-based pricing** - Calculate costs based on actual usage
- **Stripe integration** - Ready-to-use Stripe payment processor

## Quick Start

### Create an Organization

```bash
neural teams org create "Acme Corp" acme-corp --owner-email owner@acme.com --billing-plan professional
```

### Create a Team

```bash
neural teams create acme-corp "ML Team" --description "Machine Learning Research Team"
```

### Add Team Members

```bash
# Add an admin
neural teams add-member <team-id> admin@acme.com --role admin

# Add a developer
neural teams add-member <team-id> dev@acme.com --role developer

# Add a viewer
neural teams add-member <team-id> viewer@acme.com --role viewer
```

### View Team Analytics

```bash
# Show usage summary
neural teams analytics usage <team-id> --days 30

# Generate dashboard
neural teams analytics dashboard <team-id> --output dashboard.json
```

### Billing Operations

```bash
# View pricing
neural teams billing pricing

# Check subscription
neural teams billing subscription acme-corp

# List invoices
neural teams billing invoices acme-corp

# Calculate usage cost
neural teams billing usage-cost --compute-hours 150 --storage-gb 50 --api-calls 50000
```

## Python API Usage

### Organization Management

```python
from neural.teams import OrganizationManager, BillingPlan

# Create organization manager
org_manager = OrganizationManager()

# Create an organization
org = org_manager.create_organization(
    name="Acme Corp",
    slug="acme-corp",
    owner_id="user_123",
    billing_plan=BillingPlan.PROFESSIONAL,
    billing_email="billing@acme.com",
)

# List organizations
orgs = org_manager.list_organizations()

# Update organization
org_manager.update_organization(
    org_id=org.org_id,
    billing_plan=BillingPlan.ENTERPRISE,
)
```

### Team Management

```python
from neural.teams import TeamManager, Role

# Create team manager
team_manager = TeamManager()

# Create a team
team = team_manager.create_team(
    name="ML Team",
    organization_id=org.org_id,
    description="Machine Learning Research",
)

# Create user
user = team_manager.create_user(
    username="johndoe",
    email="john@acme.com",
    full_name="John Doe",
)

# Add member to team
team_manager.add_member_to_team(
    team_id=team.team_id,
    user_id=user.user_id,
    role=Role.DEVELOPER,
)
```

### Access Control

```python
from neural.teams import AccessController, Permission

# Check permissions
can_edit = AccessController.has_permission(
    user_id=user.user_id,
    team=team,
    permission=Permission.EDIT_MODELS,
)

# Require permission (raises PermissionError if denied)
AccessController.require_permission(
    user_id=user.user_id,
    team=team,
    permission=Permission.DELETE_MODELS,
)
```

### Model Registry

```python
from neural.teams import TeamModelRegistry

# Create registry
registry = TeamModelRegistry()

# Register a model
model_info = registry.register_model(
    team=team,
    user_id=user.user_id,
    model_id="model_123",
    name="Image Classifier",
    model_path="./model.h5",
    description="ResNet-50 based classifier",
    version="1.0.0",
    framework="tensorflow",
    tags=["classification", "resnet"],
)

# List models
models = registry.list_models(
    team=team,
    user_id=user.user_id,
    tags=["classification"],
)

# Download model
path = registry.download_model(
    team=team,
    user_id=user.user_id,
    model_id="model_123",
    output_dir="./downloads",
)
```

### Experiment Tracking

```python
from neural.teams import TeamExperimentTracker

# Create tracker
tracker = TeamExperimentTracker()

# Create experiment
experiment = tracker.create_experiment(
    team=team,
    user_id=user.user_id,
    experiment_id="exp_123",
    name="Hyperparameter Tuning",
    description="Finding optimal learning rate",
    tags=["tuning", "resnet"],
)

# Log hyperparameters
tracker.log_hyperparameters(
    team=team,
    user_id=user.user_id,
    experiment_id="exp_123",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
    },
)

# Log metrics
tracker.log_metrics(
    team=team,
    user_id=user.user_id,
    experiment_id="exp_123",
    metrics={"accuracy": 0.95, "loss": 0.12},
    step=100,
)

# Update status
tracker.update_experiment(
    team=team,
    user_id=user.user_id,
    experiment_id="exp_123",
    status="completed",
)
```

### Usage Analytics

```python
from neural.teams import UsageAnalytics, AnalyticsDashboard
from datetime import datetime, timedelta

# Create analytics
analytics = UsageAnalytics()

# Log compute usage
analytics.log_compute_usage(
    team_id=team.team_id,
    user_id=user.user_id,
    duration_hours=2.5,
    resource_type="training",
)

# Log storage usage
analytics.log_storage_usage(
    team_id=team.team_id,
    user_id=user.user_id,
    size_gb=5.2,
    resource_type="model",
)

# Get usage summary
summary = analytics.get_usage_summary(
    team_id=team.team_id,
    start_date=datetime.now() - timedelta(days=30),
)

# Generate dashboard
dashboard = AnalyticsDashboard(analytics)
data = dashboard.generate_team_dashboard(team, days=30)
```

### Billing Management

```python
from neural.teams import BillingManager, StripeIntegration

# Create billing manager
billing = BillingManager()

# Get plan pricing
pricing = billing.get_plan_pricing(BillingPlan.PROFESSIONAL)

# Create subscription
subscription = billing.create_subscription(
    org=org,
    plan=BillingPlan.PROFESSIONAL,
    billing_cycle="monthly",
)

# Create invoice
invoice = billing.create_invoice(
    org_id=org.org_id,
    amount=99.00,
    description="Monthly subscription - Professional Plan",
)

# Calculate usage cost
cost = billing.calculate_usage_cost(
    compute_hours=150.0,
    storage_gb=50.0,
    api_calls=50000,
)

# Stripe integration
stripe = StripeIntegration(api_key="sk_live_...")
customer_id = stripe.create_customer(org, email=org.billing_email)
```

## Billing Plans

### Free Plan
- **Price:** $0/month
- **Features:**
  - 10 models
  - 100 experiments
  - 10 GB storage
  - 100 compute hours/month
  - 5 team members
  - 10,000 API calls/day
  - 5 concurrent runs

### Starter Plan
- **Price:** $29/month ($290/year)
- **Features:**
  - 50 models
  - 500 experiments
  - 100 GB storage
  - 1,000 compute hours/month
  - 10 team members
  - 100,000 API calls/day
  - 10 concurrent runs
  - Priority support

### Professional Plan
- **Price:** $99/month ($990/year)
- **Features:**
  - 200 models
  - 2,000 experiments
  - 500 GB storage
  - 5,000 compute hours/month
  - 50 team members
  - 1,000,000 API calls/day
  - 25 concurrent runs
  - Priority support
  - Advanced analytics

### Enterprise Plan
- **Price:** $499/month ($4,990/year)
- **Features:**
  - Unlimited models
  - Unlimited experiments
  - Unlimited storage
  - Unlimited compute hours
  - Unlimited team members
  - Unlimited API calls
  - 100 concurrent runs
  - 24/7 dedicated support
  - Advanced analytics
  - Custom integrations
  - SLA guarantee

## Usage-Based Pricing

In addition to plan quotas, usage beyond limits is billed at:
- **Compute:** $0.50 per hour
- **Storage:** $0.10 per GB per month
- **API Calls:** $0.01 per 1,000 calls

## Architecture

### Data Storage Structure

```
neural_organizations/
├── organizations.json          # Organization metadata
├── {org_id}/
│   ├── teams.json             # Team metadata
│   └── users.json             # User metadata
├── registries/
│   └── {team_id}/
│       ├── models.json        # Model registry
│       └── {model_id}/        # Model storage
├── experiments/
│   └── {team_id}/
│       ├── experiments.json   # Experiment metadata
│       └── {exp_id}/          # Experiment artifacts
├── analytics/
│   └── {team_id}/
│       └── usage.json         # Usage events
└── billing/
    └── {org_id}.json          # Billing data
```

### Security Considerations

1. **Access Control:** All operations verify user permissions
2. **Data Isolation:** Organizations and teams are completely isolated
3. **Audit Logging:** All actions are logged for compliance
4. **Quota Enforcement:** Resource limits enforced at creation time
5. **Payment Security:** Stripe integration for PCI compliance

## CLI Reference

### Organization Commands

- `neural teams org create` - Create new organization
- `neural teams org list` - List organizations
- `neural teams org show` - Show organization details
- `neural teams org update` - Update organization settings

### Team Commands

- `neural teams create` - Create new team
- `neural teams list` - List teams in organization
- `neural teams show` - Show team details
- `neural teams add-member` - Add member to team
- `neural teams remove-member` - Remove member from team

### Analytics Commands

- `neural teams analytics dashboard` - Generate analytics dashboard
- `neural teams analytics usage` - Show usage summary

### Billing Commands

- `neural teams billing pricing` - Show pricing plans
- `neural teams billing subscription` - Show subscription details
- `neural teams billing invoices` - List invoices
- `neural teams billing usage-cost` - Calculate usage costs

## Integration Guide

### Integrating with Existing Code

```python
from neural.teams import TeamManager, AccessController, Permission

def train_model(team_id: str, user_id: str, model_config: dict):
    """Train a model with team access control."""
    team_manager = TeamManager()
    team = team_manager.get_team(team_id)
    
    # Check if user can create models
    if not AccessController.has_permission(
        user_id=user_id,
        team=team,
        permission=Permission.CREATE_MODELS,
    ):
        raise PermissionError("User cannot create models")
    
    # Check quota
    if not team.quota.check_quota('models'):
        raise ValueError("Team has reached model quota")
    
    # Train model...
    # Update quota
    team.quota.current_models += 1
```

## Best Practices

1. **Use proper roles** - Assign minimal required permissions
2. **Monitor quotas** - Set up alerts for quota usage
3. **Regular analytics** - Review team analytics weekly
4. **Audit access** - Regularly review team member access
5. **Cost optimization** - Monitor and optimize resource usage

## Support

For issues or questions:
- GitHub Issues: https://github.com/Lemniscate-world/Neural/issues
- Documentation: https://neural-dsl.readthedocs.io/
- Email: support@neural-dsl.io

## License

MIT License - See LICENSE.md for details
