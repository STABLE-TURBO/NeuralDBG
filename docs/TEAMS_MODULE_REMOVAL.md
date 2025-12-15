# Teams Module Removal Summary

## Overview

The `neural/teams/` module has been removed as part of codebase simplification efforts. This module was not actively used by core Neural DSL features (parser, code generation, shape propagation).

## What Was Removed

### Files Removed
- `neural/teams/billing.py` - Billing integration and Stripe payment processing
- `neural/teams/analytics.py` - Usage analytics and dashboard generation
- `neural/teams/cli_commands.py` - CLI commands for team/org management
- `neural/teams/manager.py` - Team and organization management logic
- `neural/teams/access_control.py` - Role-based access control (RBAC)
- `neural/teams/models.py` - Data models (Organization, Team, User, Role, etc.)
- `neural/teams/team_registry.py` - Team model registry
- `neural/teams/team_tracking.py` - Team experiment tracking
- `neural/teams/config.py` - Team configuration

### Features Removed
1. **Billing Integration**
   - Stripe payment processing
   - Subscription management
   - Invoice generation
   - Usage cost calculation
   - Billing plan management (Free, Starter, Professional, Enterprise)

2. **Analytics Dashboard**
   - Usage tracking and analytics
   - Dashboard generation
   - Usage summary reports
   - Event logging (compute, storage, API calls)

3. **Team Management**
   - Organization creation and management
   - Team creation within organizations
   - User management
   - Member addition/removal
   - Role assignment

4. **RBAC (Role-Based Access Control)**
   - Permission system
   - Role hierarchy (Admin, Developer, Viewer)
   - Access control checks
   - Resource quotas per team

5. **CLI Commands**
   - `neural teams` - All team management commands
   - `neural org` - Organization management commands
   - `neural analytics` - Analytics commands
   - `neural billing` - Billing commands

### Dependencies Removed from setup.py
- `TEAMS_DEPS` group removed from extras_require
- No longer installable via `pip install neural-dsl[teams]`
- Dependencies removed from "full" installation bundle

## Why It Was Removed

1. **Not Used by Core Features**: The teams module was not a dependency for:
   - DSL parsing (`neural/parser/`)
   - Code generation (`neural/code_generation/`)
   - Shape propagation (`neural/shape_propagation/`)
   - CLI core commands (compile, run, visualize, debug)

2. **Peripheral Functionality**: Multi-tenancy, billing, and analytics are:
   - SaaS-specific features
   - Not needed for local DSL usage
   - Better suited as separate services if needed

3. **Simplification Goals**: Focusing on core DSL functionality:
   - Reduce codebase complexity
   - Minimize maintenance burden
   - Improve focus on essential features

## Impact Analysis

### No Impact
- ✅ Core DSL compilation and execution
- ✅ Code generation (TensorFlow, PyTorch, ONNX)
- ✅ Shape propagation and validation
- ✅ Visualization and debugging
- ✅ HPO and AutoML features
- ✅ Integrations with ML platforms
- ✅ Experiment tracking (via experiment tracker module)

### Affected Features
- ❌ Team-based collaboration features
- ❌ Multi-tenancy in SaaS deployments
- ❌ Billing and subscription management
- ❌ Usage analytics and dashboards
- ❌ Team-specific resource quotas

## Migration Path

If you need team/billing features:

1. **For Team Collaboration**: Use Git-based workflows
   - Version control with Git
   - Code review via GitHub/GitLab
   - Team access via repository permissions

2. **For Multi-tenancy**: Build as separate service
   - Authentication/authorization service
   - User management database
   - Integration via API

3. **For Billing**: Use existing platforms
   - Stripe directly
   - Chargebee
   - Paddle
   - Other billing services

4. **For Analytics**: Use monitoring tools
   - Prometheus + Grafana
   - ELK Stack
   - Cloud provider monitoring
   - Custom analytics service

## Files Modified

1. **neural/teams/__init__.py** - Now contains only deprecation notice
2. **neural/cli/cli.py** - Removed teams command registration
3. **tests/test_teams.py** - Converted to skip all tests
4. **setup.py** - Removed TEAMS_DEPS and "teams" extra
5. **AGENTS.md** - Updated documentation
6. **ARCHITECTURE.md** - Removed teams from architecture diagram
7. **docs/installation.md** - Removed teams installation instructions

## Testing

The test file `tests/test_teams.py` now contains only a skip marker:
- All tests are skipped with reason "Teams module has been removed"
- Tests can be fully deleted in future cleanup

## Recommendations

For projects that need these features:

1. **Keep teams functionality external** to the core DSL
2. **Use microservices architecture** for SaaS features
3. **Focus DSL on compilation and code generation**
4. **Integrate with external tools** for collaboration and billing

## Version History

- **v0.3.0**: Teams module simplified and removed
- Previously: Full-featured multi-tenancy and billing system

## Related Documentation

- [REFOCUS.md](../REFOCUS.md) - Overall simplification strategy
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Updated architecture
