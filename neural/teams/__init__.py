"""
Multi-tenancy and team management module for Neural DSL.

This module provides comprehensive team and organization management including:
- Organization accounts with multi-tenancy
- Role-based access control (admin/developer/viewer)
- Resource quotas per team
- Shared experiment tracking
- Team-wide model registry
- Usage analytics dashboard
- Billing integration for SaaS model
"""

from .models import Organization, Team, User, Role, ResourceQuota, BillingPlan
from .manager import TeamManager, OrganizationManager
from .access_control import AccessController, Permission
from .team_registry import TeamModelRegistry
from .team_tracking import TeamExperimentTracker
from .analytics import UsageAnalytics, AnalyticsDashboard
from .billing import BillingManager, StripeIntegration

__all__ = [
    'Organization',
    'Team',
    'User',
    'Role',
    'ResourceQuota',
    'BillingPlan',
    'TeamManager',
    'OrganizationManager',
    'AccessController',
    'Permission',
    'TeamModelRegistry',
    'TeamExperimentTracker',
    'UsageAnalytics',
    'AnalyticsDashboard',
    'BillingManager',
    'StripeIntegration',
]
