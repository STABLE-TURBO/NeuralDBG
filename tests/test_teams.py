"""
Tests for the teams module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from neural.teams import (
    OrganizationManager,
    TeamManager,
    TeamModelRegistry,
    TeamExperimentTracker,
    UsageAnalytics,
    BillingManager,
    AccessController,
    Permission,
    BillingPlan,
    Role,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def org_manager(temp_dir):
    """Create an organization manager."""
    return OrganizationManager(base_dir=temp_dir)


@pytest.fixture
def team_manager(temp_dir):
    """Create a team manager."""
    return TeamManager(base_dir=temp_dir)


def test_create_organization(org_manager):
    """Test organization creation."""
    org = org_manager.create_organization(
        name="Test Org",
        slug="test-org",
        owner_id="owner_123",
        billing_plan=BillingPlan.PROFESSIONAL,
    )
    
    assert org.name == "Test Org"
    assert org.slug == "test-org"
    assert org.billing_plan == BillingPlan.PROFESSIONAL


def test_create_team(org_manager, team_manager):
    """Test team creation."""
    org = org_manager.create_organization(
        name="Test Org",
        slug="test-org",
        owner_id="owner_123",
    )
    
    team = team_manager.create_team(
        name="Test Team",
        organization_id=org.org_id,
        description="Test team description",
    )
    
    assert team.name == "Test Team"
    assert team.organization_id == org.org_id


def test_add_team_member(org_manager, team_manager):
    """Test adding a member to a team."""
    org = org_manager.create_organization(
        name="Test Org",
        slug="test-org",
        owner_id="owner_123",
    )
    
    team = team_manager.create_team(
        name="Test Team",
        organization_id=org.org_id,
    )
    
    user = team_manager.create_user(
        username="testuser",
        email="test@example.com",
    )
    
    success = team_manager.add_member_to_team(
        team_id=team.team_id,
        user_id=user.user_id,
        role=Role.DEVELOPER,
    )
    
    assert success
    assert team.has_member(user.user_id)
    assert team.get_member_role(user.user_id) == Role.DEVELOPER


def test_access_control(org_manager, team_manager):
    """Test role-based access control."""
    org = org_manager.create_organization(
        name="Test Org",
        slug="test-org",
        owner_id="owner_123",
    )
    
    team = team_manager.create_team(
        name="Test Team",
        organization_id=org.org_id,
    )
    
    admin = team_manager.create_user(username="admin", email="admin@example.com")
    developer = team_manager.create_user(username="dev", email="dev@example.com")
    viewer = team_manager.create_user(username="viewer", email="viewer@example.com")
    
    team_manager.add_member_to_team(team.team_id, admin.user_id, Role.ADMIN)
    team_manager.add_member_to_team(team.team_id, developer.user_id, Role.DEVELOPER)
    team_manager.add_member_to_team(team.team_id, viewer.user_id, Role.VIEWER)
    
    # Test admin permissions
    assert AccessController.has_permission(admin.user_id, team, Permission.DELETE_MODELS)
    assert AccessController.has_permission(admin.user_id, team, Permission.MANAGE_TEAM)
    
    # Test developer permissions
    assert AccessController.has_permission(developer.user_id, team, Permission.CREATE_MODELS)
    assert not AccessController.has_permission(developer.user_id, team, Permission.DELETE_MODELS)
    assert not AccessController.has_permission(developer.user_id, team, Permission.MANAGE_TEAM)
    
    # Test viewer permissions
    assert AccessController.has_permission(viewer.user_id, team, Permission.VIEW_MODELS)
    assert not AccessController.has_permission(viewer.user_id, team, Permission.CREATE_MODELS)
    assert not AccessController.has_permission(viewer.user_id, team, Permission.MANAGE_TEAM)


def test_resource_quotas(org_manager, team_manager):
    """Test resource quota management."""
    org = org_manager.create_organization(
        name="Test Org",
        slug="test-org",
        owner_id="owner_123",
        billing_plan=BillingPlan.FREE,
    )
    
    team = team_manager.create_team(
        name="Test Team",
        organization_id=org.org_id,
    )
    
    # Check initial quota
    assert team.quota.check_quota('models')
    assert team.quota.usage_percentage('models') == 0.0
    
    # Simulate resource usage
    team.quota.current_models = 5
    assert team.quota.usage_percentage('models') == 50.0


def test_billing_plans(temp_dir):
    """Test billing plan pricing."""
    billing = BillingManager(base_dir=temp_dir)
    
    free_pricing = billing.get_plan_pricing(BillingPlan.FREE)
    assert free_pricing['monthly_price'] == 0.0
    
    pro_pricing = billing.get_plan_pricing(BillingPlan.PROFESSIONAL)
    assert pro_pricing['monthly_price'] == 99.0
    assert pro_pricing['annual_price'] == 990.0


def test_usage_cost_calculation(temp_dir):
    """Test usage cost calculation."""
    billing = BillingManager(base_dir=temp_dir)
    
    cost = billing.calculate_usage_cost(
        compute_hours=100.0,
        storage_gb=50.0,
        api_calls=10000,
    )
    
    assert cost['compute_cost'] == 50.0  # 100 hours * $0.50
    assert cost['storage_cost'] == 5.0   # 50 GB * $0.10
    assert cost['api_cost'] == 0.10      # 10000 calls * $0.01/1000
    assert cost['total_cost'] == 55.10


def test_usage_analytics(temp_dir, org_manager, team_manager):
    """Test usage analytics tracking."""
    analytics = UsageAnalytics(base_dir=temp_dir)
    
    org = org_manager.create_organization(
        name="Test Org",
        slug="test-org",
        owner_id="owner_123",
    )
    
    team = team_manager.create_team(
        name="Test Team",
        organization_id=org.org_id,
    )
    
    user = team_manager.create_user(
        username="testuser",
        email="test@example.com",
    )
    
    # Log events
    analytics.log_compute_usage(
        team_id=team.team_id,
        user_id=user.user_id,
        duration_hours=5.0,
    )
    
    analytics.log_storage_usage(
        team_id=team.team_id,
        user_id=user.user_id,
        size_gb=10.0,
    )
    
    # Get summary
    summary = analytics.get_usage_summary(team.team_id)
    
    assert summary['total_events'] == 2
    assert summary['compute_usage']['total_hours'] == 5.0
    assert summary['storage_usage']['total_gb'] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
