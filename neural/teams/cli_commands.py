"""
CLI commands for team and organization management.
"""

import json
import sys
from typing import Optional

import click

from .access_control import AccessController, Permission
from .analytics import AnalyticsDashboard, UsageAnalytics
from .billing import BillingManager, StripeIntegration
from .manager import OrganizationManager, TeamManager
from .models import BillingPlan, Role
from .team_registry import TeamModelRegistry
from .team_tracking import TeamExperimentTracker


@click.group()
def teams():
    """Commands for team and organization management."""
    pass


@teams.group()
def org():
    """Organization management commands."""
    pass


@org.command('create')
@click.argument('name')
@click.argument('slug')
@click.option('--owner-email', required=True, help='Owner email address')
@click.option('--description', default='', help='Organization description')
@click.option('--billing-plan', type=click.Choice(['free', 'starter', 'professional', 'enterprise']), default='free')
def org_create(name: str, slug: str, owner_email: str, description: str, billing_plan: str):
    """Create a new organization."""
    try:
        org_manager = OrganizationManager()
        team_manager = TeamManager()
        
        owner = team_manager.get_user_by_email(owner_email)
        if not owner:
            owner = team_manager.create_user(
                username=owner_email.split('@')[0],
                email=owner_email,
                full_name="",
            )
        
        org = org_manager.create_organization(
            name=name,
            slug=slug,
            owner_id=owner.user_id,
            description=description,
            billing_plan=BillingPlan(billing_plan),
            billing_email=owner_email,
        )
        
        click.echo(f"✓ Organization created successfully!")
        click.echo(f"  ID: {org.org_id}")
        click.echo(f"  Name: {org.name}")
        click.echo(f"  Slug: {org.slug}")
        click.echo(f"  Plan: {org.billing_plan.value}")
        
    except ValueError as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@org.command('list')
@click.option('--owner-email', help='Filter by owner email')
def org_list(owner_email: Optional[str]):
    """List all organizations."""
    try:
        org_manager = OrganizationManager()
        team_manager = TeamManager()
        
        owner_id = None
        if owner_email:
            owner = team_manager.get_user_by_email(owner_email)
            if owner:
                owner_id = owner.user_id
        
        orgs = org_manager.list_organizations(owner_id=owner_id)
        
        if not orgs:
            click.echo("No organizations found.")
            return
        
        click.echo(f"\n{'Name':<25} {'Slug':<20} {'Plan':<15} {'Created':<20}")
        click.echo("-" * 80)
        
        for org in orgs:
            created = org.created_at.strftime('%Y-%m-%d %H:%M')
            click.echo(f"{org.name:<25} {org.slug:<20} {org.billing_plan.value:<15} {created:<20}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@org.command('show')
@click.argument('org_slug')
def org_show(org_slug: str):
    """Show organization details."""
    try:
        org_manager = OrganizationManager()
        org = org_manager.get_organization_by_slug(org_slug)
        
        if not org:
            click.echo(f"✗ Organization not found: {org_slug}", err=True)
            sys.exit(1)
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Organization: {org.name}")
        click.echo(f"{'='*60}")
        click.echo(f"  ID:          {org.org_id}")
        click.echo(f"  Slug:        {org.slug}")
        click.echo(f"  Plan:        {org.billing_plan.value}")
        click.echo(f"  Created:     {org.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"  Description: {org.description or 'N/A'}")
        
        quotas = org.get_plan_quotas()
        click.echo(f"\n  Plan Quotas:")
        click.echo(f"    Models:      {quotas.max_models}")
        click.echo(f"    Experiments: {quotas.max_experiments}")
        click.echo(f"    Storage:     {quotas.max_storage_gb} GB")
        click.echo(f"    Compute:     {quotas.max_compute_hours} hours")
        click.echo(f"    Members:     {quotas.max_team_members}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@org.command('update')
@click.argument('org_slug')
@click.option('--name', help='New organization name')
@click.option('--description', help='New description')
@click.option('--billing-plan', type=click.Choice(['free', 'starter', 'professional', 'enterprise']))
def org_update(org_slug: str, name: Optional[str], description: Optional[str], billing_plan: Optional[str]):
    """Update an organization."""
    try:
        org_manager = OrganizationManager()
        org = org_manager.get_organization_by_slug(org_slug)
        
        if not org:
            click.echo(f"✗ Organization not found: {org_slug}", err=True)
            sys.exit(1)
        
        plan = BillingPlan(billing_plan) if billing_plan else None
        
        org_manager.update_organization(
            org_id=org.org_id,
            name=name,
            description=description,
            billing_plan=plan,
        )
        
        click.echo(f"✓ Organization updated successfully!")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.command('create')
@click.argument('org_slug')
@click.argument('team_name')
@click.option('--description', default='', help='Team description')
def team_create(org_slug: str, team_name: str, description: str):
    """Create a new team in an organization."""
    try:
        org_manager = OrganizationManager()
        team_manager = TeamManager()
        
        org = org_manager.get_organization_by_slug(org_slug)
        if not org:
            click.echo(f"✗ Organization not found: {org_slug}", err=True)
            sys.exit(1)
        
        quota_dict = org.get_plan_quotas().to_dict()
        team = team_manager.create_team(
            name=team_name,
            organization_id=org.org_id,
            description=description,
            quota=quota_dict,
        )
        
        click.echo(f"✓ Team created successfully!")
        click.echo(f"  ID: {team.team_id}")
        click.echo(f"  Name: {team.name}")
        click.echo(f"  Organization: {org.name}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.command('list')
@click.argument('org_slug')
def team_list(org_slug: str):
    """List all teams in an organization."""
    try:
        org_manager = OrganizationManager()
        team_manager = TeamManager()
        
        org = org_manager.get_organization_by_slug(org_slug)
        if not org:
            click.echo(f"✗ Organization not found: {org_slug}", err=True)
            sys.exit(1)
        
        teams_list = team_manager.list_teams(org.org_id)
        
        if not teams_list:
            click.echo("No teams found.")
            return
        
        click.echo(f"\n{'Name':<30} {'Members':<10} {'Models':<10} {'Experiments':<15}")
        click.echo("-" * 65)
        
        for team in teams_list:
            members = len(team.members)
            models = team.quota.current_models
            experiments = team.quota.current_experiments
            click.echo(f"{team.name:<30} {members:<10} {models:<10} {experiments:<15}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.command('show')
@click.argument('team_id')
def team_show(team_id: str):
    """Show team details."""
    try:
        team_manager = TeamManager()
        team = team_manager.get_team(team_id)
        
        if not team:
            click.echo(f"✗ Team not found: {team_id}", err=True)
            sys.exit(1)
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Team: {team.name}")
        click.echo(f"{'='*60}")
        click.echo(f"  ID:          {team.team_id}")
        click.echo(f"  Description: {team.description or 'N/A'}")
        click.echo(f"  Created:     {team.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"  Members:     {len(team.members)}")
        
        click.echo(f"\n  Resource Usage:")
        click.echo(f"    Models:      {team.quota.current_models}/{team.quota.max_models} ({team.quota.usage_percentage('models'):.1f}%)")
        click.echo(f"    Experiments: {team.quota.current_experiments}/{team.quota.max_experiments} ({team.quota.usage_percentage('experiments'):.1f}%)")
        click.echo(f"    Storage:     {team.quota.current_storage_gb:.2f}/{team.quota.max_storage_gb} GB ({team.quota.usage_percentage('storage'):.1f}%)")
        click.echo(f"    Compute:     {team.quota.current_compute_hours:.2f}/{team.quota.max_compute_hours} hours ({team.quota.usage_percentage('compute'):.1f}%)")
        
        if team.members:
            click.echo(f"\n  Team Members:")
            for user_id, role in team.members.items():
                click.echo(f"    {user_id}: {role.value}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.command('add-member')
@click.argument('team_id')
@click.argument('user_email')
@click.option('--role', type=click.Choice(['admin', 'developer', 'viewer']), default='developer')
def team_add_member(team_id: str, user_email: str, role: str):
    """Add a member to a team."""
    try:
        team_manager = TeamManager()
        
        team = team_manager.get_team(team_id)
        if not team:
            click.echo(f"✗ Team not found: {team_id}", err=True)
            sys.exit(1)
        
        user = team_manager.get_user_by_email(user_email)
        if not user:
            user = team_manager.create_user(
                username=user_email.split('@')[0],
                email=user_email,
                full_name="",
            )
        
        success = team_manager.add_member_to_team(
            team_id=team_id,
            user_id=user.user_id,
            role=Role(role),
        )
        
        if success:
            click.echo(f"✓ Member added successfully!")
            click.echo(f"  User: {user_email}")
            click.echo(f"  Role: {role}")
        else:
            click.echo(f"✗ Failed to add member. Team may be at capacity.", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.command('remove-member')
@click.argument('team_id')
@click.argument('user_email')
def team_remove_member(team_id: str, user_email: str):
    """Remove a member from a team."""
    try:
        team_manager = TeamManager()
        
        team = team_manager.get_team(team_id)
        if not team:
            click.echo(f"✗ Team not found: {team_id}", err=True)
            sys.exit(1)
        
        user = team_manager.get_user_by_email(user_email)
        if not user:
            click.echo(f"✗ User not found: {user_email}", err=True)
            sys.exit(1)
        
        success = team_manager.remove_member_from_team(
            team_id=team_id,
            user_id=user.user_id,
        )
        
        if success:
            click.echo(f"✓ Member removed successfully!")
        else:
            click.echo(f"✗ Failed to remove member.", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.group()
def analytics():
    """Team analytics commands."""
    pass


@analytics.command('dashboard')
@click.argument('team_id')
@click.option('--days', default=30, help='Number of days to include')
@click.option('--output', help='Output JSON file path')
def analytics_dashboard(team_id: str, days: int, output: Optional[str]):
    """Generate analytics dashboard for a team."""
    try:
        team_manager = TeamManager()
        team = team_manager.get_team(team_id)
        
        if not team:
            click.echo(f"✗ Team not found: {team_id}", err=True)
            sys.exit(1)
        
        usage_analytics = UsageAnalytics()
        dashboard = AnalyticsDashboard(usage_analytics)
        
        data = dashboard.generate_team_dashboard(team, days=days)
        
        if output:
            dashboard.export_dashboard_json(team, output, days=days)
            click.echo(f"✓ Dashboard exported to {output}")
        else:
            click.echo(json.dumps(data, indent=2))
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@analytics.command('usage')
@click.argument('team_id')
@click.option('--days', default=30, help='Number of days to include')
def analytics_usage(team_id: str, days: int):
    """Show usage summary for a team."""
    try:
        from datetime import datetime, timedelta
        
        team_manager = TeamManager()
        team = team_manager.get_team(team_id)
        
        if not team:
            click.echo(f"✗ Team not found: {team_id}", err=True)
            sys.exit(1)
        
        usage_analytics = UsageAnalytics()
        summary = usage_analytics.get_usage_summary(
            team_id,
            start_date=datetime.now() - timedelta(days=days),
        )
        
        click.echo(f"\nUsage Summary (Last {days} days)")
        click.echo("=" * 60)
        click.echo(f"Total Events: {summary['total_events']}")
        click.echo(f"\nCompute Usage: {summary['compute_usage']['total_hours']:.2f} hours")
        click.echo(f"Storage Usage: {summary['storage_usage']['total_gb']:.2f} GB")
        click.echo(f"API Calls:     {summary['api_usage']['total_calls']}")
        
        if summary['events_by_type']:
            click.echo(f"\nEvents by Type:")
            for event_type, count in summary['events_by_type'].items():
                click.echo(f"  {event_type}: {count}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@teams.group()
def billing():
    """Billing management commands."""
    pass


@billing.command('pricing')
def billing_pricing():
    """Show pricing for all plans."""
    try:
        billing_manager = BillingManager()
        
        click.echo("\nNeural DSL Pricing Plans")
        click.echo("=" * 80)
        
        for plan in BillingPlan:
            pricing = billing_manager.get_plan_pricing(plan)
            click.echo(f"\n{plan.value.upper()}")
            click.echo(f"  Monthly: ${pricing['monthly_price']:.2f}")
            click.echo(f"  Annual:  ${pricing['annual_price']:.2f}")
            click.echo(f"  Features:")
            for feature in pricing['features']:
                click.echo(f"    • {feature}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@billing.command('subscription')
@click.argument('org_slug')
def billing_subscription(org_slug: str):
    """Show subscription details for an organization."""
    try:
        org_manager = OrganizationManager()
        billing_manager = BillingManager()
        
        org = org_manager.get_organization_by_slug(org_slug)
        if not org:
            click.echo(f"✗ Organization not found: {org_slug}", err=True)
            sys.exit(1)
        
        subscription = billing_manager.get_subscription(org.org_id)
        
        if not subscription:
            click.echo(f"No active subscription found for {org.name}")
            return
        
        click.echo(f"\nSubscription Details: {org.name}")
        click.echo("=" * 60)
        click.echo(f"  Plan:         {subscription['plan']}")
        click.echo(f"  Status:       {subscription['status']}")
        click.echo(f"  Billing:      {subscription['billing_cycle']}")
        click.echo(f"  Amount:       ${subscription['amount']:.2f} {subscription['currency']}")
        click.echo(f"  Next billing: {subscription['next_billing_date']}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@billing.command('invoices')
@click.argument('org_slug')
@click.option('--status', type=click.Choice(['pending', 'paid', 'cancelled']))
def billing_invoices(org_slug: str, status: Optional[str]):
    """List invoices for an organization."""
    try:
        org_manager = OrganizationManager()
        billing_manager = BillingManager()
        
        org = org_manager.get_organization_by_slug(org_slug)
        if not org:
            click.echo(f"✗ Organization not found: {org_slug}", err=True)
            sys.exit(1)
        
        invoices = billing_manager.get_invoices(org.org_id, status=status)
        
        if not invoices:
            click.echo("No invoices found.")
            return
        
        click.echo(f"\n{'Invoice ID':<30} {'Amount':<15} {'Status':<10} {'Created':<20}")
        click.echo("-" * 75)
        
        for invoice in invoices:
            created = invoice['created_at'][:19].replace('T', ' ')
            click.echo(f"{invoice['invoice_id']:<30} ${invoice['amount']:<14.2f} {invoice['status']:<10} {created:<20}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)


@billing.command('usage-cost')
@click.option('--compute-hours', type=float, required=True)
@click.option('--storage-gb', type=float, required=True)
@click.option('--api-calls', type=int, required=True)
def billing_usage_cost(compute_hours: float, storage_gb: float, api_calls: int):
    """Calculate cost based on usage."""
    try:
        billing_manager = BillingManager()
        
        cost = billing_manager.calculate_usage_cost(
            compute_hours=compute_hours,
            storage_gb=storage_gb,
            api_calls=api_calls,
        )
        
        click.echo("\nUsage Cost Breakdown")
        click.echo("=" * 60)
        click.echo(f"  Compute:  {compute_hours:.2f} hours × ${cost['breakdown']['compute']['rate']:.2f} = ${cost['compute_cost']:.2f}")
        click.echo(f"  Storage:  {storage_gb:.2f} GB × ${cost['breakdown']['storage']['rate']:.2f} = ${cost['storage_cost']:.2f}")
        click.echo(f"  API:      {api_calls} calls × ${cost['breakdown']['api']['rate']:.4f} = ${cost['api_cost']:.2f}")
        click.echo(f"\n  Total:    ${cost['total_cost']:.2f} {cost['currency']}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        sys.exit(1)
