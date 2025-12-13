"""
Configuration settings for team management.
"""

from typing import Dict, Any


class TeamConfig:
    """Configuration for team management system."""
    
    # Storage settings
    BASE_DIR = "neural_organizations"
    
    # Quota defaults for each plan (can be overridden)
    PLAN_QUOTAS = {
        'free': {
            'max_models': 10,
            'max_experiments': 100,
            'max_storage_gb': 10.0,
            'max_compute_hours': 100.0,
            'max_team_members': 5,
            'max_api_calls_per_day': 10000,
            'max_concurrent_runs': 5,
        },
        'starter': {
            'max_models': 50,
            'max_experiments': 500,
            'max_storage_gb': 100.0,
            'max_compute_hours': 1000.0,
            'max_team_members': 10,
            'max_api_calls_per_day': 100000,
            'max_concurrent_runs': 10,
        },
        'professional': {
            'max_models': 200,
            'max_experiments': 2000,
            'max_storage_gb': 500.0,
            'max_compute_hours': 5000.0,
            'max_team_members': 50,
            'max_api_calls_per_day': 1000000,
            'max_concurrent_runs': 25,
        },
        'enterprise': {
            'max_models': 999999,
            'max_experiments': 999999,
            'max_storage_gb': 99999.0,
            'max_compute_hours': 99999.0,
            'max_team_members': 999,
            'max_api_calls_per_day': 99999999,
            'max_concurrent_runs': 100,
        },
    }
    
    # Pricing rates (in USD)
    USAGE_RATES = {
        'compute_hour': 0.50,
        'storage_gb_month': 0.10,
        'api_call_1000': 0.01,
    }
    
    # Plan pricing
    PLAN_PRICING = {
        'free': {'monthly': 0.0, 'annual': 0.0},
        'starter': {'monthly': 29.0, 'annual': 290.0},
        'professional': {'monthly': 99.0, 'annual': 990.0},
        'enterprise': {'monthly': 499.0, 'annual': 4990.0},
    }
    
    # Analytics settings
    ANALYTICS_RETENTION_DAYS = 365
    ANALYTICS_AGGREGATION_INTERVAL = 'daily'
    
    # Billing settings
    INVOICE_DUE_DAYS = 30
    PAYMENT_GRACE_PERIOD_DAYS = 7
    
    # Security settings
    SESSION_TIMEOUT_MINUTES = 60
    MAX_LOGIN_ATTEMPTS = 5
    
    # Feature flags
    ENABLE_STRIPE_INTEGRATION = True
    ENABLE_USAGE_ANALYTICS = True
    ENABLE_AUDIT_LOGGING = True
    
    @classmethod
    def get_quota_for_plan(cls, plan: str) -> Dict[str, Any]:
        """Get quota configuration for a billing plan."""
        return cls.PLAN_QUOTAS.get(plan, cls.PLAN_QUOTAS['free'])
    
    @classmethod
    def get_pricing_for_plan(cls, plan: str) -> Dict[str, float]:
        """Get pricing for a billing plan."""
        return cls.PLAN_PRICING.get(plan, cls.PLAN_PRICING['free'])
