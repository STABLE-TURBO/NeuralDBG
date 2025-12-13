"""
Usage analytics and dashboard for team monitoring.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .models import Team, Organization


class UsageAnalytics:
    """Tracks and analyzes team usage metrics."""
    
    def __init__(self, base_dir: str = "neural_organizations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_analytics_dir(self, team_id: str) -> Path:
        """Get the analytics directory for a team."""
        analytics_dir = self.base_dir / "analytics" / team_id
        analytics_dir.mkdir(parents=True, exist_ok=True)
        return analytics_dir
    
    def _get_usage_file(self, team_id: str) -> Path:
        """Get the usage data file for a team."""
        return self._get_analytics_dir(team_id) / "usage.json"
    
    def _load_usage_data(self, team_id: str) -> List[Dict[str, Any]]:
        """Load usage data for a team."""
        usage_file = self._get_usage_file(team_id)
        if usage_file.exists():
            with open(usage_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_usage_data(self, team_id: str, usage_data: List[Dict[str, Any]]) -> None:
        """Save usage data for a team."""
        usage_file = self._get_usage_file(team_id)
        with open(usage_file, 'w') as f:
            json.dump(usage_data, f, indent=2)
    
    def log_event(
        self,
        team_id: str,
        user_id: str,
        event_type: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a usage event."""
        usage_data = self._load_usage_data(team_id)
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'event_type': event_type,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'metadata': metadata or {},
        }
        
        usage_data.append(event)
        self._save_usage_data(team_id, usage_data)
    
    def log_compute_usage(
        self,
        team_id: str,
        user_id: str,
        duration_hours: float,
        resource_type: str = "training",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log compute usage."""
        self.log_event(
            team_id=team_id,
            user_id=user_id,
            event_type="compute_usage",
            resource_type=resource_type,
            metadata={
                'duration_hours': duration_hours,
                **(metadata or {}),
            }
        )
    
    def log_storage_usage(
        self,
        team_id: str,
        user_id: str,
        size_gb: float,
        resource_type: str = "model",
        resource_id: Optional[str] = None,
    ) -> None:
        """Log storage usage."""
        self.log_event(
            team_id=team_id,
            user_id=user_id,
            event_type="storage_usage",
            resource_type=resource_type,
            resource_id=resource_id,
            metadata={'size_gb': size_gb}
        )
    
    def log_api_call(
        self,
        team_id: str,
        user_id: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Log an API call."""
        self.log_event(
            team_id=team_id,
            user_id=user_id,
            event_type="api_call",
            metadata={
                'endpoint': endpoint,
                'status_code': status_code,
                'duration_ms': duration_ms,
            }
        )
    
    def get_usage_summary(
        self,
        team_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get usage summary for a team."""
        usage_data = self._load_usage_data(team_id)
        
        if start_date:
            usage_data = [
                e for e in usage_data
                if datetime.fromisoformat(e['timestamp']) >= start_date
            ]
        
        if end_date:
            usage_data = [
                e for e in usage_data
                if datetime.fromisoformat(e['timestamp']) <= end_date
            ]
        
        total_events = len(usage_data)
        events_by_type = defaultdict(int)
        events_by_user = defaultdict(int)
        total_compute_hours = 0.0
        total_storage_gb = 0.0
        total_api_calls = 0
        
        for event in usage_data:
            events_by_type[event['event_type']] += 1
            events_by_user[event['user_id']] += 1
            
            if event['event_type'] == 'compute_usage':
                total_compute_hours += event['metadata'].get('duration_hours', 0)
            
            if event['event_type'] == 'storage_usage':
                total_storage_gb += event['metadata'].get('size_gb', 0)
            
            if event['event_type'] == 'api_call':
                total_api_calls += 1
        
        return {
            'total_events': total_events,
            'events_by_type': dict(events_by_type),
            'events_by_user': dict(events_by_user),
            'compute_usage': {
                'total_hours': total_compute_hours,
            },
            'storage_usage': {
                'total_gb': total_storage_gb,
            },
            'api_usage': {
                'total_calls': total_api_calls,
            },
            'period': {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None,
            }
        }
    
    def get_time_series_data(
        self,
        team_id: str,
        metric: str,
        days: int = 30,
    ) -> List[Tuple[str, float]]:
        """Get time series data for a metric."""
        usage_data = self._load_usage_data(team_id)
        
        start_date = datetime.now() - timedelta(days=days)
        usage_data = [
            e for e in usage_data
            if datetime.fromisoformat(e['timestamp']) >= start_date
        ]
        
        daily_data = defaultdict(float)
        
        for event in usage_data:
            date = datetime.fromisoformat(event['timestamp']).date().isoformat()
            
            if metric == 'events':
                daily_data[date] += 1
            elif metric == 'compute_hours' and event['event_type'] == 'compute_usage':
                daily_data[date] += event['metadata'].get('duration_hours', 0)
            elif metric == 'storage_gb' and event['event_type'] == 'storage_usage':
                daily_data[date] += event['metadata'].get('size_gb', 0)
            elif metric == 'api_calls' and event['event_type'] == 'api_call':
                daily_data[date] += 1
        
        return sorted(daily_data.items())
    
    def get_user_activity(
        self,
        team_id: str,
        user_id: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Get activity summary for a specific user."""
        usage_data = self._load_usage_data(team_id)
        
        start_date = datetime.now() - timedelta(days=days)
        user_data = [
            e for e in usage_data
            if e['user_id'] == user_id
            and datetime.fromisoformat(e['timestamp']) >= start_date
        ]
        
        events_by_type = defaultdict(int)
        for event in user_data:
            events_by_type[event['event_type']] += 1
        
        return {
            'user_id': user_id,
            'total_events': len(user_data),
            'events_by_type': dict(events_by_type),
            'period_days': days,
        }


class AnalyticsDashboard:
    """Generates analytics dashboards for teams."""
    
    def __init__(self, analytics: UsageAnalytics):
        self.analytics = analytics
    
    def generate_team_dashboard(
        self,
        team: Team,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Generate a comprehensive dashboard for a team."""
        usage_summary = self.analytics.get_usage_summary(
            team.team_id,
            start_date=datetime.now() - timedelta(days=days),
        )
        
        quota_status = {
            'models': {
                'current': team.quota.current_models,
                'max': team.quota.max_models,
                'usage_percent': team.quota.usage_percentage('models'),
            },
            'experiments': {
                'current': team.quota.current_experiments,
                'max': team.quota.max_experiments,
                'usage_percent': team.quota.usage_percentage('experiments'),
            },
            'storage': {
                'current_gb': team.quota.current_storage_gb,
                'max_gb': team.quota.max_storage_gb,
                'usage_percent': team.quota.usage_percentage('storage'),
            },
            'compute': {
                'current_hours': team.quota.current_compute_hours,
                'max_hours': team.quota.max_compute_hours,
                'usage_percent': team.quota.usage_percentage('compute'),
            },
            'team_members': {
                'current': team.quota.current_team_members,
                'max': team.quota.max_team_members,
                'usage_percent': team.quota.usage_percentage('team_members'),
            },
        }
        
        time_series = {
            'events': self.analytics.get_time_series_data(team.team_id, 'events', days),
            'compute_hours': self.analytics.get_time_series_data(team.team_id, 'compute_hours', days),
            'api_calls': self.analytics.get_time_series_data(team.team_id, 'api_calls', days),
        }
        
        member_activity = {}
        for user_id in team.members.keys():
            member_activity[user_id] = self.analytics.get_user_activity(team.team_id, user_id, days=7)
        
        return {
            'team_id': team.team_id,
            'team_name': team.name,
            'period_days': days,
            'quota_status': quota_status,
            'usage_summary': usage_summary,
            'time_series': time_series,
            'member_activity': member_activity,
            'generated_at': datetime.now().isoformat(),
        }
    
    def export_dashboard_json(
        self,
        team: Team,
        output_path: str,
        days: int = 30,
    ) -> None:
        """Export dashboard data to JSON file."""
        dashboard = self.generate_team_dashboard(team, days)
        
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
    
    def generate_organization_dashboard(
        self,
        org: Organization,
        teams: List[Team],
        days: int = 30,
    ) -> Dict[str, Any]:
        """Generate organization-wide dashboard."""
        team_summaries = []
        total_usage = defaultdict(float)
        
        for team in teams:
            summary = self.analytics.get_usage_summary(
                team.team_id,
                start_date=datetime.now() - timedelta(days=days),
            )
            team_summaries.append({
                'team_id': team.team_id,
                'team_name': team.name,
                'summary': summary,
            })
            
            total_usage['compute_hours'] += summary['compute_usage']['total_hours']
            total_usage['storage_gb'] += summary['storage_usage']['total_gb']
            total_usage['api_calls'] += summary['api_usage']['total_calls']
        
        return {
            'org_id': org.org_id,
            'org_name': org.name,
            'billing_plan': org.billing_plan.value,
            'period_days': days,
            'team_count': len(teams),
            'total_usage': dict(total_usage),
            'team_summaries': team_summaries,
            'generated_at': datetime.now().isoformat(),
        }
