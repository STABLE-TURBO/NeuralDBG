"""
Team and organization management functionality.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import Organization, Team, User, Role, BillingPlan


class OrganizationManager:
    """Manages organizations in a multi-tenant environment."""
    
    def __init__(self, base_dir: str = "neural_organizations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.orgs_file = self.base_dir / "organizations.json"
        self._organizations: Dict[str, Organization] = {}
        self._load_organizations()
    
    def _load_organizations(self) -> None:
        """Load organizations from storage."""
        if self.orgs_file.exists():
            with open(self.orgs_file, 'r') as f:
                data = json.load(f)
                self._organizations = {
                    org_id: Organization.from_dict(org_data)
                    for org_id, org_data in data.items()
                }
    
    def _save_organizations(self) -> None:
        """Save organizations to storage."""
        data = {
            org_id: org.to_dict()
            for org_id, org in self._organizations.items()
        }
        with open(self.orgs_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_organization(
        self,
        name: str,
        slug: str,
        owner_id: str,
        description: str = "",
        billing_plan: BillingPlan = BillingPlan.FREE,
        billing_email: str = "",
    ) -> Organization:
        """Create a new organization."""
        if self.get_organization_by_slug(slug):
            raise ValueError(f"Organization with slug '{slug}' already exists")
        
        org = Organization(
            name=name,
            slug=slug,
            description=description,
            owner_id=owner_id,
            billing_plan=billing_plan,
            billing_email=billing_email,
        )
        
        self._organizations[org.org_id] = org
        self._save_organizations()
        
        org_dir = self.base_dir / org.org_id
        org_dir.mkdir(parents=True, exist_ok=True)
        
        return org
    
    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get an organization by ID."""
        return self._organizations.get(org_id)
    
    def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get an organization by slug."""
        for org in self._organizations.values():
            if org.slug == slug:
                return org
        return None
    
    def update_organization(
        self,
        org_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        billing_plan: Optional[BillingPlan] = None,
        billing_email: Optional[str] = None,
        settings: Optional[Dict] = None,
    ) -> Optional[Organization]:
        """Update an organization."""
        org = self._organizations.get(org_id)
        if not org:
            return None
        
        if name is not None:
            org.name = name
        if description is not None:
            org.description = description
        if billing_plan is not None:
            org.billing_plan = billing_plan
        if billing_email is not None:
            org.billing_email = billing_email
        if settings is not None:
            org.settings.update(settings)
        
        org.updated_at = datetime.now()
        self._save_organizations()
        
        return org
    
    def delete_organization(self, org_id: str) -> bool:
        """Delete an organization."""
        if org_id in self._organizations:
            del self._organizations[org_id]
            self._save_organizations()
            return True
        return False
    
    def list_organizations(self, owner_id: Optional[str] = None) -> List[Organization]:
        """List all organizations, optionally filtered by owner."""
        orgs = list(self._organizations.values())
        if owner_id:
            orgs = [org for org in orgs if org.owner_id == owner_id]
        return sorted(orgs, key=lambda x: x.created_at, reverse=True)


class TeamManager:
    """Manages teams within organizations."""
    
    def __init__(self, base_dir: str = "neural_organizations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._teams: Dict[str, Team] = {}
        self._users: Dict[str, User] = {}
        self._load_data()
    
    def _get_org_dir(self, org_id: str) -> Path:
        """Get directory for an organization."""
        return self.base_dir / org_id
    
    def _get_teams_file(self, org_id: str) -> Path:
        """Get teams file for an organization."""
        return self._get_org_dir(org_id) / "teams.json"
    
    def _get_users_file(self, org_id: str) -> Path:
        """Get users file for an organization."""
        return self._get_org_dir(org_id) / "users.json"
    
    def _load_data(self) -> None:
        """Load all teams and users from storage."""
        for org_dir in self.base_dir.iterdir():
            if org_dir.is_dir():
                org_id = org_dir.name
                
                teams_file = self._get_teams_file(org_id)
                if teams_file.exists():
                    with open(teams_file, 'r') as f:
                        data = json.load(f)
                        for team_id, team_data in data.items():
                            self._teams[team_id] = Team.from_dict(team_data)
                
                users_file = self._get_users_file(org_id)
                if users_file.exists():
                    with open(users_file, 'r') as f:
                        data = json.load(f)
                        for user_id, user_data in data.items():
                            self._users[user_id] = User.from_dict(user_data)
    
    def _save_teams(self, org_id: str) -> None:
        """Save teams for an organization."""
        org_dir = self._get_org_dir(org_id)
        org_dir.mkdir(parents=True, exist_ok=True)
        
        teams_file = self._get_teams_file(org_id)
        org_teams = {
            team_id: team.to_dict()
            for team_id, team in self._teams.items()
            if team.organization_id == org_id
        }
        
        with open(teams_file, 'w') as f:
            json.dump(org_teams, f, indent=2)
    
    def _save_users(self, org_id: str) -> None:
        """Save users for an organization."""
        org_dir = self._get_org_dir(org_id)
        org_dir.mkdir(parents=True, exist_ok=True)
        
        users_file = self._get_users_file(org_id)
        org_teams = [t for t in self._teams.values() if t.organization_id == org_id]
        org_user_ids = set()
        for team in org_teams:
            org_user_ids.update(team.members.keys())
        
        org_users = {
            user_id: user.to_dict()
            for user_id, user in self._users.items()
            if user_id in org_user_ids
        }
        
        with open(users_file, 'w') as f:
            json.dump(org_users, f, indent=2)
    
    def create_team(
        self,
        name: str,
        organization_id: str,
        description: str = "",
        quota: Optional[Dict] = None,
    ) -> Team:
        """Create a new team."""
        team = Team(
            name=name,
            description=description,
            organization_id=organization_id,
        )
        
        if quota:
            team.quota = team.quota.__class__(**quota)
        
        self._teams[team.team_id] = team
        self._save_teams(organization_id)
        
        return team
    
    def get_team(self, team_id: str) -> Optional[Team]:
        """Get a team by ID."""
        return self._teams.get(team_id)
    
    def update_team(
        self,
        team_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        settings: Optional[Dict] = None,
    ) -> Optional[Team]:
        """Update a team."""
        team = self._teams.get(team_id)
        if not team:
            return None
        
        if name is not None:
            team.name = name
        if description is not None:
            team.description = description
        if settings is not None:
            team.settings.update(settings)
        
        team.updated_at = datetime.now()
        self._save_teams(team.organization_id)
        
        return team
    
    def delete_team(self, team_id: str) -> bool:
        """Delete a team."""
        team = self._teams.get(team_id)
        if team:
            org_id = team.organization_id
            del self._teams[team_id]
            self._save_teams(org_id)
            return True
        return False
    
    def list_teams(self, organization_id: str) -> List[Team]:
        """List all teams in an organization."""
        teams = [
            team for team in self._teams.values()
            if team.organization_id == organization_id
        ]
        return sorted(teams, key=lambda x: x.created_at, reverse=True)
    
    def create_user(
        self,
        username: str,
        email: str,
        full_name: str = "",
    ) -> User:
        """Create a new user."""
        user = User(
            username=username,
            email=email,
            full_name=full_name,
        )
        
        self._users[user.user_id] = user
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None
    
    def add_member_to_team(
        self,
        team_id: str,
        user_id: str,
        role: Role,
    ) -> bool:
        """Add a member to a team."""
        team = self._teams.get(team_id)
        if not team:
            return False
        
        user = self._users.get(user_id)
        if not user:
            return False
        
        success = team.add_member(user_id, role)
        if success:
            self._save_teams(team.organization_id)
            self._save_users(team.organization_id)
        
        return success
    
    def remove_member_from_team(
        self,
        team_id: str,
        user_id: str,
    ) -> bool:
        """Remove a member from a team."""
        team = self._teams.get(team_id)
        if not team:
            return False
        
        success = team.remove_member(user_id)
        if success:
            self._save_teams(team.organization_id)
            self._save_users(team.organization_id)
        
        return success
    
    def update_member_role(
        self,
        team_id: str,
        user_id: str,
        role: Role,
    ) -> bool:
        """Update a member's role in a team."""
        team = self._teams.get(team_id)
        if not team or not team.has_member(user_id):
            return False
        
        team.members[user_id] = role
        team.updated_at = datetime.now()
        self._save_teams(team.organization_id)
        
        return True
    
    def get_user_teams(self, user_id: str, organization_id: Optional[str] = None) -> List[Team]:
        """Get all teams a user belongs to."""
        teams = [
            team for team in self._teams.values()
            if team.has_member(user_id)
        ]
        
        if organization_id:
            teams = [team for team in teams if team.organization_id == organization_id]
        
        return sorted(teams, key=lambda x: x.created_at, reverse=True)
