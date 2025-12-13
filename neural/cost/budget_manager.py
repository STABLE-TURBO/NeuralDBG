"""
Budget management and alerts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neural.cost.estimator import CloudProvider


class AlertSeverity(Enum):
    """Budget alert severity."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Budget:
    """Budget configuration."""
    
    name: str
    total_amount: float
    period_days: int
    providers: List[CloudProvider] = field(default_factory=list)
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    start_time: float = field(default_factory=time.time)
    spent_amount: float = 0.0
    
    @property
    def remaining_amount(self) -> float:
        """Calculate remaining budget."""
        return max(0.0, self.total_amount - self.spent_amount)
    
    @property
    def utilization_percent(self) -> float:
        """Calculate budget utilization percentage."""
        if self.total_amount <= 0:
            return 0.0
        return (self.spent_amount / self.total_amount) * 100
    
    @property
    def days_remaining(self) -> float:
        """Calculate days remaining in budget period."""
        elapsed_seconds = time.time() - self.start_time
        elapsed_days = elapsed_seconds / 86400
        return max(0.0, self.period_days - elapsed_days)
    
    @property
    def burn_rate_per_day(self) -> float:
        """Calculate daily burn rate."""
        elapsed_seconds = time.time() - self.start_time
        elapsed_days = max(0.01, elapsed_seconds / 86400)
        return self.spent_amount / elapsed_days
    
    def projected_spend(self) -> float:
        """Project total spend at current burn rate."""
        if self.days_remaining <= 0:
            return self.spent_amount
        
        total_days = self.period_days
        return self.burn_rate_per_day * total_days
    
    def is_expired(self) -> bool:
        """Check if budget period has expired."""
        return self.days_remaining <= 0


@dataclass
class BudgetAlert:
    """Budget alert."""
    
    budget_name: str
    severity: AlertSeverity
    message: str
    utilization_percent: float
    spent_amount: float
    remaining_amount: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'budget_name': self.budget_name,
            'severity': self.severity.value,
            'message': self.message,
            'utilization_percent': self.utilization_percent,
            'spent_amount': self.spent_amount,
            'remaining_amount': self.remaining_amount,
            'timestamp': self.timestamp,
        }


class BudgetManager:
    """Manage budgets and alerts."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        alert_callback: Optional[Callable[[BudgetAlert], None]] = None
    ):
        """
        Initialize budget manager.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to store budget data
        alert_callback : callable, optional
            Callback function for alerts
        """
        self.storage_path = Path(storage_path) if storage_path else Path("budget_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.alert_callback = alert_callback
        
        self.budgets: Dict[str, Budget] = {}
        self.alerts: List[BudgetAlert] = []
        self.spending_history: List[Dict[str, Any]] = []
        
        self._load_budgets()
    
    def create_budget(
        self,
        name: str,
        total_amount: float,
        period_days: int,
        providers: Optional[List[CloudProvider]] = None,
        alert_thresholds: Optional[List[float]] = None
    ) -> Budget:
        """
        Create a new budget.
        
        Parameters
        ----------
        name : str
            Budget name
        total_amount : float
            Total budget amount
        period_days : int
            Budget period in days
        providers : list, optional
            Cloud providers to track
        alert_thresholds : list, optional
            Alert thresholds (e.g., [0.5, 0.8, 0.95])
            
        Returns
        -------
        Budget
            Created budget
        """
        budget = Budget(
            name=name,
            total_amount=total_amount,
            period_days=period_days,
            providers=providers or [],
            alert_thresholds=alert_thresholds or [0.5, 0.8, 0.95]
        )
        
        self.budgets[name] = budget
        self._save_budget(budget)
        
        return budget
    
    def record_expense(
        self,
        budget_name: str,
        amount: float,
        provider: CloudProvider,
        description: str = ""
    ):
        """
        Record an expense against a budget.
        
        Parameters
        ----------
        budget_name : str
            Budget name
        amount : float
            Expense amount
        provider : CloudProvider
            Cloud provider
        description : str, optional
            Expense description
        """
        budget = self.budgets.get(budget_name)
        
        if not budget:
            raise ValueError(f"Budget '{budget_name}' not found")
        
        if budget.providers and provider not in budget.providers:
            return
        
        old_utilization = budget.utilization_percent
        
        budget.spent_amount += amount
        
        expense_record = {
            'budget_name': budget_name,
            'amount': amount,
            'provider': provider.value,
            'description': description,
            'timestamp': time.time(),
        }
        self.spending_history.append(expense_record)
        
        self._check_thresholds(budget, old_utilization)
        
        self._save_budget(budget)
        self._save_expense(expense_record)
    
    def get_budget_status(self, budget_name: str) -> Dict[str, Any]:
        """
        Get budget status.
        
        Parameters
        ----------
        budget_name : str
            Budget name
            
        Returns
        -------
        dict
            Budget status information
        """
        budget = self.budgets.get(budget_name)
        
        if not budget:
            raise ValueError(f"Budget '{budget_name}' not found")
        
        projected = budget.projected_spend()
        projected_overspend = max(0.0, projected - budget.total_amount)
        
        return {
            'name': budget.name,
            'total_amount': budget.total_amount,
            'spent_amount': budget.spent_amount,
            'remaining_amount': budget.remaining_amount,
            'utilization_percent': budget.utilization_percent,
            'days_remaining': budget.days_remaining,
            'burn_rate_per_day': budget.burn_rate_per_day,
            'projected_spend': projected,
            'projected_overspend': projected_overspend,
            'is_expired': budget.is_expired(),
            'status': self._get_budget_status_label(budget),
        }
    
    def get_all_budgets_status(self) -> List[Dict[str, Any]]:
        """Get status of all budgets."""
        return [
            self.get_budget_status(name)
            for name in self.budgets.keys()
        ]
    
    def get_spending_report(
        self,
        budget_name: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get spending report.
        
        Parameters
        ----------
        budget_name : str, optional
            Filter by budget name
        days : int, optional
            Number of days to include
            
        Returns
        -------
        dict
            Spending report
        """
        expenses = self.spending_history
        
        if budget_name:
            expenses = [e for e in expenses if e['budget_name'] == budget_name]
        
        if days:
            cutoff_time = time.time() - (days * 86400)
            expenses = [e for e in expenses if e['timestamp'] >= cutoff_time]
        
        if not expenses:
            return {
                'total_expenses': 0,
                'total_amount': 0.0,
                'by_budget': {},
                'by_provider': {},
            }
        
        total_amount = sum(e['amount'] for e in expenses)
        
        by_budget: Dict[str, float] = {}
        by_provider: Dict[str, float] = {}
        
        for expense in expenses:
            budget = expense['budget_name']
            provider = expense['provider']
            amount = expense['amount']
            
            by_budget[budget] = by_budget.get(budget, 0.0) + amount
            by_provider[provider] = by_provider.get(provider, 0.0) + amount
        
        return {
            'total_expenses': len(expenses),
            'total_amount': total_amount,
            'by_budget': by_budget,
            'by_provider': by_provider,
            'time_period_days': days,
        }
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[BudgetAlert]:
        """
        Get recent alerts.
        
        Parameters
        ----------
        severity : AlertSeverity, optional
            Filter by severity
        limit : int
            Maximum number of alerts
            
        Returns
        -------
        list
            Recent alerts
        """
        alerts = self.alerts[-limit:]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def reset_budget(self, budget_name: str):
        """Reset budget for new period."""
        budget = self.budgets.get(budget_name)
        
        if not budget:
            raise ValueError(f"Budget '{budget_name}' not found")
        
        budget.spent_amount = 0.0
        budget.start_time = time.time()
        
        self._save_budget(budget)
    
    def delete_budget(self, budget_name: str):
        """Delete a budget."""
        if budget_name in self.budgets:
            del self.budgets[budget_name]
            
            budget_file = self.storage_path / f"budget_{budget_name}.json"
            if budget_file.exists():
                budget_file.unlink()
    
    def _check_thresholds(self, budget: Budget, old_utilization: float):
        """Check if budget thresholds are crossed."""
        new_utilization = budget.utilization_percent
        
        for threshold in budget.alert_thresholds:
            threshold_percent = threshold * 100
            
            if old_utilization < threshold_percent <= new_utilization:
                severity = self._get_alert_severity(threshold)
                
                alert = BudgetAlert(
                    budget_name=budget.name,
                    severity=severity,
                    message=f"Budget utilization reached {threshold_percent:.0f}%",
                    utilization_percent=new_utilization,
                    spent_amount=budget.spent_amount,
                    remaining_amount=budget.remaining_amount
                )
                
                self._send_alert(alert)
        
        if new_utilization >= 100:
            alert = BudgetAlert(
                budget_name=budget.name,
                severity=AlertSeverity.CRITICAL,
                message="Budget exceeded!",
                utilization_percent=new_utilization,
                spent_amount=budget.spent_amount,
                remaining_amount=budget.remaining_amount
            )
            self._send_alert(alert)
    
    def _get_alert_severity(self, threshold: float) -> AlertSeverity:
        """Get alert severity for threshold."""
        if threshold >= 0.95:
            return AlertSeverity.CRITICAL
        elif threshold >= 0.8:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _get_budget_status_label(self, budget: Budget) -> str:
        """Get budget status label."""
        if budget.is_expired():
            return "expired"
        elif budget.utilization_percent >= 100:
            return "exceeded"
        elif budget.utilization_percent >= 95:
            return "critical"
        elif budget.utilization_percent >= 80:
            return "warning"
        else:
            return "healthy"
    
    def _send_alert(self, alert: BudgetAlert):
        """Send budget alert."""
        self.alerts.append(alert)
        
        alert_file = self.storage_path / f"alert_{int(alert.timestamp)}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert.to_dict(), f, indent=2)
        
        if self.alert_callback:
            self.alert_callback(alert)
    
    def _save_budget(self, budget: Budget):
        """Save budget to file."""
        budget_file = self.storage_path / f"budget_{budget.name}.json"
        
        data = {
            'name': budget.name,
            'total_amount': budget.total_amount,
            'period_days': budget.period_days,
            'providers': [p.value for p in budget.providers],
            'alert_thresholds': budget.alert_thresholds,
            'start_time': budget.start_time,
            'spent_amount': budget.spent_amount,
        }
        
        with open(budget_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_expense(self, expense: Dict[str, Any]):
        """Save expense record."""
        expense_file = self.storage_path / "expenses.jsonl"
        
        with open(expense_file, 'a') as f:
            f.write(json.dumps(expense) + '\n')
    
    def _load_budgets(self):
        """Load budgets from disk."""
        for budget_file in self.storage_path.glob("budget_*.json"):
            try:
                with open(budget_file, 'r') as f:
                    data = json.load(f)
                
                budget = Budget(
                    name=data['name'],
                    total_amount=data['total_amount'],
                    period_days=data['period_days'],
                    providers=[CloudProvider(p) for p in data.get('providers', [])],
                    alert_thresholds=data.get('alert_thresholds', [0.5, 0.8, 0.95]),
                    start_time=data.get('start_time', time.time()),
                    spent_amount=data.get('spent_amount', 0.0)
                )
                
                self.budgets[budget.name] = budget
                
            except Exception:
                continue
        
        expense_file = self.storage_path / "expenses.jsonl"
        if expense_file.exists():
            try:
                with open(expense_file, 'r') as f:
                    for line in f:
                        expense = json.loads(line.strip())
                        self.spending_history.append(expense)
            except Exception:
                pass
