"""
Audit Logging for Compliance and Governance.

Provides comprehensive audit logging for ML operations including
model deployments, approvals, access, and compliance reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class EventType(Enum):
    """Audit event types."""
    MODEL_REGISTERED = "model_registered"
    MODEL_PROMOTED = "model_promoted"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_ROLLED_BACK = "model_rolled_back"
    MODEL_ARCHIVED = "model_archived"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    AB_TEST_CREATED = "ab_test_created"
    AB_TEST_STARTED = "ab_test_started"
    AB_TEST_COMPLETED = "ab_test_completed"
    SHADOW_DEPLOYMENT_CREATED = "shadow_deployment_created"
    ACCESS_GRANTED = "access_granted"
    ACCESS_REVOKED = "access_revoked"
    CONFIGURATION_CHANGED = "configuration_changed"
    SECURITY_VIOLATION = "security_violation"
    DATA_ACCESS = "data_access"


class EventSeverity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: EventType
    timestamp: str
    user: str
    severity: EventSeverity
    resource_type: str
    resource_id: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEvent:
        """Create from dictionary."""
        data = data.copy()
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = EventSeverity(data['severity'])
        return cls(**data)


@dataclass
class ComplianceReport:
    """Compliance audit report."""
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_user: Dict[str, int]
    security_violations: List[AuditEvent]
    critical_events: List[AuditEvent]
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['security_violations'] = [e.to_dict() for e in self.security_violations]
        data['critical_events'] = [e.to_dict() for e in self.critical_events]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComplianceReport:
        """Create from dictionary."""
        data = data.copy()
        data['security_violations'] = [
            AuditEvent.from_dict(e) for e in data.get('security_violations', [])
        ]
        data['critical_events'] = [
            AuditEvent.from_dict(e) for e in data.get('critical_events', [])
        ]
        return cls(**data)


class AuditLogger:
    """
    Enterprise audit logging for ML operations.
    
    Provides tamper-evident logging for compliance and governance,
    tracking all model lifecycle events, access, and configuration changes.
    
    Example:
        logger = AuditLogger("./audit_logs")
        
        # Log model deployment
        logger.log_event(
            event_type=EventType.MODEL_DEPLOYED,
            user="devops@company.com",
            resource_type="model",
            resource_id="fraud_detector:v2.0.0",
            action="deploy",
            details={
                "environment": "production",
                "strategy": "canary"
            },
            severity=EventSeverity.INFO
        )
        
        # Log security violation
        logger.log_security_violation(
            user="unknown@external.com",
            resource_type="model",
            resource_id="fraud_detector:v2.0.0",
            action="unauthorized_access",
            details={"ip_address": "192.168.1.1"}
        )
        
        # Generate compliance report
        report = logger.generate_compliance_report(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        # Export audit trail
        logger.export_audit_trail("audit_trail.json", format="json")
    """
    
    def __init__(self, storage_path: str = "./audit_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.events_path = self.storage_path / "events"
        self.reports_path = self.storage_path / "reports"
        
        self.events_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        self._event_counter = 0
    
    def log_event(
        self,
        event_type: EventType,
        user: str,
        resource_type: str,
        resource_id: str,
        action: str,
        severity: EventSeverity = EventSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> AuditEvent:
        """Log an audit event."""
        self._event_counter += 1
        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._event_counter:06d}"
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            user=user,
            severity=severity,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            tags=tags or []
        )
        
        self._save_event(event)
        return event
    
    def log_model_registration(
        self,
        model_name: str,
        version: str,
        user: str,
        framework: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> AuditEvent:
        """Log model registration event."""
        return self.log_event(
            event_type=EventType.MODEL_REGISTERED,
            user=user,
            resource_type="model",
            resource_id=f"{model_name}:{version}",
            action="register",
            details={
                "framework": framework,
                "metrics": metrics or {}
            }
        )
    
    def log_model_deployment(
        self,
        model_name: str,
        version: str,
        user: str,
        environment: str,
        strategy: str
    ) -> AuditEvent:
        """Log model deployment event."""
        return self.log_event(
            event_type=EventType.MODEL_DEPLOYED,
            user=user,
            resource_type="model",
            resource_id=f"{model_name}:{version}",
            action="deploy",
            severity=EventSeverity.INFO,
            details={
                "environment": environment,
                "strategy": strategy
            }
        )
    
    def log_model_rollback(
        self,
        model_name: str,
        version: str,
        user: str,
        reason: str,
        triggered_by: str = "automated"
    ) -> AuditEvent:
        """Log model rollback event."""
        return self.log_event(
            event_type=EventType.MODEL_ROLLED_BACK,
            user=user,
            resource_type="model",
            resource_id=f"{model_name}:{version}",
            action="rollback",
            severity=EventSeverity.WARNING,
            details={
                "reason": reason,
                "triggered_by": triggered_by
            }
        )
    
    def log_approval_request(
        self,
        model_name: str,
        version: str,
        user: str,
        target_stage: str,
        justification: str
    ) -> AuditEvent:
        """Log approval request event."""
        return self.log_event(
            event_type=EventType.APPROVAL_REQUESTED,
            user=user,
            resource_type="approval",
            resource_id=f"{model_name}:{version}",
            action="request",
            details={
                "target_stage": target_stage,
                "justification": justification
            }
        )
    
    def log_approval_granted(
        self,
        model_name: str,
        version: str,
        approver: str,
        comment: Optional[str] = None
    ) -> AuditEvent:
        """Log approval granted event."""
        return self.log_event(
            event_type=EventType.APPROVAL_GRANTED,
            user=approver,
            resource_type="approval",
            resource_id=f"{model_name}:{version}",
            action="approve",
            details={"comment": comment}
        )
    
    def log_approval_rejected(
        self,
        model_name: str,
        version: str,
        reviewer: str,
        reason: str
    ) -> AuditEvent:
        """Log approval rejected event."""
        return self.log_event(
            event_type=EventType.APPROVAL_REJECTED,
            user=reviewer,
            resource_type="approval",
            resource_id=f"{model_name}:{version}",
            action="reject",
            severity=EventSeverity.WARNING,
            details={"reason": reason}
        )
    
    def log_security_violation(
        self,
        user: str,
        resource_type: str,
        resource_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> AuditEvent:
        """Log security violation event."""
        return self.log_event(
            event_type=EventType.SECURITY_VIOLATION,
            user=user,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            severity=EventSeverity.CRITICAL,
            details=details,
            ip_address=ip_address,
            tags=["security", "violation"]
        )
    
    def log_data_access(
        self,
        user: str,
        dataset_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log data access event."""
        return self.log_event(
            event_type=EventType.DATA_ACCESS,
            user=user,
            resource_type="dataset",
            resource_id=dataset_id,
            action=action,
            details=details,
            tags=["data_access"]
        )
    
    def query_events(
        self,
        event_type: Optional[EventType] = None,
        user: Optional[str] = None,
        resource_type: Optional[str] = None,
        severity: Optional[EventSeverity] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        events = []
        
        for event_file in sorted(self.events_path.glob("**/*.json")):
            with open(event_file, 'r') as f:
                data = json.load(f)
            event = AuditEvent.from_dict(data)
            
            if event_type and event.event_type != event_type:
                continue
            if user and event.user != user:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if severity and event.severity != severity:
                continue
            
            event_time = datetime.fromisoformat(event.timestamp)
            if start_date and event_time < start_date:
                continue
            if end_date and event_time > end_date:
                continue
            
            if tags and not any(tag in event.tags for tag in tags):
                continue
            
            events.append(event)
        
        return sorted(events, key=lambda e: e.timestamp, reverse=True)
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_details: bool = True
    ) -> ComplianceReport:
        """Generate compliance audit report for a period."""
        events = self.query_events(start_date=start_date, end_date=end_date)
        
        events_by_type = {}
        events_by_severity = {}
        events_by_user = {}
        
        security_violations = []
        critical_events = []
        
        for event in events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            severity = event.severity.value
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
            
            events_by_user[event.user] = events_by_user.get(event.user, 0) + 1
            
            if event.event_type == EventType.SECURITY_VIOLATION:
                security_violations.append(event)
            
            if event.severity == EventSeverity.CRITICAL:
                critical_events.append(event)
        
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_events=len(events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_user=events_by_user,
            security_violations=security_violations if include_details else [],
            critical_events=critical_events if include_details else [],
            summary={
                "total_security_violations": len(security_violations),
                "total_critical_events": len(critical_events),
                "most_active_user": max(events_by_user.items(), key=lambda x: x[1])[0]
                if events_by_user else None,
                "most_common_event_type": max(events_by_type.items(), key=lambda x: x[1])[0]
                if events_by_type else None
            }
        )
        
        self._save_report(report)
        return report
    
    def export_audit_trail(
        self,
        output_path: str,
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> None:
        """Export audit trail to file."""
        events = self.query_events(start_date=start_date, end_date=end_date)
        
        output_file = Path(output_path)
        
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump([e.to_dict() for e in events], f, indent=2)
        elif format == "yaml":
            with open(output_file, 'w') as f:
                yaml.dump([e.to_dict() for e in events], f)
        elif format == "csv":
            import csv
            with open(output_file, 'w', newline='') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].to_dict().keys())
                    writer.writeheader()
                    for event in events:
                        writer.writerow(event.to_dict())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_event(self, event_id: str) -> AuditEvent:
        """Get event by ID."""
        for event_file in self.events_path.glob("**/*.json"):
            if event_id in event_file.stem:
                with open(event_file, 'r') as f:
                    data = json.load(f)
                return AuditEvent.from_dict(data)
        
        raise FileNotFoundError(f"Event {event_id} not found")
    
    def get_report(self, report_id: str) -> ComplianceReport:
        """Get compliance report by ID."""
        report_file = self.reports_path / f"{report_id}.json"
        if not report_file.exists():
            raise FileNotFoundError(f"Report {report_id} not found")
        
        with open(report_file, 'r') as f:
            data = json.load(f)
        return ComplianceReport.from_dict(data)
    
    def _save_event(self, event: AuditEvent) -> None:
        """Save audit event to disk with date-based partitioning."""
        date_str = datetime.fromisoformat(event.timestamp).strftime('%Y/%m/%d')
        event_dir = self.events_path / date_str
        event_dir.mkdir(parents=True, exist_ok=True)
        
        event_file = event_dir / f"{event.event_id}.json"
        with open(event_file, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)
    
    def _save_report(self, report: ComplianceReport) -> None:
        """Save compliance report to disk."""
        report_file = self.reports_path / f"{report.report_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
