"""
Audit Logger Module

Provides comprehensive audit logging for credit decisions
to meet regulatory and compliance requirements.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import threading


@dataclass
class AuditLogEntry:
    """Structured audit log entry."""
    log_id: str
    timestamp: str
    event_type: str
    application_id: Optional[str]
    user_id: Optional[str]
    action: str
    details: Dict[str, Any]
    outcome: Optional[str]
    risk_score: Optional[float]
    decision: Optional[str]
    processing_time_ms: Optional[float]
    ip_address: Optional[str]
    session_id: Optional[str]


class AuditLogger:
    """
    Comprehensive audit logging for credit decisions.
    
    Logs are stored in both:
    1. JSON Lines file for easy parsing
    2. Structured format for database insertion
    
    All credit decision activities must be logged for:
    - Regulatory compliance (ECOA, FCRA)
    - Internal audit
    - Dispute resolution
    - Model monitoring
    """
    
    EVENT_TYPES = {
        'APPLICATION_RECEIVED': 'Application received for processing',
        'VALIDATION_PASSED': 'Application passed validation',
        'VALIDATION_FAILED': 'Application failed validation',
        'DECISION_MADE': 'Credit decision rendered',
        'MANUAL_REVIEW_QUEUED': 'Application queued for manual review',
        'ADVERSE_ACTION_GENERATED': 'Adverse action notice generated',
        'APPLICATION_APPROVED': 'Application approved',
        'APPLICATION_REJECTED': 'Application rejected',
        'PIPELINE_ERROR': 'Pipeline processing error',
        'FALLBACK_ACTIVATED': 'Fallback mechanism activated',
        'EXPLANATION_GENERATED': 'Explanation generated for decision'
    }
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_file_prefix: str = "audit_log"
    ):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory to store log files
            log_file_prefix: Prefix for log file names
        """
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file_prefix = log_file_prefix
        self._lock = threading.Lock()
        self._buffer: List[AuditLogEntry] = []
        self._buffer_size = 50
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path (rotated daily)."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        return self.log_dir / f"{self.log_file_prefix}_{date_str}.jsonl"
    
    def log(
        self,
        event_type: str,
        action: str,
        details: Dict[str, Any],
        application_id: Optional[str] = None,
        user_id: Optional[str] = None,
        outcome: Optional[str] = None,
        risk_score: Optional[float] = None,
        decision: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (from EVENT_TYPES)
            action: Specific action performed
            details: Additional details about the event
            application_id: Associated application ID
            user_id: User who performed the action
            outcome: Result of the action
            risk_score: Calculated risk score (if applicable)
            decision: Credit decision (if applicable)
            processing_time_ms: Processing time in milliseconds
            ip_address: Client IP address
            session_id: Session identifier
            
        Returns:
            Unique log entry ID
        """
        log_id = str(uuid.uuid4())
        
        entry = AuditLogEntry(
            log_id=log_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            application_id=application_id,
            user_id=user_id,
            action=action,
            details=self._sanitize_details(details),
            outcome=outcome,
            risk_score=risk_score,
            decision=decision,
            processing_time_ms=processing_time_ms,
            ip_address=ip_address,
            session_id=session_id
        )
        
        self._write_entry(entry)
        
        return log_id
    
    def log_application_received(
        self,
        application_id: str,
        application_data: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Log receipt of a new application."""
        return self.log(
            event_type='APPLICATION_RECEIVED',
            action='Received loan application for processing',
            details={
                'loan_amount': application_data.get('loan_amount'),
                'loan_purpose': application_data.get('loan_purpose'),
                'input_fields': list(application_data.keys())
            },
            application_id=application_id,
            user_id=user_id,
            ip_address=ip_address
        )
    
    def log_decision(
        self,
        application_id: str,
        decision: str,
        risk_score: float,
        confidence: float,
        processing_time_ms: float,
        explanation_summary: str,
        component_scores: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Log a credit decision."""
        return self.log(
            event_type='DECISION_MADE',
            action=f'Credit decision: {decision}',
            details={
                'confidence': confidence,
                'explanation_summary': explanation_summary[:500],
                'component_scores': component_scores,
                'decision_rules_applied': True
            },
            application_id=application_id,
            user_id=user_id,
            outcome=decision,
            risk_score=risk_score,
            decision=decision,
            processing_time_ms=processing_time_ms
        )
    
    def log_adverse_action(
        self,
        application_id: str,
        reasons: List[Dict[str, str]],
        applicant_name: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log generation of adverse action notice."""
        return self.log(
            event_type='ADVERSE_ACTION_GENERATED',
            action='Generated ECOA-compliant adverse action notice',
            details={
                'reason_count': len(reasons),
                'reason_codes': [r.get('code') for r in reasons],
                'applicant_notified': True
            },
            application_id=application_id,
            user_id=user_id,
            decision='REJECT'
        )
    
    def log_error(
        self,
        application_id: Optional[str],
        error_type: str,
        error_message: str,
        stage: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log a pipeline error."""
        return self.log(
            event_type='PIPELINE_ERROR',
            action=f'Error in {stage}: {error_type}',
            details={
                'error_type': error_type,
                'error_message': error_message,
                'stage': stage
            },
            application_id=application_id,
            user_id=user_id,
            outcome='ERROR'
        )
    
    def log_fallback(
        self,
        application_id: str,
        fallback_mode: str,
        trigger_reason: str,
        user_id: Optional[str] = None
    ) -> str:
        """Log activation of fallback mechanism."""
        return self.log(
            event_type='FALLBACK_ACTIVATED',
            action=f'Fallback activated: {fallback_mode}',
            details={
                'fallback_mode': fallback_mode,
                'trigger_reason': trigger_reason
            },
            application_id=application_id,
            user_id=user_id
        )
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Remove PII from details before logging."""
        pii_fields = {
            'ssn', 'social_security', 'tax_id', 'email', 'phone',
            'address', 'street', 'zip', 'zipcode', 'date_of_birth',
            'dob', 'bank_account', 'routing_number', 'credit_card'
        }
        
        sanitized = {}
        for key, value in details.items():
            key_lower = key.lower()
            if any(pii in key_lower for pii in pii_fields):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _write_entry(self, entry: AuditLogEntry):
        """Write entry to log file."""
        with self._lock:
            log_file = self._get_current_log_file()
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
    
    def query_logs(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        application_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            application_id: Filter by application ID
            event_type: Filter by event type
            limit: Maximum number of records
            
        Returns:
            List of matching log entries
        """
        results = []
        
        # Get list of log files to search
        log_files = sorted(self.log_dir.glob(f"{self.log_file_prefix}_*.jsonl"))
        
        if start_date:
            log_files = [f for f in log_files if f.stem >= f"{self.log_file_prefix}_{start_date}"]
        
        if end_date:
            log_files = [f for f in log_files if f.stem <= f"{self.log_file_prefix}_{end_date}"]
        
        for log_file in reversed(log_files):  # Most recent first
            if len(results) >= limit:
                break
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        
                        try:
                            entry = json.loads(line.strip())
                            
                            # Apply filters
                            if application_id and entry.get('application_id') != application_id:
                                continue
                            if event_type and entry.get('event_type') != event_type:
                                continue
                            
                            results.append(entry)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                continue
        
        return results
    
    def get_application_history(self, application_id: str) -> List[Dict[str, Any]]:
        """
        Get complete audit history for an application.
        
        Args:
            application_id: Application ID to look up
            
        Returns:
            List of all audit events for the application, chronologically ordered
        """
        logs = self.query_logs(application_id=application_id, limit=10000)
        return sorted(logs, key=lambda x: x.get('timestamp', ''))
