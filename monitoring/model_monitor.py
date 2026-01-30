"""
Model Monitoring Module

Production model monitoring per SR 11-7 requirements.
Implements ongoing monitoring for model risk management.
"""

import json
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import sqlite3


class ModelMonitor:
    """
    Production model monitoring per regulatory requirements.
    Implements SR 11-7 ongoing monitoring standards.
    """
    
    def __init__(
        self,
        model_id: str,
        model_version: str = "1.0.0",
        db_path: Optional[str] = None
    ):
        """
        Initialize model monitor.
        
        Args:
            model_id: Unique identifier for the model
            model_version: Version of the model
            db_path: Path to SQLite database for storing decisions
        """
        self.model_id = model_id
        self.model_version = model_version
        
        # Setup database
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "decisions.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        self._init_database()
        
        # Monitoring thresholds
        self.thresholds = {
            'psi_warning': 0.10,
            'psi_critical': 0.25,
            'approval_rate_deviation': 0.05,
            'default_rate_deviation': 0.02,
            'latency_p99_ms': 5000,
            'error_rate_warning': 0.01,
            'error_rate_critical': 0.05
        }
        
        # In-memory cache for batch logging
        self._log_cache: List[Dict] = []
        self._cache_size = 100
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_log (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT,
                    input_hash TEXT,
                    risk_score REAL,
                    decision TEXT,
                    confidence REAL,
                    explanation_summary TEXT,
                    component_scores TEXT,
                    similar_case_ids TEXT,
                    latency_ms REAL,
                    pipeline_errors TEXT,
                    inputs_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migration check: Add inputs_json if missing
            try:
                cursor = conn.execute("PRAGMA table_info(decision_log)")
                columns = [c[1] for c in cursor.fetchall()]
                if 'inputs_json' not in columns:
                    conn.execute("ALTER TABLE decision_log ADD COLUMN inputs_json TEXT")
            except:
                pass
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT,
                    details TEXT,
                    acknowledged INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_decisions INTEGER,
                    approve_count INTEGER,
                    reject_count INTEGER,
                    manual_review_count INTEGER,
                    mean_risk_score REAL,
                    mean_latency_ms REAL,
                    p99_latency_ms REAL,
                    error_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_timestamp 
                ON decision_log(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_model 
                ON decision_log(model_id, model_version)
            """)
            
            conn.commit()
    
    def log_decision(self, decision_record: Dict[str, Any]) -> str:
        """
        Log every decision with full audit trail.
        Required for regulatory compliance and model monitoring.
        
        Args:
            decision_record: Decision data to log
            
        Returns:
            Unique decision ID
        """
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        record = {
            'decision_id': decision_id,
            'timestamp': timestamp,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'input_hash': self._hash_inputs(decision_record.get('inputs', {})),
            'risk_score': decision_record.get('risk_score'),
            'decision': decision_record.get('decision'),
            'confidence': decision_record.get('confidence'),
            'explanation_summary': (decision_record.get('explanation', '') or '')[:500],
            'component_scores': json.dumps(decision_record.get('component_scores', {})),
            'similar_case_ids': json.dumps(
                [c.get('id') for c in decision_record.get('similar_cases', [])[:5]]
            ),
            'latency_ms': decision_record.get('latency_ms'),
            'pipeline_errors': json.dumps(decision_record.get('errors', [])),
            'inputs_json': json.dumps(decision_record.get('inputs', {}))
        }
        
        # Add to cache
        self._log_cache.append(record)
        
        # Flush if cache is full
        if len(self._log_cache) >= self._cache_size:
            self._flush_log_cache()
        else:
            # For single inserts, write immediately
            self._write_single_record(record)
        
        return decision_id
    
    def _write_single_record(self, record: Dict[str, Any]):
        """Write a single record to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO decision_log (
                    decision_id, timestamp, model_id, model_version,
                    input_hash, risk_score, decision, confidence,
                    explanation_summary, component_scores, similar_case_ids,
                    latency_ms, pipeline_errors, inputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record['decision_id'], record['timestamp'], record['model_id'],
                record['model_version'], record['input_hash'], record['risk_score'],
                record['decision'], record['confidence'], record['explanation_summary'],
                record['component_scores'], record['similar_case_ids'],
                record['latency_ms'], record['pipeline_errors'], record['inputs_json']
            ))
            conn.commit()
    
    def _flush_log_cache(self):
        """Flush log cache to database."""
        if not self._log_cache:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO decision_log (
                    decision_id, timestamp, model_id, model_version,
                    input_hash, risk_score, decision, confidence,
                    explanation_summary, component_scores, similar_case_ids,
                    latency_ms, pipeline_errors, inputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r['decision_id'], r['timestamp'], r['model_id'],
                    r['model_version'], r['input_hash'], r['risk_score'],
                    r['decision'], r['confidence'], r['explanation_summary'],
                    r['component_scores'], r['similar_case_ids'],
                    r['latency_ms'], r['pipeline_errors'], r['inputs_json']
                )
                for r in self._log_cache
            ])
            conn.commit()
        
        self._log_cache = []
    
    def _hash_inputs(self, inputs: Dict[str, Any]) -> str:
        """Create hash of inputs for deduplication."""
        import hashlib
        # Remove PII before hashing
        sanitized = {k: v for k, v in inputs.items() 
                    if k not in ['applicant_name', 'email', 'phone', 'ssn', 'address']}
        return hashlib.sha256(json.dumps(sanitized, sort_keys=True).encode()).hexdigest()[:16]
    
    def daily_monitoring_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate daily monitoring report.
        
        Args:
            date: Date string in YYYY-MM-DD format (default: today)
            
        Returns:
            Comprehensive daily monitoring report
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            decisions = conn.execute("""
                SELECT * FROM decision_log
                WHERE DATE(timestamp) = ?
            """, [date]).fetchall()
        
        if not decisions:
            return {
                'date': date,
                'volume': 0,
                'message': 'No decisions recorded for this date'
            }
        
        decisions = [dict(d) for d in decisions]
        
        risk_scores = [d['risk_score'] for d in decisions if d['risk_score'] is not None]
        latencies = [d['latency_ms'] for d in decisions if d['latency_ms'] is not None]
        
        report = {
            'date': date,
            'volume': len(decisions),
            'decision_distribution': {
                'approve': sum(1 for d in decisions if d['decision'] == 'APPROVE'),
                'reject': sum(1 for d in decisions if d['decision'] == 'REJECT'),
                'manual_review': sum(1 for d in decisions if d['decision'] == 'MANUAL_REVIEW')
            },
            'score_statistics': {
                'mean': np.mean(risk_scores) if risk_scores else None,
                'std': np.std(risk_scores) if risk_scores else None,
                'p10': np.percentile(risk_scores, 10) if risk_scores else None,
                'p50': np.percentile(risk_scores, 50) if risk_scores else None,
                'p90': np.percentile(risk_scores, 90) if risk_scores else None
            },
            'latency': {
                'mean_ms': np.mean(latencies) if latencies else None,
                'p99_ms': np.percentile(latencies, 99) if latencies else None
            },
            'error_rate': sum(1 for d in decisions if d['pipeline_errors'] != '[]') / len(decisions)
        }
        
        # Check thresholds and create alerts
        alerts = self._check_thresholds(report)
        report['alerts'] = alerts
        
        # Store daily metrics
        self._store_daily_metrics(date, report)
        
        return report
    
    def _check_thresholds(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # Error rate check
        error_rate = report.get('error_rate', 0)
        if error_rate >= self.thresholds['error_rate_critical']:
            alerts.append({
                'type': 'ERROR_RATE_CRITICAL',
                'severity': 'CRITICAL',
                'message': f"Error rate {error_rate:.1%} exceeds critical threshold"
            })
        elif error_rate >= self.thresholds['error_rate_warning']:
            alerts.append({
                'type': 'ERROR_RATE_WARNING',
                'severity': 'WARNING',
                'message': f"Error rate {error_rate:.1%} exceeds warning threshold"
            })
        
        # Latency check
        p99_latency = report.get('latency', {}).get('p99_ms')
        if p99_latency and p99_latency > self.thresholds['latency_p99_ms']:
            alerts.append({
                'type': 'LATENCY_HIGH',
                'severity': 'WARNING',
                'message': f"P99 latency {p99_latency:.0f}ms exceeds threshold"
            })
        
        # Log alerts to database
        for alert in alerts:
            self._log_alert(alert)
        
        return alerts
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log an alert to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO monitoring_alerts (
                    alert_id, timestamp, alert_type, severity, message, details
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).isoformat(),
                alert.get('type', 'UNKNOWN'),
                alert.get('severity', 'INFO'),
                alert.get('message', ''),
                json.dumps(alert.get('details', {}))
            ))
            conn.commit()
    
    def _store_daily_metrics(self, date: str, report: Dict[str, Any]):
        """Store daily metrics for trend analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_metrics (
                    date, total_decisions, approve_count, reject_count,
                    manual_review_count, mean_risk_score, mean_latency_ms,
                    p99_latency_ms, error_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                report.get('volume', 0),
                report.get('decision_distribution', {}).get('approve', 0),
                report.get('decision_distribution', {}).get('reject', 0),
                report.get('decision_distribution', {}).get('manual_review', 0),
                report.get('score_statistics', {}).get('mean'),
                report.get('latency', {}).get('mean_ms'),
                report.get('latency', {}).get('p99_ms'),
                int(report.get('error_rate', 0) * report.get('volume', 0))
            ))
            conn.commit()
    
    def check_model_drift(self, baseline_days: int = 30, recent_days: int = 7) -> Dict[str, Any]:
        """
        Check for model drift using PSI on recent vs. baseline data.
        Should be run weekly.
        
        Args:
            baseline_days: Number of days for baseline period
            recent_days: Number of days for recent period
            
        Returns:
            Drift analysis results
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get baseline scores
            baseline = conn.execute("""
                SELECT risk_score FROM decision_log
                WHERE risk_score IS NOT NULL
                AND DATE(timestamp) BETWEEN DATE('now', ?) AND DATE('now', ?)
            """, [f'-{baseline_days + recent_days} days', f'-{recent_days} days']).fetchall()
            
            # Get recent scores
            recent = conn.execute("""
                SELECT risk_score FROM decision_log
                WHERE risk_score IS NOT NULL
                AND DATE(timestamp) >= DATE('now', ?)
            """, [f'-{recent_days} days']).fetchall()
        
        if len(baseline) < 100 or len(recent) < 50:
            return {
                'status': 'INSUFFICIENT_DATA',
                'message': 'Need more data for drift analysis',
                'baseline_count': len(baseline),
                'recent_count': len(recent)
            }
        
        baseline_scores = [b[0] for b in baseline]
        recent_scores = [r[0] for r in recent]
        
        # Calculate PSI
        score_psi = self._calculate_psi(baseline_scores, recent_scores)
        
        drift_detected = score_psi > self.thresholds['psi_warning']
        
        result = {
            'score_psi': score_psi,
            'baseline_count': len(baseline_scores),
            'recent_count': len(recent_scores),
            'drift_detected': drift_detected,
            'status': 'DRIFT_DETECTED' if drift_detected else 'OK',
            'severity': 'CRITICAL' if score_psi > self.thresholds['psi_critical'] else 
                       'WARNING' if drift_detected else 'OK'
        }
        
        if drift_detected:
            self._log_alert({
                'type': 'MODEL_DRIFT',
                'severity': result['severity'],
                'message': f'Model drift detected: PSI = {score_psi:.3f}',
                'details': result
            })
        
        return result
    
    def _calculate_psi(
        self,
        expected: List[float],
        actual: List[float],
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI < 0.10: No significant change
        0.10 <= PSI < 0.25: Moderate shift, investigation recommended
        PSI >= 0.25: Significant shift, action required
        """
        expected = np.array(expected)
        actual = np.array(actual)
        
        # Create buckets based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Add small constant to avoid division by zero
        expected_pct = (expected_counts + 0.0001) / len(expected)
        actual_pct = (actual_counts + 0.0001) / len(actual)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)
    
    def get_decision_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve decision history for analysis.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records
            
        Returns:
            List of decision records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM decision_log WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND DATE(timestamp) >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND DATE(timestamp) <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            results = conn.execute(query, params).fetchall()
        
        return [dict(r) for r in results]
    
    def get_alerts(
        self,
        acknowledged: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve monitoring alerts.
        
        Args:
            acknowledged: Filter by acknowledgment status
            severity: Filter by severity level
            limit: Maximum number of records
            
        Returns:
            List of alert records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM monitoring_alerts WHERE 1=1"
            params = []
            
            if acknowledged is not None:
                query += " AND acknowledged = ?"
                params.append(int(acknowledged))
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            results = conn.execute(query, params).fetchall()
        
        return [dict(r) for r in results]
    
    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE monitoring_alerts SET acknowledged = 1 WHERE alert_id = ?
            """, [alert_id])
            conn.commit()
