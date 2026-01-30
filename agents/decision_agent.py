"""
Decision Agent

Makes approve/reject/manual review decisions based on risk analysis.
Enhanced with weighted credit score risk and improved decision rules.
"""

from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse


class DecisionAgent(BaseAgent):
    """
    Agent responsible for making credit decisions.
    
    Enhanced with:
    - Weighted Credit Score Risk (Score-to-Penalty)
    - Integrated Risk Aggregation
    - Nuanced manual review escalation
    
    Input: Risk scores and anomalies
    Output: Decision (APPROVE/REJECT/MANUAL_REVIEW) with confidence
    """
    
    def __init__(self):
        super().__init__("decision_agent", "Credit Decision")
        
        # Calibrated decision thresholds
        self.decision_rules = {
            # Default rate thresholds
            "approve_threshold": 0.15,          # Adjusted risk < 15% for approval
            "reject_threshold": 0.40,           # Adjusted risk > 40% for rejection
            
            # Confidence requirements
            "min_confidence_for_auto": 0.60,    # Minimum confidence for auto-decision
            "high_confidence_threshold": 0.75,  # High confidence threshold
            
            # Anomaly limits
            "max_high_severity_anomalies": 1,   # Max high-severity anomalies for approval
            
            # Safety margins
            "borderline_margin": 0.08           # Margin around thresholds for manual review
        }
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Make credit decision based on risk analysis.
        """
        import time
        start_time = time.time()
        
        try:
            overall_risk = message.payload.get("overall_risk_score", {})
            anomalies = message.payload.get("anomalies", [])
            per_dimension_risk = message.payload.get("per_dimension_risk", {})
            applicant_data = message.payload.get("applicant_data", {})
            
            if not overall_risk:
                return AgentResponse(
                    success=False,
                    error="No risk scores provided",
                    processing_time=time.time() - start_time
                )
            
            # Extract risk metrics
            default_rate = overall_risk.get("overall_default_rate", 0.5)
            model_confidence = overall_risk.get("overall_confidence", 0.5)
            
            # Count anomalies by severity
            high_severity = sum(1 for a in anomalies if a.get("severity") == "HIGH")
            medium_severity = sum(1 for a in anomalies if a.get("severity") == "MEDIUM")
            
            # Make decision
            decision, confidence, reasons = self._make_decision(
                default_rate=default_rate,
                model_confidence=model_confidence,
                high_severity_anomalies=high_severity,
                medium_severity_anomalies=medium_severity,
                total_anomalies=len(anomalies),
                per_dimension_risk=per_dimension_risk,
                applicant_data=applicant_data
            )
            
            # Log processing
            self.log_message(message)
            self.update_state({
                "last_processed": time.time(),
                "total_decisions": self.state.get("total_decisions", 0) + 1,
                "decisions_by_type": {
                    **self.state.get("decisions_by_type", {}),
                    decision: self.state.get("decisions_by_type", {}).get(decision, 0) + 1
                }
            })
            
            return AgentResponse(
                success=True,
                data={
                    "recommendation": decision,
                    "confidence": confidence,
                    "default_rate": default_rate,
                    "model_confidence": model_confidence,
                    "decision_reasons": reasons,
                    "anomalies_count": {
                        "high": high_severity,
                        "medium": medium_severity,
                        "total": len(anomalies)
                    }
                },
                metadata={
                    "decision_rules_applied": self.decision_rules,
                    "threshold_used": self.decision_rules["approve_threshold"]
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Decision error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _make_decision(
        self,
        default_rate: float,
        model_confidence: float,
        high_severity_anomalies: int,
        medium_severity_anomalies: int,
        total_anomalies: int,
        per_dimension_risk: Dict[str, Any],
        applicant_data: Dict[str, Any]
    ) -> Tuple[str, float, list]:
        """
        Calculates a decision by applying weighted multipliers to the base risk.
        """
        reasons = []
        credit_score = applicant_data.get("credit_score", 0)
        dti = applicant_data.get("debt_to_income", 100)
        years_employed = applicant_data.get("years_employed", 0)

        # 1. Base Risk Aggregation (Historical retrieval data + Anomaly impact)
        anomaly_impact = (
            high_severity_anomalies * 0.15 +
            medium_severity_anomalies * 0.05 +
            total_anomalies * 0.01
        )
        base_risk = min(1.0, default_rate + anomaly_impact)
        
        # 2. Strategic Credit Multiplier
        # 1.5x for 580-600, 2.0x for < 580
        multiplier = 1.0
        if credit_score < 580:
            multiplier = 2.0
            reasons.append(f"Significant credit risk adjustment (2.0x) for score below 580")
        elif credit_score <= 600:
            multiplier = 1.5
            reasons.append(f"Moderate credit risk adjustment (1.5x) for borderline score ({credit_score})")

        # 3. Final Adjusted Risk
        final_risk = min(1.0, base_risk * multiplier)
        
        # 4. Supplemental Reasons
        if default_rate < 0.10:
            reasons.append(f"Strong historical peer stability (Base probability: {default_rate:.1%})")
        
        if dti > 43.0:
            reasons.append(f"Elevated DTI ({dti:.1f}%) requires additional oversight")
        elif dti < 20.0:
            reasons.append(f"Strong debt coverage ratio ({dti:.1f}%)")

        # 5. Threshold Calibration
        conf_factor = max(0.5, model_confidence)
        approve_threshold = self.decision_rules["approve_threshold"] / conf_factor
        reject_threshold = self.decision_rules["reject_threshold"] * conf_factor
        
        # Mapping to decision
        if final_risk > reject_threshold:
            return "REJECT", min(0.95, 0.4 + final_risk * 0.5), reasons
            
        if final_risk < approve_threshold and high_severity_anomalies == 0 and dti <= 43.0:
            return "APPROVE", min(0.95, 0.6 + (1 - final_risk) * 0.3), reasons
            
        # Default to Manual Review if not clearly approved or rejected
        if dti > 43.0:
            reasons.append("Flagged for manual review due to preferred DTI threshold (>43%)")
            
        return "MANUAL_REVIEW", 0.80, reasons
