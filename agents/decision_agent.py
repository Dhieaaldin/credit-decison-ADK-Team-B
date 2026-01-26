"""
Decision Agent

Makes approve/reject/manual review decisions based on risk analysis.
"""

from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse


class DecisionAgent(BaseAgent):
    """
    Agent responsible for making credit decisions.
    
    Input: Risk scores and anomalies
    Output: Decision (APPROVE/REJECT/MANUAL_REVIEW) with confidence
    """
    
    def __init__(self):
        super().__init__("decision_agent", "Credit Decision")
        self.decision_rules = {
            "approve_threshold": 0.15,  # Default rate < 15%
            "reject_threshold": 0.40,   # Default rate > 40%
            "max_high_severity_anomalies": 1
        }
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Make credit decision based on risk analysis.
        
        Args:
            message: Contains risk scores and anomalies
            
        Returns:
            AgentResponse with decision and confidence
        """
        import time
        start_time = time.time()
        
        try:
            overall_risk = message.payload.get("overall_risk_score", {})
            anomalies = message.payload.get("anomalies", [])
            
            if not overall_risk:
                return AgentResponse(
                    success=False,
                    error="No risk scores provided",
                    processing_time=time.time() - start_time
                )
            
            # Extract risk metrics
            default_rate = overall_risk.get("overall_default_rate", 0.5)
            high_severity_anomalies = sum(
                1 for a in anomalies if a.get("severity") == "HIGH"
            )
            medium_severity_anomalies = sum(
                1 for a in anomalies if a.get("severity") == "MEDIUM"
            )
            
            # Make decision
            decision, confidence = self._make_decision(
                default_rate,
                high_severity_anomalies,
                medium_severity_anomalies
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
                    "anomalies_count": {
                        "high": high_severity_anomalies,
                        "medium": medium_severity_anomalies,
                        "total": len(anomalies)
                    }
                },
                metadata={
                    "decision_rules_applied": self.decision_rules
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
        high_severity_anomalies: int,
        medium_severity_anomalies: int
    ) -> tuple[str, float]:
        """
        Make decision based on risk metrics.
        
        Returns:
            Tuple of (decision, confidence)
        """
        # REJECT if high default rate or multiple high-severity anomalies
        if (default_rate > self.decision_rules["reject_threshold"] or
            high_severity_anomalies >= self.decision_rules["max_high_severity_anomalies"] + 1):
            confidence = min(0.9, 0.5 + (default_rate * 0.3) + (high_severity_anomalies * 0.1))
            return "REJECT", confidence
        
        # APPROVE if low default rate and no high-severity anomalies
        elif (default_rate < self.decision_rules["approve_threshold"] and
              high_severity_anomalies == 0 and
              medium_severity_anomalies <= 1):
            confidence = min(0.9, 0.6 + (1 - default_rate) * 0.2)
            return "APPROVE", confidence
        
        # Otherwise, MANUAL_REVIEW
        else:
            return "MANUAL_REVIEW", 0.6
