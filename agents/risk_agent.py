"""
Risk Aggregation Agent

Analyzes similar cases and computes risk scores and fraud indicators.
"""

from typing import Dict, List, Any
import numpy as np
from collections import Counter
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse


class RiskAgent(BaseAgent):
    """
    Agent responsible for fraud detection and risk analysis.
    
    Input: Similar historical cases
    Output: Risk scores and anomaly flags
    """
    
    def __init__(self):
        super().__init__("risk_agent", "Fraud & Risk Analysis")
        self.risk_thresholds = {
            "high_default_rate": 0.4,
            "low_default_rate": 0.15,
            "low_similarity": 0.5,
            "high_dti": 40.0,
            "high_collections": 10000.0
        }
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Analyze similar cases and compute risk scores.
        
        Args:
            message: Contains similar cases per dimension
            
        Returns:
            AgentResponse with risk scores and anomalies
        """
        import time
        start_time = time.time()
        
        try:
            similar_cases = message.payload.get("similar_cases", {})
            original_data = message.payload.get("original_data", {})
            
            if not similar_cases:
                return AgentResponse(
                    success=False,
                    error="No similar cases provided",
                    processing_time=time.time() - start_time
                )
            
            # Analyze risk per dimension
            per_dimension_risk = {}
            for chunk_type, cases in similar_cases.items():
                risk_analysis = self._analyze_dimension_risk(chunk_type, cases)
                per_dimension_risk[chunk_type] = risk_analysis
            
            # Compute overall risk score
            overall_risk = self._compute_overall_risk(per_dimension_risk)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(per_dimension_risk, original_data)
            
            # Log processing
            self.log_message(message)
            self.update_state({
                "last_processed": time.time(),
                "total_analyzed": self.state.get("total_analyzed", 0) + 1
            })
            
            return AgentResponse(
                success=True,
                data={
                    "per_dimension_risk": per_dimension_risk,
                    "overall_risk_score": overall_risk,
                    "anomalies": anomalies
                },
                metadata={
                    "num_dimensions": len(per_dimension_risk),
                    "num_anomalies": len(anomalies)
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Risk analysis error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _analyze_dimension_risk(
        self,
        chunk_type: str,
        cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze risk for a specific dimension."""
        if not cases:
            return {
                "default_rate": 1.0,
                "avg_similarity": 0.0,
                "num_matches": 0,
                "risk_level": "HIGH"
            }
        
        # Extract scores and loan statuses
        scores = [c.get("score", 0.0) for c in cases]
        statuses = [c.get("payload", {}).get("loan_status", "Unknown") for c in cases]
        
        # Calculate default rate
        charged_off = sum(1 for s in statuses if "Charged Off" in str(s))
        fully_paid = sum(1 for s in statuses if "Fully Paid" in str(s))
        total_resolved = charged_off + fully_paid
        
        if total_resolved > 0:
            default_rate = charged_off / total_resolved
        else:
            default_rate = 0.5  # Neutral if no resolved loans
        
        # Determine risk level
        if default_rate >= self.risk_thresholds["high_default_rate"]:
            risk_level = "HIGH"
        elif default_rate <= self.risk_thresholds["low_default_rate"]:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        return {
            "default_rate": default_rate,
            "avg_similarity": np.mean(scores) if scores else 0.0,
            "num_matches": len(cases),
            "risk_level": risk_level,
            "status_distribution": dict(Counter(statuses))
        }
    
    def _compute_overall_risk(
        self,
        per_dimension_risk: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute overall risk score from per-dimension analysis."""
        default_rates = []
        weights = []
        
        for chunk_type, analysis in per_dimension_risk.items():
            default_rate = analysis.get("default_rate", 0.5)
            num_matches = analysis.get("num_matches", 0)
            
            if num_matches > 0:
                default_rates.append(default_rate)
                weights.append(num_matches)
        
        if not default_rates:
            overall_default_rate = 0.5
        else:
            overall_default_rate = np.average(default_rates, weights=weights)
        
        return {
            "overall_default_rate": overall_default_rate,
            "weighted_average": True,
            "num_dimensions": len(per_dimension_risk)
        }
    
    def _detect_anomalies(
        self,
        per_dimension_risk: Dict[str, Dict[str, Any]],
        original_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Detect anomalies and fraud indicators."""
        anomalies = []
        
        # Check for low similarity
        all_avg_similarities = [
            analysis.get("avg_similarity", 0.0)
            for analysis in per_dimension_risk.values()
        ]
        
        if all_avg_similarities:
            overall_avg_sim = np.mean(all_avg_similarities)
            if overall_avg_sim < self.risk_thresholds["low_similarity"]:
                anomalies.append({
                    "type": "LOW_SIMILARITY",
                    "severity": "HIGH",
                    "description": f"Very low similarity to historical loans (avg: {overall_avg_sim:.3f})"
                })
        
        # Check for conflicting signals
        conflicting_dims = sum(
            1 for analysis in per_dimension_risk.values()
            if analysis.get("risk_level") == "MEDIUM"
        )
        
        if conflicting_dims >= 3:
            anomalies.append({
                "type": "CONFLICTING_SIGNALS",
                "severity": "MEDIUM",
                "description": "Conflicting risk signals across multiple dimensions"
            })
        
        # Check high debt-to-income
        dti = original_data.get("debt_to_income", 0)
        if dti > self.risk_thresholds["high_dti"]:
            anomalies.append({
                "type": "HIGH_DEBT_TO_INCOME",
                "severity": "HIGH",
                "description": f"Very high debt-to-income ratio: {dti:.1f}%"
            })
        
        # Check collections
        collections = original_data.get("total_collection_amount_ever", 0)
        if collections > self.risk_thresholds["high_collections"]:
            anomalies.append({
                "type": "HIGH_COLLECTIONS",
                "severity": "MEDIUM",
                "description": f"Significant collection amounts: ${collections:,.0f}"
            })
        
        # Check current delinquencies
        current_delinq = original_data.get("current_accounts_delinq", 0)
        if current_delinq > 0:
            anomalies.append({
                "type": "CURRENT_DELINQUENCIES",
                "severity": "HIGH",
                "description": f"Currently {current_delinq} delinquent account(s)"
            })
        
        return anomalies
