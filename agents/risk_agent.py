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
    
    Enhanced with:
    - Confidence intervals for risk estimates
    - Bayesian priors for sparse data
    - Improved anomaly detection
    
    Input: Similar historical cases
    Output: Risk scores and anomaly flags
    """
    
    # Prior parameters for Bayesian estimation
    PRIOR_DEFAULT_RATE = 0.15  # Population average
    PRIOR_WEIGHT = 3.0         # Equivalent to 3 observations
    
    def __init__(self):
        super().__init__("risk_agent", "Fraud & Risk Analysis")
        self.risk_thresholds = {
            "high_default_rate": 0.35,    # Lowered from 0.4 for safety
            "low_default_rate": 0.12,     # Lowered from 0.15 for accuracy
            "low_similarity": 0.5,
            "high_dti": 40.0,
            "very_high_dti": 50.0,
            "high_collections": 10000.0,
            "min_sample_size": 3          # Minimum matches for confidence
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
                risk_analysis = self._analyze_dimension_risk(chunk_type, cases, original_data)
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
        cases: List[Dict[str, Any]],
        applicant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze risk for a specific dimension with confidence estimation.
        Uses Bayesian prior when sample size is small.
        """
        if not cases:
            return {
                "default_rate": self.PRIOR_DEFAULT_RATE,
                "avg_similarity": 0.0,
                "num_matches": 0,
                "risk_level": "MEDIUM",
                "confidence": 0.0,
                "used_prior": True
            }
        
        # Extract scores and loan statuses
        scores = [c.get("score", 0.0) for c in cases]
        statuses = [c.get("payload", {}).get("loan_status", "Unknown") for c in cases]
        
        # Calculate default rate with similarity weighting
        weighted_defaults = 0.0
        weighted_total = 0.0
        status_counts = Counter()
        
        for case, score in zip(cases, scores):
            payload = case.get("payload", {})
            status = payload.get("loan_status", "Unknown")
            status_counts[status] += 1
            
            # --- NUMERIC DISTANCE PENALTY ---
            # Penalize the semantic weight if numeric features are too different
            weight = score
            
            try:
                # 1. DTI Penalty
                hist_dti = float(payload.get("debt_to_income", 0))
                app_dti = float(applicant_data.get("debt_to_income", 0))
                if app_dti > 0:
                    dti_ratio = abs(hist_dti - app_dti) / max(app_dti, 1.0)
                    if dti_ratio > 0.25: # 25% tolerance
                        weight *= (1.0 - min(0.5, dti_ratio - 0.25))
                
                # 2. Income Penalty
                hist_inc = float(payload.get("annual_income", 0))
                app_inc = float(applicant_data.get("annual_income", 0))
                if app_inc > 0:
                    inc_ratio = abs(hist_inc - app_inc) / max(app_inc, 1.0)
                    if inc_ratio > 0.5: # 50% tolerance for income
                        weight *= (1.0 - min(0.3, (inc_ratio - 0.5) * 0.5))
            except Exception:
                pass # Fallback to semantic score if numeric data is missing/invalid
            
            # DEFAULTED: High risk indicators
            if any(x in str(status) for x in ["Charged Off", "Default", "Late"]):
                weighted_defaults += weight
                weighted_total += weight
            # GOOD STANDING: Positive risk indicators
            elif any(x in str(status) for x in ["Fully Paid", "Current", "In Grace Period", "Does not meet credit policy"]):
                # Note: "Does not meet credit policy" with status "Fully Paid" is actually good
                weighted_total += weight
            else:
                pass
        
        # Apply Bayesian prior if sample is small
        # Factor in similarity: if we have 99% similarity, we need less samples to be confident
        mean_similarity = np.mean(scores) if scores else 0.0
        similarity_bonus = 2.0 if mean_similarity > 0.9 else 0.0
        
        effective_sample_size = weighted_total + similarity_bonus
        
        if effective_sample_size < self.risk_thresholds["min_sample_size"]:
            # Bayesian posterior with prior
            default_rate = (
                (self.PRIOR_DEFAULT_RATE * self.PRIOR_WEIGHT + weighted_defaults) /
                (self.PRIOR_WEIGHT + weighted_total)
            )
            confidence = weighted_total / (self.PRIOR_WEIGHT + weighted_total + 1)
            used_prior = True
        else:
            default_rate = weighted_defaults / weighted_total if weighted_total > 0 else 0.0
            # Confidence based on sample size and similarity
            confidence = min(1.0, (weighted_total / 8.0) + (mean_similarity * 0.2))
            used_prior = False
        
        # Determine risk level
        if default_rate >= self.risk_thresholds["high_default_rate"]:
            risk_level = "HIGH"
        elif default_rate <= self.risk_thresholds["low_default_rate"]:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        
        # Summary for explanation agent
        status_summary = f"{status_counts.get('Current', 0) + status_counts.get('Fully Paid', 0)} Good Standing, {status_counts.get('Charged Off', 0) + status_counts.get('Default', 0)} Defaulted"

        return {
            "default_rate": float(default_rate),
            "avg_similarity": float(mean_similarity),
            "num_matches": len(cases),
            "risk_level": risk_level,
            "confidence": float(confidence),
            "used_prior": used_prior,
            "effective_sample_size": float(weighted_total),
            "status_distribution": dict(status_counts),
            "status_summary": status_summary
        }
    
    def _compute_overall_risk(
        self,
        per_dimension_risk: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute overall risk score from per-dimension analysis.
        Uses confidence-weighted averaging for robustness.
        """
        # Dimension importance weights
        dimension_weights = {
            'income_stability': 0.20,
            'credit_behavior': 0.25,
            'debt_obligations': 0.25,
            'recent_behavior': 0.15,
            'account_portfolio': 0.10,
            'loan_context': 0.05
        }
        
        weighted_rate = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        
        for chunk_type, analysis in per_dimension_risk.items():
            default_rate = analysis.get("default_rate", 0.5)
            confidence = analysis.get("confidence", 0.5)
            dim_weight = dimension_weights.get(chunk_type, 0.1)
            
            # Combine dimension weight with confidence
            effective_weight = dim_weight * (0.5 + 0.5 * confidence)
            
            weighted_rate += default_rate * effective_weight
            total_weight += effective_weight
            total_confidence += confidence * dim_weight
        
        if total_weight > 0:
            overall_default_rate = weighted_rate / total_weight
        else:
            overall_default_rate = self.PRIOR_DEFAULT_RATE
        
        # Normalize confidence
        overall_confidence = total_confidence / sum(dimension_weights.values())
        
        return {
            "overall_default_rate": float(overall_default_rate),
            "overall_confidence": float(overall_confidence),
            "weighted_average": True,
            "num_dimensions": len(per_dimension_risk),
            "dimension_weights_used": dimension_weights
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
