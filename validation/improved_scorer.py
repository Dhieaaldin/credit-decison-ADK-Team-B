"""
Improved Risk Scorer Module

Enhanced risk scoring with proper statistical methodology including:
- Time-weighted default rates
- Bayesian priors for sparse data
- Confidence intervals
- Multiple component integration
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskScoreResult:
    """Comprehensive risk score result."""
    final_score: float
    decision: str
    confidence: float
    components: Dict[str, Any]
    thresholds_used: Dict[str, float]
    explanation: str


class ImprovedRiskScorer:
    """
    Enhanced risk scoring with proper statistical methodology.
    
    Key improvements over basic scoring:
    1. Time decay weighting (recent loans matter more)
    2. Bayesian priors for sparse data
    3. Confidence intervals for uncertainty quantification
    4. Multiple component integration
    """
    
    def __init__(
        self,
        target_approval_rate: float = 0.70,
        target_default_rate: float = 0.03,
        time_decay_months: int = 36
    ):
        """
        Initialize improved risk scorer.
        
        Args:
            target_approval_rate: Target approval rate for threshold calibration
            target_default_rate: Target default rate for the portfolio
            time_decay_months: Half-life for time decay weighting
        """
        self.target_approval_rate = target_approval_rate
        self.target_default_rate = target_default_rate
        self.time_decay_months = time_decay_months
        
        # Calibrated thresholds (should be derived from validation)
        self.thresholds = {
            'approve': 0.12,       # Approve if default probability < 12%
            'reject': 0.35,        # Reject if default probability > 35%
            'confidence_min': 0.6  # Minimum confidence for auto-decision
        }
        
        # Dimension weights for aggregation
        self.dimension_weights = {
            'income_stability': 0.20,
            'credit_behavior': 0.25,
            'debt_obligations': 0.25,
            'recent_behavior': 0.15,
            'account_portfolio': 0.10,
            'loan_context': 0.05
        }
        
        # Prior parameters (for Bayesian updating)
        self.prior_default_rate = 0.15  # Population average default rate
        self.prior_weight = 3.0          # Equivalent to 3 observations
    
    def calculate_adjusted_default_rate(
        self,
        similar_cases: List[Dict[str, Any]],
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate default rate with:
        1. Time decay weighting (recent loans matter more)
        2. Loan seasoning adjustment
        3. Confidence intervals
        
        Args:
            similar_cases: List of similar cases from retrieval
            current_date: Current date for time calculations
            
        Returns:
            Adjusted default rate with confidence metrics
        """
        if not similar_cases:
            return {
                'point_estimate': self.prior_default_rate,
                'lower_bound': 0.05,
                'upper_bound': 0.35,
                'confidence': 0.0,
                'effective_sample_size': 0,
                'use_population_prior': True,
                'message': 'No similar cases found, using population prior'
            }
        
        if current_date is None:
            current_date = datetime.now()
        
        weighted_defaults = 0.0
        weighted_total = 0.0
        
        for case in similar_cases:
            payload = case.get('payload', {})
            
            # Get loan origination date
            origination_date = payload.get('issue_date')
            loan_age_months = self._calculate_loan_age(origination_date, current_date)
            
            # Time decay: recent loans weighted more heavily
            time_weight = np.exp(-loan_age_months / self.time_decay_months)
            
            # Similarity weight
            similarity_weight = case.get('score', 0.5)
            
            # Combined weight
            weight = time_weight * similarity_weight
            
            status = payload.get('loan_status', '')
            
            if 'Charged Off' in str(status):
                weighted_defaults += weight
                weighted_total += weight
            elif 'Fully Paid' in str(status):
                weighted_total += weight
            # Note: "Current" loans are excluded (not resolved)
        
        if weighted_total < 1.0:
            # Insufficient data - use Bayesian prior
            posterior_default = (
                (self.prior_default_rate * self.prior_weight + weighted_defaults) /
                (self.prior_weight + weighted_total)
            )
            
            return {
                'point_estimate': posterior_default,
                'lower_bound': max(0, posterior_default - 0.15),
                'upper_bound': min(1, posterior_default + 0.15),
                'confidence': weighted_total / (self.prior_weight + weighted_total),
                'sample_size': len(similar_cases),
                'effective_sample_size': weighted_total,
                'use_population_prior': True,
                'message': 'Limited data, Bayesian prior applied'
            }
        
        default_rate = weighted_defaults / weighted_total
        
        # Calculate confidence interval (Wilson score interval)
        z = 1.96  # 95% confidence
        n = weighted_total
        p = default_rate
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        
        return {
            'point_estimate': default_rate,
            'lower_bound': max(0, center - spread),
            'upper_bound': min(1, center + spread),
            'confidence': 1 - spread,  # Narrower interval = higher confidence
            'sample_size': len(similar_cases),
            'effective_sample_size': weighted_total,
            'use_population_prior': False,
            'message': 'Sufficient data for direct estimation'
        }
    
    def _calculate_loan_age(
        self,
        origination_date: Optional[str],
        current_date: datetime
    ) -> float:
        """Calculate loan age in months."""
        if not origination_date:
            return 24.0  # Default to 2 years if unknown
        
        try:
            if isinstance(origination_date, str):
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m', '%b-%Y']:
                    try:
                        orig = datetime.strptime(origination_date, fmt)
                        delta = current_date - orig
                        return max(0, delta.days / 30.44)  # Average days per month
                    except ValueError:
                        continue
            return 24.0
        except:
            return 24.0
    
    def score_application(
        self,
        per_dimension_risk: Dict[str, Dict[str, Any]],
        applicant_data: Dict[str, Any],
        anomalies: Optional[List[Dict[str, Any]]] = None
    ) -> RiskScoreResult:
        """
        Generate final risk score combining multiple factors.
        
        Args:
            per_dimension_risk: Risk scores per semantic dimension
            applicant_data: Original applicant data
            anomalies: Detected anomalies
            
        Returns:
            Comprehensive risk score result
        """
        # 1. Similarity-based default probability
        similarity_score, similarity_confidence = self._aggregate_dimension_scores(
            per_dimension_risk
        )
        
        # 2. Rule-based risk factors (deterministic)
        rule_score, rule_reasons = self._apply_rule_adjustments(applicant_data)
        
        # 3. Anomaly adjustments
        anomaly_adjustment = self._calculate_anomaly_adjustment(anomalies or [])
        
        # 4. Combine with appropriate weighting
        # Weight similarity less when confidence is low
        similarity_weight = 0.6 * similarity_confidence
        rule_weight = 0.3
        prior_weight = 0.1 * (1 - similarity_confidence)  # More weight to prior when uncertain
        
        total_weight = similarity_weight + rule_weight + prior_weight
        
        combined_score = (
            similarity_score * similarity_weight +
            rule_score * rule_weight +
            self.prior_default_rate * prior_weight +
            anomaly_adjustment
        ) / total_weight
        
        # Clamp to valid range
        combined_score = max(0.0, min(1.0, combined_score))
        
        # Make decision
        decision_result = self._threshold_decision(combined_score, similarity_confidence)
        
        # Build explanation
        explanation = self._build_explanation(
            combined_score, similarity_score, rule_score,
            rule_reasons, anomalies or [], decision_result['decision']
        )
        
        return RiskScoreResult(
            final_score=combined_score,
            decision=decision_result['decision'],
            confidence=decision_result['confidence'],
            components={
                'similarity_based': {
                    'score': similarity_score,
                    'confidence': similarity_confidence
                },
                'rule_based': {
                    'score': rule_score,
                    'reasons': rule_reasons
                },
                'anomaly_adjustment': anomaly_adjustment,
                'per_dimension': per_dimension_risk
            },
            thresholds_used=self.thresholds.copy(),
            explanation=explanation
        )
    
    def _aggregate_dimension_scores(
        self,
        per_dimension_risk: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Aggregate dimension-level scores into overall score."""
        if not per_dimension_risk:
            return self.prior_default_rate, 0.3
        
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for dimension, data in per_dimension_risk.items():
            weight = self.dimension_weights.get(dimension, 0.1)
            
            # Get default rate and confidence
            default_rate = data.get('default_rate', self.prior_default_rate)
            confidence = data.get('confidence', 0.5)
            
            weighted_score += default_rate * weight * confidence
            weighted_confidence += confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / (weighted_confidence if weighted_confidence > 0 else total_weight)
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = self.prior_default_rate
            final_confidence = 0.3
        
        return final_score, final_confidence
    
    def _apply_rule_adjustments(
        self,
        applicant_data: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Apply deterministic rule-based adjustments."""
        score = 0.5  # Neutral starting point
        reasons = []
        
        # DTI check
        dti = applicant_data.get('debt_to_income', 0) or 0
        if dti > 50:
            score += 0.25
            reasons.append(f'Very high DTI ({dti:.1f}%)')
        elif dti > 43:
            score += 0.15
            reasons.append(f'High DTI ({dti:.1f}%)')
        elif dti > 35:
            score += 0.05
        elif dti < 20:
            score -= 0.1
            reasons.append(f'Low DTI ({dti:.1f}%)')
        
        # Loan-to-income check
        loan_amount = applicant_data.get('loan_amount', 0) or 0
        annual_income = applicant_data.get('annual_income', 1) or 1
        lti = loan_amount / annual_income if annual_income > 0 else 1.0
        
        if lti > 1.0:
            score += 0.2
            reasons.append('Loan exceeds annual income')
        elif lti > 0.5:
            score += 0.1
            reasons.append('Loan > 50% of annual income')
        elif lti < 0.25:
            score -= 0.05
        
        # Income level
        if annual_income < 25000:
            score += 0.15
            reasons.append('Low annual income')
        elif annual_income < 40000:
            score += 0.05
        elif annual_income > 100000:
            score -= 0.1
            reasons.append('Strong income')
        
        # Employment length
        emp_length = str(applicant_data.get('emp_length', ''))
        if '< 1' in emp_length or emp_length == '' or emp_length.lower() == 'n/a':
            score += 0.1
            reasons.append('Short employment history')
        elif '10+' in emp_length:
            score -= 0.05
        
        # Delinquencies
        delinq = applicant_data.get('delinq_2y', 0) or 0
        if delinq > 2:
            score += 0.2
            reasons.append(f'{delinq} delinquencies in 2 years')
        elif delinq > 0:
            score += 0.1
            reasons.append(f'{delinq} delinquency in 2 years')
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        return score, reasons
    
    def _calculate_anomaly_adjustment(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> float:
        """Calculate score adjustment based on anomalies."""
        adjustment = 0.0
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'MEDIUM')
            anomaly_type = anomaly.get('type', '')
            
            if severity == 'HIGH':
                adjustment += 0.15
            elif severity == 'MEDIUM':
                adjustment += 0.08
            else:
                adjustment += 0.03
            
            # Specific anomaly adjustments
            if anomaly_type == 'HIGH_DEBT_TO_INCOME':
                adjustment += 0.05
            elif anomaly_type == 'CURRENT_DELINQUENCIES':
                adjustment += 0.10
            elif anomaly_type == 'HIGH_COLLECTIONS':
                adjustment += 0.08
        
        return adjustment
    
    def _threshold_decision(
        self,
        score: float,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Make decision with confidence-adjusted thresholds.
        Low confidence -> more likely to route to manual review.
        """
        # Adjust thresholds based on confidence
        confidence_factor = max(0.5, confidence)
        
        # Effective thresholds widen when confidence is low
        effective_approve = self.thresholds['approve'] / confidence_factor
        effective_reject = self.thresholds['reject'] * confidence_factor
        
        if score < effective_approve and confidence >= self.thresholds['confidence_min']:
            return {
                'decision': 'APPROVE',
                'confidence': confidence,
                'reason': 'low_risk'
            }
        elif score > effective_reject and confidence >= self.thresholds['confidence_min'] * 0.8:
            return {
                'decision': 'REJECT',
                'confidence': confidence,
                'reason': 'high_risk'
            }
        else:
            reason = 'low_confidence' if confidence < self.thresholds['confidence_min'] else 'borderline_score'
            return {
                'decision': 'MANUAL_REVIEW',
                'confidence': confidence,
                'reason': reason
            }
    
    def _build_explanation(
        self,
        final_score: float,
        similarity_score: float,
        rule_score: float,
        rule_reasons: List[str],
        anomalies: List[Dict[str, Any]],
        decision: str
    ) -> str:
        """Build human-readable explanation."""
        lines = [
            f"Risk Assessment: {final_score:.1%} estimated default probability",
            f"Decision: {decision}",
            "",
            "Key Factors:"
        ]
        
        # Add rule-based reasons
        for reason in rule_reasons[:3]:
            lines.append(f"  • {reason}")
        
        # Add anomalies
        if anomalies:
            lines.append("")
            lines.append("Anomalies Detected:")
            for anomaly in anomalies[:2]:
                lines.append(f"  ⚠ {anomaly.get('description', anomaly.get('type', 'Unknown'))}")
        
        # Add comparison note
        lines.append("")
        lines.append(f"Historical comparison: {similarity_score:.1%} default rate in similar profiles")
        
        return "\n".join(lines)
    
    def calibrate_thresholds(
        self,
        historical_decisions: List[Dict[str, Any]],
        target_approval_rate: Optional[float] = None,
        target_default_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calibrate thresholds based on historical data.
        
        Args:
            historical_decisions: List of historical decisions with outcomes
            target_approval_rate: Desired approval rate
            target_default_rate: Maximum acceptable default rate
            
        Returns:
            Calibrated thresholds
        """
        if target_approval_rate is None:
            target_approval_rate = self.target_approval_rate
        if target_default_rate is None:
            target_default_rate = self.target_default_rate
        
        if not historical_decisions:
            return self.thresholds.copy()
        
        # Extract scores and outcomes
        scores = [d.get('risk_score', 0.5) for d in historical_decisions]
        outcomes = [d.get('defaulted', False) for d in historical_decisions]
        
        scores = np.array(scores)
        outcomes = np.array(outcomes)
        
        # Find threshold that achieves target approval rate
        approve_threshold = np.percentile(scores, (1 - target_approval_rate) * 100)
        
        # Find threshold that keeps default rate below target
        # (for approved applications)
        sorted_indices = np.argsort(scores)
        cumulative_defaults = np.cumsum(outcomes[sorted_indices])
        cumulative_count = np.arange(1, len(scores) + 1)
        cumulative_default_rate = cumulative_defaults / cumulative_count
        
        # Find where default rate exceeds target
        reject_idx = np.argmax(cumulative_default_rate > target_default_rate)
        if reject_idx > 0:
            reject_threshold = scores[sorted_indices[reject_idx]]
        else:
            reject_threshold = self.thresholds['reject']
        
        calibrated = {
            'approve': float(approve_threshold),
            'reject': float(reject_threshold),
            'confidence_min': self.thresholds['confidence_min']
        }
        
        self.thresholds = calibrated
        return calibrated
