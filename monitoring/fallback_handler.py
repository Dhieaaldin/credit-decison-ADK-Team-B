"""
Fallback Handler Module

Handles pipeline failures gracefully to ensure credit decisions
can still be made when components fail.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass


@dataclass
class FallbackResult:
    """Result of a fallback decision."""
    success: bool
    fallback_mode: str
    decision: str
    confidence: float
    risk_score: Optional[float]
    explanation: str
    priority: str
    requires_manual_review: bool


class FallbackHandler:
    """
    Handles pipeline failures gracefully.
    Ensures credit decisions can still be made when components fail.
    
    Fallback strategy hierarchy:
    1. Use partial results if available
    2. Use rule-based scoring as fallback
    3. Route to manual review as last resort
    """
    
    # Rule-based thresholds for fallback scoring
    FALLBACK_THRESHOLDS = {
        'dti_high': 43.0,         # DTI above 43% is high risk
        'dti_very_high': 50.0,    # DTI above 50% is very high risk
        'lti_high': 0.5,          # Loan-to-income ratio above 50%
        'income_low': 30000,      # Annual income below $30K
        'income_very_low': 20000, # Annual income below $20K
        'loan_high': 50000,       # Loan amount above $50K
    }
    
    def __init__(self):
        """Initialize fallback handler."""
        self.fallback_count = 0
    
    def handle_pipeline_failure(
        self,
        failed_stage: str,
        application_data: Dict[str, Any],
        partial_results: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> FallbackResult:
        """
        Handle failure at any pipeline stage.
        
        Args:
            failed_stage: Name of the stage that failed
            application_data: Original application data
            partial_results: Any partial results from earlier stages
            error_message: Error message from the failure
            
        Returns:
            FallbackResult with decision and explanation
        """
        self.fallback_count += 1
        
        # Log the failure (in production, send to monitoring)
        self._log_failure(failed_stage, application_data, error_message)
        
        # Stage-specific fallback strategies
        fallback_strategies = {
            'ingestion': self._fallback_validation_error,
            'chunking': self._fallback_no_chunks,
            'retrieval': self._fallback_no_similar_cases,
            'risk_analysis': self._fallback_rule_based_risk,
            'decision': self._fallback_manual_review,
            'explanation': self._fallback_template_explanation
        }
        
        strategy = fallback_strategies.get(failed_stage, self._fallback_manual_review)
        return strategy(application_data, partial_results or {})
    
    def _fallback_validation_error(
        self,
        application_data: Dict[str, Any],
        partial_results: Dict[str, Any]
    ) -> FallbackResult:
        """Fallback when validation fails."""
        return FallbackResult(
            success=True,
            fallback_mode='validation_error',
            decision='MANUAL_REVIEW',
            confidence=0.0,
            risk_score=None,
            explanation=(
                'Application data validation failed. '
                'Required fields may be missing or invalid. '
                'Manual review needed to verify application completeness.'
            ),
            priority='HIGH',
            requires_manual_review=True
        )
    
    def _fallback_no_chunks(
        self,
        application_data: Dict[str, Any],
        partial_results: Dict[str, Any]
    ) -> FallbackResult:
        """Fallback when chunking fails."""
        # Use basic rule-based assessment
        score, reasons = self._quick_rule_assessment(application_data)
        
        return FallbackResult(
            success=True,
            fallback_mode='no_chunks',
            decision='MANUAL_REVIEW',
            confidence=0.3,
            risk_score=score,
            explanation=(
                'Unable to segment application into risk dimensions. '
                f'Basic assessment: {"; ".join(reasons)}. '
                'Full analysis requires manual review.'
            ),
            priority='MEDIUM',
            requires_manual_review=True
        )
    
    def _fallback_no_similar_cases(
        self,
        application_data: Dict[str, Any],
        partial_results: Dict[str, Any]
    ) -> FallbackResult:
        """
        Fallback when vector search fails.
        Use rule-based scoring only.
        """
        score, reasons = self._quick_rule_assessment(application_data)
        
        return FallbackResult(
            success=True,
            fallback_mode='rule_based_only',
            decision='MANUAL_REVIEW',
            confidence=0.4,
            risk_score=score,
            explanation=(
                'Unable to find similar historical cases for comparison. '
                f'Rule-based pre-screening: {"; ".join(reasons)}. '
                'Recommend manual review for comprehensive assessment.'
            ),
            priority='MEDIUM',
            requires_manual_review=True
        )
    
    def _fallback_rule_based_risk(
        self,
        application_data: Dict[str, Any],
        partial_results: Dict[str, Any]
    ) -> FallbackResult:
        """
        Fallback when risk analysis fails.
        Use simple DTI and credit-based rules.
        """
        score, reasons = self._quick_rule_assessment(application_data)
        
        # Determine if any clear decision can be made
        if score > 0.7:
            decision = 'REJECT'
            confidence = 0.5
            priority = 'HIGH'
        elif score < 0.3:
            decision = 'MANUAL_REVIEW'  # Still need human verification in fallback
            confidence = 0.5
            priority = 'LOW'
        else:
            decision = 'MANUAL_REVIEW'
            confidence = 0.4
            priority = 'MEDIUM'
        
        return FallbackResult(
            success=True,
            fallback_mode='rule_based_risk',
            decision=decision,
            confidence=confidence,
            risk_score=score,
            explanation=(
                'Risk analysis using fallback rules. '
                f'{"; ".join(reasons)}. '
                'Automated similarity analysis unavailable.'
            ),
            priority=priority,
            requires_manual_review=True
        )
    
    def _fallback_manual_review(
        self,
        application_data: Dict[str, Any],
        partial_results: Dict[str, Any]
    ) -> FallbackResult:
        """
        Ultimate fallback: route everything to manual review.
        """
        # Get basic scores if available
        risk_score = partial_results.get('risk_score')
        
        return FallbackResult(
            success=True,
            fallback_mode='full_manual',
            decision='MANUAL_REVIEW',
            confidence=0.0,
            risk_score=risk_score,
            explanation=(
                'Automated analysis unavailable due to system error. '
                'Application requires full manual underwriting review.'
            ),
            priority='HIGH',
            requires_manual_review=True
        )
    
    def _fallback_template_explanation(
        self,
        application_data: Dict[str, Any],
        partial_results: Dict[str, Any]
    ) -> FallbackResult:
        """Fallback when explanation generation fails."""
        # Use the decision from partial results if available
        decision = partial_results.get('decision', 'MANUAL_REVIEW')
        risk_score = partial_results.get('risk_score')
        confidence = partial_results.get('confidence', 0.5)
        
        # Generate basic template explanation
        if decision == 'APPROVE':
            explanation = (
                f'Application approved with risk score of {risk_score:.2%}. '
                'The applicant profile is consistent with historically successful loans.'
            )
        elif decision == 'REJECT':
            explanation = (
                f'Application declined with risk score of {risk_score:.2%}. '
                'Risk factors identified require attention.'
            )
        else:
            explanation = (
                f'Application requires manual review. Risk score: {risk_score:.2%}. '
                'Mixed signals require human judgment.'
            )
        
        return FallbackResult(
            success=True,
            fallback_mode='template_explanation',
            decision=decision,
            confidence=confidence,
            risk_score=risk_score,
            explanation=explanation,
            priority='MEDIUM' if decision == 'MANUAL_REVIEW' else 'LOW',
            requires_manual_review=(decision == 'MANUAL_REVIEW')
        )
    
    def _quick_rule_assessment(
        self,
        application_data: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """
        Quick rule-based risk assessment.
        
        Returns:
            Tuple of (risk_score, list_of_reasons)
        """
        score = 0.5  # Neutral starting point
        reasons = []
        
        # Get key metrics
        dti = application_data.get('debt_to_income', 0) or 0
        annual_income = application_data.get('annual_income', 0) or 0
        loan_amount = application_data.get('loan_amount', 0) or 0
        
        # DTI check
        if dti > self.FALLBACK_THRESHOLDS['dti_very_high']:
            score += 0.25
            reasons.append(f'Very high debt-to-income ratio ({dti:.1f}%)')
        elif dti > self.FALLBACK_THRESHOLDS['dti_high']:
            score += 0.15
            reasons.append(f'High debt-to-income ratio ({dti:.1f}%)')
        elif dti < 25:
            score -= 0.1
            reasons.append(f'Manageable debt-to-income ratio ({dti:.1f}%)')
        
        # Loan-to-income check
        if annual_income > 0:
            lti = loan_amount / annual_income
            if lti > 1.0:
                score += 0.2
                reasons.append('Loan amount exceeds annual income')
            elif lti > self.FALLBACK_THRESHOLDS['lti_high']:
                score += 0.1
                reasons.append('Loan amount is significant relative to income')
            elif lti < 0.25:
                score -= 0.1
                reasons.append('Conservative loan amount relative to income')
        
        # Income level check
        if annual_income < self.FALLBACK_THRESHOLDS['income_very_low']:
            score += 0.15
            reasons.append('Very low annual income')
        elif annual_income < self.FALLBACK_THRESHOLDS['income_low']:
            score += 0.1
            reasons.append('Low annual income')
        elif annual_income > 100000:
            score -= 0.1
            reasons.append('Strong income level')
        
        # Loan amount check
        if loan_amount > self.FALLBACK_THRESHOLDS['loan_high']:
            score += 0.05
            reasons.append('Large loan amount requested')
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        if not reasons:
            reasons.append('Standard risk profile')
        
        return score, reasons
    
    def _log_failure(
        self,
        failed_stage: str,
        application_data: Dict[str, Any],
        error_message: Optional[str]
    ):
        """Log pipeline failure for monitoring."""
        # In production, this would send to centralized logging
        timestamp = datetime.now().isoformat()
        application_id = application_data.get('id', 'unknown')
        
        print(f"[FALLBACK] {timestamp} | Stage: {failed_stage} | "
              f"App: {application_id} | Error: {error_message or 'Unknown'}")
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get statistics about fallback usage."""
        return {
            'total_fallbacks': self.fallback_count,
            'message': 'Fallback handler active'
        }
