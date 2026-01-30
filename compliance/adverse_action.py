"""
Adverse Action Notice Generator

Generates ECOA-compliant adverse action notices per:
- Regulation B §1002.9
- CFPB Model Forms B-1 and B-2

When taking adverse action on a credit application, lenders must:
1. Provide specific reasons for the action
2. Reasons must relate to the actual credit decision
3. Must be disclosed in writing within 30 days
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AdverseActionReason:
    """A specific reason for adverse action."""
    code: str
    reason: str
    score: float
    detail: str


class AdverseActionGenerator:
    """
    Generate ECOA-compliant adverse action notices.
    
    Per Regulation B §1002.9, when taking adverse action:
    1. Must provide specific reasons for the action
    2. Reasons must relate to the credit decision
    3. Must be disclosed in writing within 30 days
    
    The CFPB Sample Forms provide templates, but reasons must be
    derived from the actual factors that influenced the decision.
    """
    
    # Standard reason codes per CFPB Model Form B-1
    REASON_CODES = {
        'DTI_HIGH': 'Excessive obligations in relation to income',
        'INCOME_LOW': 'Unable to verify income',
        'INCOME_INSUFFICIENT': 'Income insufficient for amount of credit requested',
        'EMPLOYMENT_SHORT': 'Length of employment',
        'EMPLOYMENT_UNSTABLE': 'Temporary or irregular employment',
        'EMPLOYMENT_TYPE': 'Type of employment',
        'CREDIT_HISTORY_SHORT': 'Insufficient credit history',
        'CREDIT_HISTORY_NONE': 'No credit file',
        'CREDIT_HISTORY_DELINQ': 'Delinquent past or present credit obligations',
        'CREDIT_HISTORY_SERIOUS_DELINQ': 'Serious delinquency, derogatory public record, or collection',
        'CREDIT_HISTORY_COLLECTIONS': 'Collection action or judgment',
        'CREDIT_HISTORY_BANKRUPTCY': 'Bankruptcy',
        'CREDIT_RATIO_HIGH': 'Proportion of balances to credit limits too high',
        'CREDIT_ACCOUNTS_TOO_FEW': 'Too few accounts currently paid as agreed',
        'CREDIT_INQUIRIES': 'Too many recent inquiries on credit report',
        'CREDIT_RECENT_ACCOUNTS': 'Too many accounts recently opened',
        'LOAN_AMOUNT_HIGH': 'Amount of loan requested exceeds guidelines',
        'LOAN_COLLATERAL': 'Value or type of collateral not sufficient',
        'LOAN_PURPOSE': 'Unacceptable loan purpose',
        'RESIDENCE_UNSTABLE': 'Length of residence',
        'AGE_OF_ACCOUNTS': 'Lack of recently active credit accounts',
        'GARNISHMENT': 'Garnishment, attachment, foreclosure, repossession, or suit',
    }
    
    # Mapping from risk analysis dimensions to reason codes
    DIMENSION_TO_REASONS = {
        'income_stability': ['INCOME_INSUFFICIENT', 'EMPLOYMENT_SHORT', 'EMPLOYMENT_UNSTABLE'],
        'debt_obligations': ['DTI_HIGH', 'CREDIT_HISTORY_DELINQ'],
        'credit_behavior': ['CREDIT_HISTORY_DELINQ', 'CREDIT_RATIO_HIGH', 'CREDIT_INQUIRIES'],
        'recent_behavior': ['CREDIT_RECENT_ACCOUNTS', 'CREDIT_INQUIRIES'],
        'account_portfolio': ['CREDIT_HISTORY_SHORT', 'CREDIT_ACCOUNTS_TOO_FEW'],
        'loan_context': ['LOAN_AMOUNT_HIGH', 'LOAN_PURPOSE']
    }
    
    # Mapping from anomaly types to reason codes
    ANOMALY_TO_REASONS = {
        'HIGH_DEBT_TO_INCOME': 'DTI_HIGH',
        'HIGH_COLLECTIONS': 'CREDIT_HISTORY_COLLECTIONS',
        'CURRENT_DELINQUENCIES': 'CREDIT_HISTORY_DELINQ',
        'LOW_SIMILARITY': None,  # Not a specific adverse reason
        'CONFLICTING_SIGNALS': None  # Not a specific adverse reason
    }
    
    def __init__(self, lender_name: str = "Regional Bank", lender_address: str = ""):
        """
        Initialize adverse action generator.
        
        Args:
            lender_name: Name of the lending institution
            lender_address: Address of the lending institution
        """
        self.lender_name = lender_name
        self.lender_address = lender_address
    
    def generate_adverse_action_reasons(
        self,
        decision_data: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        applicant_data: Dict[str, Any],
        max_reasons: int = 4
    ) -> List[AdverseActionReason]:
        """
        Generate principal reasons for adverse action.
        
        Per ECOA, must provide up to 4 principal reasons.
        Reasons must be specific and based on actual factors.
        
        Args:
            decision_data: Decision output from decision agent
            risk_analysis: Risk analysis output from risk agent
            applicant_data: Original application data
            max_reasons: Maximum number of reasons to return (default: 4 per ECOA)
            
        Returns:
            List of AdverseActionReason objects
        """
        reason_scores: List[AdverseActionReason] = []
        
        per_dim = risk_analysis.get('per_dimension_risk', {})
        anomalies = risk_analysis.get('anomalies', [])
        
        # Analyze debt obligations dimension
        reason_scores.extend(
            self._analyze_debt_obligations(per_dim, applicant_data)
        )
        
        # Analyze income stability dimension
        reason_scores.extend(
            self._analyze_income_stability(per_dim, applicant_data)
        )
        
        # Analyze credit behavior dimension
        reason_scores.extend(
            self._analyze_credit_behavior(per_dim, applicant_data)
        )
        
        # Analyze recent behavior dimension
        reason_scores.extend(
            self._analyze_recent_behavior(per_dim, applicant_data)
        )
        
        # Analyze anomalies
        reason_scores.extend(
            self._analyze_anomalies(anomalies, applicant_data)
        )
        
        # Analyze loan context
        reason_scores.extend(
            self._analyze_loan_context(per_dim, applicant_data)
        )
        
        # Sort by impact score and deduplicate
        reason_scores.sort(key=lambda x: x.score, reverse=True)
        
        seen_codes = set()
        unique_reasons = []
        for reason in reason_scores:
            if reason.code not in seen_codes:
                seen_codes.add(reason.code)
                unique_reasons.append(reason)
        
        return unique_reasons[:max_reasons]
    
    def _analyze_debt_obligations(
        self,
        per_dim: Dict[str, Any],
        applicant_data: Dict[str, Any]
    ) -> List[AdverseActionReason]:
        """Analyze debt obligations dimension for adverse reasons."""
        reasons = []
        
        dim = per_dim.get('debt_obligations', {})
        if dim.get('risk_level') == 'HIGH':
            dti = applicant_data.get('debt_to_income', 0)
            if dti > 40:
                reasons.append(AdverseActionReason(
                    code='DTI_HIGH',
                    reason=self.REASON_CODES['DTI_HIGH'],
                    score=0.9,
                    detail=f'Debt-to-income ratio of {dti:.1f}% exceeds guidelines'
                ))
            elif dti > 30:
                reasons.append(AdverseActionReason(
                    code='DTI_HIGH',
                    reason=self.REASON_CODES['DTI_HIGH'],
                    score=0.7,
                    detail=f'Debt-to-income ratio of {dti:.1f}% is above preferred range'
                ))
        
        return reasons
    
    def _analyze_income_stability(
        self,
        per_dim: Dict[str, Any],
        applicant_data: Dict[str, Any]
    ) -> List[AdverseActionReason]:
        """Analyze income stability dimension for adverse reasons."""
        reasons = []
        
        dim = per_dim.get('income_stability', {})
        if dim.get('risk_level') in ['HIGH', 'MEDIUM']:
            income = applicant_data.get('annual_income', 0)
            loan_amount = applicant_data.get('loan_amount', 0)
            emp_length = str(applicant_data.get('emp_length', ''))
            
            # Check income sufficiency
            if income > 0 and loan_amount / income > 0.5:
                reasons.append(AdverseActionReason(
                    code='INCOME_INSUFFICIENT',
                    reason=self.REASON_CODES['INCOME_INSUFFICIENT'],
                    score=0.85,
                    detail='Annual income insufficient for requested loan amount'
                ))
            
            # Check employment length
            if emp_length:
                emp_lower = emp_length.lower()
                if '< 1' in emp_lower or 'month' in emp_lower:
                    reasons.append(AdverseActionReason(
                        code='EMPLOYMENT_SHORT',
                        reason=self.REASON_CODES['EMPLOYMENT_SHORT'],
                        score=0.7,
                        detail='Limited employment history at current position'
                    ))
                elif '1 year' in emp_lower or '2 year' in emp_lower:
                    reasons.append(AdverseActionReason(
                        code='EMPLOYMENT_SHORT',
                        reason=self.REASON_CODES['EMPLOYMENT_SHORT'],
                        score=0.5,
                        detail='Short employment history at current position'
                    ))
        
        return reasons
    
    def _analyze_credit_behavior(
        self,
        per_dim: Dict[str, Any],
        applicant_data: Dict[str, Any]
    ) -> List[AdverseActionReason]:
        """Analyze credit behavior dimension for adverse reasons."""
        reasons = []
        
        dim = per_dim.get('credit_behavior', {})
        if dim.get('risk_level') in ['HIGH', 'MEDIUM']:
            delinq = applicant_data.get('delinq_2y', 0)
            inquiries = applicant_data.get('inquiries_last_12m', 0)
            credit_util = applicant_data.get('total_credit_utilized', 0)
            credit_limit = applicant_data.get('total_credit_limit', 1)
            
            # Delinquencies
            if delinq and delinq > 0:
                reasons.append(AdverseActionReason(
                    code='CREDIT_HISTORY_DELINQ',
                    reason=self.REASON_CODES['CREDIT_HISTORY_DELINQ'],
                    score=0.95,
                    detail=f'{delinq} delinquencies in past 2 years'
                ))
            
            # Credit inquiries
            if inquiries and inquiries > 6:
                reasons.append(AdverseActionReason(
                    code='CREDIT_INQUIRIES',
                    reason=self.REASON_CODES['CREDIT_INQUIRIES'],
                    score=0.6,
                    detail=f'{inquiries} credit inquiries in past 12 months'
                ))
            elif inquiries and inquiries > 3:
                reasons.append(AdverseActionReason(
                    code='CREDIT_INQUIRIES',
                    reason=self.REASON_CODES['CREDIT_INQUIRIES'],
                    score=0.4,
                    detail=f'{inquiries} credit inquiries in past 12 months'
                ))
            
            # Credit utilization
            if credit_limit > 0:
                util_ratio = credit_util / credit_limit
                if util_ratio > 0.7:
                    reasons.append(AdverseActionReason(
                        code='CREDIT_RATIO_HIGH',
                        reason=self.REASON_CODES['CREDIT_RATIO_HIGH'],
                        score=0.75,
                        detail=f'Credit utilization of {util_ratio:.0%} is above recommended levels'
                    ))
        
        return reasons
    
    def _analyze_recent_behavior(
        self,
        per_dim: Dict[str, Any],
        applicant_data: Dict[str, Any]
    ) -> List[AdverseActionReason]:
        """Analyze recent behavior dimension for adverse reasons."""
        reasons = []
        
        dim = per_dim.get('recent_behavior', {})
        if dim.get('risk_level') in ['HIGH', 'MEDIUM']:
            accounts_opened = applicant_data.get('accounts_opened_24m', 0)
            collections = applicant_data.get('num_collections_last_12m', 0)
            past_due_30 = applicant_data.get('num_accounts_30d_past_due', 0)
            
            if accounts_opened and accounts_opened > 5:
                reasons.append(AdverseActionReason(
                    code='CREDIT_RECENT_ACCOUNTS',
                    reason=self.REASON_CODES['CREDIT_RECENT_ACCOUNTS'],
                    score=0.55,
                    detail=f'{accounts_opened} new accounts opened in past 24 months'
                ))
            
            if collections and collections > 0:
                reasons.append(AdverseActionReason(
                    code='CREDIT_HISTORY_COLLECTIONS',
                    reason=self.REASON_CODES['CREDIT_HISTORY_COLLECTIONS'],
                    score=0.85,
                    detail=f'{collections} collection actions in past 12 months'
                ))
            
            if past_due_30 and past_due_30 > 0:
                reasons.append(AdverseActionReason(
                    code='CREDIT_HISTORY_DELINQ',
                    reason=self.REASON_CODES['CREDIT_HISTORY_DELINQ'],
                    score=0.9,
                    detail=f'{past_due_30} account(s) currently 30+ days past due'
                ))
        
        return reasons
    
    def _analyze_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        applicant_data: Dict[str, Any]
    ) -> List[AdverseActionReason]:
        """Analyze detected anomalies for adverse reasons."""
        reasons = []
        
        for anomaly in anomalies:
            anomaly_type = anomaly.get('type', '')
            reason_code = self.ANOMALY_TO_REASONS.get(anomaly_type)
            
            if reason_code and reason_code in self.REASON_CODES:
                severity = anomaly.get('severity', 'MEDIUM')
                score = 0.9 if severity == 'HIGH' else 0.6
                
                reasons.append(AdverseActionReason(
                    code=reason_code,
                    reason=self.REASON_CODES[reason_code],
                    score=score,
                    detail=anomaly.get('description', '')
                ))
        
        return reasons
    
    def _analyze_loan_context(
        self,
        per_dim: Dict[str, Any],
        applicant_data: Dict[str, Any]
    ) -> List[AdverseActionReason]:
        """Analyze loan context dimension for adverse reasons."""
        reasons = []
        
        dim = per_dim.get('loan_context', {})
        if dim.get('risk_level') in ['HIGH', 'MEDIUM']:
            loan_amount = applicant_data.get('loan_amount', 0)
            income = applicant_data.get('annual_income', 1)
            
            # Loan amount relative to income
            if income > 0 and loan_amount / income > 1.0:
                reasons.append(AdverseActionReason(
                    code='LOAN_AMOUNT_HIGH',
                    reason=self.REASON_CODES['LOAN_AMOUNT_HIGH'],
                    score=0.65,
                    detail='Requested loan amount exceeds annual income'
                ))
        
        return reasons
    
    def format_notice(
        self,
        applicant_name: str,
        reasons: List[AdverseActionReason],
        application_date: datetime,
        decision_date: Optional[datetime] = None,
        application_id: Optional[str] = None
    ) -> str:
        """
        Generate formatted adverse action notice per CFPB Model Form B-1.
        
        Args:
            applicant_name: Name of the applicant
            reasons: List of adverse action reasons
            application_date: Date of the original application
            decision_date: Date of the decision (default: now)
            application_id: Optional application reference number
            
        Returns:
            Formatted adverse action notice text
        """
        if decision_date is None:
            decision_date = datetime.now()
        
        reference = f"Application Reference: {application_id}\n" if application_id else ""
        
        notice = f"""
================================================================================
                        NOTICE OF ADVERSE ACTION
================================================================================

Date: {decision_date.strftime('%B %d, %Y')}
Applicant: {applicant_name}
Application Date: {application_date.strftime('%B %d, %Y')}
{reference}
Lender: {self.lender_name}
{self.lender_address}

--------------------------------------------------------------------------------

Dear {applicant_name},

This notice is to inform you that your recent application for credit has been 
denied. This decision was based on our review of your credit application and 
credit report.

PRINCIPAL REASONS FOR THIS DECISION:

"""
        for i, reason in enumerate(reasons, 1):
            notice += f"  {i}. {reason.reason}\n"
        
        notice += """

--------------------------------------------------------------------------------
                            YOUR RIGHTS
--------------------------------------------------------------------------------

CREDIT REPORT INFORMATION

You have the right to obtain a free copy of your credit report from any 
consumer reporting agency that provided information used in this decision. 
You must request your free report within 60 days of receiving this notice.

Principal consumer reporting agencies:
  • Equifax: 1-800-685-1111, www.equifax.com
  • Experian: 1-888-397-3742, www.experian.com  
  • TransUnion: 1-800-888-4213, www.transunion.com

DISPUTE RIGHTS

You have the right to dispute the accuracy or completeness of any 
information in your credit report. The consumer reporting agency must 
investigate your dispute within 30 days.

EQUAL CREDIT OPPORTUNITY ACT NOTICE

The federal Equal Credit Opportunity Act prohibits creditors from 
discriminating against credit applicants on the basis of race, color, 
religion, national origin, sex, marital status, age (provided the applicant 
has the capacity to enter into a binding contract); because all or part of 
the applicant's income derives from any public assistance program; or 
because the applicant has in good faith exercised any right under the 
Consumer Credit Protection Act. The federal agency that administers 
compliance with this law concerning this creditor is:

  Consumer Financial Protection Bureau
  1700 G Street NW
  Washington, DC 20552
  www.consumerfinance.gov/complaint

--------------------------------------------------------------------------------

If you have questions about this notice, please contact us at:
{self.lender_name}
{self.lender_address}

================================================================================
"""
        return notice
    
    def generate_structured_notice(
        self,
        applicant_name: str,
        reasons: List[AdverseActionReason],
        application_date: datetime,
        decision_date: Optional[datetime] = None,
        application_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured adverse action notice for API/database storage.
        
        Args:
            applicant_name: Name of the applicant
            reasons: List of adverse action reasons
            application_date: Date of the original application
            decision_date: Date of the decision (default: now)
            application_id: Optional application reference number
            
        Returns:
            Structured notice data for storage/transmission
        """
        if decision_date is None:
            decision_date = datetime.now()
        
        return {
            'notice_type': 'ADVERSE_ACTION',
            'application_id': application_id,
            'applicant_name': applicant_name,
            'application_date': application_date.isoformat(),
            'decision_date': decision_date.isoformat(),
            'lender_name': self.lender_name,
            'lender_address': self.lender_address,
            'reasons': [
                {
                    'code': r.code,
                    'reason': r.reason,
                    'detail': r.detail
                }
                for r in reasons
            ],
            'reason_count': len(reasons),
            'formatted_notice': self.format_notice(
                applicant_name, reasons, application_date, decision_date, application_id
            )
        }
