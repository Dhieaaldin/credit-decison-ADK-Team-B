"""
Semantic Chunking Agent

Decomposes loan applications into semantic chunks using FastEmbed.
No PyTorch/Torch dependencies - pure ONNX-based embeddings.
"""

from typing import Dict, Any, Optional
import time
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from embeddings import EmbeddingModel


class ChunkingAgent(BaseAgent):
    """
    Agent responsible for semantic chunking of loan applications.
    
    Uses FastEmbed for embedding-based semantic segmentation.
    Input: Cleaned loan application data
    Output: Semantic chunks (6 dimensions)
    """
    
    def __init__(self, use_chonkie: bool = True, embedding_model_name: Optional[str] = None):
        """
        Initialize chunking agent.
        
        Args:
            use_chonkie: Ignored (FastEmbed is used instead)
            embedding_model_name: FastEmbed model name (default: BAAI/bge-small-en-v1.5)
        """
        super().__init__("chunking_agent", "Semantic Chunking")
        
        # Initialize FastEmbed embedding model
        try:
            self.embedding_model = EmbeddingModel(embedding_model_name)
            print("✓ ChunkingAgent initialized with FastEmbed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {e}")
    
    def _create_chunks(self, cleaned_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create semantic chunks from loan application data.
        
        Args:
            cleaned_data: Cleaned loan application
            
        Returns:
            Dictionary of chunk_type -> chunk_text
        """
        return {
            'income_stability': (
                # NOTE: emp_title removed for fair lending compliance (ECOA)
                # Job titles can serve as proxies for protected characteristics
                f"Employment length: {self._categorize_emp_length(cleaned_data.get('emp_length', 'Unknown'))}, "
                f"Income: ${cleaned_data.get('annual_income', 0):,.0f}, "
                f"Verified: {cleaned_data.get('verified_income', 'Unknown')}, "
                f"Income stability score: {self._calculate_income_stability(cleaned_data)}"
            ),
            'credit_behavior': (
                f"Credit history since {cleaned_data.get('earliest_credit_line', 'Unknown')}. "
                f"Total lines: {cleaned_data.get('total_credit_lines', 0)}, "
                f"Open: {cleaned_data.get('open_credit_lines', 0)}, "
                f"Limit: ${cleaned_data.get('total_credit_limit', 0):,.0f}, "
                f"Utilized: ${cleaned_data.get('total_credit_utilized', 0):,.0f}, "
                f"Delinquencies (2y): {cleaned_data.get('delinq_2y', 0)}, "
                f"Inquiries (12m): {cleaned_data.get('inquiries_last_12m', 0)}"
            ),
            'debt_obligations': (
                f"Debt-to-income: {cleaned_data.get('debt_to_income', 0):.1f}%. "
                f"Failed payments: {cleaned_data.get('num_historical_failed_to_pay', 0)}, "
                f"Months since 90d late: {cleaned_data.get('months_since_90d_late', 'N/A')}, "
                f"Current delinquent accounts: {cleaned_data.get('current_accounts_delinq', 0)}, "
                f"Total collections: ${cleaned_data.get('total_collection_amount_ever', 0):,.0f}"
            ),
            'recent_behavior': (
                f"Accounts opened (24m): {cleaned_data.get('accounts_opened_24m', 0)}, "
                f"Collections (12m): {cleaned_data.get('num_collections_last_12m', 0)}, "
                f"Months since last inquiry: {cleaned_data.get('months_since_last_credit_inquiry', 'N/A')}, "
                f"Accounts 30d past due: {cleaned_data.get('num_accounts_30d_past_due', 0)}, "
                f"Accounts 120d past due: {cleaned_data.get('num_accounts_120d_past_due', 0)}"
            ),
            'account_portfolio': (
                f"Total CC accounts: {cleaned_data.get('num_total_cc_accounts', 0)}, "
                f"Open CC: {cleaned_data.get('num_open_cc_accounts', 0)}, "
                f"CC with balance: {cleaned_data.get('num_cc_carrying_balance', 0)}, "
                f"Mortgage accounts: {cleaned_data.get('num_mort_accounts', 0)}, "
                f"Installment accounts: {cleaned_data.get('current_installment_accounts', 0)}"
            ),
            'loan_context': (
                # NOTE: State removed for fair lending compliance (ECOA/Fair Housing Act)
                # Geographic features can serve as proxies for race/ethnicity
                f"Loan purpose: {cleaned_data.get('loan_purpose', 'Unknown')}, "
                f"Type: {cleaned_data.get('application_type', 'Individual')}, "
                f"Amount: ${cleaned_data.get('loan_amount', 0):,.0f}, "
                f"Term: {cleaned_data.get('term', 'Unknown')} months, "
                f"Homeownership: {cleaned_data.get('homeownership', 'Unknown')}, "
                f"Loan-to-income ratio: {self._calculate_lti_ratio(cleaned_data):.2f}"
            )
        }
    
    def _categorize_emp_length(self, emp_length: str) -> str:
        """
        Categorize employment length into fair lending compliant buckets.
        Uses broad categories instead of specific values to reduce discrimination risk.
        """
        if not emp_length or emp_length == 'Unknown' or emp_length == 'n/a':
            return 'Unknown'
        
        emp_str = str(emp_length).lower().strip()
        
        if '10+' in emp_str or '10 year' in emp_str:
            return 'Long-term (10+ years)'
        elif any(x in emp_str for x in ['5 year', '6 year', '7 year', '8 year', '9 year']):
            return 'Established (5-9 years)'
        elif any(x in emp_str for x in ['2 year', '3 year', '4 year']):
            return 'Moderate (2-4 years)'
        elif any(x in emp_str for x in ['1 year', '< 1']):
            return 'Recent (< 2 years)'
        else:
            return 'Unknown'
    
    def _calculate_income_stability(self, cleaned_data: Dict[str, Any]) -> str:
        """
        Calculate income stability score based on permissible factors only.
        Does not use job title or industry to avoid discrimination.
        """
        score = 0
        
        # Employment length factor
        emp_length = str(cleaned_data.get('emp_length', '')).lower()
        if '10+' in emp_length:
            score += 3
        elif any(x in emp_length for x in ['5', '6', '7', '8', '9']):
            score += 2
        elif any(x in emp_length for x in ['2', '3', '4']):
            score += 1
        
        # Income verification
        verified = cleaned_data.get('verified_income', '').lower()
        if verified == 'verified':
            score += 2
        elif verified == 'source_verified':
            score += 1
        
        # Income level relative to loan
        income = cleaned_data.get('annual_income', 0) or 0
        loan = cleaned_data.get('loan_amount', 0) or 0
        
        if income > 0 and loan > 0:
            lti = loan / income
            if lti < 0.25:
                score += 2
            elif lti < 0.5:
                score += 1
        
        # Convert to category
        if score >= 6:
            return 'High'
        elif score >= 4:
            return 'Moderate'
        elif score >= 2:
            return 'Low'
        else:
            return 'Insufficient Data'
    
    def _calculate_lti_ratio(self, cleaned_data: Dict[str, Any]) -> float:
        """Calculate loan-to-income ratio."""
        income = cleaned_data.get('annual_income', 0) or 0
        loan = cleaned_data.get('loan_amount', 0) or 0
        
        if income > 0:
            return loan / income
        return 0.0
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process cleaned data and create semantic chunks.
        
        Args:
            message: Contains cleaned loan application data
            
        Returns:
            AgentResponse with semantic chunks
        """
        start_time = time.time()
        
        try:
            cleaned_data = message.payload.get("cleaned_data", {})
            
            if not cleaned_data:
                return AgentResponse(
                    success=False,
                    error="No cleaned data provided",
                    processing_time=time.time() - start_time
                )
            
            # Create semantic chunks
            chunks = self._create_chunks(cleaned_data)
            
            # Log processing
            self.log_message(message)
            self.update_state({
                "last_processed": time.time(),
                "total_chunked": self.state.get("total_chunked", 0) + 1
            })
            
            return AgentResponse(
                success=True,
                data={
                    "chunks": chunks,
                    "chunk_types": list(chunks.keys()),
                    "original_data": cleaned_data
                },
                metadata={
                    "num_chunks": len(chunks),
                    "chunk_types": list(chunks.keys())
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Chunking error: {str(e)}",
                processing_time=time.time() - start_time
            )
