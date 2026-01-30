"""
Multi-Agent Pipeline Orchestrator

Orchestrates the flow of loan applications through the multi-agent system.
Enhanced with monitoring, audit logging, and fallback handling.
"""

from typing import Dict, List, Any, Optional
import time
import uuid
from agents.base_agent import AgentMessage, AgentResponse
from agents.ingestion_agent import IngestionAgent
from agents.chunking_agent import ChunkingAgent
from agents.retrieval_agent import RetrievalAgent
from agents.risk_agent import RiskAgent
from agents.decision_agent import DecisionAgent
from agents.explanation_agent import ExplanationAgent


class MultiAgentPipeline:
    """
    Orchestrates the multi-agent loan screening pipeline.
    
    Enhanced with:
    - Monitoring and audit logging
    - Fallback handling for failures
    - Latency tracking
    - Application ID tracking
    
    Flow:
    1. IngestionAgent -> Cleaned Data
    2. ChunkingAgent -> Semantic Chunks
    3. RetrievalAgent -> Similar Cases
    4. RiskAgent -> Risk Scores
    5. DecisionAgent -> Decision
    6. ExplanationAgent -> Explanation
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        use_google_cloud: bool = False,
        use_google_ai: bool = False,
        enable_monitoring: bool = True,
        enable_audit_logging: bool = True
    ):
        """
        Initialize the multi-agent pipeline.
        
        Args:
            project_id: Google Cloud project ID (required if use_google_cloud=True)
            location: GCP region
            use_google_cloud: Use Google Vertex AI Vector Search
            use_google_ai: Use Google Gemini for explanations
            enable_monitoring: Enable model monitoring
            enable_audit_logging: Enable audit logging
        """
        # Initialize agents
        self.ingestion_agent = IngestionAgent()
        self.chunking_agent = ChunkingAgent(use_chonkie=True)
        
        # Retrieval agent (Qdrant primary, Google Cloud optional)
        self.retrieval_agent = RetrievalAgent(
            project_id=project_id if use_google_cloud else None,
            location=location,
            use_google_cloud=use_google_cloud,
            use_local_embeddings=True,
            qdrant_url="http://localhost:6333"
        )
        
        self.risk_agent = RiskAgent()
        self.decision_agent = DecisionAgent()
        self.explanation_agent = ExplanationAgent(use_google_ai=use_google_ai)
        
        # Monitoring and logging
        self.enable_monitoring = enable_monitoring
        self.enable_audit_logging = enable_audit_logging
        self.monitor = None
        self.audit_logger = None
        self.fallback_handler = None
        
        # Initialize monitoring components
        self._init_monitoring()
        
        # Pipeline state
        self.pipeline_state: Dict[str, Any] = {
            "total_applications": 0,
            "successful_applications": 0,
            "failed_applications": 0,
            "average_latency_ms": 0
        }
    
    def _init_monitoring(self):
        """Initialize monitoring components."""
        try:
            if self.enable_monitoring:
                from monitoring.model_monitor import ModelMonitor
                self.monitor = ModelMonitor(
                    model_id="credit_decision_adk_v1"
                )
                print("✓ Model monitoring enabled")
            
            if self.enable_audit_logging:
                from monitoring.audit_logger import AuditLogger
                self.audit_logger = AuditLogger(
                    log_dir="./audit_logs"
                )
                print("✓ Audit logging enabled")
            
            # Always initialize fallback handler
            from monitoring.fallback_handler import FallbackHandler
            self.fallback_handler = FallbackHandler()
            print("✓ Fallback handler initialized")
            
        except ImportError as e:
            print(f"⚠ Monitoring components not available: {e}")
        except Exception as e:
            print(f"⚠ Error initializing monitoring: {e}")
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle monitoring alerts."""
        print(f"🚨 ALERT: {alert.get('type')} - {alert.get('message')}")
    
    def _route_to_manual_review(self, application_id: str, reason: str) -> Dict[str, Any]:
        """Route application to manual review."""
        return {
            "application_id": application_id,
            "routed_to": "manual_review_queue",
            "reason": reason
        }
    
    def process_application(
        self,
        loan_application: Dict[str, Any],
        application_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a loan application through the entire pipeline.
        
        Args:
            loan_application: Raw loan application data
            application_id: Optional application ID for tracking
            
        Returns:
            Complete screening result with decision and explanation
        """
        pipeline_start = time.time()
        
        # Generate application ID if not provided
        if not application_id:
            application_id = f"APP-{uuid.uuid4().hex[:12].upper()}"
        
        self.pipeline_state["total_applications"] += 1
        
        try:
            # --- STRATEGIC SAFETY GATE (Pre-Analysis) ---
            # Reject mathematically unviable loans before complex processing
            debt = float(loan_application.get("existing_debt", 0))
            income = float(loan_application.get("annual_income", 1))
            loan = float(loan_application.get("loan_amount", 0))
            
            # 1. Hard DTI Ceiling (Total Debt / Total Income)
            # Monthly DTI calculation
            monthly_income = income / 12
            estimated_monthly_payment = (loan * 0.15) / 12 # Aggressive 15% APR estimate for safety
            # existing_debt is already monthly provided by UI
            total_monthly_debt = debt + estimated_monthly_payment
            dti = (total_monthly_debt / monthly_income) * 100 if monthly_income > 0 else 100.0
            
            # Inject calculated DTI for downstream agents
            loan_application['debt_to_income'] = dti
            
            if dti > 50.0:
                return {
                    "success": True,
                    "application_id": application_id,
                    "recommendation": "REJECT",
                    "confidence": 1.0,
                    "decision_reasons": [
                        f"Critical DTI ratio ({dti:.1f}%) exceeds absolute safety ceiling (50.0%)",
                        "Application fails fundamental affordability criteria before secondary analysis"
                    ],
                    "overall_risk_score": {"overall_default_rate": 1.0},
                    "explanation": f"REJECTED: DTI excessive ({dti:.1f}%). Maximum allowed is 50%.",
                    "processing_metadata": {"total_latency_seconds": time.time() - pipeline_start}
                }

            # 2. Income Sufficiency (Loan-to-Income)
            if loan > income * 2.0: # Relaxed to 2.0x to allow reasonable large loans if DTI fits
                return {
                    "success": True,
                    "application_id": application_id,
                    "recommendation": "REJECT",
                    "confidence": 1.0,
                    "decision_reasons": [
                        f"Loan amount (${loan:,.0f}) is excessive compared to annual income (${income:,.0f})",
                        "Insufficient income to support requested capital amount"
                    ],
                    "overall_risk_score": {"overall_default_rate": 1.0},
                    "explanation": "REJECTED: Requested loan amount exceeds sustainable income multipliers.",
                    "processing_metadata": {"total_latency_seconds": time.time() - pipeline_start}
                }

            # Step 1: Ingestion & Validation
            ingestion_msg = AgentMessage(
                sender="pipeline",
                receiver=self.ingestion_agent.agent_name,
                message_type="process_application",
                payload={"data": loan_application}
            )
            ingestion_response = self.ingestion_agent.process(ingestion_msg)
            
            if not ingestion_response.success:
                return self._handle_failure(
                    application_id=application_id,
                    stage="ingestion",
                    error=ingestion_response.error,
                    loan_application=loan_application,
                    start_time=pipeline_start
                )
            
            cleaned_data = ingestion_response.data["cleaned_data"]
            
            # Step 2: Semantic Chunking
            chunking_msg = AgentMessage(
                sender=self.ingestion_agent.agent_name,
                receiver=self.chunking_agent.agent_name,
                message_type="chunk_data",
                payload={"cleaned_data": cleaned_data}
            )
            chunking_response = self.chunking_agent.process(chunking_msg)
            
            if not chunking_response.success:
                return self._handle_failure(
                    application_id=application_id,
                    stage="chunking",
                    error=chunking_response.error,
                    loan_application=loan_application,
                    start_time=pipeline_start
                )
            
            chunks = chunking_response.data["chunks"]
            
            # Step 3: Similarity Retrieval
            from embeddings import EmbeddingModel
            embedding_model = EmbeddingModel()
            chunk_embeddings = embedding_model.embed_chunks(chunks)
            
            retrieval_msg = AgentMessage(
                sender=self.chunking_agent.agent_name,
                receiver=self.retrieval_agent.agent_name,
                message_type="search_similar",
                payload={
                    "chunks": chunks,
                    "chunk_embeddings": chunk_embeddings,
                    "top_k": 20
                }
            )
            retrieval_response = self.retrieval_agent.process(retrieval_msg)
            
            if not retrieval_response.success:
                return self._handle_failure(
                    application_id=application_id,
                    stage="retrieval",
                    error=retrieval_response.error,
                    loan_application=loan_application,
                    start_time=pipeline_start
                )
            
            similar_cases = retrieval_response.data["similar_cases"]
            
            # Step 4: Risk Analysis
            risk_msg = AgentMessage(
                sender=self.retrieval_agent.agent_name,
                receiver=self.risk_agent.agent_name,
                message_type="analyze_risk",
                payload={
                    "similar_cases": similar_cases,
                    "original_data": cleaned_data
                }
            )
            risk_response = self.risk_agent.process(risk_msg)
            
            if not risk_response.success:
                return self._handle_failure(
                    application_id=application_id,
                    stage="risk_analysis",
                    error=risk_response.error,
                    loan_application=loan_application,
                    start_time=pipeline_start
                )
            
            per_dimension_risk = risk_response.data["per_dimension_risk"]
            overall_risk = risk_response.data["overall_risk_score"]
            anomalies = risk_response.data["anomalies"]
            
            # Step 5: Decision (now includes applicant_data for prime overrides)
            decision_msg = AgentMessage(
                sender=self.risk_agent.agent_name,
                receiver=self.decision_agent.agent_name,
                message_type="make_decision",
                payload={
                    "overall_risk_score": overall_risk,
                    "anomalies": anomalies,
                    "per_dimension_risk": per_dimension_risk,
                    "applicant_data": cleaned_data  # Added for prime applicant overrides
                }
            )
            decision_response = self.decision_agent.process(decision_msg)
            
            if not decision_response.success:
                return self._handle_failure(
                    application_id=application_id,
                    stage="decision",
                    error=decision_response.error,
                    loan_application=loan_application,
                    start_time=pipeline_start
                )
            
            decision_data = decision_response.data
            
            # Step 6: Explanation
            explanation_msg = AgentMessage(
                sender=self.decision_agent.agent_name,
                receiver=self.explanation_agent.agent_name,
                message_type="generate_explanation",
                payload={
                    "decision": decision_data,
                    "risk_analysis": {
                        "per_dimension_risk": per_dimension_risk,
                        "overall_risk_score": overall_risk,
                        "anomalies": anomalies
                    }
                }
            )
            explanation_response = self.explanation_agent.process(explanation_msg)
            
            if not explanation_response.success:
                # Use template explanation if AI fails
                from agents.explanation_agent import ExplanationAgent
                template_agent = ExplanationAgent(use_google_ai=False)
                explanation_response = template_agent.process(explanation_msg)
            
            explanation = explanation_response.data.get("explanation", "")
            
            # Calculate total latency
            total_latency = time.time() - pipeline_start
            
            # Compile final result
            result = {
                "success": True,
                "application_id": application_id,
                "recommendation": decision_data["recommendation"],
                "confidence": decision_data["confidence"],
                "model_confidence": decision_data.get("model_confidence", decision_data["confidence"]),
                "decision_reasons": decision_data.get("decision_reasons", []),
                "per_dimension_analysis": per_dimension_risk,
                "overall_risk_score": overall_risk,
                "anomalies": anomalies,
                "explanation": explanation,
                # Flatten and format similar cases for UI/Orchestrator
                "similar_cases": self._format_similar_cases(similar_cases),
                "processing_metadata": {
                    "application_id": application_id,
                    "total_latency_seconds": total_latency,
                    "ingestion_time": ingestion_response.processing_time,
                    "chunking_time": chunking_response.processing_time,
                    "retrieval_time": retrieval_response.processing_time,
                    "risk_analysis_time": risk_response.processing_time,
                    "decision_time": decision_response.processing_time,
                    "explanation_time": explanation_response.processing_time
                }
            }
            
            # Log to monitoring
            monitor_id = self._log_decision(result, cleaned_data)
            result["monitor_id"] = monitor_id
            
            # Update pipeline stats
            self.pipeline_state["successful_applications"] += 1
            self._update_average_latency(total_latency * 1000)
            
            return result
            
        except Exception as e:
            return self._handle_failure(
                application_id=application_id,
                stage="unknown",
                error=str(e),
                loan_application=loan_application,
                start_time=pipeline_start
            )
    
    def _handle_failure(
        self,
        application_id: str,
        stage: str,
        error: str,
        loan_application: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle pipeline failure with fallback."""
        self.pipeline_state["failed_applications"] += 1
        
        # Try fallback handler
        if self.fallback_handler:
            try:
                fallback_result = self.fallback_handler.handle_failure(
                    application_id=application_id,
                    stage=stage,
                    error=error,
                    applicant_data=loan_application
                )
                
                return {
                    "success": False,
                    "application_id": application_id,
                    "error": f"{stage} failed: {error}",
                    "stage": stage,
                    "fallback_applied": True,
                    "fallback_action": fallback_result.get("action", "manual_review"),
                    "recommendation": "MANUAL_REVIEW",
                    "confidence": 0.0,
                    "processing_metadata": {
                        "application_id": application_id,
                        "total_latency_seconds": time.time() - start_time,
                        "failure_stage": stage
                    }
                }
            except Exception as fallback_error:
                print(f"Fallback handler error: {fallback_error}")
        
        # Standard failure response
        return {
            "success": False,
            "application_id": application_id,
            "error": f"{stage} failed: {error}",
            "stage": stage,
            "fallback_applied": False,
            "recommendation": "MANUAL_REVIEW",
            "confidence": 0.0,
            "decision_reasons": [f"Pipeline Error: {error}"],
            "processing_metadata": {
                "application_id": application_id,
                "total_latency_seconds": time.time() - start_time,
                "failure_stage": stage
            }
        }
    
    def _log_decision(self, result: Dict[str, Any], applicant_data: Dict[str, Any]):
        """Log decision to monitoring and audit systems."""
        try:
            # Log to model monitor
            monitor_id = None
            if self.monitor:
                monitor_id = self.monitor.log_decision({
                    "application_id": result["application_id"],
                    "risk_score": result.get("risk_score", result["overall_risk_score"].get("overall_default_rate", 0)),
                    "decision": result["recommendation"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"],
                    "component_scores": result["per_dimension_analysis"],
                    "similar_cases": result["similar_cases"],
                    "latency_ms": result["processing_metadata"].get("total_latency_seconds", 0) * 1000,
                    "inputs": applicant_data
                })
            
            # Log to audit logger
            if self.audit_logger:
                # Extract numeric risk for the audit entry
                numeric_risk = result.get("risk_score")
                if isinstance(numeric_risk, dict):
                    numeric_risk = numeric_risk.get("overall_default_rate", 0.5)
                elif numeric_risk is None:
                    numeric_risk = result["overall_risk_score"].get("overall_default_rate", 0.5)

                self.audit_logger.log_decision(
                    application_id=result["application_id"],
                    decision=result["recommendation"],
                    confidence=result["confidence"],
                    risk_score=numeric_risk,
                    processing_time_ms=result["processing_metadata"].get("total_latency_seconds", 0) * 1000,
                    explanation_summary=result["explanation"],
                    component_scores=result["per_dimension_analysis"],
                    user_id=None
                )
            return monitor_id
        except Exception as e:
            print(f"Logging error: {e}")
            return None
    
    def _update_average_latency(self, latency_ms: float):
        """Update running average latency."""
        n = self.pipeline_state["successful_applications"]
        current_avg = self.pipeline_state["average_latency_ms"]
        
        # Incremental average update
        self.pipeline_state["average_latency_ms"] = (
            current_avg + (latency_ms - current_avg) / n
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "ingestion_agent": self.ingestion_agent.get_state(),
            "chunking_agent": self.chunking_agent.get_state(),
            "retrieval_agent": self.retrieval_agent.get_state(),
            "risk_agent": self.risk_agent.get_state(),
            "decision_agent": self.decision_agent.get_state(),
            "explanation_agent": self.explanation_agent.get_state()
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = self.pipeline_state["total_applications"]
        successful = self.pipeline_state["successful_applications"]
        
        return {
            "total_applications": total,
            "successful_applications": successful,
            "failed_applications": self.pipeline_state["failed_applications"],
            "success_rate": successful / total if total > 0 else 0,
            "average_latency_ms": self.pipeline_state["average_latency_ms"],
            "monitoring_enabled": self.enable_monitoring,
            "audit_logging_enabled": self.enable_audit_logging
        }
    
    def get_monitoring_report(self) -> Optional[Dict[str, Any]]:
        """Get monitoring report if monitoring is enabled."""
        if self.monitor:
            try:
                return self.monitor.get_metrics_report()
            except Exception as e:
                return {"error": str(e)}
        return None

    def _format_similar_cases(self, similar_cases_dict: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Flatten, deduplicate and format multi-dimensional search results.
        
        Transforms technical Qdrant format into display format:
        - score -> similarity
        - payload[loan_amount] -> amount
        - payload[defaulted] -> defaulted
        """
        unique_cases = {}
        
        for dimension, cases in similar_cases_dict.items():
            if not isinstance(cases, list):
                continue
                
            for case in cases:
                case_id = str(case.get("id"))
                score = case.get("score", 0.0)
                payload = case.get("payload", {})
                
                # If seen before, keep the one with higher similarity score
                if case_id in unique_cases:
                    if score > unique_cases[case_id]["similarity"]:
                        unique_cases[case_id]["similarity"] = score
                    continue
                
                # Format for display
                formatted_case = {
                    "id": case_id,
                    "similarity": score,
                    "amount": payload.get("loan_amount", 0),
                    "defaulted": payload.get("loan_status") == "Charged Off" or payload.get("loan_status") == "Default",
                    "outcome": "Defaulted" if (payload.get("loan_status") == "Charged Off" or payload.get("loan_status") == "Default") else "Good Standing",
                    "credit_score": payload.get("credit_score", "N/A"),
                    "dimension": dimension # Track which dimension found it
                }
                unique_cases[case_id] = formatted_case
        
        # Sort by similarity and return top 5
        sorted_cases = sorted(unique_cases.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_cases[:5]

