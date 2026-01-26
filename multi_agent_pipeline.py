"""
Multi-Agent Pipeline Orchestrator

Orchestrates the flow of loan applications through the multi-agent system.
"""

from typing import Dict, Any, Optional
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
        use_google_ai: bool = False
    ):
        """
        Initialize the multi-agent pipeline.
        
        Args:
            project_id: Google Cloud project ID (required if use_google_cloud=True)
            location: GCP region
            use_google_cloud: Use Google Vertex AI Vector Search
            use_google_ai: Use Google Gemini for explanations
        """
        # Initialize agents
        self.ingestion_agent = IngestionAgent()
        self.chunking_agent = ChunkingAgent(use_chonkie=True)  # Use Chonkie for semantic chunking
        
        # Retrieval agent (Qdrant primary, Google Cloud optional)
        self.retrieval_agent = RetrievalAgent(
            project_id=project_id if use_google_cloud else None,
            location=location,
            use_google_cloud=use_google_cloud,
            use_local_embeddings=True,  # Always use local embeddings
            qdrant_url="http://localhost:6333"  # Qdrant as primary
        )
        
        self.risk_agent = RiskAgent()
        self.decision_agent = DecisionAgent()
        self.explanation_agent = ExplanationAgent(use_google_ai=use_google_ai)
        
        # Pipeline state
        self.pipeline_state: Dict[str, Any] = {}
    
    def process_application(self, loan_application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a loan application through the entire pipeline.
        
        Args:
            loan_application: Raw loan application data
            
        Returns:
            Complete screening result with decision and explanation
        """
        try:
            # Step 1: Ingestion & Validation
            ingestion_msg = AgentMessage(
                sender="pipeline",
                receiver=self.ingestion_agent.agent_name,
                message_type="process_application",
                payload={"data": loan_application}
            )
            ingestion_response = self.ingestion_agent.process(ingestion_msg)
            
            if not ingestion_response.success:
                return {
                    "success": False,
                    "error": f"Ingestion failed: {ingestion_response.error}",
                    "stage": "ingestion"
                }
            
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
                return {
                    "success": False,
                    "error": f"Chunking failed: {chunking_response.error}",
                    "stage": "chunking"
                }
            
            chunks = chunking_response.data["chunks"]
            
            # Step 3: Similarity Retrieval
            # Generate embeddings for chunks
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
                return {
                    "success": False,
                    "error": f"Retrieval failed: {retrieval_response.error}",
                    "stage": "retrieval"
                }
            
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
                return {
                    "success": False,
                    "error": f"Risk analysis failed: {risk_response.error}",
                    "stage": "risk_analysis"
                }
            
            per_dimension_risk = risk_response.data["per_dimension_risk"]
            overall_risk = risk_response.data["overall_risk_score"]
            anomalies = risk_response.data["anomalies"]
            
            # Step 5: Decision
            decision_msg = AgentMessage(
                sender=self.risk_agent.agent_name,
                receiver=self.decision_agent.agent_name,
                message_type="make_decision",
                payload={
                    "overall_risk_score": overall_risk,
                    "anomalies": anomalies
                }
            )
            decision_response = self.decision_agent.process(decision_msg)
            
            if not decision_response.success:
                return {
                    "success": False,
                    "error": f"Decision failed: {decision_response.error}",
                    "stage": "decision"
                }
            
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
            
            # Compile final result
            result = {
                "success": True,
                "recommendation": decision_data["recommendation"],
                "confidence": decision_data["confidence"],
                "per_dimension_analysis": per_dimension_risk,
                "overall_risk_score": overall_risk,
                "anomalies": anomalies,
                "explanation": explanation,
                "processing_metadata": {
                    "ingestion_time": ingestion_response.processing_time,
                    "chunking_time": chunking_response.processing_time,
                    "retrieval_time": retrieval_response.processing_time,
                    "risk_analysis_time": risk_response.processing_time,
                    "decision_time": decision_response.processing_time,
                    "explanation_time": explanation_response.processing_time
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Pipeline error: {str(e)}",
                "stage": "unknown"
            }
    
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
