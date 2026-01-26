"""
Similarity Retrieval Agent

Performs vector similarity search using Qdrant (primary).
Google Vertex AI Vector Search available as optional backend.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse

# Lazy import for Google Cloud (optional)
GOOGLE_CLOUD_AVAILABLE = False


def _init_google_cloud():
    """Lazy initialize Google Cloud imports (optional)."""
    global GOOGLE_CLOUD_AVAILABLE
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform import vector_search
        GOOGLE_CLOUD_AVAILABLE = True
        return True
    except Exception as e:
        print(f"Warning: Google Cloud AI Platform not available. Install with: pip install google-cloud-aiplatform")
        GOOGLE_CLOUD_AVAILABLE = False
        return False


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for similarity search using Google Vertex AI Vector Search.
    
    Input: Semantic chunks with embeddings
    Output: Similar historical loan cases
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        index_endpoint_id: Optional[str] = None,
        index_id: Optional[str] = None,
        use_google_cloud: bool = False,
        use_local_embeddings: bool = True,
        qdrant_url: str = "http://localhost:6333"
    ):
        """
        Initialize retrieval agent with Qdrant (primary) or Google Vertex AI (optional).
        
        Args:
            project_id: Google Cloud project ID (required if use_google_cloud=True)
            location: GCP region (default: us-central1)
            index_endpoint_id: Vertex AI Vector Search index endpoint ID
            index_id: Vertex AI Vector Search index ID
            use_google_cloud: If True, use Google Vertex AI (otherwise use Qdrant)
            use_local_embeddings: If True, use local embeddings (sentence-transformers)
            qdrant_url: Qdrant server URL (default: localhost:6333)
        """
        super().__init__("retrieval_agent", "Vector Similarity Search")
        
        self.project_id = project_id
        self.location = location
        self.index_endpoint_id = index_endpoint_id
        self.index_id = index_id
        self.use_google_cloud = use_google_cloud
        self.use_local_embeddings = use_local_embeddings
        self.qdrant_url = qdrant_url
        
        # Initialize Qdrant (primary vector store)
        try:
            from vector_store import QdrantVectorStore
            self.qdrant_store = QdrantVectorStore(url=qdrant_url)
            self.qdrant_available = True
            print(f"Qdrant vector store initialized at {qdrant_url}")
        except Exception as e:
            print(f"Warning: Could not initialize Qdrant: {e}")
            self.qdrant_store = None
            self.qdrant_available = False
        
        # Initialize Google Cloud client if requested
        if use_google_cloud:
            if _init_google_cloud() and GOOGLE_CLOUD_AVAILABLE and project_id:
                try:
                    from google.cloud import aiplatform
                    aiplatform.init(project=project_id, location=location)
                    self.client_initialized = True
                    print("Google Vertex AI initialized")
                except Exception as e:
                    print(f"Warning: Could not initialize Google Cloud: {e}")
                    self.client_initialized = False
            else:
                self.client_initialized = False
        else:
            self.client_initialized = False
        
        # For local embeddings
        if use_local_embeddings:
            try:
                from embeddings import EmbeddingModel
                self.embedding_model = EmbeddingModel()
            except Exception as e:
                print(f"Warning: Could not load local embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Perform similarity search for semantic chunks.
        
        Args:
            message: Contains semantic chunks and optionally embeddings
            
        Returns:
            AgentResponse with similar cases per chunk dimension
        """
        import time
        start_time = time.time()
        
        try:
            chunks = message.payload.get("chunks", {})
            chunk_embeddings = message.payload.get("chunk_embeddings", {})
            top_k = message.payload.get("top_k", 20)
            
            if not chunks:
                return AgentResponse(
                    success=False,
                    error="No chunks provided",
                    processing_time=time.time() - start_time
                )
            
            # Generate embeddings if not provided
            if not chunk_embeddings and self.embedding_model:
                chunk_embeddings = self.embedding_model.embed_chunks(chunks)
            
            # Perform similarity search for each chunk
            similar_cases = {}
            for chunk_type, chunk_text in chunks.items():
                if chunk_type in chunk_embeddings:
                    embedding = chunk_embeddings[chunk_type]
                    similar_loans = self._search_similar(
                        query_embedding=embedding,
                        chunk_type=chunk_type,
                        limit=top_k
                    )
                    similar_cases[chunk_type] = similar_loans
            
            # Log processing
            self.log_message(message)
            self.update_state({
                "last_processed": time.time(),
                "total_searches": self.state.get("total_searches", 0) + len(chunks)
            })
            
            return AgentResponse(
                success=True,
                data={"similar_cases": similar_cases},
                metadata={
                    "num_dimensions_searched": len(similar_cases),
                    "total_results": sum(len(cases) for cases in similar_cases.values())
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Retrieval error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _search_similar(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for similar loans using Qdrant (primary) or Google Vertex AI (optional).
        
        Args:
            query_embedding: Query embedding vector
            chunk_type: Type of chunk being searched
            limit: Maximum number of results
            
        Returns:
            List of similar loan cases
        """
        # Use Qdrant as primary vector store
        if self.qdrant_available and self.qdrant_store:
            return self._search_qdrant(query_embedding, chunk_type, limit)
        elif self.use_google_cloud and self.client_initialized and self.index_endpoint_id:
            # Use Google Vertex AI Vector Search as alternative
            return self._search_vertex_ai(query_embedding, chunk_type, limit)
        else:
            # Final fallback
            return []
    
    def _search_vertex_ai(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search using Google Vertex AI Vector Search."""
        try:
            from google.cloud import aiplatform
            
            # Convert numpy array to list
            query_vector = query_embedding.tolist()
            
            # Create index client
            index_client = aiplatform.MatchingEngineIndex(
                index_endpoint_name=self.index_endpoint_id,
                index_id=self.index_id
            )
            
            # Perform search
            # Note: This is a simplified example - actual implementation depends on
            # your Vertex AI Vector Search setup
            results = index_client.find_neighbors(
                deployed_index_id=self.index_id,
                queries=[query_vector],
                num_neighbors=limit
            )
            
            # Format results
            formatted_results = []
            for result in results[0] if results else []:
                formatted_results.append({
                    "id": result.get("id", ""),
                    "score": result.get("distance", 0.0),
                    "payload": result.get("payload", {})
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Vertex AI search error: {e}. Falling back to Qdrant.")
            if self.qdrant_available and self.qdrant_store:
                return self._search_qdrant(query_embedding, chunk_type, limit)
            return []
    
    def _search_qdrant(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search using Qdrant vector store (primary method)."""
        try:
            return self.qdrant_store.search_similar_loans(
                query_embedding=query_embedding,
                chunk_type=chunk_type,
                limit=limit
            )
        except Exception as e:
            print(f"Qdrant search error: {e}")
            return []
