"""
Qdrant Vector Store Module

Manages a Qdrant collection with named vectors for loan applications.
Each loan is stored as a single point with multiple semantic vectors.
"""

from typing import Dict, Any, Optional, List
import uuid
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    NamedVector,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantVectorStore:
    COLLECTION_NAME = "lending_club_loans"
    VECTOR_SIZE = 384  # FastEmbed BAAI/bge-small-en-v1.5
    DISTANCE_METRIC = Distance.COSINE

    CHUNK_TYPES = [
        "income_stability",
        "credit_behavior",
        "debt_obligations",
        "recent_behavior",
        "account_portfolio",
        "loan_context",
    ]

    def __init__(self, url: Optional[str] = None, path: Optional[str] = None, location: Optional[str] = None):
        """
        Initialize Qdrant client with prioritized connection methods.
        
        Args:
            url: Qdrant server URL (e.g., http://localhost:6333)
            path: Local path for disk storage (e.g., "./qdrant_db")
            location: Qdrant location (e.g., ":memory:")
        """
        self.url = url or "http://localhost:6333"
        
        try:
            if location or path:
                # Use local storage if explicitly requested
                self.client = QdrantClient(path=path, location=location)
                print(f"✓ Connected to Qdrant (Local Storage: {location or path})")
            else:
                # Try server with a short timeout to avoid hanging
                print(f"⌛ Connecting to Qdrant server at {self.url}...")
                self.client = QdrantClient(url=self.url, timeout=5.0)
                # Test connection
                self.client.get_collections()
                print(f"✓ Connected to Qdrant server at {self.url}")
        except Exception as e:
            print(f"⚠️ Could not connect to Qdrant server: {e}")
            print(f"🔄 Falling back to in-memory storage (:memory:)...")
            self.client = QdrantClient(location=":memory:")
            print(f"✓ Initialized Qdrant in-memory storage")

    # ------------------------
    # Collection Management
    # ------------------------

    def create_collection(self, force_recreate: bool = False):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)

        if exists and not force_recreate:
            print(f"Collection '{self.COLLECTION_NAME}' already exists")
            return

        if exists and force_recreate:
            self.client.delete_collection(self.COLLECTION_NAME)
            print(f"Deleted existing collection '{self.COLLECTION_NAME}'")

        vectors_config = {
            chunk: VectorParams(
                size=self.VECTOR_SIZE,
                distance=self.DISTANCE_METRIC,
            )
            for chunk in self.CHUNK_TYPES
        }

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=vectors_config,
        )

        print(f"✓ Created collection '{self.COLLECTION_NAME}' with {len(vectors_config)} named vectors")

    # ------------------------
    # Ingestion (Generic API)
    # ------------------------

    def upsert(
        self,
        id: int,
        named_vectors: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ):
        """
        Generic upsert method used by DatasetLoader.
        """

        vectors = {
            name: vector.tolist()
            for name, vector in named_vectors.items()
        }

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=id,
                    vector=vectors,
                    payload=metadata,
                )
            ],
        )

    # ------------------------
    # Search
    # ------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection using named vectors.
        
        Args:
            query_embedding: Query embedding vector
            chunk_type: Type of chunk (named vector name) to search
            limit: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar results with scores and metadata
        """
        if chunk_type not in self.CHUNK_TYPES:
            raise ValueError(f"Invalid chunk type: {chunk_type}")

        try:
            # Modern Qdrant (1.10+) uses unified query_points API
            # Search using the named vector (passed via 'using' parameter)
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_embedding.tolist(),
                using=chunk_type,
                limit=limit,
                query_filter=filters,
            )
            
            return [
                {
                    "id": p.id,
                    "score": p.score,
                    "payload": p.payload,
                }
                for p in response.points
            ]
        except Exception as e:
            # Fallback for older versions if query_points is not available
            print(f"Unified query API error or unavailable: {e}")
            try:
                # Try the legacy search method if it exists (though check_methods showed it doesn't)
                results = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=NamedVector(
                        name=chunk_type,
                        vector=query_embedding.tolist(),
                    ),
                    limit=limit,
                    query_filter=filters,
                )
                
                return [
                    {
                        "id": r.id,
                        "score": r.score,
                        "payload": r.payload,
                    }
                    for r in results
                ]
            except Exception as e2:
                print(f"Legacy search also failed: {e2}")
                return []
    
    def search_similar_loans(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Alias for search() method for backward compatibility with retrieval agent.
        
        Args:
            query_embedding: Embedding vector to search for
            chunk_type: Type of chunk (dimension) to search
            limit: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of similar loans with scores
        """
        return self.search(query_embedding, chunk_type, limit, filters)