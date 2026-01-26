"""
Embedding Module for Loan Application Chunks

This module handles text embeddings using FastEmbed - a lightweight,
fast embedding library optimized for local deployment.
"""

import os
from typing import List, Dict, Optional
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except Exception as e:
    FASTEMBED_AVAILABLE = False
    raise ImportError(f"FastEmbed is required. Install with: pip install fastembed. Error: {e}")


class EmbeddingModel:
    """
    Text embedding model using FastEmbed backend.
    
    FastEmbed provides fast, lightweight embeddings suitable for local deployment.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the FastEmbed embedding model.
        
        Args:
            model_name: Model name (default: BAAI/bge-small-en-v1.5)
                       Other options: BAAI/bge-base-en-v1.5, sentence-transformers/all-MiniLM-L6-v2
        """
        self.model_name = model_name or "BAAI/bge-small-en-v1.5"
        
        try:
            self.model = TextEmbedding(model_name=self.model_name)
            # Most embedding models return 384-dimensional vectors
            self.embedding_size = 384
            print(f"✓ FastEmbed initialized with model: {self.model_name}")
            print(f"✓ Embedding dimension: {self.embedding_size}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FastEmbed: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embeddings = list(self.model.embed([text]))
            return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Embedding error: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors (shape: [n_texts, embedding_size])
        """
        try:
            embeddings = list(self.model.embed(texts))
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Batch embedding error: {e}")
    
    def embed_chunks(self, chunks: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all chunks in a loan application.
        
        Args:
            chunks: Dictionary mapping chunk type to chunk text
            
        Returns:
            Dictionary mapping chunk type to embedding vector
        """
        chunk_embeddings = {}
        for chunk_type, chunk_text in chunks.items():
            chunk_embeddings[chunk_type] = self.embed_text(chunk_text)
        
        return chunk_embeddings
