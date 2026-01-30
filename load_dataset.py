"""
Dataset Ingestion Script

Loads loans_full_schema.csv into Qdrant using parallel processing for maximum CPU efficiency.
Uses multiprocessing to parallelize embedding generation and batch uploads.

Usage:
    python load_dataset.py --csv_path loans_full_schema.csv --limit 1000 --workers 4 --batch_size 64
"""

import argparse
import sys
import os
from typing import Dict, Any, Optional, List, Tuple
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import numpy as np

from vector_store import QdrantVectorStore


# Global embedding model for worker processes
_embedding_model = None


def init_worker(embedding_model_name: Optional[str] = None):
    """Initialize embedding model in worker process."""
    global _embedding_model
    from embeddings import EmbeddingModel
    _embedding_model = EmbeddingModel(embedding_model_name)


def create_chunks(row: Dict[str, Any]) -> Dict[str, str]:
    """Create semantic chunks from loan record."""
    return {
        "income_stability": (
            f"Employment: {row.get('emp_title', 'Unknown')}, "
            f"Length: {row.get('emp_length', 'Unknown')}, "
            f"Income: ${row.get('annual_income', 0):,.0f}, "
            f"Verified: {row.get('verified_income', 'Unknown')}"
        ),
        "credit_behavior": (
            f"Credit history since {row.get('earliest_credit_line', 'Unknown')}. "
            f"Total lines: {row.get('total_credit_lines', 0)}, "
            f"Open lines: {row.get('open_credit_lines', 0)}, "
            f"Credit limit: ${row.get('total_credit_limit', 0):,.0f}"
        ),
        "debt_obligations": (
            f"Debt-to-income ratio: {row.get('debt_to_income', 0):.1f}%. "
            f"Failed payments: {row.get('num_historical_failed_to_pay', 0)}"
        ),
        "recent_behavior": (
            f"Accounts opened (24m): {row.get('accounts_opened_24m', 0)}, "
            f"Inquiries (12m): {row.get('inquiries_last_12m', 0)}"
        ),
        "account_portfolio": (
            f"Credit cards: {row.get('num_total_cc_accounts', 0)}, "
            f"Mortgages: {row.get('num_mort_accounts', 0)}, "
            f"Installments: {row.get('current_installment_accounts', 0)}"
        ),
        "loan_context": (
            f"Purpose: {row.get('loan_purpose', 'Unknown')}, "
            f"Amount: ${row.get('loan_amount', 0):,.0f}, "
            f"Term: {row.get('term', 'Unknown')} months, "
            f"State: {row.get('state', 'Unknown')}"
        ),
    }


def process_batch(batch_data: List[Tuple[int, Dict[str, Any]]]) -> List[Tuple[int, Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]]:
    """
    Process a batch of rows in parallel worker.
    Embeds all chunk texts for the entire batch in a single call.
    
    Args:
        batch_data: List of (idx, row_dict) tuples
        
    Returns:
        List of (idx, chunks, embeddings, metadata) tuples
    """
    global _embedding_model
    results = []

    all_texts = []
    row_chunk_counts = []  # store (idx, num_chunks) per row to split embeddings later
    rows_chunks = []       # store chunks dict for each row for later mapping

    # Collect all chunk texts from all rows in batch
    for idx, row in batch_data:
        chunks = create_chunks(row)
        rows_chunks.append(chunks)
        texts = list(chunks.values())
        row_chunk_counts.append((idx, len(texts)))
        all_texts.extend(texts)

    # Embed all chunk texts in one batch call
    all_embeddings = _embedding_model.embed_batch(all_texts)

    # Split embeddings back per row
    pos = 0
    for (idx, chunk_count), chunks in zip(row_chunk_counts, rows_chunks):
        chunk_embeddings = all_embeddings[pos : pos + chunk_count]
        pos += chunk_count

        embeddings = {
            chunk_type: chunk_embeddings[i]
            for i, chunk_type in enumerate(chunks.keys())
        }

        # Use the row corresponding to idx for metadata
        row_data = next(row for i, row in batch_data if i == idx)

        metadata = {
            "loan_id": row_data.get("id", "unknown"),
            "loan_amount": float(row_data.get("loan_amount", 0)),
            "annual_income": float(row_data.get("annual_income", 0)),
            "state": row_data.get("state", "unknown"),
            "purpose": row_data.get("loan_purpose", "unknown"),
            "loan_status": row_data.get("loan_status", "Unknown"),
            "grade": row_data.get("grade", "N/A"),
        }

        results.append((idx, chunks, embeddings, metadata))

    return results


class DatasetLoader:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        embedding_model_name: Optional[str] = None,
        num_workers: int = None,
        batch_size: int = 64,
    ):
        """
        Initialize dataset loader with multiprocessing support.
        
        Args:
            qdrant_url: Qdrant server URL
            embedding_model_name: Embedding model name
            num_workers: Number of parallel workers (default: CPU count - 1)
            batch_size: Batch size for embedding generation
        """
        self.vector_store = QdrantVectorStore(qdrant_url)
        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, cpu_count() - 1)

        print(f"✓ Parallel workers: {self.num_workers}")
        print(f"✓ Batch size: {self.batch_size}")

    # ------------------------
    # Ingestion (Parallel)
    # ------------------------

    def ingest(self, csv_path: str, limit: Optional[int] = None):
        """
        Ingest dataset with parallel embedding generation.
        
        Uses multiprocessing to parallelize embedding computation across CPU cores.
        """
        print(f"\n📂 Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} records")

        if limit:
            df = df.head(limit)
            print(f"⚠️ Limited to {limit} rows")

        self.vector_store.create_collection(force_recreate=True)

        print(f"\n⏳ Processing {len(df)} records with {self.num_workers} workers...")
        print(f"   Batch size: {self.batch_size}\n")

        # Convert rows to list of (idx, row_dict) tuples
        rows_list = [(idx, row.to_dict()) for idx, (_, row) in enumerate(df.iterrows())]

        # Create batches for parallel processing
        batches = [
            rows_list[i : i + self.batch_size]
            for i in range(0, len(rows_list), self.batch_size)
        ]

        success = 0
        total = len(df)

        # Process batches in parallel with worker pool
        with Pool(
            processes=self.num_workers,
            initializer=init_worker,
            initargs=(self.embedding_model_name,)
        ) as pool:
            for batch_results in tqdm(
                pool.imap_unordered(process_batch, batches),
                total=len(batches),
                desc="Embedding & uploading",
            ):
                try:
                    self.vector_store.upsert_batch([
                        (idx, embeddings, metadata)
                        for idx, _, embeddings, metadata in batch_results
                    ])
                    success += len(batch_results)
                except Exception as e:
                    print(f"⚠️ Error uploading batch: {e}")

        print(f"\n✅ Successfully ingested {success}/{total} records")


# ------------------------
# CLI
# ------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load loan dataset into Qdrant with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_dataset.py --limit 100
  python load_dataset.py --workers 8 --batch_size 128
  python load_dataset.py --csv_path data.csv --workers auto
        """
    )
    parser.add_argument("--csv_path", default="loans_full_schema.csv", help="CSV file path")
    parser.add_argument("--qdrant_url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--limit", type=int, default=None, help="Max records to load")
    parser.add_argument(
        "--workers",
        type=str,
        default=str(max(1, cpu_count() - 1)),
        help=f"Number of parallel workers (default: {max(1, cpu_count() - 1)}, 'auto' = CPU count - 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per worker (default: 64, increase for more parallelism)"
    )

    args = parser.parse_args()

    # Parse workers
    if args.workers.lower() == "auto":
        num_workers = max(1, cpu_count() - 1)
    else:
        num_workers = int(args.workers)

    try:
        loader = DatasetLoader(
            qdrant_url=args.qdrant_url,
            num_workers=num_workers,
            batch_size=args.batch_size,
        )
        loader.ingest(args.csv_path, args.limit)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
