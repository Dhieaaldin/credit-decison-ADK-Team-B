# Technical Report: Credit Decision Screening System (ADK)

## 1. Project Idea & Problem Statement
### The Problem
Traditional credit scoring models often rely on static formulas that may miss the qualitative nuances of a borrower's financial history. For modern lenders, understanding the "semantic similarity" between a new applicant and historical borrowers—ones who successfully repaid and ones who defaulted—is a powerful but underutilized signal.

### The Solution
The **Credit Decision ADK** is a cognitive screening system that uses a multi-agent architecture to analyze loan applications. Instead of just looking at raw numbers, it breaks down applications into semantic dimensions (e.g., "Income Stability", "Credit Behavior") and uses Vector Search to find historical parallels, providing a similarity-based risk assessment.

---

## 2. Architecture & Qdrant Integration
### Multi-Agent Pipeline
The system is built on a modular pipeline where specialized agents handle different stages of the decision-making process:

- **Ingestion Agent**: Validates and normalizes incoming loan data.
- **Chunking Agent**: Uses `Chonkie` (Semantic Chunking) to group related financial attributes.
- **Retrieval Agent**: Communicates with the vector store to find similar historical cases.
- **Risk Agent**: Aggregates similarity scores and calculates a weighted risk profile.
- **Decision Agent**: Evaluates risk against lending thresholds.
- **Explanation Agent**: Uses LLMs (via OpenRouter/Vertex AI) to generate natural language justifications.

---

### 🛡️ The Core: Qdrant Vector Intelligence
**Qdrant** is not just a database here—it is the system's **Cognitive Memory**. By integrating Qdrant, the system transforms from a static rule-engine into a dynamic learning pipeline.

#### Key Qdrant Capabilities Utilized:
- **Multi-Vector Semantic Search**: Unlike basic search, we use **Named Vectors**. This allows the system to perform simultaneous comparisons across different financial "dimensions" (e.g., searching for "Income Stability" and "Debt-to-Income" behavior separately but in parallel).
- **Unified Query API (`query_points`)**: We leverage the latest unified interface to perform complex lookups, ensuring we find the most relevant historical "neighbors" for any new application.
- **Zero-Latency Retrieval**: Qdrant's HNSW indexing enables the system to screen applications against thousands of historical records in milliseconds.
- **Fail-Safe Versatility**: The implementation is designed for production flexibility:
    1. **Remote Qdrant Server**: High-performance Docker/Cloud clusters.
    2. **Local Disk Storage**: Secure, on-premise persistence for sensitive financial data.
    3. **In-Memory Logic**: Lightning-fast ephemeral sessions for real-time validation.

---

## 3. Data Pipeline
The transformation from raw data to actionable intelligence follows this path:
1. **Raw CSV/JSON**: Historical loan data is loaded.
2. **Schema Alignment**: Features are mapped to a standardized credit schema.
3. **Semantic Partitioning**: Attributes are grouped into logical "chunks" (e.g., `loan_amount` and `purpose` into "Loan Context").
4. **Embedding Generation**: `FastEmbed` (BAAI/bge-small-en-v1.5) converts text chunks into 384-dimensional vectors.
5. **Vector Upsert**: Points are stored in Qdrant with the original loan data as the payload.
6. **Querying**: New applications are embedded in the same space to retrieve the k-nearest neighbors.

---

## 4. Project Timeline & Future Work
### Work Accomplished
- ✅ **Core Infrastructure**: Multi-agent orchestration layer implemented.
- ✅ **Vector Storage**: Complete Qdrant integration with version-aware API support.
- ✅ **Local Inference**: Integrated FastEmbed for torch-less, low-latency embedding generation.
- ✅ **Interactive CLI**: Orchestrator tool for manual testing and batch processing.
- ✅ **Explainability Layer**: Semantic risk explanation generator.

### Future Work
- 🔮 **Edge Deployment**: Expanding local disk storage capabilities for offline enterprise use.
- 🔮 **Google Vertex AI Deep Integration**: Adding full support for Vertex AI Vector Search for massive scale.
- 🔮 **Dashboard UI**: A web-based interface for loan officers to visualize "Risk Clusters".
- 🔮 **Real-time Monitoring**: A "Guardian Agent" that monitors for drift in similarity distributions.

---
**Date**: January 26, 2026
**Version**: 1.0.0
