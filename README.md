# Credit Decision ADK - Multi-Agent Loan Screening System

## 🚀 Overview
Credit Decision ADK is an advanced, multi-agent automated loan screening system designed to provide deep cognitive analysis of credit applications. It leverages semantic vector search and a pipeline of specialized AI agents to evaluate risk, retrieve similar historical cases, and provide human-readable explanations for credit decisions.

## 🧠 Key Features
- **Multi-Agent Orchestration**: A modular pipeline of 6+ specialized agents (Ingestion, Chunking, Retrieval, Risk, Decision, Explanation).
- **Semantic Risk Assessment**: Uses `Chonkie` for semantic chunking and `FastEmbed` for local high-performance embeddings.
- **Vector Intelligence**: Integrated with **Qdrant** for high-speed similarity search, allowing the system to "remember" and compare new applications against historical outcomes.
- **Local & Cloud Flexible**: Supports local execution (in-memory/disk) and scales to Google Cloud (Vertex AI) for production.
- **Automated Explainability**: Generates detailed justifications for every "Approve" or "Reject" decision.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "credit decision ADK"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For Windows users, it is recommended to install PyTorch separately if needed, or stick to the default FastEmbed backend.*

3. **Environment Setup**:
   Create a `.env` file with your API keys:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

## 🏃 Usage

### 📊 Load Historical Data
To populate the vector store with historical loan data:
```bash
python load_dataset.py
```

### 📋 Screen an Application
To run a test screening using example data:
```bash
python orchestrator.py --test
```

### 🎯 Interactive Mode
To interactively evaluate applications:
```bash
python orchestrator.py --interactive
```

## 🏗️ Architecture
The system follows a sequential multi-agent pipeline:
1. **Ingestion Agent**: Validates and cleans raw application data.
2. **Chunking Agent**: Breaks data into semantic dimensions (Income Stability, Credit Behavior, etc.).
3. **Retrieval Agent**: Queries **Qdrant** for similar historical cases based on these dimensions.
4. **Risk Agent**: Analyzes similarities and calculates a multi-dimensional risk score.
5. **Decision Agent**: Makes the final Approved/Rejected recommendation.
6. **Explanation Agent**: Synthesizes the findings into a natural language report.

## 📄 License
MIT
