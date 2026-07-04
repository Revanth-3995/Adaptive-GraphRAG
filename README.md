# 🕸️ Adaptive GraphRAG

A hybrid retrieval-augmented generation (RAG) system that combines multiple retrieval strategies, query intent classification, and an advanced verification framework for accurate and contextually relevant question answering from PDF documents.

## ✨ Features

- **Hybrid Retrieval**: Combines BM25 (lexical), FAISS (semantic), and Knowledge Graph (context) retrieval.
- **Semantic Graph**: Builds knowledge graphs from document chunks using cosine similarity thresholds for multi-hop context expansion. Offers BFS, PPR (Personalized PageRank), Random Walk, and PPR+Random Walk hybrid traversal strategies.
- **Adaptive Retrieval Planning**: Classifies query intent into `SIMPLE`, `MODERATE`, `COMPLEX`, `ALGORITHM`, or `RESEARCH` to formulate a tailored retrieval plan.
- **Query Decomposition & HyDE**: Automatically decomposes complex queries into focused sub-questions for parallel execution, and utilizes HyDE (Hypothetical Document Embeddings) to improve vector retrieval.
- **Cross-Encoder Reranking**: Refines results using deep attention-based relevance scoring with intent-based keyword boosts.
- **Phase 4 Answer Intelligence & Trust Framework**:
  - **Claim Extraction**: Extracts factual claims from generated answers.
  - **Claim Verification**: Automatically verifies claims against retrieved chunks, calculating a **Grounding Score**, **Trust Level**, and **Hallucination Risk**.
  - **Citation Verification**: Automatically verifies and injects inline source citations, classifying them as valid or failed.
  - **Modes**: Choose between **FAST** (direct generation) and **VERIFIED** (full claim verification and trust framework) modes.
- **Performance Diagnostics & Logging**:
  - Global, thread-safe `PerformanceTracker` logging execution times, LLM calls, bottleneck identification, and verification overhead to `performance_logs/performance_trace.json`.
  - Detailed answer quality reports saved to `graph/answer_quality_report.json`.
- **Multi-Tenant / Multi-Workspace Support**: SQLite-based workspace and chat history database (`storage.py`) for maintaining isolated workspaces.
- **Multi-Provider LLM Manager**: Real-time provider health checking and compatibility with Groq, OpenAI, Gemini, and Claude APIs (`providers.py`).
- **Interactive UI**: Streamlit-based workspace dashboard featuring visual graph rendering, real-time performance bottleneck charts, chat history, and document management.

## 🏗️ Architecture

```
User Query
    ↓
┌────────────────────────────────────────────────────────┐
│               Retrieval Planner / Intent               │
│                        ↓                               │
│              Query Decomposition & HyDE                │
│                        ↓                               │
│  ┌──────────────────┬──────────────────┬────────────┐  │
│  │   BM25 Search    │  Vector (FAISS)  │ Graph Walk │  │
│  │     (Lexical)    │    (Semantic)    │ (Context)  │  │
│  └────────┬─────────┴────────┬─────────┴─────┬──────┘  │
│           └──────────────────┼───────────────┘         │
│                              ▼                         │
│                        Result Fusion                   │
│                              ▼                         │
│                          Reranker                      │
│                       (Cross-Encoder)                  │
└──────────────────────────────┬─────────────────────────┘
                               ↓
                         LLM Generator
                               ↓
┌────────────────────────────────────────────────────────┐
│             Answer Intelligence & Trust                │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Claim Extraction & Verification                 │  │
│  │  (Grounding Score, Trust Level, Hallucination)   │  │
│  ├──────────────────────────────────────────────────┤  │
│  │  Citation Verification & Formatting              │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────┬─────────────────────────┘
                               ↓
                          Final Answer
```

## 📁 Project Structure

```
.
├── embeddings/                   # Dense vectors and FAISS indices
├── graph/                        # Graph structures, BM25 indices, reports, and traces
├── retrieval/                    # Retrieval modules
│   ├── bm25_index.py             # BM25 keyword search
│   ├── fusion.py                 # Multi-method result fusion
│   ├── graph_retriever.py        # Graph traversal (BFS, PPR, Random Walk, etc.)
│   ├── hyde.py                   # Hypothetical Document Embeddings (HyDE) generator
│   ├── query_classifier.py       # Query intent classification
│   ├── query_decomposer.py       # Multi-query decomposition
│   ├── reranker.py               # Cross-Encoder scoring
│   └── vector_search.py          # FAISS semantic search
├── evaluation/                   # Evaluation & auditing scripts
│   ├── answer_evaluator.py       # Evaluation metrics
│   ├── test_answer_intelligence.py # Tests for Phase 4 answer intelligence
│   └── performance_audit.py      # Automated performance auditing tool
├── performance_logs/             # Execution and latency logs
├── chunker.py                    # PDF extraction and chunking
├── embedder.py                   # SentenceTransformer embeddings
├── graph_builder.py              # Semantic graph creation
├── pipeline.py                   # End-to-end retrieval orchestrator
├── complexity_analysis.py        # Benchmarking and complexity analysis
├── generate_test_data.py         # Synthetic test PDF generator
├── ingest.py                     # Ingestion pipeline entry point
├── app.py                        # Streamlit web interface and workspace dashboard
├── llm.py                        # LLM interface and generation helper
├── providers.py                  # Multi-provider LLM manager
├── storage.py                    # SQLite DB for workspaces, documents, and chat history
├── performance_tracker.py        # Thread-safe performance tracking system
├── citation_verifier.py          # Inline citation verification
├── claim_extractor.py            # Answer claim extraction
├── claim_verifier.py             # Claim verification against sources
└── requirements.txt              # Project dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Revanth-3995/Adaptive-GraphRAG.git
cd Adaptive-GraphRAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Environment**:
Create a `.env` file in the root directory:
```env
# Supported LLM Keys (Configure at least one)
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
CLAUDE_API_KEY=your_anthropic_api_key
```

## 📖 Usage

### Running the Dashboard UI

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

Inside the Dashboard:
1. **Workspace Management**: Create isolated workspaces or select an existing one.
2. **Document Ingestion**: Upload PDF documents in the sidebar. The UI will show real-time progress bars for chunking, embedding, indexing, and graph-building steps.
3. **Interactive Chat**: Query your workspace. Select between **FAST** and **VERIFIED** modes in the configuration settings.
4. **Answer Diagnostics**: Expand details to inspect the **Grounding Score**, **Trust Level**, **Hallucination Risk**, and detailed claims analysis.
5. **Performance Insights**: View a visual breakdown of execution latencies per stage (retrieval, generation, verification) to identify system bottlenecks.

### CLI Ingestion Pipeline

To ingest a document via the command line:
```bash
python ingest.py path/to/document.pdf
```

This constructs and persists:
- `graph/chunk_store.json` (Document chunks)
- `embeddings/embeddings.npy` (Dense vectors)
- `embeddings/faiss.index` (FAISS similarity index)
- `graph/bm25_index.pkl` (BM25 lexical index)
- `graph/graph.pkl` (Knowledge graph object)

### Running Performance Benchmarks

To analyze the complexity and latency of the system components:
```bash
python complexity_analysis.py
```
This output is saved to `graph/complexity_report.json`.

To run the automated performance audit:
```bash
python evaluation/performance_audit.py
```

## 🧪 Testing

To run the unit tests for the Answer Intelligence layer:
```bash
pytest evaluation/test_answer_intelligence.py
```

## 🔬 Technical Details

### Models & Libraries
- **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Graph Utility**: NetworkX with custom similarity-based edge construction
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Database**: SQLite for thread-safe metadata and chat persistence
- **GUI Components**: Streamlit for dashboard rendering, PyVis for interactive graph visualization

---
**Built with ❤️ for intelligent document retrieval**
