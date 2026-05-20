# Adaptive GraphRAG

A robust, multi-hop Retrieval-Augmented Generation (RAG) pipeline that transcends traditional vector search by building a **Semantic Knowledge Graph** over educational PDF documents. 

By combining exact-keyword matching (BM25), dense semantic search (FAISS), and graph-based neighbourhood traversal (BFS/DFS), the system excels at retrieving broad, contextually rich information for complex reasoning tasks. All candidate results are then jointly scored via a Cross-Encoder Transformer for maximum precision.

## 🚀 Features

* **Global Sliding-Window Chunking**: Extracts and cleans PDF text, maintaining semantic continuity across page boundaries.
* **Semantic Graph Construction**: Computes pairwise cosine similarity across chunks to build a fully connected, traversable semantic network.
* **Hybrid Retrieval System**:
  * **BM25**: Probabilistic term-frequency indexing for exact keyword precision.
  * **FAISS (Vector Search)**: `all-MiniLM-L6-v2` embeddings for semantic similarity and paraphrase matching.
  * **Graph Traversal**: BFS/DFS expansion to retrieve topically adjacent context (multi-hop reasoning).
* **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to jointly score query and chunk pairs for state-of-the-art relevance ranking.
* **Empirical Complexity Benchmarking**: Includes tools to prove $O(N^2)$ graph construction scaling and $O(b^d)$ traversal timings.

---

## 🛠️ Project Structure

```text
.
├── data/                         # Place your raw PDF documents here
├── embeddings/                   # Cached dense vectors and FAISS index
├── graph/                        # Cached chunk JSON, pickled NetworkX graph, BM25 index
├── retrieval/                    # Phase 2 retrieval modules
│   ├── bm25_index.py             # BM25 keyword search
│   ├── fusion.py                 # Multi-method result deduplication & boosting
│   ├── graph_retriever.py        # BFS/DFS context expansion
│   ├── reranker.py               # Cross-Encoder scoring
│   └── vector_search.py          # FAISS semantic search
├── chunker.py                    # Phase 1: PDF extraction and window chunking
├── embedder.py                   # Phase 1: SentenceTransformer dense embedding
├── graph_builder.py              # Phase 1: Cosine similarity semantic graph creation
├── pipeline.py                   # End-to-end Phase 2 hybrid retrieval orchestrator
├── complexity_analysis.py        # DAA benchmark timings and complexity reports
├── generate_test_data.py         # Utility to generate synthetic test PDFs
└── requirements.txt              # Project dependencies
```

---

## 💻 Installation

1. **Clone the repository** (or navigate to your working directory).
2. **Ensure Python 3.10+ is installed**.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or install manually if requirements.txt is missing:
   pip install pymupdf sentence-transformers networkx scikit-learn numpy faiss-cpu rank-bm25
   ```

---

## 🏃 How to Run

The pipeline must be executed sequentially. Phase 1 processes the documents and builds the data structures. Phase 2 queries those structures.

### Step 0: Test Data (Optional)
If you do not have real PDFs to test with, you can generate clean, synthetic, text-rich PDFs (Machine Learning, Graph Algorithms, NLP) by running:
```bash
python generate_test_data.py
```

### Step 1: Phase 1 — Data Ingestion & Graph Construction
Run these scripts in exact order to build the chunk store, the embedding matrix, and the semantic graph.
```bash
python chunker.py
python embedder.py
python graph_builder.py
```
*(Note: Initial embedding and model downloads may take a few minutes. Subsequent runs use cached disk artifacts).*

### Step 2: Phase 2 — Hybrid Retrieval Pipeline
Once Phase 1 artifacts exist, use the orchestrator script to run the hybrid retrieval pipeline (BM25 + FAISS + Graph + Fusion + Reranking).

**Run the default test queries:**
```bash
python pipeline.py
```

**Run an interactive, custom query:**
```bash
python pipeline.py "What is dynamic programming?"
```

---

## 📊 Complexity Benchmarking

This project includes a dedicated benchmarking script for Design and Analysis of Algorithms (DAA) validation. It empirically measures:
1. BFS vs DFS traversal timings across different depths.
2. The $O(N^2)$ scaling behavior of the pairwise cosine similarity matrix.

To run the benchmarks (ensure Phase 1 is completed first):
```bash
python complexity_analysis.py
```

## 🧠 Architecture Details

* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, fast CPU inference)
* **Reranking Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Joint query-document self-attention)
* **Similarity Threshold**: Calibrated to `0.50` for educational slide/PDF content to preserve graph connectivity without introducing severe noise.
