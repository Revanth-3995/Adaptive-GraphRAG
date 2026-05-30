# 🕸️ Adaptive GraphRAG

A hybrid retrieval-augmented generation system that combines multiple retrieval strategies for accurate and contextually relevant question answering from PDF documents.

## ✨ Features

- **Hybrid Retrieval**: Combines BM25 (lexical), FAISS (semantic), and Knowledge Graph (context) retrieval
- **Semantic Graph**: Builds knowledge graphs from document chunks for multi-hop context expansion
- **Cross-Encoder Reranking**: Refines results using deep attention-based relevance scoring
- **Result Fusion**: Intelligent weighted combination of multiple retrieval methods
- **Interactive UI**: Streamlit-based web interface for easy querying
- **Scalable Architecture**: Modular design supporting optimization and scaling
- **Complexity Analysis**: Built-in benchmarking and performance profiling tools

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────────────┐
│         Retrieval Pipeline               │
│  ┌────────┐  ┌────────┐  ┌────────┐   │
│  │  BM25  │  │ FAISS  │  │ Graph  │   │
│  │Lexical │  │Semantic│  │Context │   │
│  └───┬────┘  └───┬────┘  └───┬────┘   │
│      └───────────┼───────────┘          │
│                  ▼                       │
│           Result Fusion                  │
│                  ▼                       │
│            Reranker                      │
│         (Cross-Encoder)                  │
└──────────────────┬──────────────────────┘
                   ↓
              LLM Generator
                   ↓
              Final Answer
```

## 📁 Project Structure

```
.
├── data/                         # Raw PDF documents
├── embeddings/                   # Dense vectors and FAISS index
├── graph/                        # Graph and BM25 data
├── retrieval/                    # Phase 2 retrieval modules
│   ├── bm25_index.py             # BM25 keyword search
│   ├── fusion.py                 # Multi-method result fusion
│   ├── graph_retriever.py        # BFS/DFS context expansion
│   ├── reranker.py               # Cross-Encoder scoring
│   └── vector_search.py          # FAISS semantic search
├── chunker.py                    # PDF extraction and chunking
├── embedder.py                   # SentenceTransformer embeddings
├── graph_builder.py              # Semantic graph creation
├── pipeline.py                   # End-to-end retrieval orchestrator
├── complexity_analysis.py        # Benchmarking and complexity analysis
├── generate_test_data.py         # Synthetic test PDF generation
├── ingest.py                     # Ingestion pipeline entry point
├── app.py                        # Streamlit web interface
├── llm.py                        # LLM answer generation
└── requirements.txt              # Project dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd RAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API key**:
```bash
# Windows
set GROQ_API_KEY=your_api_key_here

# Linux/Mac
export GROQ_API_KEY=your_api_key_here
```

## 📖 Usage

### Step 1: Prepare Your Data

Place your PDF documents in the `data/` directory:
```bash
data/
├── document1.pdf
├── document2.pdf
└── ...
```

Or generate synthetic test data:
```bash
python generate_test_data.py
```

### Step 2: Ingest Documents

Run the ingestion pipeline (one-time per document):
```bash
python ingest.py data/your_document.pdf
```

This process:
- Chunks the PDF into overlapping text pieces
- Generates semantic embeddings
- Builds FAISS vector index
- Builds BM25 keyword index
- Creates semantic knowledge graph

**Output**:
- `graph/chunk_store.json` - Document chunks
- `embeddings/embeddings.npy` - Embedding vectors
- `embeddings/faiss.index` - FAISS index
- `graph/bm25_index.pkl` - BM25 index
- `graph/graph.pkl` - Knowledge graph

### Step 3: Run the Application

Launch the web interface:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Step 4: Query Your Documents

1. Enter your question in the text input
2. Adjust retrieval parameters (optional):
   - Initial Top-K: Number of candidates from each method
   - Graph Depth: Depth for graph traversal
   - Final Top-K: Number of final results
3. Click "Search"
4. View the generated answer and retrieved context

## 💻 Programmatic Usage

```python
from pipeline import RetrievalPipeline

# Initialize pipeline
pipeline = RetrievalPipeline(
    bm25_weight=0.3,
    vector_weight=0.5,
    graph_weight=0.2,
    graph_depth=1,
    use_bfs=True
)

# Load components
pipeline.load()

# Retrieve relevant chunks
results = pipeline.retrieve(
    query="What are the key features of the project?",
    top_k_initial=10,
    top_k_final=5
)

# View results
for chunk, score in results:
    print(f"Score: {score:.4f}")
    print(f"Source: {chunk['source_filename']} (Page {chunk['page_number']})")
    print(f"Text: {chunk['text'][:200]}...\n")
```

## 🔧 Configuration

### Retrieval Weights

Adjust the importance of each retrieval method in `pipeline.py` or via the UI:

- **BM25 Weight**: 0.3 (keyword matching)
- **Vector Weight**: 0.5 (semantic matching)
- **Graph Weight**: 0.2 (context expansion)

### Chunking Parameters

Modify in `ingest.py`:

```python
chunker = DocumentChunker(
    chunk_size_words=200,  # Words per chunk
    overlap_words=50       # Overlap between chunks
)
```

### Graph Threshold

Adjust in `graph_builder.py`:

```python
# Edge creation threshold (cosine similarity)
SIMILARITY_THRESHOLD = 0.7
```

## 📊 Performance Analysis

Run complexity benchmarks:

```bash
python complexity_analysis.py
```

This generates a detailed report with:
- Ingestion timing (chunking, embedding, indexing)
- Retrieval timing (BM25, FAISS, graph, fusion, reranking)
- System statistics (chunk count, graph size, etc.)

Output: `complexity_report.json`

## 🧪 Testing

Generate synthetic test documents:

```bash
# Single test PDF
python generate_test_data.py

# Multiple test PDFs
python -c "
from generate_test_data import TestDataGenerator
generator = TestDataGenerator()
generator.generate_multiple_test_pdfs('data', num_pdfs=3)
"
```

## 📚 How It Works

### Ingestion Pipeline

1. **Chunking**: Split PDF into overlapping text chunks
2. **Embedding**: Convert chunks to 384-dimensional vectors
3. **FAISS Index**: Build efficient vector similarity index
4. **BM25 Index**: Build keyword-based search index
5. **Graph Building**: Create semantic knowledge graph

### Retrieval Pipeline

1. **BM25 Search**: Find exact keyword matches
2. **Vector Search**: Find semantically similar chunks
3. **Graph Traversal**: Expand context via graph neighbors
4. **Result Fusion**: Combine and normalize scores
5. **Reranking**: Refine with Cross-Encoder for precision

### Scoring

- **BM25 Score**: Keyword relevance (unbounded)
- **Vector Score**: Semantic similarity (-1 to 1)
- **Graph Score**: Proximity to seeds (0 to 0.9)
- **Fusion Score**: Weighted combination (0 to 1)
- **Reranker Score**: Deep attention relevance (0 to 1)

## 🎯 Use Cases

- **Document Q&A**: Answer questions from technical documentation
- **Research Assistance**: Find relevant information across papers
- **Knowledge Base**: Build searchable knowledge repositories
- **Legal/Medical**: Retrieve precise information from domain documents
- **Technical Support**: Find solutions in documentation

## 🔬 Technical Details

### Models

- **Embedding**: `all-MiniLM-L6-v2` (384 dimensions)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: Groq API (LLaMA models)

### Algorithms

- **BM25**: Best Matching 25 ranking function
- **FAISS**: Facebook AI Similarity Search (IndexFlatIP)
- **Graph**: NetworkX with cosine similarity edges
- **Fusion**: Min-Max normalization with weighted sum

### Complexity

- **Ingestion**: O(N²) for graph building (dominant factor)
- **Retrieval**: O(N) for BM25/FAISS, O(V+E) for graph
- **Reranking**: O(K) where K is candidates (typically 20)

## 🛠️ Development

### Adding New Retrieval Methods

1. Create new module in `retrieval/`
2. Implement search interface
3. Add to `fusion.py` weights
4. Update `pipeline.py` to include new method

### Modifying Graph Construction

Edit `graph_builder.py`:
- Adjust similarity threshold
- Change edge weight calculation
- Add node attributes

### Custom LLM Integration

Edit `llm.py`:
- Replace Groq API with your provider
- Adjust prompt template
- Modify response parsing

## 📝 Documentation

For detailed technical documentation, see:
- [SYSTEM_REPORT.md](SYSTEM_REPORT.md) - Comprehensive technical report
- Code comments - Inline documentation
- Docstrings - Function/class documentation

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **Sentence Transformers**: For embedding models
- **FAISS**: For efficient vector search
- **Rank BM25**: For keyword retrieval
- **NetworkX**: For graph operations
- **Streamlit**: For the web interface
- **Groq**: For LLM API

## 🐛 Troubleshooting

### Common Issues

**Import Error**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**API Key Error**: Set GROQ_API_KEY environment variable
```bash
set GROQ_API_KEY=your_key
```

**FAISS Error**: Ensure embeddings.npy exists (run ingestion first)
```bash
python ingest.py data/your_document.pdf
```

**Memory Error**: Reduce chunk size or use smaller documents
```python
chunker = DocumentChunker(chunk_size_words=100)  # Smaller chunks
```

## 📈 Performance Tips

1. **Use GPU**: Enable GPU for FAISS if available
2. **Approximate Search**: Use HNSW instead of IndexFlatIP for large datasets
3. **Limit Graph Depth**: Reduce graph traversal depth for faster queries
4. **Cache Queries**: Implement caching for frequent queries
5. **Batch Processing**: Process multiple documents in batches

## 🔮 Future Enhancements

- [ ] Approximate FAISS indices (HNSW, IVF)
- [ ] Query expansion and understanding
- [ ] Dynamic weight adjustment
- [ ] Multi-document support
- [ ] Real-time index updates
- [ ] Evaluation metrics (precision, recall)
- [ ] User feedback integration
- [ ] Multi-modal support (images, tables)

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review code comments

---

**Built with ❤️ for intelligent document retrieval**
