# Adaptive GraphRAG System - Comprehensive Technical Report

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Phase 1: Ingestion Pipeline](#phase-1-ingestion-pipeline)
5. [Phase 2: Retrieval Pipeline](#phase-2-retrieval-pipeline)
6. [Scoring Mechanisms](#scoring-mechanisms)
7. [Data Flow](#data-flow)
8. [Component Details](#component-details)
9. [Usage Guide](#usage-guide)
10. [Performance Characteristics](#performance-characteristics)

---

## System Overview

Adaptive GraphRAG is a hybrid retrieval-augmented generation system that combines multiple retrieval strategies to provide accurate and contextually relevant answers from PDF documents. The system uses a multi-stage approach:

- **Lexical Retrieval**: BM25 keyword matching for exact term matches
- **Semantic Retrieval**: FAISS vector search for concept-based matching
- **Graph Retrieval**: Knowledge graph traversal for multi-hop context expansion
- **Result Fusion**: Weighted combination of all retrieval methods
- **Reranking**: Cross-Encoder refinement for final precision

This hybrid approach addresses limitations of individual methods and provides robust retrieval across different query types.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   BM25       │  │   FAISS      │  │   Graph      │          │
│  │  (Lexical)   │  │  (Semantic)  │  │  (Context)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           ▼                                     │
│                    Result Fusion                                │
│                    (Weighted Sum)                                │
│                           ▼                                     │
│                    Reranker                                     │
│                 (Cross-Encoder)                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Generator                               │
│                    (Answer Generation)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Answer                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── data/                         # Raw PDF documents
├── embeddings/                   # Dense vectors and FAISS index
│   ├── embeddings.npy           # NumPy array of embeddings
│   ├── faiss.index              # FAISS vector index
│   └── faiss_chunks.pkl         # Chunk metadata for FAISS
├── graph/                        # Graph and BM25 data
│   ├── chunk_store.json         # All chunks with metadata
│   ├── graph.pkl                # NetworkX knowledge graph
│   └── bm25_index.pkl           # BM25 index and chunks
├── retrieval/                    # Phase 2 retrieval modules
│   ├── bm25_index.py            # BM25 keyword search
│   ├── fusion.py                # Multi-method result fusion
│   ├── graph_retriever.py       # BFS/DFS context expansion
│   ├── reranker.py              # Cross-Encoder scoring
│   └── vector_search.py        # FAISS semantic search
├── chunker.py                    # PDF extraction and chunking
├── embedder.py                   # SentenceTransformer embeddings
├── graph_builder.py              # Semantic graph creation
├── pipeline.py                   # End-to-end retrieval orchestrator
├── complexity_analysis.py        # Benchmarking and complexity analysis
├── generate_test_data.py         # Synthetic test PDF generation
├── ingest.py                     # Ingestion pipeline entry point
├── app.py                        # Streamlit web interface
├── llm.py                        # LLM answer generation
└── requirements.txt              # Dependencies
```

---

## Phase 1: Ingestion Pipeline

The ingestion pipeline processes raw PDF documents into indexed structures for efficient retrieval. This is a one-time setup process per document.

### Step 1: Document Chunking (`chunker.py`)

**Purpose**: Split PDF documents into overlapping text chunks for processing.

**Process**:
1. Extract text from PDF using PyPDF2
2. Split text into chunks based on word count (default: 200 words)
3. Add overlap between chunks (default: 50 words) to maintain context
4. Track metadata: source filename, page number, chunk ID

**Algorithm**:
```
Input: PDF document
Parameters: chunk_size=200, overlap=50

For each page in PDF:
    Extract text
    Split into words
    Create sliding window chunks:
        Chunk 1: words[0:200]
        Chunk 2: words[150:350]  # 50 word overlap
        Chunk 3: words[300:500]
        ...

Output: List of chunk dictionaries with metadata
```

**Complexity**: O(N) where N is total word count

**Output**: `graph/chunk_store.json` - JSON array of all chunks

### Step 2: Embedding Generation (`embedder.py`)

**Purpose**: Convert text chunks into dense vector representations.

**Process**:
1. Load SentenceTransformer model (`all-MiniLM-L6-v2`)
2. Generate 384-dimensional embeddings for each chunk
3. Store embeddings as NumPy array

**Model**: `all-MiniLM-L6-v2`
- Dimension: 384
- Type: Bi-Encoder (processes query and document separately)
- Speed: Fast, suitable for large-scale retrieval

**Algorithm**:
```
Input: List of chunks
Model: all-MiniLM-L6-v2

For each chunk:
    embedding = model.encode(chunk["text"])
    Store embedding in array

Output: embeddings.npy (shape: [num_chunks, 384])
```

**Complexity**: O(N × D) where N is chunks, D is embedding dimension

**Output**: `embeddings/embeddings.npy` - NumPy array of embeddings

### Step 3: FAISS Index Building (`retrieval/vector_search.py`)

**Purpose**: Build efficient vector similarity search index.

**Process**:
1. L2-normalize all embedding vectors
2. Build FAISS IndexFlatIP (Inner Product index)
3. Add normalized vectors to index
4. Save index and chunk mappings

**Why FAISS?**
- Efficient similarity search and clustering
- Supports both exact and approximate search
- Scales to millions of vectors
- IndexFlatIP computes inner product (equivalent to cosine similarity for normalized vectors)

**Algorithm**:
```
Input: embeddings array, chunks
Index Type: IndexFlatIP (Exact Search)

Normalize vectors:
    vectors = vectors / ||vectors||

Build index:
    index = IndexFlatIP(384)
    index.add(normalized_vectors)

Save:
    faiss.write_index(index, "faiss.index")
    pickle.dump(chunks, "faiss_chunks.pkl")
```

**Complexity**: O(N × D) for index build

**Output**: `embeddings/faiss.index`, `embeddings/faiss_chunks.pkl`

### Step 4: BM25 Index Building (`retrieval/bm25_index.py`)

**Purpose**: Build keyword-based search index for exact term matching.

**Process**:
1. Tokenize all chunks (lowercase, remove punctuation)
2. Build BM25Okapi index from tokenized corpus
3. Save index and chunk references

**What is BM25?**
- Evolution of TF-IDF (Term Frequency-Inverse Document Frequency)
- Adds term frequency saturation (diminishing returns for repeated terms)
- Normalizes by document length
- Excellent for exact keyword matches, acronyms, names

**Algorithm**:
```
Input: chunks
For each chunk:
    tokens = lowercase(chunk.text)
    tokens = remove_punctuation(tokens)
    tokens = split(tokens)

Build BM25 index:
    bm25 = BM25Okapi(tokenized_corpus)

Save:
    pickle.dump({bm25, chunks}, "bm25_index.pkl")
```

**Complexity**: O(N × L) where N is chunks, L is average chunk length

**Output**: `graph/bm25_index.pkl`

### Step 5: Knowledge Graph Building (`graph_builder.py`)

**Purpose**: Create semantic knowledge graph linking related chunks.

**Process**:
1. Compute pairwise cosine similarity between all chunks
2. Create edges between chunks with similarity > threshold (0.7)
3. Store as NetworkX graph with node metadata

**Graph Structure**:
- **Nodes**: Document chunks
- **Edges**: Semantic similarity relationships
- **Node Attributes**: text, source, page, chunk_id
- **Edge Weights**: Cosine similarity scores

**Algorithm**:
```
Input: chunks, embeddings
Threshold: 0.7

Compute similarity matrix:
    similarities = cosine_similarity(embeddings, embeddings)

Build graph:
    For each pair (i, j):
        if similarities[i,j] > 0.7:
            add_edge(i, j, weight=similarities[i,j])

Add node metadata:
    For each chunk:
        add_node(chunk_id, attr=chunk_metadata)

Save:
    nx.write_gpickle(graph, "graph.pkl")
```

**Complexity**: O(N² × D) for all-pairs similarity

**Output**: `graph/graph.pkl`

---

## Phase 2: Retrieval Pipeline

The retrieval pipeline executes at query time to find relevant document chunks.

### Step 1: Lexical Search (BM25)

**File**: `retrieval/bm25_index.py`

**Process**:
1. Tokenize query (same preprocessing as documents)
2. Compute BM25 scores for all chunks
3. Return top-k chunks with highest scores

**Scoring**:
```
BM25 score = IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl))

Where:
- qi: query term
- f(qi,D): frequency of term in document
- |D|: document length
- avgdl: average document length
- k1, b: tuning parameters (default: k1=1.5, b=0.75)
- IDF: inverse document frequency
```

**Strengths**: Exact keyword matches, acronyms, names
**Weaknesses**: Misses semantic relationships

**Output**: List of (chunk, bm25_score) tuples

### Step 2: Semantic Search (FAISS)

**File**: `retrieval/vector_search.py`

**Process**:
1. Generate query embedding using same model
2. L2-normalize query embedding
3. Search FAISS index for nearest neighbors
4. Return top-k chunks with highest cosine similarity

**Scoring**:
```
Cosine Similarity = (A · B) / (||A|| × ||B||)

Since vectors are L2-normalized:
Cosine Similarity = A · B (dot product)
```

**Range**: -1.0 to 1.0 (typically 0 to 1 for text)

**Strengths**: Concept-based matching, synonym understanding
**Weaknesses**: May miss exact keyword matches

**Output**: List of (chunk, cosine_similarity) tuples

### Step 3: Graph Expansion

**File**: `retrieval/graph_retriever.py`

**Process**:
1. Use top vector search results as seed nodes
2. Traverse graph from seeds using BFS/DFS
3. Collect neighboring chunks as additional context
4. Score based on traversal depth

**Traversal Algorithm**:
```
Input: seed_chunks, max_depth, use_bfs

Initialize:
    queue = [(seed_id, 0)]
    visited = {seed_ids}

While queue not empty:
    current_id, depth = queue.pop()
    
    If depth > 0 and current_id not visited:
        Add to results with score = 0.9^depth
        visited.add(current_id)
    
    If depth < max_depth:
        For each neighbor of current_id:
            If neighbor not visited:
                queue.append((neighbor, depth + 1))

Output: List of (chunk, graph_score) tuples
```

**Scoring**:
- Depth 1: 0.9
- Depth 2: 0.81
- Depth 3: 0.729
- Formula: `score = 0.9^depth`

**Strengths**: Multi-hop context, related concepts
**Weaknesses**: Dependent on graph quality

**Output**: List of (chunk, graph_score) tuples

### Step 4: Result Fusion

**File**: `retrieval/fusion.py`

**Purpose**: Combine results from all retrieval methods with normalized scores.

**Process**:
1. Min-Max normalize scores within each retrieval stream
2. Apply weights to each stream
3. Deduplicate chunks (same chunk from multiple sources)
4. Combine weighted scores
5. Sort by combined score

**Normalization**:
```
Min-Max Normalization:
normalized_score = (score - min_score) / (max_score - min_score)

Result: All scores in range [0, 1]
```

**Fusion Formula**:
```
final_score = (bm25_norm × 0.3) + 
              (vector_norm × 0.5) + 
              (graph_norm × 0.2)
```

**Weights**:
- BM25: 0.3 (keyword matching)
- Vector: 0.5 (semantic matching - highest weight)
- Graph: 0.2 (context expansion)

**Deduplication**:
- Chunks identified by `chunk_id`
- If chunk appears in multiple streams, scores are summed
- Prevents duplicate results in final output

**Output**: List of (chunk, fused_score) tuples, sorted by score

### Step 5: Reranking

**File**: `retrieval/reranker.py`

**Purpose**: Refine top candidates using Cross-Encoder for higher precision.

**Process**:
1. Take top-k fused results (e.g., top 20)
2. Pass (query, document) pairs through Cross-Encoder
3. Get relevance scores for each pair
4. Re-sort by Cross-Encoder scores
5. Return top-k final results (e.g., top 5)

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Type: Cross-Encoder (processes query and document together)
- Advantage: Deep attention between query and document words
- Disadvantage: Slow, can't process all documents

**Bi-Encoder vs Cross-Encoder**:
```
Bi-Encoder (Embedder):
    Query → [Model] → Query Vector
    Document → [Model] → Document Vector
    Similarity = cosine(query_vec, doc_vec)
    Speed: O(1) with index
    Accuracy: Good

Cross-Encoder (Reranker):
    [Query, Document] → [Model] → Relevance Score
    Speed: O(N) where N is candidates
    Accuracy: Excellent
```

**Pipeline Rationale**:
1. Fast retrieval (BM25 + FAISS + Graph) → 50 candidates
2. Reranker (Cross-Encoder) → 5 final results
3. Combines speed of Bi-Encoder with accuracy of Cross-Encoder

**Output**: List of (chunk, cross_encoder_score) tuples

---

## Scoring Mechanisms

### BM25 Score
- **Range**: Can be > 1.0 (unbounded)
- **Meaning**: Keyword relevance
- **Threshold**: > 0 (filters no matches)
- **Formula**: TF-IDF with saturation and length normalization

### Vector/FAISS Score
- **Range**: -1.0 to 1.0 (typically 0 to 1)
- **Meaning**: Semantic similarity
- **Threshold**: None (returns top-k)
- **Formula**: Cosine similarity of normalized embeddings

### Graph Score
- **Range**: 0 to 0.9
- **Meaning**: Proximity to seed nodes
- **Threshold**: None
- **Formula**: `0.9^depth` (depth-based heuristic)

### Fusion Score
- **Range**: 0 to 1
- **Meaning**: Combined relevance across methods
- **Threshold**: None
- **Formula**: Weighted sum of normalized scores

### Reranker Score (Final)
- **Range**: Typically 0 to 1 (can be negative)
- **Meaning**: Deep attention-based relevance
- **Threshold**: None (returns top-k)
- **Formula**: Cross-Encoder prediction

---

## Data Flow

### Ingestion Flow
```
PDF Document
    ↓
Chunker (text extraction + chunking)
    ↓
chunk_store.json (chunks with metadata)
    ↓
Embedder (SentenceTransformer)
    ↓
embeddings.npy (384-dim vectors)
    ↓
├─→ FAISS Index Builder
│   ↓
│   faiss.index + faiss_chunks.pkl
│
├─→ BM25 Index Builder
│   ↓
│   bm25_index.pkl
│
└─→ Graph Builder
    ↓
    graph.pkl (NetworkX graph)
```

### Retrieval Flow
```
User Query
    ↓
├─→ BM25 Search → bm25_results (keyword matches)
├─→ Vector Search → vector_results (semantic matches)
└─→ Graph Traversal → graph_results (context expansion)
    ↓
Result Fusion (normalize + weight + deduplicate)
    ↓
fused_results (top 20 candidates)
    ↓
Reranker (Cross-Encoder)
    ↓
final_results (top 5 chunks)
    ↓
LLM Generator (answer generation)
    ↓
Final Answer
```

---

## Component Details

### chunker.py
**Class**: `DocumentChunker`
**Methods**:
- `process_pdf(pdf_path)`: Extract and chunk PDF
- `save_chunks(output_path)`: Save chunks to JSON

**Parameters**:
- `chunk_size_words`: Words per chunk (default: 200)
- `overlap_words`: Overlap between chunks (default: 50)

### embedder.py
**Class**: `Embedder`
**Methods**:
- `generate_embeddings(chunks)`: Generate embeddings for chunks
- `generate_query_embedding(query)`: Generate embedding for query

**Model**: `all-MiniLM-L6-v2` (384 dimensions)

### retrieval/vector_search.py
**Class**: `VectorSearch`
**Methods**:
- `build_faiss_index(chunks, embeddings)`: Build FAISS index
- `vector_search(query_embedding, top_k)`: Search similar vectors
- `save_index()`: Save index to disk
- `load_index()`: Load index from disk

**Index Type**: `IndexFlatIP` (exact search)

### retrieval/bm25_index.py
**Class**: `BM25Retriever`
**Methods**:
- `build_bm25_index(chunks)`: Build BM25 index
- `bm25_search(query, top_k)`: Search by keywords
- `save_index()`: Save index to disk
- `load_index()`: Load index from disk

**Algorithm**: BM25Okapi

### retrieval/graph_retriever.py
**Class**: `GraphRetriever`
**Methods**:
- `graph_based_retrieval(seed_chunks, max_depth, use_bfs)`: Traverse graph

**Traversal**: BFS or DFS with depth-based scoring

### retrieval/fusion.py
**Class**: `ResultFusion`
**Methods**:
- `fuse_results(bm25_results, vector_results, graph_results, top_k)`: Combine results

**Normalization**: Min-Max
**Weights**: BM25=0.3, Vector=0.5, Graph=0.2

### retrieval/reranker.py
**Class**: `Reranker`
**Methods**:
- `rerank(query, candidates, top_k)`: Re-rank candidates

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### graph_builder.py
**Class**: `GraphBuilder`
**Methods**:
- `build_graph(chunks, embeddings)`: Build semantic graph
- `save_graph()`: Save graph to disk
- `load_graph()`: Load graph from disk

**Threshold**: 0.7 cosine similarity for edge creation

### pipeline.py
**Class**: `RetrievalPipeline`
**Methods**:
- `load()`: Load all components
- `retrieve(query, top_k_initial, top_k_final)`: Execute full pipeline
- `get_retrieval_stats()`: Get system statistics

**Purpose**: Orchestrates entire retrieval process

### llm.py
**Class**: `LLMGenerator`
**Methods**:
- `generate_answer(query, retrieved_chunks)`: Generate answer using LLM

**Provider**: Groq API (uses LLaMA models)

---

## Usage Guide

### Initial Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set API Key**:
```bash
set GROQ_API_KEY=your_api_key_here
```

### Ingestion (One-time per document)

```bash
python ingest.py data/your_document.pdf
```

**Output**:
- `graph/chunk_store.json` - Chunks
- `embeddings/embeddings.npy` - Embeddings
- `embeddings/faiss.index` - FAISS index
- `embeddings/faiss_chunks.pkl` - FAISS chunks
- `graph/bm25_index.pkl` - BM25 index
- `graph/graph.pkl` - Knowledge graph

### Running the Application

```bash
streamlit run app.py
```

**Interface**:
- Enter query in text input
- Adjust retrieval parameters (top-k, graph depth, etc.)
- View retrieved chunks with scores
- View local graph neighborhood
- Get generated answer

### Programmatic Usage

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

# Retrieve
results = pipeline.retrieve(
    query="What are the key features?",
    top_k_initial=10,
    top_k_final=5
)

# View results
for chunk, score in results:
    print(f"Score: {score:.4f}")
    print(f"Text: {chunk['text'][:200]}...")
```

### Generating Test Data

```bash
python generate_test_data.py
```

**Output**: Synthetic PDFs in `data/` directory

### Complexity Analysis

```bash
python complexity_analysis.py
```

**Output**: `complexity_report.json` with timing benchmarks

---

## Performance Characteristics

### Time Complexity

**Ingestion**:
- Chunking: O(N) where N is word count
- Embedding: O(N × D) where D is embedding dimension
- FAISS Build: O(N × D)
- BM25 Build: O(N × L) where L is avg chunk length
- Graph Build: O(N² × D) for all-pairs similarity

**Retrieval**:
- BM25 Search: O(N × L) where L is query length
- FAISS Search: O(N × D) for exact, O(log N) for approximate
- Graph Traversal: O(V + E) where V is vertices, E is edges
- Fusion: O(K) where K is candidates
- Reranking: O(K × M) where M is model complexity

### Space Complexity

**Storage**:
- Chunks: O(N × L) where L is avg chunk length
- Embeddings: O(N × D) where D is 384
- FAISS Index: O(N × D)
- BM25 Index: O(N × L)
- Graph: O(N²) in worst case (dense graph)

### Scalability

**Current Setup**:
- Suitable for: Hundreds to thousands of documents
- Embedding dimension: 384 (compact)
- Index type: Exact search (IndexFlatIP)

**Scaling Options**:
- For millions of chunks: Use `IndexIVFFlat` or HNSW (approximate search)
- For larger models: Increase embedding dimension
- For distributed: Use FAISS GPU indices or distributed search

### Bottlenecks

**Ingestion**:
- Graph building (O(N²)) - slow for large documents
- Embedding generation - compute intensive

**Retrieval**:
- Reranking - slow but only on top candidates
- Graph traversal - depends on graph density

**Optimizations**:
- Use approximate FAISS indices for large scale
- Limit graph traversal depth
- Cache query embeddings
- Use GPU for embedding generation

---

## Design Decisions

### Why Hybrid Retrieval?

**Single Method Limitations**:
- BM25 only: Misses semantic relationships
- Vector only: Misses exact keyword matches
- Graph only: Requires high-quality graph

**Hybrid Advantages**:
- BM25: Exact matches, acronyms, names
- Vector: Concept matching, synonyms
- Graph: Multi-hop context, related concepts
- Combined: Robust across query types

### Why Cross-Encoder Reranking?

**Trade-off**:
- Bi-Encoder: Fast but less accurate
- Cross-Encoder: Slow but more accurate

**Solution**:
- Use Bi-Encoder for initial retrieval (fast)
- Use Cross-Encoder for refinement (accurate)
- Two-stage pipeline balances speed and accuracy

### Why Graph Traversal?

**Problem**:
- Direct matches may miss related context
- Answer may require information from multiple sections

**Solution**:
- Graph links semantically related chunks
- Traversal finds multi-hop relationships
- Provides broader context for complex queries

### Why Min-Max Normalization?

**Problem**:
- Different scoring scales (BM25 > 10, FAISS -1 to 1, Graph 0 to 0.9)
- Direct addition would bias toward larger scales

**Solution**:
- Normalize each stream to [0, 1]
- Apply weights for relative importance
- Fair combination of all methods

---

## Future Enhancements

### Potential Improvements

1. **Approximate Search**: Use HNSW or IVF for FAISS
2. **Query Expansion**: Expand queries with related terms
3. **Dynamic Weighting**: Adjust weights based on query type
4. **Graph Pruning**: Remove weak edges for efficiency
5. **Caching**: Cache frequent queries
6. **Parallel Retrieval**: Run BM25 and FAISS in parallel
7. **More Rerankers**: Try different Cross-Encoder models
8. **Evaluation**: Add relevance metrics (precision, recall)
9. **User Feedback**: Learn from user interactions
10. **Multi-modal**: Support images, tables, code

### Advanced Features

1. **Query Understanding**: Classify query type (factual, conceptual, etc.)
2. **Answer Verification**: Check answer against retrieved context
3. **Citation Generation**: Provide source citations
4. **Multi-document**: Search across multiple PDFs
5. **Real-time Updates**: Incremental index updates
6. **Distributed Deployment**: Scale across multiple machines

---

## Conclusion

Adaptive GraphRAG implements a sophisticated hybrid retrieval system that combines the strengths of multiple approaches:

- **BM25** for exact keyword matching
- **FAISS** for semantic similarity
- **Knowledge Graph** for multi-hop context
- **Result Fusion** for balanced combination
- **Cross-Encoder Reranking** for precision

The system is designed to be:
- **Accurate**: Multiple retrieval methods + reranking
- **Efficient**: Fast retrieval with FAISS, selective reranking
- **Flexible**: Adjustable weights and parameters
- **Scalable**: Modular architecture supports optimization
- **Maintainable**: Clean separation of concerns

This architecture provides robust retrieval across diverse query types while maintaining reasonable performance characteristics.
