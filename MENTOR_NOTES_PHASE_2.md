# Adaptive GraphRAG: Phase 2 Mentor Notes
## Hybrid Retrieval Pipeline

Phase 2 moves from building our foundation (chunks, vectors, graph) to actually querying it. A robust retrieval pipeline cannot rely on just one method. Here is why we use a hybrid approach.

### 1. BM25 (Sparse Retrieval)

**The Concept:**
Vector embeddings are amazing at understanding semantics. But what if a user asks: "What is error code `ERR-00512-B`?"
An embedding model might not have seen that specific string during training. It might map it to a generic "error code" vector, losing the exact identifier.

BM25 is a sparse retrieval algorithm based on exact keyword matching.
*   **Sparse:** It uses vocabulary vectors where 99% of the entries are zero.
*   **Math:** It improves upon TF-IDF (Term Frequency-Inverse Document Frequency) by adding term saturation (a word appearing 10 times isn't 10x better than appearing once) and document length normalization (a short document with the keyword is better than a huge document with the keyword).

By combining Vector Search and BM25, we catch both semantic intent and exact lexical matches.

---

### 2. FAISS (Vector DB)

**The Concept:**
In Phase 1, we calculated Cosine Similarity using `np.dot` over an $N \times N$ matrix. That is an $O(N^2)$ operation. During query time, computing the distance of the query against $N$ chunks is $O(N)$.
For 100 chunks, $O(N)$ is instantaneous. For 10 million chunks, your users will be waiting 5 minutes for an answer.

**FAISS:**
Facebook AI Similarity Search (FAISS) builds specialized data structures to make vector search blazingly fast.
While we are using `IndexFlatIP` (Exact Search, $O(N)$) for our small dataset, FAISS allows swapping to HNSW (Hierarchical Navigable Small World) or IVF (Inverted File Index) to perform Approximate Nearest Neighbor (ANN) search.
ANN searches drop the time complexity to $O(\log N)$ by sacrificing a tiny bit of accuracy, which is required for scaling to millions of chunks.

---

### 3. Graph Retrieval

**The Concept:**
When BM25 and FAISS return the top chunks, they only return what was directly asked for.
If Chunk A mentions "Project Apollo" and Chunk B (which mentions "Project Apollo was successful") is retrieved, we might miss Chunk C which says "The mission was led by Neil Armstrong" if it doesn't explicitly mention "Project Apollo" but is linked to Chunk B.

Our `graph_retriever.py` takes the chunks found by BM25 and FAISS (the "seeds") and runs a BFS traversal on our semantic graph. It pulls in the neighbors of these chunks. This is how we achieve multi-hop reasoning.

---

### 4. Fusion and Reranking

**The Problem:**
We now have three streams of results:
1. BM25 (Scores might be 5.2, 12.8, etc.)
2. FAISS (Scores are Cosine Similarities: 0.1 to 1.0)
3. Graph (Heuristic depth scores: 0.9, 0.81)

We cannot just add `12.8 + 0.8 + 0.9`. BM25 would overpower everything.

**The Solution:**
1.  **Normalization (`fusion.py`):** We use Min-Max normalization within each stream to force all scores to a `[0, 1]` scale.
2.  **Linear Combination:** We apply weights (e.g., Vector=0.5, BM25=0.3, Graph=0.2) and sum them up. We also deduplicate chunks found by multiple streams.
3.  **Cross-Encoder Reranking (`reranker.py`):** The final step. The fusion step gives us a top 50 list. We pass these 50 chunks to a Cross-Encoder.
    *   *Bi-Encoder (FAISS):* Encodes Query -> Vector. Encodes Doc -> Vector. Compares Vectors. Very fast.
    *   *Cross-Encoder:* Passes `(Query, Document)` together through the Transformer. The attention heads look at how every word in the query relates to every word in the document simultaneously.
    *   *Tradeoff:* Cross-Encoders are extremely accurate but very slow. That's why we only run them on the top 50 candidates, filtering down to the absolute best 5 chunks for the LLM context.