# Adaptive GraphRAG: Phase 3 Mentor Notes
## System Integration & LLMs

Welcome to the final phase! We are now tying together our complex retrieval pipeline with a Large Language Model and a user interface. 

### 1. LLM Integration and Grounding (`llm.py`)

**The Concept:**
LLMs are powerful, but they hallucinate (make things up). In an enterprise RAG system, hallucination is unacceptable. If a user asks "What is our company's refund policy?", the LLM must answer using *our* policy, not a generic one it learned during training.

**Prompt Engineering for Grounding:**
We use a specific technique called "Grounding". We pass a System Prompt that strictly instructs the LLM:
1.  **Strict Boundary:** "Answer the user's question based ONLY on the provided Context."
2.  **Safe Fallback:** "If the answer is not contained in the Context, say 'I cannot answer this'."
3.  **Attribution:** "Cite the source number."

This dramatically reduces hallucinations. If the retrieval pipeline fails to find the right chunk, the system safely says "I don't know" instead of lying.

**Token Limits and Reranking:**
Why did we build the Cross-Encoder in Phase 2? Because LLMs have context windows, and pricing is often per-token. We can't pass 100 chunks to the LLM. The Reranker ensures that the 5 chunks we *do* pass to the LLM (which cost money and token space) are the absolute highest quality.

---

### 2. Frontend Visualization (`app.py`)

**The Concept:**
A RAG system is a black box to a user. They ask a question, they get an answer. But how do they trust it?

**Explainability:**
We use Streamlit to provide an interactive UI that opens the black box:
1.  **Source Citations:** We display the exact text chunks used to generate the answer, along with their source document name, page number, and retrieval score. 
2.  **Graph Visualization (`pyvis`):** We render an interactive graph of the semantic neighborhood. The user can visually see *why* a chunk was retrieved. If Node A (directly matched the query) is connected to Node B, the user sees that connection line, proving that the system understood the relational context.

### 3. System Architecture & Scalability Considerations

As a Senior AI Engineer, you must always think about what happens when the data grows from 1 PDF to 10,000 PDFs.

*   **Ingestion Bottleneck:** Generating embeddings for 1M chunks takes time. *Solution:* Asynchronous task queues (Celery/RabbitMQ) and horizontal scaling of embedding workers.
*   **Vector Search Scaling:** `IndexFlatL2` works for a few thousand chunks. For millions, you must use HNSW or IVF indices in FAISS, or a dedicated vector database (Pinecone, Milvus, Weaviate).
*   **Graph Scaling:** NetworkX is an in-memory graph. It will crash on a 10M node graph. *Solution:* Migrate to a dedicated Graph Database like Neo4j or Amazon Neptune.
*   **Real-time Updates:** Currently, our system rebuilds the graph from scratch. In production, you need dynamic insertion (updating FAISS and adding nodes/edges to the graph incrementally).

Congratulations! You have built the core engine of an Adaptive GraphRAG system. You understand the math behind embeddings, the complexity of graph traversals, and the architecture of hybrid retrieval.