# Adaptive GraphRAG: Phase 1 Mentor Notes
## Core Graph Engine Fundamentals

Welcome to the foundation of the Adaptive GraphRAG system! As your mentor, I want to ensure you deeply understand *why* we are doing things, not just *how*. We are building a system that doesn't just retrieve text, but understands relationships between pieces of text.

### 1. Document Chunking (`chunker.py`)

**The Concept:**
LLMs and embedding models have strict token limits (context windows). If we feed a whole 100-page PDF into an embedding model, it will fail or truncate the text, losing massive amounts of information. We solve this by breaking the document into "chunks".

**Why Overlapping Chunks?**
Imagine chunking purely by word count (e.g., 200 words). What if word 200 is the start of a crucial sentence explaining a concept, and word 201 is the end of it? 
* Chunk A has the setup.
* Chunk B has the punchline.
If a user queries for the punchline, Chunk B is retrieved, but it lacks the setup context. 
By introducing an *overlap* (e.g., 50 words), Chunk B starts 50 words *before* Chunk A ended. This ensures that boundary concepts are captured fully in at least one chunk.

**Data Structure:**
We store chunks as a list of dictionaries. Each dict contains a `chunk_id` (UUID), `text`, and metadata (`source_filename`, `page_number`). The UUID is crucial because it acts as the Primary Key for our upcoming Graph nodes and Vector DB entries.

---

### 2. Embeddings (`embedder.py`)

**The Concept:**
How do we make a computer understand that "The canine barked" and "The dog woofed" mean the same thing, even though they share no common keywords? 
*Embeddings*.

An embedding model (like `all-MiniLM-L6-v2`) is a neural network trained to map sentences into a high-dimensional continuous vector space. 
* "Dog" might map to coordinates `[0.1, 0.5, -0.2, ...]` in a 384-dimensional space.
* "Puppy" will map to coordinates very close to "Dog".
* "Car" will map to coordinates far away.

**Why all-MiniLM-L6-v2?**
In systems engineering, we constantly balance performance (latency) and accuracy. `all-MiniLM-L6-v2` maps text to a 384-dimensional vector. Larger models (like OpenAI's `text-embedding-ada-002`) use 1536 dimensions. 
* 384 dimensions means faster distance calculations (important for our O(N^2) graph building) and less memory usage.
* It still retains excellent semantic representation for most retrieval tasks.

**Batch Processing:**
When generating embeddings, we pass `batch_size=32`. Pushing data through a neural network has overhead. Doing it one sentence at a time is like carrying groceries from your car to your house one apple at a time. Batching uses matrix multiplication to process many sentences simultaneously, massively speeding up execution, especially on GPUs.

---

### 3. Graph Construction & Traversal (`graph_builder.py`)

**The Concept:**
Standard Vector DBs are "flat". You query, and it returns the top-K closest vectors. 
But what if the answer requires connecting dots? 
* Chunk A: "Company X acquired Company Y."
* Chunk B: "Company Y created Product Z."
If I ask "Who owns Product Z?", a standard vector search might only find Chunk B, missing the ownership link in Chunk A.

A Semantic Graph solves this. The nodes are chunks. The edges represent high semantic similarity between chunks.

**Cosine Similarity Mathematics:**
To draw edges, we calculate how close two vectors are. We use Cosine Similarity:
`cos(θ) = (A · B) / (||A|| * ||B||)`
Where:
* `A · B` is the dot product (sum of pairwise multiplications).
* `||A||` is the L2 norm (magnitude or length) of the vector.
This measures the *angle* between the vectors, ignoring their magnitude. An angle of 0 degrees means a cosine of 1.0 (perfectly similar).

**Time Complexity of Graph Building:**
Calculating the similarity of every chunk against every other chunk requires an all-pairs comparison.
* Time Complexity: `O(N^2 * D)`, where `N` is the number of chunks and `D` is the embedding dimension.
* *Tradeoff Note:* For 1,000 chunks, this is 1,000,000 operations—very fast. For 1 million chunks, this is 1 trillion operations—too slow. In an enterprise system, we would use an Approximate Nearest Neighbor (ANN) index like FAISS to find nearest neighbors in `O(N log N)` time instead of `O(N^2)`.

**Traversals:**
Once the graph is built, how do we find related context?
1.  **Breadth-First Search (BFS):** Explores all immediate neighbors (1-hop) before moving to 2-hop neighbors. Uses a `Queue` data structure. 
    *   *Why use it?* Excellent for exploring immediate context. If Chunk A is relevant, BFS ensures we grab the chunks most similar to A first.
    *   *Complexity:* `O(V + E)` where V is vertices, E is edges.
2.  **Depth-First Search (DFS):** Explores a path as deeply as possible before backtracking. Uses the Call Stack (recursion).
    *   *Why use it?* Good for following a chain of reasoning across highly connected concepts.

By implementing BFS and DFS, we enable multi-hop retrieval, allowing the system to pull in context that the initial vector search might have missed.