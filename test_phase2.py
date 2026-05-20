import json
import numpy as np
from embedder import Embedder
from graph_builder import GraphBuilder
from bm25_index import BM25Retriever
from vector_search import VectorSearch
from graph_retriever import GraphRetriever
from fusion import ResultFusion
from reranker import Reranker

def run_phase2_test():
    print("=== Starting Phase 2 Integration Test ===")
    
    # 1. Load Data
    print("\n--- Loading Data ---")
    with open("graph/test_chunk_store.json", "r") as f:
        chunks = json.load(f)
    embeddings = np.load("embeddings/test_embeddings.npy")
    
    embedder = Embedder() # Used for query embedding
    gb = GraphBuilder()
    gb.load_graph("graph/test_graph.pkl")
    
    query = "test document"
    
    # 2. BM25 Search
    print("\n--- Testing BM25 Search ---")
    bm25 = BM25Retriever()
    bm25.build_bm25_index(chunks)
    bm25_res = bm25.bm25_search(query, top_k=2)
    print(f"BM25 found {len(bm25_res)} chunks.")
    
    # 3. Vector Search
    print("\n--- Testing Vector Search ---")
    vs = VectorSearch()
    vs.build_faiss_index(chunks, embeddings)
    q_emb = embedder.generate_query_embedding(query)
    vec_res = vs.vector_search(q_emb, top_k=2)
    print(f"Vector search found {len(vec_res)} chunks.")
    
    # 4. Graph Search
    print("\n--- Testing Graph Retrieval ---")
    gr = GraphRetriever(gb)
    # Use top vector result as seed
    seeds = [vec_res[0][0]] if vec_res else []
    graph_res = gr.graph_based_retrieval(seeds, max_depth=1)
    print(f"Graph retrieval expanded {len(graph_res)} chunks.")
    
    # 5. Fusion
    print("\n--- Testing Result Fusion ---")
    fusion = ResultFusion()
    fused_res = fusion.fuse_results(bm25_res, vec_res, graph_res, top_k=5)
    print(f"Fusion yielded {len(fused_res)} unique chunks.")
    
    # 6. Reranking
    print("\n--- Testing Reranker ---")
    reranker = Reranker()
    final_res = reranker.rerank(query, fused_res, top_k=2)
    print(f"Reranker returned {len(final_res)} top chunks.")
    for i, (chunk, score) in enumerate(final_res):
        print(f"  {i+1}. Score: {score:.4f} | ID: {chunk['chunk_id']}")

    print("\n=== Phase 2 Integration Test Complete ===")

if __name__ == "__main__":
    run_phase2_test()
