import os
import json
import numpy as np

# Import all core components
from chunker import DocumentChunker
from embedder import Embedder
from graph_builder import GraphBuilder
from retrieval.bm25_index import BM25Retriever
from retrieval.vector_search import VectorSearch
from retrieval.graph_retriever import GraphRetriever
from retrieval.fusion import ResultFusion
from retrieval.reranker import Reranker
from llm import LLMGenerator

def test_full_pipeline():
    print("=== Starting End-to-End Pipeline Test ===")
    
    # 1. Setup a dummy PDF if it doesn't exist
    test_pdf = "test_dummy.pdf"
    if not os.path.exists(test_pdf):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "GraphRAG is a novel retrieval architecture. It combines semantic search with graph traversal. This allows multi-hop reasoning. The system was developed in 2024.")
        doc.save(test_pdf)
        doc.close()
        print(f"Created dummy PDF: {test_pdf}")

    # --- INGESTION PHASE ---
    print("\n--- Ingestion Phase ---")
    
    # Chunking
    chunker = DocumentChunker(chunk_size_words=10, overlap_words=2)
    chunks = chunker.process_pdf(test_pdf)
    chunker.save_chunks("graph/test_chunk_store.json")
    print(f"Created {len(chunks)} chunks.")
    
    # Embedding
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(chunks)
    embedder.save_embeddings(embeddings, "embeddings/test_embeddings.npy")
    print("Embeddings generated.")
    
    # Graph Building
    gb = GraphBuilder(similarity_threshold=0.3, max_edges_per_node=3)
    gb.build_graph(chunks, embeddings)
    gb.save_graph("graph/test_graph.pkl")
    print("Graph built.")
    
    # BM25 Indexing
    bm25 = BM25Retriever()
    bm25.build_bm25_index(chunks)
    bm25.save_index("graph/test_bm25_index.pkl")
    print("BM25 indexed.")
    
    # Vector Indexing
    vs = VectorSearch()
    vs.build_faiss_index(chunks, embeddings)
    vs.save_index("embeddings/test_faiss.index", "embeddings/test_faiss_chunks.pkl")
    print("Vector index built.")
    
    # --- RETRIEVAL PHASE ---
    print("\n--- Retrieval Phase ---")
    query = "When was the architecture developed?"
    
    q_emb = embedder.generate_query_embedding(query)
    
    bm25_res = bm25.bm25_search(query, top_k=2)
    vec_res = vs.vector_search(q_emb, top_k=2)
    
    gr = GraphRetriever(gb)
    seeds = [res[0] for res in vec_res[:1]] if vec_res else []
    graph_res = gr.graph_based_retrieval(seeds, max_depth=1)
    
    print(f"BM25 hits: {len(bm25_res)}, Vector hits: {len(vec_res)}, Graph hits: {len(graph_res)}")
    
    fusion = ResultFusion()
    fused_res = fusion.fuse_results(bm25_res, vec_res, graph_res, top_k=5)
    print(f"Fused candidates: {len(fused_res)}")
    
    reranker = Reranker()
    final_res = reranker.rerank(query, fused_res, top_k=2)
    
    print("Top retrieved chunks:")
    for i, (chunk, score) in enumerate(final_res):
        print(f" [{i+1}] {chunk['text']} (Score: {score:.4f})")
        
    # --- LLM PHASE ---
    print("\n--- LLM Generation Phase ---")
    # Use dummy key for test so we don't hit API limits/errors if key is missing in environment
    os.environ["OPENAI_API_KEY"] = "dummy"
    llm = LLMGenerator()
    
    # We won't actually call the client to save money/prevent errors with dummy key, 
    # but we will test the context building which is the core logic.
    context = llm.build_context(final_res)
    print("Generated Context for LLM:")
    print(context)
    
    print("=== End-to-End Pipeline Test Complete ===")

if __name__ == "__main__":
    test_full_pipeline()
