import os
import json
import numpy as np
from chunker import DocumentChunker
from embedder import Embedder
from graph_builder import GraphBuilder

def run_phase1_test():
    print("=== Starting Phase 1 Integration Test ===")
    
    # 1. Chunker
    print("\n--- Testing Document Chunker ---")
    chunker = DocumentChunker(chunk_size_words=20, overlap_words=5) # Small chunks for fast test
    
    # We use the dummy pdf created earlier
    test_pdf = "test_dummy.pdf"
    if not os.path.exists(test_pdf):
        print(f"Error: {test_pdf} not found. Did you run the dummy creation script?")
        return
        
    chunks = chunker.process_pdf(test_pdf)
    print(f"Generated {len(chunks)} chunks.")
    chunker.save_chunks("graph/test_chunk_store.json")
    
    # 2. Embedder
    print("\n--- Testing Embedder ---")
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(chunks)
    embedder.save_embeddings(embeddings, "embeddings/test_embeddings.npy")
    print(f"Embeddings generated with shape: {embeddings.shape}")
    
    # 3. Graph Builder
    print("\n--- Testing Graph Builder ---")
    gb = GraphBuilder(similarity_threshold=0.8, max_edges_per_node=3)
    graph = gb.build_graph(chunks, embeddings)
    gb.save_graph("graph/test_graph.pkl")
    
    # Test traversals
    if chunks:
        seed_id = chunks[0]["chunk_id"]
        print(f"\nTesting traversals from seed node: {seed_id}")
        
        bfs_nodes = gb.bfs_traversal(seed_id, max_depth=1)
        print(f"BFS 1-hop nodes found: {len(bfs_nodes)}")
        
        dfs_nodes = gb.dfs_traversal(seed_id, max_depth=2)
        print(f"DFS depth-2 nodes found: {len(dfs_nodes)}")
        
        multi_hop = gb.multi_hop_retrieval([seed_id], max_depth=1, use_bfs=True)
        print(f"Multi-hop (BFS) nodes found: {len(multi_hop)}")

    print("\n=== Phase 1 Integration Test Complete ===")

if __name__ == "__main__":
    run_phase1_test()
