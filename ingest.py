"""
ingest.py — Run this ONCE before starting the app.

Usage:
    python ingest.py path/to/your_document.pdf

This script:
1. Chunks the PDF into overlapping text pieces
2. Generates embeddings (semantic vectors)
3. Builds the FAISS vector index
4. Builds the BM25 keyword index
5. Builds the semantic knowledge graph

All outputs are saved to graph/ and embeddings/ folders.
"""

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/your_document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Ingesting: {pdf_path}")
    print(f"{'='*50}\n")

    # Step 1: Chunk the PDF
    print("[1/5] Chunking PDF...")
    from chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size_words=200, overlap_words=50)
    chunks = chunker.process_pdf(pdf_path)
    chunker.save_chunks("graph/chunk_store.json")
    print(f"      ✓ {len(chunks)} chunks created\n")

    # Step 2: Generate embeddings
    print("[2/5] Generating embeddings (this may take a minute)...")
    from embedder import Embedder
    import numpy as np
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(chunks)
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/embeddings.npy", embeddings)
    print(f"      ✓ Embeddings shape: {embeddings.shape}\n")

    # Step 3: Build FAISS vector index
    print("[3/5] Building FAISS vector index...")
    from retrieval.vector_search import VectorSearch
    vs = VectorSearch()
    vs.build_faiss_index(chunks, embeddings)
    vs.save_index()
    print("      ✓ FAISS index saved\n")

    # Step 4: Build BM25 keyword index
    print("[4/5] Building BM25 keyword index...")
    from retrieval.bm25_index import BM25Retriever
    bm25 = BM25Retriever()
    bm25.build_bm25_index(chunks)
    bm25.save_index("graph/bm25_index.pkl")
    print("      ✓ BM25 index saved\n")

    # Step 5: Build semantic knowledge graph
    print("[5/5] Building knowledge graph (may take a moment for large docs)...")
    from graph_builder import GraphBuilder
    gb = GraphBuilder()
    gb.build_graph(chunks, embeddings)
    gb.save_graph()
    print("      ✓ Graph saved\n")

    print(f"{'='*50}")
    print("✅ Ingestion complete! You can now run: streamlit run app.py")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()