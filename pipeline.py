"""
pipeline.py — End-to-end Phase 2 hybrid retrieval orchestrator

This module orchestrates the complete retrieval pipeline:
1. Lexical Search (BM25)
2. Semantic Search (FAISS)
3. Graph Expansion (BFS/DFS)
4. Result Fusion (weighted combination)
5. Reranking (Cross-Encoder)

Usage:
    from pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from embedder import Embedder
from graph_builder import GraphBuilder
from retrieval.bm25_index import BM25Retriever
from retrieval.vector_search import VectorSearch
from retrieval.graph_retriever import GraphRetriever
from retrieval.fusion import ResultFusion
from retrieval.reranker import Reranker


class RetrievalPipeline:
    """
    RetrievalPipeline orchestrates the complete hybrid retrieval pipeline.
    
    The Pipeline Flow:
    1. Load all indices and models
    2. Execute parallel retrieval (BM25 + Vector)
    3. Expand context using graph traversal
    4. Fuse results using weighted combination
    5. Rerank top candidates using Cross-Encoder
    """
    
    def __init__(
        self,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.5,
        graph_weight: float = 0.2,
        graph_depth: int = 1,
        use_bfs: bool = True
    ):
        """
        Initialize the retrieval pipeline.
        
        Args:
            bm25_weight: Weight for BM25 results in fusion (default: 0.3)
            vector_weight: Weight for vector results in fusion (default: 0.5)
            graph_weight: Weight for graph results in fusion (default: 0.2)
            graph_depth: Depth for graph traversal (default: 1)
            use_bfs: Use BFS for graph traversal (True) or DFS (False)
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.graph_depth = graph_depth
        self.use_bfs = use_bfs
        
        # Components (loaded on demand)
        self.chunks: Optional[List[Dict[str, Any]]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.embedder: Optional[Embedder] = None
        self.graph_builder: Optional[GraphBuilder] = None
        self.bm25: Optional[BM25Retriever] = None
        self.vector_search: Optional[VectorSearch] = None
        self.graph_retriever: Optional[GraphRetriever] = None
        self.fusion: Optional[ResultFusion] = None
        self.reranker: Optional[Reranker] = None
        
        self._loaded = False
    
    def load(self):
        """Load all indices and models into memory."""
        if self._loaded:
            return
            
        print("Loading retrieval pipeline components...")
        
        # Load data
        try:
            with open("graph/chunk_store.json", "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.embeddings = np.load("embeddings/embeddings.npy")
        except Exception as e:
            raise FileNotFoundError(f"Data not found. Please run the ingestion pipeline first. ({e})")
        
        # Initialize components
        self.embedder = Embedder()
        
        self.graph_builder = GraphBuilder()
        try:
            self.graph_builder.load_graph()
        except Exception as e:
            raise FileNotFoundError(f"Graph not found. Run ingestion first. ({e})")
        
        self.bm25 = BM25Retriever()
        try:
            self.bm25.load_index()
        except Exception as e:
            raise FileNotFoundError(f"BM25 index not found. Run ingestion first. ({e})")
        
        self.vector_search = VectorSearch()
        try:
            self.vector_search.load_index()
        except Exception as e:
            raise FileNotFoundError(f"FAISS index not found. Run ingestion first. ({e})")
        
        self.graph_retriever = GraphRetriever(self.graph_builder)
        self.fusion = ResultFusion(
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight
        )
        self.reranker = Reranker()
        
        self._loaded = True
        print("Retrieval pipeline loaded successfully.")
    
    def retrieve(
        self,
        query: str,
        top_k_initial: int = 10,
        top_k_final: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute the complete retrieval pipeline.
        
        Args:
            query: User's question
            top_k_initial: Number of candidates to retrieve from each method
            top_k_final: Number of final results after reranking
            
        Returns:
            List of (chunk_dict, reranker_score) tuples
        """
        if not self._loaded:
            self.load()
        
        # Step 1: Lexical Search (BM25)
        bm25_results = self.bm25.bm25_search(query, top_k=top_k_initial)
        
        # Step 2: Semantic Search (FAISS)
        query_embedding = self.embedder.generate_query_embedding(query)
        vector_results = self.vector_search.vector_search(query_embedding, top_k=top_k_initial)
        
        # Step 3: Graph Expansion
        graph_results = []
        if self.graph_depth > 0 and vector_results:
            # Use top 3 vector hits as seeds
            seeds = [res[0] for res in vector_results[:3]]
            graph_results = self.graph_retriever.graph_based_retrieval(
                seeds,
                max_depth=self.graph_depth,
                use_bfs=self.use_bfs
            )
        
        # Step 4: Fusion
        fused_results = self.fusion.fuse_results(
            bm25_results,
            vector_results,
            graph_results,
            top_k=20
        )
        
        # Step 5: Reranking
        final_results = self.reranker.rerank(query, fused_results, top_k=top_k_final)
        
        return final_results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with system statistics
        """
        if not self._loaded:
            self.load()
        
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1],
            "graph_nodes": self.graph_builder.graph.number_of_nodes(),
            "graph_edges": self.graph_builder.graph.number_of_edges(),
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "graph_depth": self.graph_depth,
            "use_bfs": self.use_bfs
        }


if __name__ == "__main__":
    # Example usage
    pipeline = RetrievalPipeline()
    pipeline.load()
    
    query = "What are the key features of the project?"
    results = pipeline.retrieve(query, top_k_initial=10, top_k_final=5)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} Results:")
    for i, (chunk, score) in enumerate(results):
        print(f"\n[{i+1}] Score: {score:.4f}")
        print(f"Source: {chunk.get('source_filename')} (Page {chunk.get('page_number')})")
        print(f"Text: {chunk.get('text')[:200]}...")
