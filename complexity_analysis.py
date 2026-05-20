"""
complexity_analysis.py — DAA benchmark timings and complexity reports

This module provides tools for analyzing the computational complexity
and performance of the RAG system components.

Usage:
    from complexity_analysis import ComplexityAnalyzer
    analyzer = ComplexityAnalyzer()
    report = analyzer.analyze_pipeline()
"""

import time
import json
import numpy as np
from typing import Dict, Any, List, Callable, Optional
from functools import wraps

from pipeline import RetrievalPipeline


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


class ComplexityAnalyzer:
    """
    ComplexityAnalyzer benchmarks and reports on the computational complexity
    of various RAG system components.
    
    Complexity Analysis:
    - Chunking: O(N * L) where N is document length, L is chunk size
    - Embedding: O(N * D) where N is chunks, D is embedding dimension
    - FAISS Index Build: O(N * D) for IndexFlatIP
    - FAISS Search: O(N * D) for exact search, O(log N) for approximate
    - BM25 Index Build: O(N * L) where N is chunks, L is avg chunk length
    - BM25 Search: O(N * L) where N is chunks, L is query length
    - Graph Build: O(N^2 * D) for all-pairs similarity
    - Graph Traversal: O(V + E) where V is vertices, E is edges
    - Fusion: O(K) where K is number of candidates
    - Reranking: O(K * D) where K is candidates, D is model complexity
    """
    
    def __init__(self):
        self.pipeline: Optional[RetrievalPipeline] = None
        self.benchmark_results: Dict[str, Any] = {}
    
    def load_pipeline(self):
        """Load the retrieval pipeline for benchmarking."""
        self.pipeline = RetrievalPipeline()
        self.pipeline.load()
    
    @timing_decorator
    def benchmark_chunking(self, pdf_path: str) -> Dict[str, Any]:
        """Benchmark the chunking process."""
        from chunker import DocumentChunker
        
        chunker = DocumentChunker(chunk_size_words=200, overlap_words=50)
        chunks = chunker.process_pdf(pdf_path)
        
        return {
            "num_chunks": len(chunks),
            "avg_chunk_length": np.mean([len(c["text"].split()) for c in chunks])
        }
    
    @timing_decorator
    def benchmark_embedding_generation(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark embedding generation."""
        from embedder import Embedder
        
        embedder = Embedder()
        embeddings = embedder.generate_embeddings(chunks)
        
        return {
            "embedding_shape": embeddings.shape,
            "embedding_dimension": embeddings.shape[1]
        }
    
    @timing_decorator
    def benchmark_faiss_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        """Benchmark FAISS index building."""
        from retrieval.vector_search import VectorSearch
        
        vs = VectorSearch()
        vs.build_faiss_index(chunks, embeddings)
        
        return {
            "num_vectors": vs.index.ntotal
        }
    
    @timing_decorator
    def benchmark_faiss_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark FAISS search."""
        if not self.pipeline:
            self.load_pipeline()
        
        query_embedding = self.pipeline.embedder.generate_query_embedding(query)
        results = self.pipeline.vector_search.vector_search(query_embedding, top_k=top_k)
        
        return {
            "num_results": len(results),
            "top_score": results[0][1] if results else 0
        }
    
    @timing_decorator
    def benchmark_bm25_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark BM25 index building."""
        from retrieval.bm25_index import BM25Retriever
        
        bm25 = BM25Retriever()
        bm25.build_bm25_index(chunks)
        
        return {
            "num_chunks": len(chunks)
        }
    
    @timing_decorator
    def benchmark_bm25_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark BM25 search."""
        if not self.pipeline:
            self.load_pipeline()
        
        results = self.pipeline.bm25.bm25_search(query, top_k=top_k)
        
        return {
            "num_results": len(results),
            "top_score": results[0][1] if results else 0
        }
    
    @timing_decorator
    def benchmark_graph_build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        """Benchmark graph building."""
        from graph_builder import GraphBuilder
        
        gb = GraphBuilder()
        gb.build_graph(chunks, embeddings)
        
        return {
            "num_nodes": gb.graph.number_of_nodes(),
            "num_edges": gb.graph.number_of_edges()
        }
    
    @timing_decorator
    def benchmark_graph_traversal(self, query: str, max_depth: int = 1) -> Dict[str, Any]:
        """Benchmark graph traversal."""
        if not self.pipeline:
            self.load_pipeline()
        
        query_embedding = self.pipeline.embedder.generate_query_embedding(query)
        vector_results = self.pipeline.vector_search.vector_search(query_embedding, top_k=3)
        seeds = [res[0] for res in vector_results]
        
        results = self.pipeline.graph_retriever.graph_based_retrieval(
            seeds,
            max_depth=max_depth,
            use_bfs=True
        )
        
        return {
            "num_results": len(results),
            "max_depth": max_depth
        }
    
    @timing_decorator
    def benchmark_fusion(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Benchmark result fusion."""
        if not self.pipeline:
            self.load_pipeline()
        
        bm25_results = self.pipeline.bm25.bm25_search(query, top_k=top_k)
        query_embedding = self.pipeline.embedder.generate_query_embedding(query)
        vector_results = self.pipeline.vector_search.vector_search(query_embedding, top_k=top_k)
        
        fused = self.pipeline.fusion.fuse_results(bm25_results, vector_results, [], top_k=top_k)
        
        return {
            "num_results": len(fused)
        }
    
    @timing_decorator
    def benchmark_reranking(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """Benchmark reranking."""
        if not self.pipeline:
            self.load_pipeline()
        
        bm25_results = self.pipeline.bm25.bm25_search(query, top_k=top_k)
        query_embedding = self.pipeline.embedder.generate_query_embedding(query)
        vector_results = self.pipeline.vector_search.vector_search(query_embedding, top_k=top_k)
        
        fused = self.pipeline.fusion.fuse_results(bm25_results, vector_results, [], top_k=top_k)
        reranked = self.pipeline.reranker.rerank(query, fused, top_k=5)
        
        return {
            "num_candidates": len(fused),
            "num_final": len(reranked)
        }
    
    @timing_decorator
    def benchmark_full_pipeline(self, query: str, top_k_initial: int = 10, top_k_final: int = 5) -> Dict[str, Any]:
        """Benchmark the full retrieval pipeline."""
        if not self.pipeline:
            self.load_pipeline()
        
        results = self.pipeline.retrieve(query, top_k_initial=top_k_initial, top_k_final=top_k_final)
        
        return {
            "num_results": len(results),
            "top_score": results[0][1] if results else 0
        }
    
    def run_full_benchmark(self, pdf_path: str, queries: List[str]) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark of the entire system.
        
        Args:
            pdf_path: Path to PDF document for ingestion benchmarks
            queries: List of queries for retrieval benchmarks
            
        Returns:
            Dictionary with all benchmark results
        """
        print("Running full complexity analysis...")
        
        results = {
            "ingestion": {},
            "retrieval": {},
            "system_info": self.pipeline.get_retrieval_stats() if self.pipeline else {}
        }
        
        # Ingestion benchmarks
        print("\n--- Ingestion Benchmarks ---")
        
        _, chunking_time = self.benchmark_chunking(pdf_path)
        results["ingestion"]["chunking_time"] = chunking_time
        
        # Load chunks for subsequent benchmarks
        from chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size_words=200, overlap_words=50)
        chunks = chunker.process_pdf(pdf_path)
        
        _, embedding_time = self.benchmark_embedding_generation(chunks)
        results["ingestion"]["embedding_time"] = embedding_time
        
        from embedder import Embedder
        embedder = Embedder()
        embeddings = embedder.generate_embeddings(chunks)
        
        _, faiss_index_time = self.benchmark_faiss_index(chunks, embeddings)
        results["ingestion"]["faiss_index_time"] = faiss_index_time
        
        _, bm25_index_time = self.benchmark_bm25_index(chunks)
        results["ingestion"]["bm25_index_time"] = bm25_index_time
        
        _, graph_build_time = self.benchmark_graph_build(chunks, embeddings)
        results["ingestion"]["graph_build_time"] = graph_build_time
        
        # Retrieval benchmarks
        print("\n--- Retrieval Benchmarks ---")
        self.load_pipeline()
        
        retrieval_times = []
        for query in queries:
            print(f"  Benchmarking query: {query[:50]}...")
            
            _, faiss_search_time = self.benchmark_faiss_search(query)
            _, bm25_search_time = self.benchmark_bm25_search(query)
            _, graph_traversal_time = self.benchmark_graph_traversal(query)
            _, fusion_time = self.benchmark_fusion(query)
            _, reranking_time = self.benchmark_reranking(query)
            _, pipeline_time = self.benchmark_full_pipeline(query)
            
            retrieval_times.append({
                "query": query,
                "faiss_search_time": faiss_search_time,
                "bm25_search_time": bm25_search_time,
                "graph_traversal_time": graph_traversal_time,
                "fusion_time": fusion_time,
                "reranking_time": reranking_time,
                "pipeline_time": pipeline_time
            })
        
        results["retrieval"]["query_benchmarks"] = retrieval_times
        
        # Calculate averages
        avg_faiss = np.mean([t["faiss_search_time"] for t in retrieval_times])
        avg_bm25 = np.mean([t["bm25_search_time"] for t in retrieval_times])
        avg_graph = np.mean([t["graph_traversal_time"] for t in retrieval_times])
        avg_fusion = np.mean([t["fusion_time"] for t in retrieval_times])
        avg_rerank = np.mean([t["reranking_time"] for t in retrieval_times])
        avg_pipeline = np.mean([t["pipeline_time"] for t in retrieval_times])
        
        results["retrieval"]["averages"] = {
            "avg_faiss_search_time": avg_faiss,
            "avg_bm25_search_time": avg_bm25,
            "avg_graph_traversal_time": avg_graph,
            "avg_fusion_time": avg_fusion,
            "avg_reranking_time": avg_rerank,
            "avg_pipeline_time": avg_pipeline
        }
        
        self.benchmark_results = results
        return results
    
    def generate_report(self, output_path: str = "complexity_report.json"):
        """Generate and save a complexity analysis report."""
        if not self.benchmark_results:
            raise ValueError("No benchmark results available. Run run_full_benchmark first.")
        
        with open(output_path, "w") as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        print(f"\nComplexity report saved to {output_path}")
        
        # Print summary
        print("\n--- Complexity Summary ---")
        print("Ingestion Times:")
        for key, value in self.benchmark_results["ingestion"].items():
            print(f"  {key}: {value:.4f}s")
        
        print("\nRetrieval Times (Average):")
        for key, value in self.benchmark_results["retrieval"]["averages"].items():
            print(f"  {key}: {value:.4f}s")
        
        print("\nSystem Info:")
        for key, value in self.benchmark_results["system_info"].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    # Example usage
    analyzer = ComplexityAnalyzer()
    
    # Run benchmark (requires a PDF and queries)
    pdf_path = "data/test.pdf"
    queries = [
        "What are the key features of the project?",
        "How does the retrieval system work?",
        "What is the purpose of the graph?"
    ]
    
    try:
        results = analyzer.run_full_benchmark(pdf_path, queries)
        analyzer.generate_report()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        print("Make sure to run ingestion first and have test data available.")
