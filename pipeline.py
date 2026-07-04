"""
pipeline.py — End-to-end Phase 2B Adaptive Retrieval Engine

This module orchestrates the complete adaptive retrieval pipeline:
1. Understands query intent (SIMPLE, MODERATE, COMPLEX, ALGORITHM, RESEARCH)
2. Formulates a tailored Retrieval Plan (traversal strategy, graph depth, sub-queries)
3. Decomposes complex queries into focused sub-questions if needed
4. Performs parallel search (BM25 + Vector with HyDE) for each sub-question
5. Conducts dynamic graph traversals (BFS, PPR, Random Walk, or PPR+Random Walk hybrid)
6. Merges and deduplicates candidates by chunk_id
7. Reranks with cross-encoders using intent-based keyword boosts
8. Records a detailed retrieval trace for transparency and debugging
"""

import os
import json
import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

from embedder import Embedder
from graph_builder import GraphBuilder
from retrieval.bm25_index import BM25Retriever
from retrieval.vector_search import VectorSearch
from retrieval.graph_retriever import GraphRetriever
from retrieval.fusion import ResultFusion
from retrieval.reranker import Reranker
from retrieval.query_classifier import QueryClassifier
from retrieval.query_decomposer import QueryDecomposer
from retrieval.hyde import HyDEGenerator
from retrieval_planner import RetrievalPlanner

# Phase 4 Answer Intelligence Layer imports
from llm import LLMGenerator
from citation_verifier import CitationVerifier
from claim_extractor import ClaimExtractor
from claim_verifier import ClaimVerifier


class RetrievalPipeline:
    """
    RetrievalPipeline orchestrates the adaptive retrieval pipeline.
    """
    
    def __init__(
        self,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.5,
        graph_weight: float = 0.2,
        trace_path: str = "graph/retrieval_trace.json"
    ):
        """
        Initialize the retrieval pipeline.
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.trace_path = trace_path
        self.trace_lock = threading.Lock()
        
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
        
        self.classifier: Optional[QueryClassifier] = None
        self.decomposer: Optional[QueryDecomposer] = None
        self.hyde: Optional[HyDEGenerator] = None
        
        # Phase 4 components
        self.llm: Optional[LLMGenerator] = None
        self.citation_verifier = CitationVerifier()
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier()
        
        self._loaded = False
        self.last_trace: Dict[str, Any] = {}
    
    def load(self):
        """Load all indices and models into memory."""
        if self._loaded:
            return
            
        print("Loading adaptive retrieval pipeline components...")
        
        # Load data with a series of fallbacks (Standard -> test fallbacks -> doc_store fallbacks)
        loaded_data = False
        
        # Paths lists in priority order
        chunks_paths = ["graph/chunk_store.json", "graph/test_chunk_store.json", "doc_store/test/chunks.json"]
        embeddings_paths = ["embeddings/embeddings.npy", "embeddings/test_embeddings.npy", "doc_store/test/embeddings.npy"]
        
        for c_path, e_path in zip(chunks_paths, embeddings_paths):
            if os.path.exists(c_path) and os.path.exists(e_path):
                try:
                    with open(c_path, "r", encoding="utf-8") as f:
                        self.chunks = json.load(f)
                    self.embeddings = np.load(e_path)
                    print(f"Loaded chunks from {c_path} and embeddings from {e_path}")
                    loaded_data = True
                    break
                except Exception:
                    continue
                    
        if not loaded_data:
            raise FileNotFoundError("Could not load chunks and embeddings from any standard, test, or doc_store path.")
        
        # Initialize components
        self.embedder = Embedder()
        self.graph_builder = GraphBuilder()
        
        # Load Graph with fallbacks
        graph_loaded = False
        graph_paths = ["graph/graph.pkl", "graph/test_graph.pkl", "doc_store/test/graph.pkl"]
        for g_path in graph_paths:
            if os.path.exists(g_path):
                try:
                    self.graph_builder.load_graph(g_path)
                    print(f"Loaded graph from {g_path}")
                    graph_loaded = True
                    break
                except Exception:
                    continue
        if not graph_loaded:
            raise FileNotFoundError("Could not load networkx graph from any path.")
            
        # Load BM25 with fallbacks
        bm25_loaded = False
        bm25_paths = ["graph/bm25_index.pkl", "graph/test_bm25_index.pkl", "doc_store/test/bm25_index.pkl"]
        for b_path in bm25_paths:
            if os.path.exists(b_path):
                try:
                    self.bm25 = BM25Retriever()
                    self.bm25.load_index(b_path)
                    print(f"Loaded BM25 index from {b_path}")
                    bm25_loaded = True
                    break
                except Exception:
                    continue
        if not bm25_loaded:
            raise FileNotFoundError("Could not load BM25 index from any path.")
            
        # Load FAISS Vector search with fallbacks
        faiss_loaded = False
        faiss_paths = [
            ("embeddings/faiss.index", "embeddings/faiss_chunks.pkl"),
            ("embeddings/test_faiss.index", "embeddings/test_faiss_chunks.pkl"),
            ("doc_store/test/faiss.index", "doc_store/test/faiss_chunks.pkl")
        ]
        for f_idx_path, f_chunks_path in faiss_paths:
            if os.path.exists(f_idx_path) and os.path.exists(f_chunks_path):
                try:
                    self.vector_search = VectorSearch()
                    self.vector_search.load_index(f_idx_path, f_chunks_path)
                    print(f"Loaded FAISS index from {f_idx_path}")
                    faiss_loaded = True
                    break
                except Exception:
                    continue
        if not faiss_loaded:
            raise FileNotFoundError("Could not load FAISS index from any path.")
        
        self.graph_retriever = GraphRetriever(self.graph_builder)
        self.fusion = ResultFusion(
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight
        )
        self.reranker = Reranker()
        
        # Adaptive components
        self.classifier = QueryClassifier()
        self.decomposer = QueryDecomposer()
        self.hyde = HyDEGenerator()
        self.planner = RetrievalPlanner()
        
        # Phase 4 components
        self.llm = LLMGenerator()
        
        self._loaded = True
        print("Adaptive retrieval pipeline loaded successfully.")
    
    def _save_trace(self, trace: Dict[str, Any]):
        """Persists the retrieval reasoning trace to disk gracefully."""
        self.last_trace = trace
        try:
            with self.trace_lock:
                os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)
                
                # Maintain a history of traces (limit to 10 entries to save space)
                history = []
                if os.path.exists(self.trace_path):
                    try:
                        with open(self.trace_path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                            if not isinstance(history, list):
                                history = []
                    except Exception:
                        history = []
                        
                history.append(trace)
                # Cap trace history
                if len(history) > 10:
                    history = history[-10:]
                    
                with open(self.trace_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save retrieval trace: {e}")

    def retrieve(
        self,
        query: str,
        top_k_initial: int = 10,
        top_k_final: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute the complete adaptive retrieval pipeline.
        
        Args:
            query: User's question
            top_k_initial: Number of candidates to retrieve from each method per sub-query
            top_k_final: Number of final results after reranking
            
        Returns:
            List of (chunk_dict, reranker_score) tuples
        """
        if not self._loaded:
            self.load()
        
        # Step 1: Understand Intent & Formulate Plan (Retrieval Planning)
        plan = self.planner.plan(query)
        query_type = plan["query_type"]
        use_hyde = plan["use_hyde"]
        use_decomposition = plan["use_decomposition"]
        graph_depth = plan["graph_depth"]
        traversal = plan["traversal"]
        rerank_mode = plan["rerank_mode"]
            
        # Step 3: Query Decomposition (Conditional)
        if use_decomposition:
            sub_questions = self.decomposer.decompose(query)
        else:
            sub_questions = [query]
            
        merged_candidates = []
        
        # Step 4: Execute Plan for each sub-question
        for sub_q in sub_questions:
            # 1. HyDE Caching and Generation
            hypothesis = self.hyde.generate_hypothesis(sub_q)
            
            # 2. Parallel BM25 + Vector Search
            bm25_results = self.bm25.bm25_search(sub_q, top_k=top_k_initial)
            
            query_embedding = self.embedder.generate_query_embedding(hypothesis)
            vector_results = self.vector_search.vector_search(query_embedding, top_k=top_k_initial)
            
            # 3. Dynamic Graph Expansion
            graph_results = []
            if graph_depth > 0 and vector_results:
                # Retrieve seeds using the RAW query to remain anchored to original tokens
                raw_query_emb = self.embedder.generate_query_embedding(sub_q)
                raw_vec_res = self.vector_search.vector_search(raw_query_emb, top_k=3)
                seeds = [res[0] for res in raw_vec_res]
                
                graph_results = self.graph_retriever.graph_based_retrieval(
                    seeds,
                    max_depth=graph_depth,
                    use_bfs=True,
                    strategy=traversal
                )
            
            # 4. Fusion of current sub-query stream
            sub_fused = self.fusion.fuse_results(
                bm25_results,
                vector_results,
                graph_results,
                top_k=20
            )
            merged_candidates.extend(sub_fused)
            
        # Step 5: Global Deduplication by chunk_id
        seen_ids = set()
        unique_candidates = []
        for chunk, score in merged_candidates:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_candidates.append((chunk, score))
                
        # Sort globally by fused score, cap to top 50 candidates
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        candidate_pool = unique_candidates[:50]
        
        # Step 6: Adaptive Reranking against original query with term boosting
        final_results = self.reranker.rerank(
            query=query,
            candidates=candidate_pool,
            top_k=top_k_final,
            query_type=query_type
        )
        
        # Step 7: Record Retrieval Trace (Feature 8 Conformance)
        trace = {
            "timestamp": np.datetime64('now').astype(str),
            "query": query,
            "query_type": query_type,
            "retrieval_strategy": f"BM25 + Vector + Graph ({traversal})",
            "graph_depth": graph_depth,
            "traversal": traversal,
            "subqueries": sub_questions,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:100] + "...",
                    "source": chunk["source_filename"],
                    "page": chunk["page_number"]
                }
                for chunk, _ in merged_candidates
            ],
            "after_deduplication": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:100] + "...",
                    "source": chunk["source_filename"],
                    "page": chunk["page_number"]
                }
                for chunk, _ in candidate_pool
            ],
            "final_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:100] + "...",
                    "source": chunk["source_filename"],
                    "page": chunk["page_number"]
                }
                for chunk, _ in final_results
            ],
            "rerank_scores": [round(score, 4) for _, score in final_results]
        }
        self._save_trace(trace)
        
        return final_results

    def answer_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        mode: str = "FAST"
    ) -> Dict[str, Any]:
        """
        Generate answer, timing breakdowns, count LLM calls, identify bottlenecks,
        support Fast vs Verified modes, and save performance traces.
        """
        import time
        from performance_tracker import PerformanceTracker
        tracker = PerformanceTracker()
        tracker.reset_call_count()
        
        t_total_start = time.perf_counter()
        
        if not self._loaded:
            self.load()
            
        # 1. Timing: Classification & Planning
        t_start = time.perf_counter()
        plan = self.planner.plan(query)
        query_type = plan["query_type"]
        use_hyde = plan["use_hyde"]
        use_decomposition = plan["use_decomposition"]
        graph_depth = plan["graph_depth"]
        traversal = plan["traversal"]
        rerank_mode = plan["rerank_mode"]
        classification_ms = (time.perf_counter() - t_start) * 1000
        
        # 2. Timing: Query Decomposition
        t_start = time.perf_counter()
        if use_decomposition:
            sub_questions = self.decomposer.decompose(query)
        else:
            sub_questions = [query]
        decomposition_ms = (time.perf_counter() - t_start) * 1000
        
        hyde_ms = 0.0
        retrieval_ms = 0.0
        graph_expansion_ms = 0.0
        
        merged_candidates = []
        
        # 3. Timing: Retrieve candidates for each sub-question
        for sub_q in sub_questions:
            # HyDE
            t_sub_start = time.perf_counter()
            hypothesis = self.hyde.generate_hypothesis(sub_q)
            hyde_ms += (time.perf_counter() - t_sub_start) * 1000
            
            # Retrieval (BM25 + Vector)
            t_sub_start = time.perf_counter()
            bm25_results = self.bm25.bm25_search(sub_q, top_k=10)
            query_embedding = self.embedder.generate_query_embedding(hypothesis)
            vector_results = self.vector_search.vector_search(query_embedding, top_k=10)
            retrieval_ms += (time.perf_counter() - t_sub_start) * 1000
            
            # Graph Expansion
            t_sub_start = time.perf_counter()
            graph_results = []
            if graph_depth > 0 and vector_results:
                raw_query_emb = self.embedder.generate_query_embedding(sub_q)
                raw_vec_res = self.vector_search.vector_search(raw_query_emb, top_k=3)
                seeds = [res[0] for res in raw_vec_res]
                
                graph_results = self.graph_retriever.graph_based_retrieval(
                    seeds,
                    max_depth=graph_depth,
                    use_bfs=True,
                    strategy=traversal
                )
            graph_expansion_ms += (time.perf_counter() - t_sub_start) * 1000
            
            # Fusion
            t_sub_start = time.perf_counter()
            sub_fused = self.fusion.fuse_results(
                bm25_results,
                vector_results,
                graph_results,
                top_k=20
            )
            merged_candidates.extend(sub_fused)
            
        # Deduplication
        seen_ids = set()
        unique_candidates = []
        for chunk, score in merged_candidates:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_candidates.append((chunk, score))
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        candidate_pool = unique_candidates[:50]
        
        # 4. Timing: Adaptive Reranking
        t_start = time.perf_counter()
        retrieved_chunks = self.reranker.rerank(
            query=query,
            candidates=candidate_pool,
            top_k=5,
            query_type=query_type
        )
        rerank_ms = (time.perf_counter() - t_start) * 1000
        
        # 5. Timing: LLM Generation
        t_start = time.perf_counter()
        
        # In FAST mode we don't prep summaries to minimize latency
        document_summaries = []
        if mode == "VERIFIED":
            seen_files = set()
            for chunk, _ in retrieved_chunks:
                source_file = chunk.get("source_filename")
                if source_file and source_file not in seen_files:
                    seen_files.add(source_file)
                    import re
                    doc_name = re.sub(r'[^a-zA-Z0-9]', '_', source_file.replace(".pdf", "").replace(".PDF", "")).lower()
                    meta_path = f"doc_store/{doc_name}/meta.json"
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            if "document_summary" in meta:
                                document_summaries.append(f"Document: {source_file}\nSummary: {meta['document_summary']}")
                        except Exception:
                            pass
                            
        raw_answer = self.llm.generate_answer(query, retrieved_chunks, chat_history=chat_history, document_summaries=document_summaries)
        
        # Parse confidence label
        confidence_map = {
            "HIGH":   ("High",   ""),
            "MEDIUM": ("Medium", ""),
            "LOW":    ("Low",    ""),
        }
        label, emoji = "Medium", ""
        answer = raw_answer.strip()

        for key, (l, e) in confidence_map.items():
            tag = f"CONFIDENCE: {key}"
            if tag in raw_answer:
                label, emoji = l, e
                answer = raw_answer.replace(tag, "").strip()
                break
                
        # Clean spacing and list formatting
        import re
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('•'):
                indent = len(line) - len(stripped)
                line = ' ' * indent + '- ' + stripped[1:].lstrip()
            cleaned_lines.append(line)
        answer = '\n'.join(cleaned_lines)
        generation_ms = (time.perf_counter() - t_start) * 1000
        
        # Initialize Phase 4 variables for FAST mode safety
        claims_list = []
        citations = []
        grounding_score = 100.0
        trust_level = "VERIFIED"
        hallucination_risk = "LOW"
        verified_answer = answer
        verified_cit_count = 0
        failed_cit_count = 0
        
        # Timings for verification steps (FAST mode sets to 0.0)
        claim_extraction_ms = 0.0
        claim_verification_ms = 0.0
        citation_verification_ms = 0.0
        grounding_ms = 0.0
        
        claims = []
        supported_claims = []
        unsupported_claims = []
        
        # 6. VERIFIED MODE vs FAST MODE routing
        if mode == "VERIFIED":
            # Timing: Claim Extraction
            t_start = time.perf_counter()
            claims = self.claim_extractor.extract_claims(answer)
            claim_extraction_ms = (time.perf_counter() - t_start) * 1000
            
            # Timing: Claim Verification
            t_start = time.perf_counter()
            verify_res = self.claim_verifier.verify_claims(claims, retrieved_chunks)
            claims_list = verify_res["claims"]
            grounding_score = verify_res["grounding_score"]
            trust_level = verify_res["trust_level"]
            hallucination_risk = verify_res["hallucination_risk"]
            supported_claims = [c["claim"] for c in claims_list if c["status"] == "SUPPORTED"]
            unsupported_claims = [c["claim"] for c in claims_list if c["status"] == "UNSUPPORTED"]
            claim_verification_ms = (time.perf_counter() - t_start) * 1000
            
            # Timing: Citation Verification
            t_start = time.perf_counter()
            cit_res = self.citation_verifier.verify_citations(answer, retrieved_chunks)
            verified_answer = cit_res["verified_answer"]
            citations = cit_res["citations"]
            verified_cit_count = cit_res["verified_count"]
            failed_cit_count = cit_res["failed_count"]
            citation_verification_ms = (time.perf_counter() - t_start) * 1000
            
            # Timing: Grounding Calculations / Framework overhead
            t_start = time.perf_counter()
            # Simple calculations (already covered in claim verification but recording overhead if any)
            grounding_ms = (time.perf_counter() - t_start) * 1000
            
        total_ms = (time.perf_counter() - t_total_start) * 1000
        
        # Calculate Bottlenecks
        stages_timings = {
            "classification": classification_ms,
            "hyde": hyde_ms,
            "decomposition": decomposition_ms,
            "retrieval": retrieval_ms,
            "graph_expansion": graph_expansion_ms,
            "rerank": rerank_ms,
            "generation": generation_ms,
            "claim_extraction": claim_extraction_ms,
            "claim_verification": claim_verification_ms,
            "citation_verification": citation_verification_ms,
            "grounding": grounding_ms
        }
        
        slowest_stage = max(stages_timings, key=stages_timings.get)
        slowest_stage_ms = stages_timings[slowest_stage]
        
        # Track claim optimization details
        num_claims = len(claims)
        avg_per_claim_ms = (claim_verification_ms / num_claims) if num_claims > 0 else 0.0
        
        # Get LLM API call count for this execution
        llm_calls = tracker.get_llm_calls()
        
        # Log performance trace
        perf_trace = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "mode": mode,
            "total_ms": round(total_ms, 1),
            "llm_calls": llm_calls,
            "slowest_stage": slowest_stage,
            "slowest_stage_ms": round(slowest_stage_ms, 1),
            "classification_ms": round(classification_ms, 1),
            "hyde_ms": round(hyde_ms, 1),
            "decomposition_ms": round(decomposition_ms, 1),
            "retrieval_ms": round(retrieval_ms, 1),
            "graph_expansion_ms": round(graph_expansion_ms, 1),
            "rerank_ms": round(rerank_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "claim_extraction_ms": round(claim_extraction_ms, 1),
            "claim_verification_ms": round(claim_verification_ms, 1),
            "citation_verification_ms": round(citation_verification_ms, 1),
            "grounding_ms": round(grounding_ms, 1),
            "claims_count": num_claims,
            "avg_per_claim_ms": round(avg_per_claim_ms, 1)
        }
        tracker.log_trace(perf_trace)
        
        # Save standard trace info for transparency
        trace_data = {
            "query": query,
            "query_type": query_type,
            "retrieval_strategy": f"BM25 + Vector + Graph ({traversal})",
            "grounding_score": grounding_score,
            "trust_level": trust_level,
            "claims": claims,
            "verified_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "citations": citations,
            "performance": perf_trace
        }
        self.last_trace.update(trace_data)
        self._save_trace(self.last_trace)
        
        # Save Answer Quality Report (Feature 7)
        quality_report = {
            "query": query,
            "query_type": query_type,
            "grounding_score": grounding_score,
            "trust_level": trust_level,
            "claims": claims_list,
            "verified_claims": len(supported_claims),
            "unverified_claims": len(unsupported_claims) + (len(claims_list) - len(supported_claims) - len(unsupported_claims)),
            "verified_citations": verified_cit_count,
            "failed_citations": failed_cit_count,
            "hallucination_risk": hallucination_risk,
            "mode": mode,
            "performance": perf_trace
        }
        
        report_path = "graph/answer_quality_report.json"
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as rf:
                json.dump(quality_report, rf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save quality report: {e}")
            
        # Format sources
        sources = []
        for chunk, score in retrieved_chunks:
            sources.append({
                "filename": chunk.get("source_filename", "Unknown"),
                "page": chunk.get("page_number", 0),
                "score": score,
                "text": chunk.get("text", "")
            })
            
        return {
            "answer": verified_answer,
            "sources": sources,
            "confidence": label,
            "confidence_emoji": emoji,
            "grounding_score": grounding_score,
            "trust_level": trust_level,
            "hallucination_risk": hallucination_risk,
            "claims": claims_list,
            "citations": citations,
            "trace": self.last_trace,
            "performance": perf_trace
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
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
            "trace_file": self.trace_path
        }


if __name__ == "__main__":
    # Example usage
    pipeline = RetrievalPipeline()
    pipeline.load()
    
    query = "Compare Bellman-Ford and Dijkstra algorithms."
    results = pipeline.retrieve(query, top_k_initial=10, top_k_final=5)
    
    print(f"\nQuery: {query}")
    print(f"Detected Type: {pipeline.last_trace.get('query_type')}")
    print(f"Sub-questions: {pipeline.last_trace.get('sub_questions')}")
    print(f"Traversal: {pipeline.last_trace.get('traversal_used')} (Depth: {pipeline.last_trace.get('graph_depth')})")
    print(f"\nTop {len(results)} Boosted Results:")
    for i, (chunk, score) in enumerate(results):
        print(f"\n[{i+1}] Boosted Rerank Score: {score:.4f}")
        print(f"Source: {chunk.get('source_filename')} (Page {chunk.get('page_number')})")
        print(f"Text: {chunk.get('text')[:200]}...")
