from typing import List, Dict, Any, Tuple

class ResultFusion:
    """
    ResultFusion merges the results from our 3 retrieval streams:
    1. BM25 (Sparse/Lexical)
    2. FAISS (Dense/Semantic)
    3. Graph (Multi-hop Context)
    
    Why Fusion?
    Different retrieval methods have different scoring scales. BM25 scores can be > 10.0, 
    FAISS cosine similarities are between -1.0 and 1.0, and Graph scores are heuristics.
    If we just add them together, BM25 will dominate. 
    
    We need to normalize the scores. For our baseline, we'll use Min-Max normalization 
    within each result set, and then use a weighted sum (Linear Combination) to fuse them.
    We also deduplicate results (a chunk might be found by all 3 methods).
    """
    
    def __init__(self, bm25_weight: float = 0.3, vector_weight: float = 0.5, graph_weight: float = 0.2):
        self.weights = {
            "bm25": bm25_weight,
            "vector": vector_weight,
            "graph": graph_weight
        }

    def _normalize_scores(self, results: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Applies Min-Max normalization so all scores are between 0 and 1.
        Formula: (score - min) / (max - min)
        """
        if not results:
            return []
            
        scores = [r[1] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # If all scores are identical, set them all to 1.0
            return [(r[0], 1.0) for r in results]
            
        normalized = []
        for chunk, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((chunk, norm_score))
            
        return normalized

    def fuse_results(
        self, 
        bm25_results: List[Tuple[Dict[str, Any], float]], 
        vector_results: List[Tuple[Dict[str, Any], float]], 
        graph_results: List[Tuple[Dict[str, Any], float]],
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Deduplicates and fuses normalized scores.
        """
        
        # 1. Normalize each stream
        norm_bm25 = self._normalize_scores(bm25_results)
        norm_vector = self._normalize_scores(vector_results)
        norm_graph = self._normalize_scores(graph_results)
        
        # 2. Accumulate scores in a dictionary keyed by chunk_id
        # We also need to keep the actual chunk payload
        fused_scores = {}
        chunk_payloads = {}
        
        def _add_to_fusion(normalized_results, source_name):
            weight = self.weights[source_name]
            for chunk, score in normalized_results:
                cid = chunk["chunk_id"]
                if cid not in fused_scores:
                    fused_scores[cid] = 0.0
                    chunk_payloads[cid] = chunk
                fused_scores[cid] += score * weight

        _add_to_fusion(norm_bm25, "bm25")
        _add_to_fusion(norm_vector, "vector")
        _add_to_fusion(norm_graph, "graph")
        
        # 3. Sort by combined score descending
        sorted_fused = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 4. Return top-k results formatted as (chunk, score)
        final_results = []
        for cid, score in sorted_fused[:top_k]:
            final_results.append((chunk_payloads[cid], score))
            
        return final_results

if __name__ == "__main__":
    pass
