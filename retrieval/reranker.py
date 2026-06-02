from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranker uses a Cross-Encoder to finely score the top candidate chunks against the query.
    
    Bi-Encoder vs Cross-Encoder:
    - Our Embedder (`all-MiniLM-L6-v2`) is a Bi-Encoder. It processes the query and the 
      document separately to create vectors, and we just compare the vectors (Cosine Sim). 
      It's fast but lacks deep attention between query and document words.
    - A Cross-Encoder takes both the query and the document AT THE SAME TIME and passes 
      them together through the Transformer's self-attention layers. This allows the model 
      to see how every word in the query relates to every word in the document.
      
    Adaptive Boosting:
    Depending on the query type, specific domain vocabulary (e.g., "pseudocode" for ALGORITHM)
    is boosted to ensure the most aligned context style surfaces to the top.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        print(f"Loading CrossEncoder model: {self.model_name}...")
        self.model = CrossEncoder(self.model_name)
        print("CrossEncoder loaded.")
        
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_k: int = 5,
        query_type: str = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Reranks a list of candidate chunks based on the query, with optional intent boosting.
        
        Args:
            query: The user's question.
            candidates: List of (chunk_dict, previous_fusion_score).
            top_k: Number of final results to return.
            query_type: The classified intent (e.g. SIMPLE, MODERATE, COMPLEX, ALGORITHM, RESEARCH).
            
        Returns:
            List of (chunk_dict, boosted_cross_encoder_score) sorted by score.
        """
        if not candidates:
            return []
            
        # We need to construct pairs of (Query, Document) for the CrossEncoder
        pairs = []
        for chunk, _ in candidates:
            pairs.append((query, chunk["text"]))
            
        # Predict scores for all pairs (typically between -10.0 and 10.0)
        scores = self.model.predict(pairs)
        
        # Combine the chunks with their new scores and apply adaptive boosting
        reranked_results = []
        
        boost_words = {
            "ALGORITHM": ["algorithm", "pseudocode", "steps", "procedure", "workflow"],
            "COMPLEX": ["comparison", "difference", "advantage", "disadvantage", "tradeoff"],
            "RESEARCH": ["overview", "survey", "method", "approach", "framework"]
        }
        
        for i, (chunk, _) in enumerate(candidates):
            raw_score = float(scores[i])
            boost = 0.0
            
            # Apply boosting if query_type matches our criteria
            if query_type in boost_words:
                text_lower = chunk["text"].lower()
                matches = sum(1 for word in boost_words[query_type] if word in text_lower)
                if matches > 0:
                    # Provide an additive boost of +1.0 per unique matched boost keyword
                    boost = 1.0 * matches
            
            boosted_score = raw_score + boost
            reranked_results.append((chunk, boosted_score))
            
        # Sort by the boosted score descending
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]


if __name__ == "__main__":
    pass
