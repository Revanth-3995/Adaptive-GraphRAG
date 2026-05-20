from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Reranker uses a Cross-Encoder to finely score the top candidate chunks against the query.
    
    Bi-Encoder vs Cross-Encoder:
    - Our Embedder (`all-MiniLM-L6-v2`) is a Bi-Encoder. It processes the query and the 
      document separately to create vectors, and we just compare the vectors (Cosine Sim). 
      It's fast (O(1) query time if index exists) but lacks deep attention between query 
      and document words.
    - A Cross-Encoder takes both the query and the document AT THE SAME TIME and passes 
      them together through the Transformer's self-attention layers. This allows the model 
      to see how every word in the query relates to every word in the document.
      
    Why use it?
    Cross-Encoders are incredibly accurate, but very slow. We can't run a Cross-Encoder 
    over 10,000 documents. So we use a pipeline:
    1. Fast retrieval (BM25 + FAISS + Graph) gets the top 50 candidates.
    2. Reranker (Cross-Encoder) carefully re-scores these 50 candidates and gives us the top 5.
    This pattern improves precision dramatically.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        print(f"Loading CrossEncoder model: {self.model_name}...")
        self.model = CrossEncoder(self.model_name)
        print("CrossEncoder loaded.")
        
    def rerank(self, query: str, candidates: List[Tuple[Dict[str, Any], float]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Reranks a list of candidate chunks based on the query.
        
        Args:
            query: The user's question.
            candidates: List of (chunk_dict, previous_fusion_score).
            
        Returns:
            List of (chunk_dict, cross_encoder_score) sorted by score.
        """
        if not candidates:
            return []
            
        # We need to construct pairs of (Query, Document) for the CrossEncoder
        pairs = []
        for chunk, _ in candidates:
            pairs.append((query, chunk["text"]))
            
        # Predict scores for all pairs
        scores = self.model.predict(pairs)
        
        # Combine the chunks with their new scores
        reranked_results = []
        for i, (chunk, _) in enumerate(candidates):
            reranked_results.append((chunk, float(scores[i])))
            
        # Sort by the new CrossEncoder score descending
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]

if __name__ == "__main__":
    pass
