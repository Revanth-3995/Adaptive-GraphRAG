import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple

class VectorSearch:
    """
    VectorSearch uses FAISS to perform highly efficient semantic similarity retrieval.
    
    What is FAISS?
    FAISS (Facebook AI Similarity Search) is a library for efficient similarity search 
    and clustering of dense vectors.
    
    Why not just use np.dot() like we did in GraphBuilder?
    Our GraphBuilder computes O(N^2) similarity offline, which is fine during ingestion. 
    But at query time, if we have 10 million chunks, doing a linear scan `O(N)` with 
    np.dot() over 10 million vectors every time a user asks a question is too slow.
    
    FAISS allows us to create an index. We use `IndexFlatL2` (Exact Search) for 
    simplicity here since our data is small, but FAISS shines with `IndexIVFFlat` or HNSW 
    which perform Approximate Nearest Neighbor (ANN) search, dropping query time to `O(log N)`.
    """
    
    def __init__(self, embedding_dimension: int = 384):
        self.dimension = embedding_dimension
        # IndexFlatIP computes inner product. 
        # If vectors are L2 normalized, Inner Product is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalizes the vectors. 
        FAISS IndexFlatIP computes the dot product. 
        Dot product of two L2-normalized vectors is exactly their Cosine Similarity.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_faiss_index(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Builds the FAISS index from the provided embeddings.
        """
        print(f"Building FAISS index for {len(chunks)} vectors...")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}")
            
        self.chunks = chunks
        normalized_embeddings = self._normalize_vectors(embeddings)
        self.index.add(np.ascontiguousarray(normalized_embeddings))
        print(f"FAISS index built. Total vectors: {self.index.ntotal}")

    def vector_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the FAISS index for the vectors most similar to the query embedding.
        
        Returns:
            List of tuples: (chunk_dict, cosine_similarity_score)
        """
        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. Call build_faiss_index first.")
            
        # Reshape query to 2D array: (1, D)
        query_embedding_2d = query_embedding.reshape(1, -1)
        normalized_query = self._normalize_vectors(query_embedding_2d)
        
        # search() returns distances (or similarities for IP) and indices
        similarities, indices = self.index.search(np.ascontiguousarray(normalized_query), top_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1: # FAISS returns -1 if there aren't enough vectors
                # sim is the cosine similarity because we normalized
                results.append((self.chunks[idx], float(sim)))
                
        return results

    def save_index(self, index_path: str = "embeddings/faiss.index", chunks_path: str = "embeddings/faiss_chunks.pkl"):
        """Saves the FAISS index and chunk mappings."""
        import pickle
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"FAISS index saved to {index_path}")

    def load_index(self, index_path: str = "embeddings/faiss.index", chunks_path: str = "embeddings/faiss_chunks.pkl"):
        """Loads the FAISS index and chunk mappings."""
        import pickle
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError("FAISS index or chunks file not found.")
            
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"FAISS index loaded. Total vectors: {self.index.ntotal}")

if __name__ == "__main__":
    pass
