import os
import sys
# Mask tensorflow to prevent import conflicts with streamlit's protobuf requirements
sys.modules['tensorflow'] = None

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Embedder is responsible for generating dense vector representations (embeddings) 
    for the text chunks using SentenceTransformers.
    
    What are Embeddings?
    Embeddings are high-dimensional numerical vectors that capture the semantic meaning 
    of text. Words or sentences with similar meanings will have vectors that are closer 
    together in this high-dimensional space.
    
    Why all-MiniLM-L6-v2?
    This is a fast, lightweight model that provides excellent semantic search quality 
    while keeping the dimensionality relatively low (384 dimensions). This balances 
    performance (speed) with accuracy.
    
    Uses a singleton pattern to ensure the model is loaded only once across all
    components (pipeline, claim verifier, HyDE, etc.), preventing redundant
    model loads especially during parallel claim verification.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if Embedder._initialized:
            return
        self.model_name = model_name
        print(f"Loading SentenceTransformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        # Typically 384 for all-MiniLM-L6-v2
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
        Embedder._initialized = True
        
    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> np.ndarray:
        """
        Generates embeddings for a list of text chunks in batches.
        
        Why Batching?
        Passing multiple sentences to the model at once (batching) is much more 
        computationally efficient (especially on GPUs) than processing them one by one.
        
        Time Complexity: O(N * L) where N is number of chunks and L is max sequence length.
        """
        texts = [chunk["text"] for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # We use show_progress_bar=True if we want visual feedback, but keeping it simple here
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
        
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates a single embedding for a user query.
        """
        # encode() returns a 1D array for a single string
        return self.model.encode(query, convert_to_numpy=True)

    def save_embeddings(self, embeddings: np.ndarray, output_path: str = "embeddings/embeddings.npy"):
        """
        Persists the embeddings to disk as a NumPy array (.npy format).
        
        Why .npy?
        NumPy's native binary format is highly optimized for saving and loading 
        large numerical arrays. It is much faster and takes less space than JSON or CSV.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Successfully saved {embeddings.shape[0]} embeddings to {output_path} with shape {embeddings.shape}")

    def load_embeddings(self, input_path: str = "embeddings/embeddings.npy") -> np.ndarray:
        """
        Loads embeddings from disk.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")
            
        embeddings = np.load(input_path)
        print(f"Loaded embeddings of shape: {embeddings.shape}")
        return embeddings

    def process_and_save_chunks(self, chunks_path: str = "graph/chunk_store.json", output_path: str = "embeddings/embeddings.npy"):
        """
        Helper method to read chunks from disk, embed them, and save the embeddings.
        """
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunk store not found: {chunks_path}")
            
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        embeddings = self.generate_embeddings(chunks)
        self.save_embeddings(embeddings, output_path)
        return embeddings

if __name__ == "__main__":
    # Example usage
    pass
