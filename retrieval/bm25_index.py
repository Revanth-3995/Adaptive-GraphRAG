import json
import pickle
import os
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import string

class BM25Retriever:
    """
    BM25Retriever implements sparse, keyword-based retrieval using the BM25 algorithm.
    
    What is BM25?
    BM25 (Best Matching 25) is a ranking function used by search engines to estimate 
    the relevance of documents to a given search query. It's an evolution of TF-IDF.
    
    TF-IDF vs BM25:
    - TF (Term Frequency): How often a word appears in a document.
    - IDF (Inverse Document Frequency): How rare the word is across all documents.
    - BM25 improves on TF-IDF by adding term frequency saturation (seeing a word 10 times 
      is better than 1 time, but not 10x better) and document length normalization 
      (a 5-word document matching "dog" is more relevant than a 500-word doc matching "dog").
      
    Why use it alongside Vectors?
    Vector search (semantic) is great at finding concepts ("puppy" matches "dog").
    BM25 (lexical) is great at exact keyword matches (e.g., specific acronyms, names, 
    or unique identifiers like "JIRA-1234"). A hybrid approach combines the best of both.
    """
    
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer: lowercases text and removes punctuation.
        In a production system, you might use NLTK or SpaCy for better stemming/lemmatization.
        """
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def build_bm25_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 index from a list of chunks.
        
        Complexity: O(N * L) where N is chunks and L is avg chunk length.
        """
        print(f"Building BM25 index for {len(chunks)} chunks...")
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully.")

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the BM25 index for a given query.
        
        Returns:
            List of tuples: (chunk_dict, bm25_score)
        """
        if self.bm25 is None:
            raise ValueError("BM25 index is not built. Call build_bm25_index first.")
            
        tokenized_query = self._tokenize(query)
        # get_scores() returns an array of scores corresponding to each document
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k highest scores
        # argsort() sorts ascending, so we take the last `top_k` and reverse
        top_n_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_n_indices:
            score = scores[idx]
            if score > 0: # Only return chunks that have at least some overlap
                # BM25 scores can be greater than 1, we don't normalize here.
                # Fusion layer will handle normalization.
                results.append((self.chunks[idx], float(score)))
                
        return results

    def save_index(self, output_path: str = "graph/bm25_index.pkl"):
        """Saves the BM25 object and chunk references."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)
        print(f"BM25 index saved to {output_path}")

    def load_index(self, input_path: str = "graph/bm25_index.pkl"):
        """Loads the BM25 object."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"BM25 index not found: {input_path}")
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']
        print("BM25 index loaded.")

if __name__ == "__main__":
    pass
