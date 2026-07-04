"""
HyDE — Hypothetical Document Embeddings

Instead of embedding the raw query (which is short and query-like),
generate a hypothetical answer (which is long and document-like) and
embed that instead. This dramatically improves vector search quality.

Reference: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without
Relevance Labels"
"""
import os
import json
import threading
from typing import Dict
from groq import Groq


class HyDEGenerator:
    """
    Generates hypothetical document embeddings for improved retrieval.
    Uses a fast, cheap model to generate the hypothesis — quality doesn't
    matter here, just semantic proximity to the document domain.

    Features a thread-safe, file-backed cache to prevent redundant Groq calls
    and minimize latency.
    """

    def __init__(self, cache_path: str = "graph/hyde_cache.json"):
        self.client = Groq()
        # Use the fastest/cheapest model for hypothesis generation
        self.model = "llama-3.1-8b-instant"
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.cache: Dict[str, str] = {}
        
        # Load persistent cache
        self._load_cache()

    def _load_cache(self):
        """Loads the cache from disk gracefully."""
        if not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            print(f"Loaded {len(self.cache)} entries from HyDE cache.")
        except Exception as e:
            print(f"Warning: Failed to load HyDE cache from {self.cache_path}: {e}")
            self.cache = {}

    def _save_cache(self):
        """Saves the cache to disk gracefully."""
        try:
            # Ensure containing directory exists
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save HyDE cache to {self.cache_path}: {e}")

    def generate_hypothesis(self, query: str) -> str:
        """
        Generates a hypothetical document passage that would answer the query.

        Uses cached results if available to optimize Groq usage.
        Returns the hypothesis if successful, original query if it fails.
        """
        cleaned_query = query.strip().lower()
        
        # 1. Thread-safe Cache Lookup
        with self.lock:
            if cleaned_query in self.cache:
                return self.cache[cleaned_query]

        # 2. Cache Miss: Generate using LLM
        try:
            self.client.api_key = os.environ.get("GROQ_API_KEY") or self.client.api_key
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document passage generator. "
                            "Given a question, write a short passage (2-4 sentences) "
                            "that would appear in a textbook and directly answer the question. "
                            "Write as if you are the document, not as an assistant. "
                            "Do not say 'Here is' or 'The answer is' — just write the passage."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=150  # Keep it short — just enough for semantic proximity
            )
            hypothesis = response.choices[0].message.content.strip()
            
            from performance_tracker import PerformanceTracker
            PerformanceTracker().increment_llm_calls()
            
            if hypothesis:
                # 3. Thread-safe Cache Update & Persist
                with self.lock:
                    self.cache[cleaned_query] = hypothesis
                    self._save_cache()
                return hypothesis
            
            return query

        except Exception as e:
            print(f"Warning: HyDE hypothesis generation failed ({e}). Falling back to raw query.")
            # Silently fall back to original query on any error
            return query
