"""
HyDE — Hypothetical Document Embeddings

Instead of embedding the raw query (which is short and query-like),
generate a hypothetical answer (which is long and document-like) and
embed that instead. This dramatically improves vector search quality.

Reference: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without
Relevance Labels"
"""
import os
from groq import Groq


class HyDEGenerator:
    """
    Generates hypothetical document embeddings for improved retrieval.
    Uses a fast, cheap model to generate the hypothesis — quality doesn't
    matter here, just semantic proximity to the document domain.
    """

    def __init__(self):
        self.client = Groq()
        # Use the fastest/cheapest model for hypothesis generation
        # Quality doesn't matter — just needs to be in the right domain
        self.model = "llama-3.1-8b-instant"
        self.cache = {}

    def generate_hypothesis(self, query: str) -> str:
        """
        Generates a hypothetical document passage that would answer the query.

        Returns the hypothesis if successful, original query if it fails.
        Failure is acceptable — we fall back to the original query.
        """
        if query in self.cache:
            return self.cache[query]

        try:
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
            result = hypothesis if hypothesis else query
            self.cache[query] = result
            return result
        except Exception:
            # Silently fall back to original query on any error
            return query
