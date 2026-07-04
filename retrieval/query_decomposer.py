"""
Query Decomposer — breaks complex questions into focused sub-questions.

For simple questions ("what is routing"), returns the original query unchanged.
For complex questions ("compare X and Y"), breaks into focused sub-questions
and retrieves for each one separately.
"""
import os
import json
from groq import Groq
from retrieval.query_classifier import QueryClassifier


class QueryDecomposer:

    def __init__(self):
        self.client = Groq()
        self.model = "llama-3.1-8b-instant"  # fast cheap model for decomposition
        self.classifier = QueryClassifier()

    def decompose(self, query: str) -> list:
        """
        Returns a list of sub-questions.
        
        For SIMPLE, MODERATE, or ALGORITHM questions, bypasses LLM call and 
        returns [original_query].
        
        For COMPLEX or RESEARCH questions, uses LLM to split into focused sub-questions.
        Always includes the original query as the last item to ensure
        the full question is also retrieved for.
        """
        # 1. Deterministic Classifier Check
        q_type = self.classifier.classify(query)
        if q_type in ("SIMPLE", "MODERATE", "ALGORITHM"):
            # Do not decompose these query types
            return [query]

        # 2. Decompose only when beneficial (COMPLEX or RESEARCH)
        try:
            self.client.api_key = os.environ.get("GROQ_API_KEY") or self.client.api_key
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a query analyzer.
Determine if a question is simple (answerable with one retrieval) or complex
(requires multiple retrievals to fully answer).

If SIMPLE: respond with exactly: SIMPLE
If COMPLEX: respond with a JSON array of 2-4 focused sub-questions that together
cover the full question. Each sub-question should be self-contained.

Examples:
Q: "what is routing" → SIMPLE
Q: "Compare distance vector and link state routing" → ["What is distance vector routing?", "What is link state routing?", "What are the differences between distance vector and link state routing?"]
Q: "explain heapsort algorithm and its time complexity" → ["What is the heapsort algorithm?", "What is the time complexity of heapsort?"]

Respond with ONLY "SIMPLE" or a valid JSON array. Nothing else."""
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=200
            )

            from performance_tracker import PerformanceTracker
            PerformanceTracker().increment_llm_calls()

            result = response.choices[0].message.content.strip()

            if result == "SIMPLE":
                return [query]

            # Try to parse JSON array
            sub_questions = json.loads(result)
            if isinstance(sub_questions, list) and len(sub_questions) > 0:
                # Always add original query to ensure full coverage
                if query not in sub_questions:
                    sub_questions.append(query)
                return sub_questions

            return [query]  # fallback

        except Exception as e:
            print(f"Warning: Query decomposition failed ({e}). Falling back to original query.")
            return [query]  # always fallback to original query on any error