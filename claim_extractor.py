import os
import json
import re
from typing import List
from groq import Groq

# Reuse models from llm.py
MODELS_CLAIM = [
    "llama-3.1-8b-instant",      # Fast, high rate limits
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

class ClaimExtractor:
    """
    ClaimExtractor extracts atomic, self-contained factual statements
    from an answer text using LLM generation with a fallback to regex-based
    sentence splitting.
    """

    def __init__(self):
        # We initialize Groq client, which reads from GROQ_API_KEY env var
        try:
            self.client = Groq()
        except Exception:
            self.client = None

    def extract_claims(self, answer: str) -> List[str]:
        """
        Extracts a list of atomic factual claims from the text.
        
        Args:
            answer: Generated text response
            
        Returns:
            List of string claims
        """
        # Clean answer first: remove metadata/formatting and confidence tag
        cleaned_text = re.sub(r'CONFIDENCE:\s*\w+', '', answer).strip()
        cleaned_text = re.sub(r'\[[^\]]+\]\s*(?:\[(Verified|Unverified)\])?', '', cleaned_text).strip()
        
        if not cleaned_text:
            return []

        # If Groq is not configured, use fallback immediately
        if not self.client or not os.environ.get("GROQ_API_KEY"):
            return self._fallback_extract(cleaned_text)

        prompt = f"""You are a Fact Extraction Assistant.
Your task is to extract a list of all distinct, atomic, self-contained factual claims made in the text below.
An atomic claim is a single statement that can be verified independently. Do not extract opinions, questions, formatting, or transitions.
Each claim must be complete and understandable on its own (e.g. replace pronouns like "it" or "this algorithm" with their actual referents).

Text:
"{cleaned_text}"

Return the claims EXACTLY as a JSON array of strings. Do not include any other text, markdown blocks, or explanation.
Example Output:
[
  "Bellman-Ford computes shortest paths in a graph.",
  "Bellman-Ford supports negative edge weights."
]
"""
        
        # Try LLM models
        for model in MODELS_CLAIM:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional fact extraction bot that outputs only JSON arrays."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=800
                )
                res_content = response.choices[0].message.content.strip()
                
                # Clean potential markdown wrapping
                if res_content.startswith("```json"):
                    res_content = res_content[7:]
                if res_content.startswith("```"):
                    res_content = res_content[3:]
                if res_content.endswith("```"):
                    res_content = res_content[:-3]
                res_content = res_content.strip()
                
                claims = json.loads(res_content)
                if isinstance(claims, list):
                    # Filter and clean claims
                    return [str(c).strip() for c in claims if c and len(str(c).strip()) > 5]
            except Exception as e:
                # Log or print warning (ASCII safe)
                # print(f"Warning: Claim extraction failed for model {model}: {e}")
                continue

        # If LLM failed or JSON parsing failed, run fallback
        return self._fallback_extract(cleaned_text)

    def _fallback_extract(self, text: str) -> List[str]:
        """Regex-based sentence splitter fallback."""
        # Split by periods/exclamation/question marks followed by whitespace
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Clean up markdown headers/bullets
            s_clean = re.sub(r'^[-*•#\s\d+\.]+', '', s_clean)
            if len(s_clean) > 10:
                claims.append(s_clean)
        return claims

if __name__ == "__main__":
    # Test harness
    from dotenv import load_dotenv
    load_dotenv()
    
    extractor = ClaimExtractor()
    test_ans = (
        "Bellman-Ford computes shortest paths and supports negative weights. "
        "However, Dijkstra's algorithm is faster but fails on negative edges. [test.pdf, Page 3]"
    )
    extracted = extractor.extract_claims(test_ans)
    print("Answer:", test_ans)
    print("Extracted Claims:")
    for i, c in enumerate(extracted, 1):
        print(f"  {i}. {c}")
