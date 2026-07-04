# NOTE: This module contains Experimental / Advanced Features that are currently disabled from the standard user-facing product experience.
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
        
        # Strip HTML tags and UI symbols to prevent LLM confusion
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        cleaned_text = cleaned_text.replace("✓ Verified", "").replace("⚠️ Unverified", "")
        cleaned_text = cleaned_text.strip()
        
        if not cleaned_text:
            return []

        if self.client:
            self.client.api_key = os.environ.get("GROQ_API_KEY") or self.client.api_key

        # If Groq is not configured, use fallback immediately
        if not self.client or not os.environ.get("GROQ_API_KEY"):
            return self._fallback_extract(cleaned_text)

        prompt = f"""You are a Fact Extraction Assistant.
Your task is to extract a list of all distinct, atomic, self-contained factual claims made in the text below.
An atomic claim is a single statement that can be verified independently.

RULES:
- Do NOT extract opinions, questions, formatting, or transitions.
- Do NOT extract pseudocode lines, algorithm steps, loop constructs, variable assignments, or return statements (e.g. "return true", "for i = 0 to n", "if A[i] = A[i+1]", "sort the array").
- Do NOT extract individual steps of an algorithm as separate claims.
- ONLY extract high-level factual assertions about concepts, definitions, properties, complexities, or comparisons.
- Each claim must be complete and understandable on its own (e.g. replace pronouns like "it" or "this algorithm" with their actual referents).

Text:
"{cleaned_text}"

Return the claims EXACTLY as a JSON array of strings. Do not include any other text, markdown blocks, or explanation.
Example Output:
[
  "Bellman-Ford computes shortest paths in a graph.",
  "The time complexity of the presort-based algorithm is O(n log n)."
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
                
                from performance_tracker import PerformanceTracker
                PerformanceTracker().increment_llm_calls()
                
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
                    # Filter and clean claims, then remove pseudocode lines
                    cleaned = [str(c).strip() for c in claims if c and len(str(c).strip()) > 5]
                    return [c for c in cleaned if not self._is_pseudocode(c)]
            except Exception as e:
                # Log or print warning (ASCII safe)
                # print(f"Warning: Claim extraction failed for model {model}: {e}")
                continue

        # If LLM failed or JSON parsing failed, run fallback
        return self._fallback_extract(cleaned_text)

    def _is_pseudocode(self, text: str) -> bool:
        """Detects if a claim is actually a pseudocode/algorithm step rather than a factual assertion."""
        t = text.strip().lower()
        # Common pseudocode patterns
        pseudocode_patterns = [
            r'^(return|print|output|input)\b',          # return true, print x
            r'^(for|while|repeat|do|loop)\b',           # for i = 0 to n
            r'^(if|else|then|elsif|elif)\b',            # if A[i] = A[j]
            r'^(sort|swap|set|let|initialize)\b',       # sort the array
            r'^(call|invoke|execute|run)\b',            # call function
            r'\bdo$',                                    # ends with 'do'
            r'^\w+\s*\(.*\)\s*$',                       # function call pattern
            r'^\w+\[.*\]\s*[=<>!]',                     # array access comparison
        ]
        for pat in pseudocode_patterns:
            if re.search(pat, t):
                return True
        # Very short statements that look like code
        if len(t) < 25 and any(kw in t for kw in ['return ', 'break', 'continue', '= true', '= false']):
            return True
        return False

    def _fallback_extract(self, text: str) -> List[str]:
        """Regex-based sentence splitter fallback."""
        # Split by periods/exclamation/question marks followed by whitespace
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Clean up markdown headers/bullets
            s_clean = re.sub(r'^[-*•#\s\d+\.]+', '', s_clean)
            if len(s_clean) > 10 and not self._is_pseudocode(s_clean):
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
