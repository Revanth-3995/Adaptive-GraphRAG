# NOTE: This module contains Experimental / Advanced Features that are currently disabled from the standard user-facing product experience.
import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from groq import Groq

# Models for verification
MODELS_VERIFY = [
    "llama-3.1-8b-instant",      # Fast, high rate limits
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

class ClaimVerifier:
    """
    ClaimVerifier evaluates each extracted claim against retrieved evidence chunks.
    It computes Grounding Score, Trust Level, and Hallucination Risk.
    """

    def __init__(self):
        try:
            self.client = Groq()
        except Exception:
            self.client = None

    def verify_claims(
        self,
        claims: List[str],
        retrieved_chunks: List[Tuple[Dict[str, Any], float]]
    ) -> Dict[str, Any]:
        """
        Verifies a list of claims against retrieved context chunks.
        
        Args:
            claims: List of atomic factual claims
            retrieved_chunks: List of (chunk_dict, score) tuples
            
        Returns:
            Dict containing:
                "claims": List of claim verifications (claim, status, explanation)
                "grounding_score": Float percentage of supported claims
                "trust_level": String (VERIFIED, HIGH CONFIDENCE, etc.)
                "hallucination_risk": String (LOW, MEDIUM, HIGH)
                "supported_count": Int
                "partially_supported_count": Int
                "unsupported_count": Int
        """
        if not claims:
            return {
                "claims": [],
                "grounding_score": 100.0,
                "trust_level": "VERIFIED",
                "hallucination_risk": "LOW",
                "supported_count": 0,
                "partially_supported_count": 0,
                "unsupported_count": 0
            }

        # Build context block
        context_parts = []
        for i, (chunk, _) in enumerate(retrieved_chunks):
            source = chunk.get("source_filename", "Unknown")
            page = chunk.get("page_number", "Unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Document Chunk {i+1} | Source: {source}, Page: {page}]\n{text}")
        context = "\n\n".join(context_parts)

        verifications = []
        supported_count = 0
        partially_supported_count = 0
        unsupported_count = 0

        # Try to verify via Groq LLM
        use_llm = self.client is not None and os.environ.get("GROQ_API_KEY") is not None
        if use_llm:
            self.client.api_key = os.environ.get("GROQ_API_KEY")
        
        # We run the claim verification in parallel using a ThreadPoolExecutor
        # to prevent high sequential latency from multiple LLM calls.
        from concurrent.futures import ThreadPoolExecutor

        # Define single claim verifier function
        def verify_single_claim(claim: str) -> Dict[str, Any]:
            # Clean claim text from any HTML or UI tags
            clean_claim = re.sub(r'<[^>]+>', '', claim)
            clean_claim = clean_claim.replace("✓ Verified", "").replace("⚠️ Unverified", "").strip()
            
            verified = False
            status = "UNSUPPORTED"
            explanation = "No explanation provided."
            
            if use_llm:
                prompt = f"""You are a Fact Verification Assistant.
Analyze whether the given Claim is supported by the provided Context.
Classify the support level into exactly one of the following categories:

- SUPPORTED: The claim is fully or substantially supported by the context. This includes:
  * Minor rephrasings, synonyms, or direct logical inferences
  * Natural-language descriptions of algorithms/pseudocode found in the context
  * Complexity statements matching algorithms in the context
  * Paraphrased steps of an algorithm that is present in the context
- PARTIALLY_SUPPORTED: The claim is only partially supported, or adds significant new details, numbers, or assumptions not found in the context.
- UNSUPPORTED: The claim is clearly not supported, is contradicted, or discusses topics completely absent from the context.

IMPORTANT RULES:
1. If the context contains an algorithm's pseudocode and the claim describes the same algorithm in natural language, classify as SUPPORTED.
2. Treat natural language descriptions of code constructs (e.g. 'for each element, check if...' describing 'for i = 0 to n do if A[i]...') as SUPPORTED.
3. Focus on whether the FACTUAL CONTENT is correct, not whether the exact words appear.
4. When in doubt between SUPPORTED and PARTIALLY_SUPPORTED, lean toward SUPPORTED.

Context:
{context}

Claim:
"{clean_claim}"

Output exactly one line containing ONLY the classification (SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED) followed by a short one-line explanation on the next line.
Example:
SUPPORTED
The context explicitly states that Bellman-Ford supports negative weights.
"""
                for model in MODELS_VERIFY:
                    try:
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a fact verifier that focuses on factual correctness rather than exact wording. Natural language descriptions of algorithms or pseudocode should be considered supported if they accurately describe the logic. Output only the status and short explanation."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.0,
                            max_tokens=200
                        )
                        output = response.choices[0].message.content.strip().split('\n')
                        from performance_tracker import PerformanceTracker
                        PerformanceTracker().increment_llm_calls()
                        status = output[0].strip().upper()
                        
                        # Validate status
                        if status not in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]:
                            found = False
                            for line in output:
                                clean_line = line.strip().upper()
                                for val in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]:
                                    if val in clean_line:
                                        status = val
                                        found = True
                                        break
                                if found:
                                    break
                            if not found:
                                status = "UNSUPPORTED"
                        
                        explanation = "\n".join(output[1:]).strip() if len(output) > 1 else "No explanation provided."
                        verified = True
                        break
                    except Exception:
                        continue
            
            if not verified:
                try:
                    status, explanation = self._fallback_verify(claim, retrieved_chunks)
                except Exception as e:
                    status = "UNSUPPORTED"
                    explanation = f"Fallback failed ({e}). Defaulting to unsupported."
            
            return {
                "claim": claim,
                "status": status,
                "explanation": explanation
            }

        # Run verification tasks concurrently using ThreadPoolExecutor
        max_workers = min(len(claims), 12)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(verify_single_claim, claim) for claim in claims]
            for future in futures:
                try:
                    res = future.result()
                    verifications.append(res)
                    status = res["status"]
                    if status == "SUPPORTED":
                        supported_count += 1
                    elif status == "PARTIALLY_SUPPORTED":
                        partially_supported_count += 1
                    else:
                        unsupported_count += 1
                except Exception as e:
                    verifications.append({
                        "claim": "Error",
                        "status": "UNSUPPORTED",
                        "explanation": f"Concurrency error during verification: {e}"
                    })
                    unsupported_count += 1

        # Compute grounding score: supported + 0.5 * partially_supported / total
        total_claims = len(claims)
        
        # Feature 5: Weighted grounding score
        # SUPPORTED = 100%, PARTIALLY_SUPPORTED = 50%, UNSUPPORTED = 0%
        grounding_score = ((supported_count + 0.5 * partially_supported_count) / total_claims) * 100.0 if total_claims > 0 else 100.0
        
        # Feature 6: Trust Framework mapping
        if grounding_score >= 90:
            trust_level = "VERIFIED"
        elif grounding_score >= 80:
            trust_level = "HIGH CONFIDENCE"
        elif grounding_score >= 60:
            trust_level = "MEDIUM CONFIDENCE"
        elif grounding_score >= 40:
            trust_level = "LOW CONFIDENCE"
        else:
            trust_level = "UNTRUSTED"
            
        # Feature 9: Hallucination Risk assessment
        # High risk if there is a medium/large portion of unsupported claims
        if grounding_score < 60 or unsupported_count > 1:
            hallucination_risk = "HIGH"
        elif grounding_score < 90 or unsupported_count > 0:
            hallucination_risk = "MEDIUM"
        else:
            hallucination_risk = "LOW"

        return {
            "claims": verifications,
            "grounding_score": round(grounding_score, 1),
            "trust_level": trust_level,
            "hallucination_risk": hallucination_risk,
            "supported_count": supported_count,
            "partially_supported_count": partially_supported_count,
            "unsupported_count": unsupported_count
        }

    def _fallback_verify(self, claim: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> Tuple[str, str]:
        """Dense embedding similarity fallback."""
        from embedder import Embedder
        embedder = Embedder()
        
        # Clean claim text of any HTML or formatting tags
        clean_claim = re.sub(r'<[^>]+>', '', claim)
        clean_claim = clean_claim.replace("✓ Verified", "").replace("⚠️ Unverified", "").strip()
        
        claim_emb = embedder.generate_query_embedding(clean_claim)
        claim_norm = claim_emb / (np.linalg.norm(claim_emb) + 1e-10)
        
        best_sim = 0.0
        best_chunk_idx = 0
        best_chunk = None
        
        for idx, (chunk, _) in enumerate(retrieved_chunks):
            text = chunk.get("text", "")
            chunk_emb = embedder.generate_query_embedding(text)
            chunk_norm = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-10)
            
            sim = float(np.dot(claim_norm, chunk_norm))
            if sim > best_sim:
                best_sim = sim
                best_chunk_idx = idx
                best_chunk = chunk
                
        # Classify based on cosine similarity thresholds
        # Typically MiniLM-L6-v2 cosine similarities are:
        # > 0.58 indicates strong semantic similarity (Supported)
        # > 0.40 indicates partial overlap (Partially Supported)
        # < 0.40 indicates poor alignment (Unsupported)
        if best_sim >= 0.58:
            status = "SUPPORTED"
            filename = best_chunk.get("source_filename", "Unknown") if best_chunk else "Unknown"
            page = best_chunk.get("page_number", "Unknown") if best_chunk else "Unknown"
            explanation = f"Semantic match of {best_sim:.2f} with Document Chunk {best_chunk_idx+1} ({filename}, Page {page})."
        elif best_sim >= 0.40:
            status = "PARTIALLY_SUPPORTED"
            filename = best_chunk.get("source_filename", "Unknown") if best_chunk else "Unknown"
            page = best_chunk.get("page_number", "Unknown") if best_chunk else "Unknown"
            explanation = f"Moderate semantic overlap of {best_sim:.2f} with Chunk {best_chunk_idx+1} ({filename}, Page {page})."
        else:
            status = "UNSUPPORTED"
            explanation = f"Low semantic correlation (max similarity {best_sim:.2f}) with all retrieved document chunks."
            
        return status, explanation

if __name__ == "__main__":
    # Quick test harness
    from dotenv import load_dotenv
    load_dotenv()
    
    verifier = ClaimVerifier()
    test_claims = [
        "Bellman-Ford supports negative edge weights.",
        "Dijkstra algorithm uses Fibonacci heaps to achieve better complexity.",
        "The moon is made of green cheese."
    ]
    test_chunks = [
        ({"source_filename": "bellman.pdf", "page_number": 2, "text": "Bellman-Ford computes shortest paths. Unlike Dijkstra, Bellman-Ford works with negative weight edges."}, 0.9),
        ({"source_filename": "dijkstra.pdf", "page_number": 4, "text": "Dijkstra algorithm is a greedy search algorithm for shortest path. It can be implemented with a binary heap or Fibonacci heap."}, 0.8),
    ]
    res = verifier.verify_claims(test_claims, test_chunks)
    print("Verification Report:")
    print(json.dumps(res, indent=2))
