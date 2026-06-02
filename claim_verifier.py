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
        
        # We can optimize LLM verification by processing claims in parallel or sequentially.
        # Since groq is extremely fast, a simple loop is highly reliable.
        for claim in claims:
            verified = False
            
            if use_llm:
                prompt = f"""You are a Fact Verification Assistant.
Analyze whether the given Claim is supported by the provided Context.
Classify the support level into exactly one of the following categories:
- SUPPORTED: The claim is fully and directly supported by the context.
- PARTIALLY_SUPPORTED: The claim is partially supported, but contains minor details, inferences, or numbers not directly in the context.
- UNSUPPORTED: The claim is not supported, is contradicted, or cannot be verified using only the context.

Context:
{context}

Claim:
"{claim}"

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
                                {"role": "system", "content": "You are a precise fact verifier. Be honest and strict. Output only the status and short explanation."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.0,
                            max_tokens=200
                        )
                        output = response.choices[0].message.content.strip().split('\n')
                        status = output[0].strip().upper()
                        
                        # Validate status
                        if status not in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]:
                            # Attempt clean parsing if not exact line
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
                        
                        verifications.append({
                            "claim": claim,
                            "status": status,
                            "explanation": explanation
                        })
                        
                        if status == "SUPPORTED":
                            supported_count += 1
                        elif status == "PARTIALLY_SUPPORTED":
                            partially_supported_count += 1
                        else:
                            unsupported_count += 1
                            
                        verified = True
                        break
                    except Exception as e:
                        # Log error internally and try next model
                        continue
            
            # Semantic search similarity fallback
            if not verified:
                try:
                    status, explanation = self._fallback_verify(claim, retrieved_chunks)
                    verifications.append({
                        "claim": claim,
                        "status": status,
                        "explanation": explanation
                    })
                    
                    if status == "SUPPORTED":
                        supported_count += 1
                    elif status == "PARTIALLY_SUPPORTED":
                        partially_supported_count += 1
                    else:
                        unsupported_count += 1
                except Exception as e:
                    # Absolute safety default
                    verifications.append({
                        "claim": claim,
                        "status": "UNSUPPORTED",
                        "explanation": f"Fallback failed ({e}). Defaulting to unsupported."
                    })
                    unsupported_count += 1

        # Compute grounding score: supported claims + 0.5 * partially supported / total
        # To align with strict Supported Claims / Total Claims formula:
        total_claims = len(claims)
        
        # Feature 5: Supported Claims / Total Claims (percentage)
        # We'll calculate strict score:
        grounding_score = (supported_count / total_claims) * 100.0 if total_claims > 0 else 100.0
        
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
        
        claim_emb = embedder.generate_query_embedding(claim)
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
        # > 0.65 indicates strong semantic similarity (Supported)
        # > 0.45 indicates partial overlap (Partially Supported)
        # < 0.45 indicates poor alignment (Unsupported)
        if best_sim >= 0.65:
            status = "SUPPORTED"
            filename = best_chunk.get("source_filename", "Unknown") if best_chunk else "Unknown"
            page = best_chunk.get("page_number", "Unknown") if best_chunk else "Unknown"
            explanation = f"Semantic match of {best_sim:.2f} with Document Chunk {best_chunk_idx+1} ({filename}, Page {page})."
        elif best_sim >= 0.45:
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
