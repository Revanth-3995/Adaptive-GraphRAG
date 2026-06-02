import unittest
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from citation_verifier import CitationVerifier
from claim_extractor import ClaimExtractor
from claim_verifier import ClaimVerifier

class TestAnswerIntelligence(unittest.TestCase):
    """
    Test suite for Phase 4 Answer Intelligence Layer.
    """

    def setUp(self):
        self.citation_verifier = CitationVerifier()
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimVerifier()

    def test_citation_verification(self):
        """Verify that CitationVerifier parses and matches references correctly."""
        # Setup simulated retrieved chunks
        chunks = [
            ({"source_filename": "network_routing.pdf", "page_number": 5, "text": "Dijkstra is link-state"}, 0.9),
            ({"source_filename": "algorithms.pdf", "page_number": 12, "text": "Bellman-Ford supports negative weights"}, 0.8),
        ]
        
        # Test Case 1: Valid Index Citation
        ans_1 = "Bellman-Ford is in [Source 2]."
        res_1 = self.citation_verifier.verify_citations(ans_1, chunks)
        self.assertIn("[Source 2] [Verified]", res_1["verified_answer"])
        self.assertEqual(res_1["verified_count"], 1)
        self.assertEqual(res_1["failed_count"], 0)
        self.assertTrue(res_1["citations"][0]["verified"])
        
        # Test Case 2: Invalid Index Citation (out of bounds)
        ans_2 = "RIP is in [Source 3]."
        res_2 = self.citation_verifier.verify_citations(ans_2, chunks)
        self.assertIn("[Source 3] [Unverified]", res_2["verified_answer"])
        self.assertEqual(res_2["failed_count"], 1)
        self.assertFalse(res_2["citations"][0]["verified"])

        # Test Case 3: Valid File & Page Citation
        ans_3 = "Dijkstra is defined on [network_routing.pdf, Page 5]."
        res_3 = self.citation_verifier.verify_citations(ans_3, chunks)
        self.assertIn("[network_routing.pdf, Page 5] [Verified]", res_3["verified_answer"])
        self.assertEqual(res_3["verified_count"], 1)
        
        # Test Case 4: Invalid Page Citation
        ans_4 = "Dijkstra on [network_routing.pdf, Page 7]."
        res_4 = self.citation_verifier.verify_citations(ans_4, chunks)
        self.assertIn("[network_routing.pdf, Page 7] [Unverified]", res_4["verified_answer"])
        self.assertEqual(res_4["failed_count"], 1)

    def test_claim_extraction_fallback(self):
        """Verify that the regex fallback splits sentences and cleans bullets correctly."""
        text = "- Bellman-Ford computes shortest paths.\n* Dijkstra is greedy.\n- Pronoun it is replaced."
        extracted = self.claim_extractor._fallback_extract(text)
        self.assertEqual(len(extracted), 3)
        self.assertEqual(extracted[0], "Bellman-Ford computes shortest paths.")
        self.assertEqual(extracted[1], "Dijkstra is greedy.")
        self.assertEqual(extracted[2], "Pronoun it is replaced.")

    def test_claim_verifier_fallback_matching(self):
        """Verify that the cosine similarity fallback scores semantic matches correctly."""
        chunks = [
            ({"source_filename": "text.pdf", "page_number": 1, "text": "Bellman-Ford works with negative weight edges in routing graphs."}, 0.9),
        ]
        
        # Strong semantic match
        status, exp = self.claim_verifier._fallback_verify("Bellman-Ford supports negative weights.", chunks)
        self.assertEqual(status, "SUPPORTED")
        self.assertIn("Semantic match", exp)
        
        # Weak/no match
        status_un, exp_un = self.claim_verifier._fallback_verify("The solar eclipse is next week.", chunks)
        self.assertEqual(status_un, "UNSUPPORTED")
        self.assertIn("Low semantic correlation", exp_un)

    def test_grounding_and_trust_mapping(self):
        """Assert grounding scores map to correct trust categories and hallucination risks."""
        # 100% Grounded
        res_high = self.claim_verifier.verify_claims([], [])
        self.assertEqual(res_high["grounding_score"], 100.0)
        self.assertEqual(res_high["trust_level"], "VERIFIED")
        self.assertEqual(res_high["hallucination_risk"], "LOW")
        
        # Simulate custom results via mock inputs
        # Test Case: 1 supported claim, 2 unsupported claims = 33.3% grounding
        claims = ["C1", "C2", "C3"]
        chunks = [
            ({"source_filename": "text.pdf", "page_number": 1, "text": "Details for C1"}, 0.9)
        ]
        
        # Force LLM skip to trigger fallback verification (C1 will match first chunk, C2/C3 will be unsupported)
        import os
        # Temporarily unset GROQ key to force fallback path
        old_key = os.environ.get("GROQ_API_KEY")
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
            
        try:
            # We mock the _fallback_verify to get deterministic results
            orig_fallback = self.claim_verifier._fallback_verify
            def mock_fallback(claim, retrieved_chunks):
                if claim == "C1":
                    return "SUPPORTED", "Mocked support"
                elif claim == "C2":
                    return "PARTIALLY_SUPPORTED", "Mocked partial"
                else:
                    return "UNSUPPORTED", "Mocked unsupported"
            
            self.claim_verifier._fallback_verify = mock_fallback
            report = self.claim_verifier.verify_claims(claims, chunks)
            
            # Grounding = 1 supported / 3 total = 33.3%
            self.assertEqual(report["grounding_score"], 33.3)
            self.assertEqual(report["trust_level"], "UNTRUSTED")
            self.assertEqual(report["hallucination_risk"], "HIGH")
            
            # Restore fallback
            self.claim_verifier._fallback_verify = orig_fallback
        finally:
            if old_key is not None:
                os.environ["GROQ_API_KEY"] = old_key

if __name__ == "__main__":
    unittest.main()
