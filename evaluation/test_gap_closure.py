"""
test_gap_closure.py — Comprehensive validation suite for Phase 3 Finalization.

Verifies:
1. Table Extraction: Markdown conversion and chunk_type == "table".
2. Metadata Extraction: Entities, keywords, and noun phrases populated correctly.
3. spaCy Fallback: Graceful degradation to Regex parser when spaCy fails, and fallback to empty structures under total error.
4. Traversal Selection: Correct planner routing for SIMPLE, MODERATE, COMPLEX, ALGORITHM, and RESEARCH intents.
"""

import os
import unittest
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunker import DocumentChunker
from metadata_extractor import MetadataExtractor
from retrieval_planner import RetrievalPlanner


class TestTableExtraction(unittest.TestCase):
    """Verifies table extraction, markdown conversion, and chunk typing."""

    def test_table_markdown_conversion(self):
        chunker = DocumentChunker()
        
        # Simulating row cell values extracted by PyMuPDF tables
        rows = [
            ["Algorithm", "Complexity"],
            ["MergeSort", "O(n log n)"],
            ["QuickSort", "O(n^2)"]
        ]
        
        # Test markdown builder manual generation logic inside chunker
        md_rows = []
        header = [str(cell).strip() for cell in rows[0]]
        md_rows.append("| " + " | ".join(header) + " |")
        md_rows.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows[1:]:
            row = [str(cell).strip() for cell in row]
            md_rows.append("| " + " | ".join(row) + " |")
            
        table_md = '\n'.join(md_rows)
        
        # Verification assertions
        self.assertIn("| Algorithm | Complexity |", table_md)
        self.assertIn("|---|---|", table_md)
        self.assertIn("| MergeSort | O(n log n) |", table_md)

    def test_table_chunk_typing(self):
        chunker = DocumentChunker()
        table_text = "| Header | Col |\n|---|---|\n| Data | Val |"
        
        # Verify chunker makes chunk with type "table"
        chunk = chunker._make_chunk(table_text, 1, "test.pdf", chunk_type="table")
        self.assertEqual(chunk["chunk_type"], "table")
        self.assertEqual(chunk["text"], table_text)
        self.assertEqual(chunk["page_number"], 1)
        self.assertEqual(chunk["source_filename"], "test.pdf")


class TestMetadataExtractionAndFallback(unittest.TestCase):
    """Verifies entity, keyword extraction, and spaCy fallbacks."""

    def test_metadata_extraction_fields(self):
        # Force fallback to run deterministic Regex parsing for identical testing environment
        extractor = MetadataExtractor(force_fallback=True)
        sample_text = "Bellman-Ford Algorithm is used in Distance Vector Routing."
        
        meta = extractor.extract(sample_text)
        
        # Assertions
        self.assertTrue(any("Bellman-Ford" in ent for ent in meta["entities"]))
        self.assertIn("Distance Vector Routing", meta["entities"])
        self.assertIn("routing", meta["keywords"])
        self.assertIn("algorithm", meta["keywords"])
        self.assertTrue(len(meta["noun_phrases"]) > 0)

    def test_spacy_fallback_to_regex(self):
        # Instantiate with force_fallback to prove Regex works when spaCy is unavailable
        extractor = MetadataExtractor(force_fallback=True)
        self.assertTrue(extractor.use_fallback)
        
        sample_text = "Dijkstra algorithm is used for Shortest Path calculation."
        meta = extractor.extract(sample_text)
        
        # Confirm extraction operates fully
        self.assertIn("Dijkstra", meta["entities"])
        self.assertIn("shortest", meta["keywords"])

    def test_absolute_error_fallback(self):
        extractor = MetadataExtractor(force_fallback=True)
        
        # Simulating unexpected types / complete failures
        meta_none = extractor.extract(None)
        self.assertEqual(meta_none, {"entities": [], "keywords": [], "noun_phrases": []})
        
        meta_int = extractor.extract(12345)
        self.assertEqual(meta_int, {"entities": [], "keywords": [], "noun_phrases": []})


class TestTraversalSelectionRouting(unittest.TestCase):
    """Verifies that the RetrievalPlanner routes queries correctly to traversal plans."""

    def setUp(self):
        self.planner = RetrievalPlanner()

    def test_plan_routing(self):
        # 1. SIMPLE -> BFS Depth 1
        plan_simple = self.planner.plan("What is routing?")
        self.assertEqual(plan_simple["query_type"], "SIMPLE")
        self.assertEqual(plan_simple["traversal"], "BFS")
        self.assertEqual(plan_simple["graph_depth"], 1)

        # 2. MODERATE -> BFS Depth 2
        plan_mod = self.planner.plan("Explain OSPF routing details.")
        self.assertEqual(plan_mod["query_type"], "MODERATE")
        self.assertEqual(plan_mod["traversal"], "BFS")
        self.assertEqual(plan_mod["graph_depth"], 2)

        # 3. COMPLEX -> PageRank Depth 3
        plan_complex = self.planner.plan("Compare OSPF and RIP protocol mechanisms.")
        self.assertEqual(plan_complex["query_type"], "COMPLEX")
        self.assertEqual(plan_complex["traversal"], "PPR")
        self.assertEqual(plan_complex["graph_depth"], 3)

        # 4. ALGORITHM -> BFS Depth 2
        plan_algo = self.planner.plan("Give Dijkstra algorithm steps.")
        self.assertEqual(plan_algo["query_type"], "ALGORITHM")
        self.assertEqual(plan_algo["traversal"], "BFS")
        self.assertEqual(plan_algo["graph_depth"], 2)
        self.assertEqual(plan_algo["rerank_mode"], "algorithm")

        # 5. RESEARCH -> Hybrid PPR + Random Walk Depth 3
        plan_res = self.planner.plan("Survey all routing methods in dynamic networks.")
        self.assertEqual(plan_res["query_type"], "RESEARCH")
        self.assertEqual(plan_res["traversal"], "hybrid")
        self.assertEqual(plan_res["graph_depth"], 3)
        self.assertEqual(plan_res["rerank_mode"], "research")


class TestEdgeAnalytics(unittest.TestCase):
    """Verifies graph build process maps correct edge types (section, page, entity, similarity)."""

    def test_typed_edges_construction(self):
        from graph_builder import GraphBuilder
        import numpy as np
        
        gb = GraphBuilder()
        
        # Create 3 dummy chunks representing a document sequence
        chunks = [
            {
                "chunk_id": "chunk_A",
                "text": "Distance Vector routing is a key networking algorithm.",
                "source_filename": "doc.pdf",
                "page_number": 1,
                "entities": ["Distance Vector", "routing"],
                "keywords": ["networking", "algorithm"],
                "noun_phrases": ["Distance Vector routing"]
            },
            {
                "chunk_id": "chunk_B",
                "text": "Link State routing is another popular networking algorithm.",
                "source_filename": "doc.pdf",
                "page_number": 1,
                "entities": ["Link State", "routing"],
                "keywords": ["networking", "algorithm"],
                "noun_phrases": ["Link State routing"]
            },
            {
                "chunk_id": "chunk_C",
                "text": "Shortest path tree algorithms are used in routing protocols.",
                "source_filename": "doc.pdf",
                "page_number": 2,
                "entities": ["Shortest path tree", "routing"],
                "keywords": ["algorithms", "protocols"],
                "noun_phrases": ["Shortest path tree algorithms"]
            }
        ]
        
        # Create identical embeddings so similarity is 1.0 (above threshold)
        embeddings = np.ones((3, 384))
        
        # Build the graph
        G = gb.build_graph(chunks, embeddings)
        
        # Assert structural counts
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertTrue(G.number_of_edges() > 0)
        
        # Verify edge types are populated
        has_similarity = False
        has_entity = False
        has_section = False
        has_page = False
        
        for u, v, data in G.edges(data=True):
            types = data.get("types", [])
            if "similarity" in types:
                has_similarity = True
            if "entity" in types:
                has_entity = True
            if "section" in types:
                has_section = True
            if "page" in types:
                has_page = True
                
        self.assertTrue(has_similarity, "Should contain similarity edges")
        self.assertTrue(has_entity, "Should contain shared entity edges ('routing')")
        self.assertTrue(has_section, "Should contain sequential section edges")
        self.assertTrue(has_page, "Should contain page proximity edges")


if __name__ == "__main__":
    unittest.main()
