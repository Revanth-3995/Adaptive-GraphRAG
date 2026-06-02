"""
QueryClassifier — Deterministic intent understanding for adaptive GraphRAG retrieval.

Classifies incoming queries into five distinct intents in a strict priority order:
1. RESEARCH
2. ALGORITHM
3. COMPLEX
4. MODERATE
5. SIMPLE

This ensures the retrieval pipeline dynamically allocates the appropriate strategy,
traversal algorithms, and depths based on the query complexity.
"""
import re


class QueryClassifier:
    """
    Deterministic query intent classifier.
    Ensures stable, fast, and repeatable query classification.
    """

    def __init__(self):
        # 1. RESEARCH Patterns: Exploratory, overview, synthesize, survey, literature
        self.research_patterns = [
            r"\banalyze\b",
            r"\banalysis\b",
            r"\bsurvey\b",
            r"\breview\b",
            r"\boverview\b",
            r"\bsynthesize\b",
            r"\bsynthesis\b",
            r"\bapproaches\b",
            r"\bmethods\b",
            r"\btechniques\b",
            r"\bliterature\b",
            r"\ball\s+routing\b"
        ]

        # 2. ALGORITHM Patterns: Chunks with procedures, code, steps, specific algorithms
        self.algorithm_patterns = [
            r"\balgorithm\b",
            r"\bpseudocode\b",
            r"\bcode\b",
            r"\bprocedure\b",
            r"\bstep-by-step\b",
            r"\bsteps\b",
            r"\bworkflow\b",
            r"\bdijkstra\b",
            r"\bbellman-ford\b",
            r"\bprim\b",
            r"\bkruskal\b"
        ]

        # 3. COMPLEX Patterns: Comparison, contrast, versus, relations
        self.complex_patterns = [
            r"\bcompare\b",
            r"\bcontrast\b",
            r"\bdifference\b",
            r"\bdifferences\b",
            r"\bvs\b",
            r"\bversus\b",
            r"\bdistinguish\b",
            r"\brelationship\b",
            r"\bdifferentiate\b",
            r"\bbetter\b",
            r"\bperform\b"
        ]

        # 4. MODERATE Patterns: Explanation, descriptions, mechanics
        self.moderate_patterns = [
            r"\bexplain\b",
            r"\bdescribe\b",
            r"\bhow\s+(?:does|do|is|are|can)\b",
            r"\bdiscuss\b",
            r"\bwhy\s+(?:does|do|is|are|was)\b",
            r"\bmechanism\b",
            r"\bdetails\b"
        ]

        # 5. SIMPLE Patterns: Basic lookups and definitions
        self.simple_patterns = [
            r"\bwhat\s+(?:is|are|was|were)\b",
            r"\bwho\s+(?:is|are|was|were)\b",
            r"\bwhere\s+(?:is|are|was|were)\b",
            r"\bwhen\s+(?:was|were|is)\b",
            r"\bdefine\b",
            r"\bdefinition\b"
        ]

    def classify(self, query: str) -> str:
        """
        Classifies the input query deterministically using a strict priority order.
        
        Priority Order:
        RESEARCH -> ALGORITHM -> COMPLEX -> MODERATE -> SIMPLE
        
        Args:
            query: The user's input question.
            
        Returns:
            The uppercase string representing the classification:
            'RESEARCH', 'ALGORITHM', 'COMPLEX', 'MODERATE', or 'SIMPLE'.
        """
        if not query or not isinstance(query, str):
            return "SIMPLE"

        # Normalize the query: lowercase and remove leading/trailing whitespace
        query_normalized = query.strip().lower()

        # 1. Check RESEARCH
        for pattern in self.research_patterns:
            if re.search(pattern, query_normalized):
                return "RESEARCH"

        # 2. Check ALGORITHM
        for pattern in self.algorithm_patterns:
            if re.search(pattern, query_normalized):
                return "ALGORITHM"

        # 3. Check COMPLEX
        for pattern in self.complex_patterns:
            if re.search(pattern, query_normalized):
                return "COMPLEX"

        # 4. Check MODERATE
        for pattern in self.moderate_patterns:
            if re.search(pattern, query_normalized):
                return "MODERATE"

        # 5. Check SIMPLE (explicit matches)
        for pattern in self.simple_patterns:
            if re.search(pattern, query_normalized):
                return "SIMPLE"

        # Fallback to SIMPLE if no rules matched
        return "SIMPLE"


if __name__ == "__main__":
    # Self-test code
    classifier = QueryClassifier()
    test_cases = [
        ("What is routing?", "SIMPLE"),
        ("Explain routing", "MODERATE"),
        ("Compare RIP and OSPF", "COMPLEX"),
        ("Give Dijkstra algorithm", "ALGORITHM"),
        ("Analyze routing approaches", "RESEARCH"),
        ("Survey all routing methods", "RESEARCH"),
    ]
    print("Testing QueryClassifier...")
    all_passed = True
    for q, expected in test_cases:
        res = classifier.classify(q)
        status = "PASSED" if res == expected else f"FAILED (got {res})"
        if res != expected:
            all_passed = False
        print(f"  Query: '{q}' -> Predicted: {res} | Expected: {expected} | {status}")
    print(f"Classifier result: {'All tests PASSED!' if all_passed else 'Some tests FAILED.'}")
