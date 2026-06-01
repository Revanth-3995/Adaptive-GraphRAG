"""
Query Classifier - categorizes user queries into specific retrieval strategies.
"""
import re

class QueryClassifier:
    """
    Classifies a user query into one of the following categories:
    - SIMPLE: Definition lookup (What is X?)
    - MODERATE: Detailed explanation (Explain X in detail, Describe how Y works)
    - ALGORITHM: Requesting pseudocode, procedures, steps, algorithms (Give Dijkstra algorithm, Write Bellman-Ford)
    - COMPLEX: Comparisons, multi-hop retrieval (Compare X and Y)
    - RESEARCH: Exploration, synthesis across documents (Analyze the evolution of X, What are all approaches to Y?)
    """
    def __init__(self):
        # We can implement a simple heuristic-based classifier for now.
        pass

    def classify(self, query: str) -> str:
        query_lower = query.lower()

        # 1. RESEARCH
        research_keywords = ['analyze', 'all approaches', 'evolution', 'explore', 'synthesis', 'review', 'comprehensive', 'survey']
        if any(keyword in query_lower for keyword in research_keywords):
            return "RESEARCH"

        # 2. ALGORITHM
        algorithm_keywords = ['algorithm', 'pseudocode', 'procedure', 'steps', 'implement', 'code']
        if any(keyword in query_lower for keyword in algorithm_keywords):
            return "ALGORITHM"

        # 3. COMPLEX
        complex_keywords = ['compare', 'difference', 'vs', 'versus', 'pros and cons', 'better']
        if any(keyword in query_lower for keyword in complex_keywords):
            return "COMPLEX"

        # 4. MODERATE
        moderate_keywords = ['explain', 'describe', 'how does', 'detail']
        if any(keyword in query_lower for keyword in moderate_keywords):
            return "MODERATE"

        # 5. SIMPLE (Fallback)
        return "SIMPLE"

if __name__ == "__main__":
    qc = QueryClassifier()
    assert qc.classify("What is routing?") == "SIMPLE"
    assert qc.classify("Compare Bellman-Ford and Dijkstra") == "COMPLEX"
    assert qc.classify("Give Dijkstra algorithm") == "ALGORITHM"
    assert qc.classify("Analyze the evolution of routing algorithms") == "RESEARCH" # Research priority first
    assert qc.classify("Survey all routing methods") == "RESEARCH"
    print("All tests passed.")