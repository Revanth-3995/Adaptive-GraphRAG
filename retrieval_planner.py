"""
retrieval_planner.py — Central planning and strategy brain for GraphRAG retrieval.

Takes a user query, invokes the QueryClassifier, and formulates a structured
Retrieval Plan detailing parameter selections:
- use_hyde (caching enabled)
- use_decomposition (conditional)
- graph_depth (1 to 3+)
- traversal strategy (BFS, PageRank, Hybrid, etc.)
- rerank_mode (algorithm, comparison, research, standard)

Ensures retrieval is observable, trace-logged, and highly explainable.
"""
from typing import Dict, Any
from retrieval.query_classifier import QueryClassifier


class RetrievalPlanner:
    """
    RetrievalPlanner translates classified query intents into explainable
    retrieval execution plans.
    """

    def __init__(self):
        self.classifier = QueryClassifier()

    def plan(self, query: str) -> Dict[str, Any]:
        """
        Formulates a Retrieval Plan for the given query.
        
        Args:
            query: The user's input question.
            
        Returns:
            A dictionary containing the structured Retrieval Plan parameters.
        """
        # Determine intent first
        query_type = self.classifier.classify(query)
        
        # Plan defaults
        plan_params = {
            "query_type": query_type,
            "use_hyde": True,
            "use_decomposition": False,
            "graph_depth": 1,
            "traversal": "BFS",
            "rerank_mode": "standard"
        }

        # Apply strategy specifications
        if query_type == "SIMPLE":
            plan_params.update({
                "use_hyde": True,
                "use_decomposition": False,
                "graph_depth": 1,
                "traversal": "BFS",
                "rerank_mode": "standard"
            })
        elif query_type == "MODERATE":
            plan_params.update({
                "use_hyde": True,
                "use_decomposition": False,
                "graph_depth": 2,
                "traversal": "BFS",
                "rerank_mode": "standard"
            })
        elif query_type == "COMPLEX":
            plan_params.update({
                "use_hyde": True,
                "use_decomposition": True,
                "graph_depth": 3,
                "traversal": "PPR",
                "rerank_mode": "comparison"
            })
        elif query_type == "ALGORITHM":
            plan_params.update({
                "use_hyde": True,
                "use_decomposition": False,
                "graph_depth": 2,
                "traversal": "BFS",  # BFS with Algorithm Boosting
                "rerank_mode": "algorithm"
            })
        elif query_type == "RESEARCH":
            plan_params.update({
                "use_hyde": True,
                "use_decomposition": True,
                "graph_depth": 3,
                "traversal": "hybrid",  # Random Walk + Personalized PageRank
                "rerank_mode": "research"
            })

        return plan_params


if __name__ == "__main__":
    # Test planning
    planner = RetrievalPlanner()
    test_queries = [
        "What is routing?",
        "Compare OSPF and RIP protocols.",
        "Give Dijkstra algorithm.",
        "Survey all routing methods in literature."
    ]
    print("Testing RetrievalPlanner...")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        p = planner.plan(q)
        for k, v in p.items():
            print(f"  - {k}: {v}")
