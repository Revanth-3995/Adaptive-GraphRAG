from typing import List, Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_builder import GraphBuilder

class GraphRetriever:
    """
    GraphRetriever is the bridge between our standard searches (Vector/BM25) and our Graph.
    
    The Flow:
    1. A user query retrieves "seed" nodes from Vector and BM25 search.
    2. These seed nodes might only be the tip of the iceberg.
    3. The GraphRetriever takes these seed nodes and uses our GraphBuilder's traversals 
       (BFS/DFS) to explore the surrounding semantic neighborhood.
    4. It returns these neighborhood nodes as additional context.
    
    Why this matters?
    This is what makes "GraphRAG" different from standard "RAG". It finds related context 
    that might not contain the explicit query terms but is heavily linked to the chunks that do.
    """
    
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder
        
    def graph_based_retrieval(self, seed_chunks: List[Dict[str, Any]], max_depth: int = 1, use_bfs: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """
        Expands the retrieval context by traversing the graph from seed chunks.
        
        Note on Scoring:
        Graph retrieved nodes aren't directly scored against the query. Instead, we 
        score them based on their proximity/edge-weights to the seed nodes. For Phase 1/2 
        baseline, we'll assign a diminishing score based on depth.
        
        Returns:
            List of tuples: (chunk_dict, graph_score)
        """
        if not self.graph_builder.graph or self.graph_builder.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty or not loaded.")
            return []
            
        seed_ids = [chunk["chunk_id"] for chunk in seed_chunks]
        expanded_results = []
        visited_ids = set(seed_ids) # We don't want to return the seeds themselves as "graph results"
        
        # We'll use BFS manually here so we can keep track of depth for scoring
        from collections import deque
        
        for seed_id in seed_ids:
            if seed_id not in self.graph_builder.graph:
                continue
                
            queue = deque([(seed_id, 0)])
            
            while queue:
                current_id, depth = queue.popleft()
                
                if depth > 0:
                    if current_id not in visited_ids:
                        visited_ids.add(current_id)
                        # Get chunk metadata from the graph node
                        node_data = self.graph_builder.graph.nodes[current_id]
                        
                        # Reconstruct chunk dict
                        chunk_dict = {
                            "chunk_id": current_id,
                            "text": node_data.get("text", ""),
                            "source_filename": node_data.get("source", ""),
                            "page_number": node_data.get("page", 0)
                        }
                        
                        # Heuristic score: closer nodes get higher scores.
                        # Depth 1: 0.9, Depth 2: 0.81, etc.
                        graph_score = 0.9 ** depth
                        expanded_results.append((chunk_dict, graph_score))
                    else:
                        # Already visited, skip adding its neighbors again
                        continue
                    
                if depth < max_depth:
                    for neighbor in self.graph_builder.graph.neighbors(current_id):
                        if neighbor not in visited_ids:
                            queue.append((neighbor, depth + 1))
                            
        return expanded_results

if __name__ == "__main__":
    pass
