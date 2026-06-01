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
        
    def graph_based_retrieval(
        self,
        seed_chunks: List[Dict[str, Any]],
        max_depth: int = 1,
        use_bfs: bool = True,
        query_type: str = "SIMPLE"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Expands the retrieval context by adaptively traversing the graph from seed chunks
        based on the classified query type.
        """
        if not self.graph_builder.graph or self.graph_builder.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty or not loaded.")
            return []
            
        seed_ids = [chunk["chunk_id"] for chunk in seed_chunks if chunk["chunk_id"] in self.graph_builder.graph]
        if not seed_ids:
            return []

        expanded_results = []
        visited_ids = set(seed_ids)
        import networkx as nx

        # --- Adaptive Traversal Strategies ---
        if query_type == "SIMPLE":
            # BFS Depth 1
            from collections import deque
            for seed_id in seed_ids:
                queue = deque([(seed_id, 0)])
                while queue:
                    current_id, depth = queue.popleft()
                    if depth == 1 and current_id not in visited_ids:
                        visited_ids.add(current_id)
                        node_data = self.graph_builder.graph.nodes[current_id]
                        chunk_dict = {"chunk_id": current_id, "text": node_data.get("text", ""), "source_filename": node_data.get("source", ""), "page_number": node_data.get("page", 0)}
                        expanded_results.append((chunk_dict, 0.9))
                    if depth < 1:
                        for neighbor in self.graph_builder.graph.neighbors(current_id):
                            if neighbor not in visited_ids:
                                queue.append((neighbor, depth + 1))

        elif query_type == "MODERATE":
            # BFS Depth 2
            from collections import deque
            for seed_id in seed_ids:
                queue = deque([(seed_id, 0)])
                while queue:
                    current_id, depth = queue.popleft()
                    if depth > 0 and current_id not in visited_ids:
                        visited_ids.add(current_id)
                        node_data = self.graph_builder.graph.nodes[current_id]
                        chunk_dict = {"chunk_id": current_id, "text": node_data.get("text", ""), "source_filename": node_data.get("source", ""), "page_number": node_data.get("page", 0)}
                        expanded_results.append((chunk_dict, 0.9 ** depth))
                    if depth < 2:
                        for neighbor in self.graph_builder.graph.neighbors(current_id):
                            if neighbor not in visited_ids:
                                queue.append((neighbor, depth + 1))

        elif query_type == "ALGORITHM":
            # BFS Depth 2, prioritizing SECTION_EDGE and ENTITY_EDGE
            from collections import deque
            for seed_id in seed_ids:
                queue = deque([(seed_id, 0)])
                while queue:
                    current_id, depth = queue.popleft()
                    if depth > 0 and current_id not in visited_ids:
                        visited_ids.add(current_id)
                        node_data = self.graph_builder.graph.nodes[current_id]
                        chunk_dict = {"chunk_id": current_id, "text": node_data.get("text", ""), "source_filename": node_data.get("source", ""), "page_number": node_data.get("page", 0)}
                        # Base score on depth
                        base_score = 0.9 ** depth
                        # Boost score slightly if we reached this node via a preferred edge
                        # Since BFS queue doesn't store edge path easily here, we just apply base score.
                        # True prioritization happens implicitly by taking these edges.
                        expanded_results.append((chunk_dict, base_score))
                    if depth < 2:
                        # Prioritize specific edges
                        neighbors = []
                        for neighbor in self.graph_builder.graph.neighbors(current_id):
                            if neighbor not in visited_ids:
                                edge_data = self.graph_builder.graph[current_id][neighbor]
                                e_type = edge_data.get("edge_type", "SIMILARITY_EDGE")
                                priority = 1 if e_type in ["SECTION_EDGE", "ENTITY_EDGE"] else 0
                                neighbors.append((priority, neighbor))
                        
                        # Sort by priority descending
                        neighbors.sort(key=lambda x: x[0], reverse=True)
                        for _, neighbor in neighbors:
                            queue.append((neighbor, depth + 1))

        elif query_type == "COMPLEX":
            # Personalized PageRank
            personalization = {node: (1.0 if node in seed_ids else 0.0) for node in self.graph_builder.graph.nodes}
            try:
                pr = nx.pagerank(self.graph_builder.graph, alpha=0.85, personalization=personalization, weight='weight')
                # Sort and take top 10 non-seed nodes
                sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
                for node_id, score in sorted_pr:
                    if node_id not in visited_ids and len(expanded_results) < 10:
                        visited_ids.add(node_id)
                        node_data = self.graph_builder.graph.nodes[node_id]
                        chunk_dict = {"chunk_id": node_id, "text": node_data.get("text", ""), "source_filename": node_data.get("source", ""), "page_number": node_data.get("page", 0)}
                        # PR scores are small, normalize up to ~0.9 range roughly
                        expanded_results.append((chunk_dict, min(score * 10, 0.9)))
            except Exception as e:
                print(f"PageRank failed: {e}. Falling back to BFS.")
                return self.graph_based_retrieval(seed_chunks, max_depth=2, use_bfs=True, query_type="MODERATE")

        elif query_type == "RESEARCH":
            # Random Walk + PageRank approximation (using standard PageRank with lower alpha for broader exploration)
            personalization = {node: (1.0 if node in seed_ids else 0.0) for node in self.graph_builder.graph.nodes}
            try:
                pr = nx.pagerank(self.graph_builder.graph, alpha=0.70, personalization=personalization, weight='weight')
                sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
                for node_id, score in sorted_pr:
                    if node_id not in visited_ids and len(expanded_results) < 15:
                        visited_ids.add(node_id)
                        node_data = self.graph_builder.graph.nodes[node_id]
                        chunk_dict = {"chunk_id": node_id, "text": node_data.get("text", ""), "source_filename": node_data.get("source", ""), "page_number": node_data.get("page", 0)}
                        expanded_results.append((chunk_dict, min(score * 15, 0.9)))
            except Exception as e:
                print(f"PageRank failed: {e}. Falling back to BFS.")
                return self.graph_based_retrieval(seed_chunks, max_depth=2, use_bfs=True, query_type="MODERATE")

        else:
            # Fallback BFS depth 1
            return self.graph_based_retrieval(seed_chunks, max_depth=1, use_bfs=True, query_type="SIMPLE")

        return expanded_results

if __name__ == "__main__":
    pass
