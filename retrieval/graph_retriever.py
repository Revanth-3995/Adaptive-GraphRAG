import sys
import os
import random
import networkx as nx
from typing import List, Dict, Any, Tuple, Set
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_builder import GraphBuilder


class GraphRetriever:
    """
    GraphRetriever is the bridge between our standard searches (Vector/BM25) and our Graph.
    
    The Flow:
    1. A user query retrieves "seed" nodes from Vector and BM25 search.
    2. These seed nodes might only be the tip of the iceberg.
    3. The GraphRetriever takes these seed nodes and uses our GraphBuilder's traversals 
       (BFS/DFS), or advanced traversals (PPR, Random Walk) to explore the neighborhood.
    4. It returns these neighborhood nodes as additional context.
    """
    
    def __init__(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder

    def _get_reachable_nodes(self, seed_ids: List[str], max_depth: int) -> Set[str]:
        """Utility to get all nodes reachable within max_depth hops from seeds."""
        reachable = set()
        if not self.graph_builder.graph:
            return reachable
            
        for seed_id in seed_ids:
            if seed_id not in self.graph_builder.graph:
                continue
            queue = deque([(seed_id, 0)])
            visited = {seed_id}
            while queue:
                curr, depth = queue.popleft()
                reachable.add(curr)
                if depth < max_depth:
                    for neighbor in self.graph_builder.graph.neighbors(curr):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
        return reachable

    def personalized_pagerank_traversal(
        self,
        seed_chunks: List[Dict[str, Any]],
        max_depth: int = 3,
        top_n: int = 15
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Runs Personalized PageRank (PPR) on the graph personalized to the seed chunks.
        Filters results to only nodes within max_depth hops and excludes seeds.
        """
        if not self.graph_builder.graph or self.graph_builder.graph.number_of_nodes() == 0:
            return []

        seed_ids = [chunk["chunk_id"] for chunk in seed_chunks]
        valid_seeds = [sid for sid in seed_ids if sid in self.graph_builder.graph]
        if not valid_seeds:
            return []

        # Find valid subgraph nodes within depth to filter
        reachable = self._get_reachable_nodes(valid_seeds, max_depth)
        if not reachable:
            return []

        # Create personalization dictionary
        personalization = {sid: 1.0 for sid in valid_seeds}

        try:
            # Calculate PageRank
            scores = nx.pagerank(
                self.graph_builder.graph,
                alpha=0.85,
                personalization=personalization,
                weight='weight'
            )

            # Filter and sort
            expanded_results = []
            for node_id, score in scores.items():
                if node_id in reachable and node_id not in seed_ids:
                    node_data = self.graph_builder.graph.nodes[node_id]
                    chunk_dict = {
                        "chunk_id": node_id,
                        "text": node_data.get("text", ""),
                        "source_filename": node_data.get("source", ""),
                        "page_number": node_data.get("page", 0)
                    }
                    expanded_results.append((chunk_dict, float(score)))

            expanded_results.sort(key=lambda x: x[1], reverse=True)
            return expanded_results[:top_n]
        except Exception as e:
            print(f"Warning: Personalized PageRank failed ({e}). Falling back to BFS.")
            return self.bfs_graph_traversal(seed_chunks, max_depth)

    def random_walk_traversal(
        self,
        seed_chunks: List[Dict[str, Any]],
        max_depth: int = 3,
        walk_length: int = 10,
        num_walks: int = 15,
        top_n: int = 15
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Runs semantic-similarity biased Random Walks starting from seed nodes.
        Filters results to only nodes within max_depth hops and excludes seeds.
        """
        if not self.graph_builder.graph or self.graph_builder.graph.number_of_nodes() == 0:
            return []

        seed_ids = [chunk["chunk_id"] for chunk in seed_chunks]
        valid_seeds = [sid for sid in seed_ids if sid in self.graph_builder.graph]
        if not valid_seeds:
            return []

        reachable = self._get_reachable_nodes(valid_seeds, max_depth)
        
        visited_counts = {}
        for _ in range(num_walks):
            for seed in valid_seeds:
                curr = seed
                for _ in range(walk_length):
                    neighbors = list(self.graph_builder.graph.neighbors(curr))
                    if not neighbors:
                        break
                    
                    # Compute weights (cosine similarities from edges)
                    weights = [self.graph_builder.graph[curr][nbr].get('weight', 1.0) for nbr in neighbors]
                    sum_w = sum(weights)
                    probs = [w / sum_w for w in weights] if sum_w > 0 else [1.0 / len(neighbors)] * len(neighbors)
                    
                    # Choose next node biased by semantic similarity weights
                    curr = random.choices(neighbors, weights=probs, k=1)[0]
                    
                    # Track visit counts for nodes in reachable set (excluding seeds)
                    if curr in reachable and curr not in seed_ids:
                        visited_counts[curr] = visited_counts.get(curr, 0) + 1

        if not visited_counts:
            return []

        # Normalize and package results
        expanded_results = []
        max_count = max(visited_counts.values())
        for node_id, count in visited_counts.items():
            node_data = self.graph_builder.graph.nodes[node_id]
            chunk_dict = {
                "chunk_id": node_id,
                "text": node_data.get("text", ""),
                "source_filename": node_data.get("source", ""),
                "page_number": node_data.get("page", 0)
            }
            # Normalize heuristic score between 0.1 and 0.95
            score = 0.1 + 0.85 * (count / max_count)
            expanded_results.append((chunk_dict, score))

        expanded_results.sort(key=lambda x: x[1], reverse=True)
        return expanded_results[:top_n]

    def bfs_graph_traversal(
        self,
        seed_chunks: List[Dict[str, Any]],
        max_depth: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Classic BFS traversal (original logic)."""
        if not self.graph_builder.graph or self.graph_builder.graph.number_of_nodes() == 0:
            return []
            
        seed_ids = [chunk["chunk_id"] for chunk in seed_chunks]
        expanded_results = []
        visited_ids = set(seed_ids)
        
        for seed_id in seed_ids:
            if seed_id not in self.graph_builder.graph:
                continue
                
            queue = deque([(seed_id, 0)])
            
            while queue:
                current_id, depth = queue.popleft()
                
                if depth > 0:
                    if current_id not in visited_ids:
                        visited_ids.add(current_id)
                        node_data = self.graph_builder.graph.nodes[current_id]
                        
                        chunk_dict = {
                            "chunk_id": current_id,
                            "text": node_data.get("text", ""),
                            "source_filename": node_data.get("source", ""),
                            "page_number": node_data.get("page", 0)
                        }
                        
                        graph_score = 0.9 ** depth
                        expanded_results.append((chunk_dict, graph_score))
                    else:
                        continue
                    
                if depth < max_depth:
                    for neighbor in self.graph_builder.graph.neighbors(current_id):
                        if neighbor not in visited_ids:
                            queue.append((neighbor, depth + 1))
                            
        return expanded_results

    def graph_based_retrieval(
        self,
        seed_chunks: List[Dict[str, Any]],
        max_depth: int = 1,
        use_bfs: bool = True,
        strategy: str = "BFS"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Expands the retrieval context by traversing the graph using various adaptive strategies.
        
        Supported Strategies:
        - BFS: Standard breadth-first search.
        - PPR: Personalized PageRank.
        - RANDOM_WALK: Similarity biased random walks.
        - HYBRID: Combined PageRank + Random Walk (perfect for RESEARCH exploration).
        
        Maintains complete backward compatibility with the original signature.
        """
        # Map strategy string to uppercase
        strat = strategy.upper() if strategy else "BFS"
        
        if not use_bfs and strat == "BFS":
            strat = "DFS"  # Fallback to original DFS if use_bfs=False

        if strat == "DFS":
            # Original DFS heuristic using BFS but sorted by weights if needed
            # We will use simple DFS traversal from graph_builder or fallback to BFS
            return self.bfs_graph_traversal(seed_chunks, max_depth)
            
        elif strat == "PPR":
            return self.personalized_pagerank_traversal(seed_chunks, max_depth)
            
        elif strat == "RANDOM_WALK":
            return self.random_walk_traversal(seed_chunks, max_depth)
            
        elif strat == "HYBRID":
            # Combined PPR + Random Walk
            ppr_res = self.personalized_pagerank_traversal(seed_chunks, max_depth)
            rw_res = self.random_walk_traversal(seed_chunks, max_depth)
            
            # Fuse them
            fused = {}
            for chunk, score in ppr_res:
                cid = chunk["chunk_id"]
                fused[cid] = [chunk, score, 0.0]
            for chunk, score in rw_res:
                cid = chunk["chunk_id"]
                if cid in fused:
                    fused[cid][2] = score
                else:
                    fused[cid] = [chunk, 0.0, score]
            
            combined = []
            for cid, (chunk, ppr_s, rw_s) in fused.items():
                # Average non-zero scores
                avg_score = (ppr_s + rw_s) / (2.0 if ppr_s > 0 and rw_s > 0 else 1.0)
                combined.append((chunk, avg_score))
                
            combined.sort(key=lambda x: x[1], reverse=True)
            return combined[:15]
            
        else:
            # Default to BFS
            return self.bfs_graph_traversal(seed_chunks, max_depth)


if __name__ == "__main__":
    pass
