import os
import json
import pickle
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Set
from collections import deque

class GraphBuilder:
    """
    GraphBuilder constructs a semantic graph from text chunks and their embeddings.
    
    Why a Graph?
    Traditional vector search retrieves isolated chunks. A graph connects semantically 
    related chunks, allowing us to traverse the graph and discover relevant context 
    that might not directly match the query but is highly related to the matched chunks.
    
    Internal Data Structure:
    We use NetworkX to represent the graph. Under the hood, NetworkX uses an 
    Adjacency List (specifically, a dict-of-dicts) to store nodes and edges.
    - Nodes represent text chunks.
    - Edges represent a strong semantic similarity (cosine similarity > threshold).
    """
    
    def __init__(self, similarity_threshold: float = 0.6, max_edges_per_node: int = 10):
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        self.graph = nx.Graph()
        
    def _page_proximity_bonus(self, page_a: int, page_b: int) -> float:
        """
        Returns a small bonus for chunks on nearby pages.
        Same page = 0.1 bonus, adjacent page = 0.05 bonus, far = 0
        """
        diff = abs(page_a - page_b)
        if diff == 0:
            return 0.1
        elif diff == 1:
            return 0.05
        return 0.0

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates the Cosine Similarity between two vectors.
        
        Math: cos(theta) = (A dot B) / (||A|| * ||B||)
        It measures the cosine of the angle between two vectors in a multi-dimensional space.
        1.0 means perfectly similar, 0.0 means orthogonal (unrelated), -1.0 means opposite.
        
        Complexity: O(D) where D is the dimension of the vectors.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def build_graph(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> nx.Graph:
        """
        Builds the semantic graph by computing pairwise similarities.
        
        Time Complexity: O(N^2 * D) where N is number of chunks, D is embedding dimension.
        *Optimization Note:* For large datasets, N^2 is too slow. In production, 
        we'd use FAISS or Annoy to efficiently find nearest neighbors instead of exhaustive search.
        """
        print(f"Building graph for {len(chunks)} nodes...")
        
        # 1. Add all nodes
        for i, chunk in enumerate(chunks):
            self.graph.add_node(
                chunk["chunk_id"], 
                text=chunk["text"], 
                source=chunk["source_filename"],
                page=chunk["page_number"],
                chunk_type=chunk.get("chunk_type", "text"),
                entities=chunk.get("entities", []),
                keywords=chunk.get("keywords", []),
                noun_phrases=chunk.get("noun_phrases", []),
                idx=i # Store the embedding index for reference
            )
            
    def _add_typed_edge(self, u, v, weight, edge_type):
        """Helper to safely add/update edges with types metadata."""
        if self.graph.has_edge(u, v):
            current_types = self.graph[u][v].get("types", [])
            if edge_type not in current_types:
                current_types.append(edge_type)
            self.graph[u][v]["types"] = current_types
            self.graph[u][v]["weight"] = max(self.graph[u][v].get("weight", 0.0), weight)
        else:
            self.graph.add_edge(u, v, weight=weight, types=[edge_type])

    def build_graph(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> nx.Graph:
        """
        Builds the semantic graph by computing pairwise similarities and typing edges.
        """
        print(f"Building graph for {len(chunks)} nodes...")
        
        # 1. Add all nodes
        for i, chunk in enumerate(chunks):
            self.graph.add_node(
                chunk["chunk_id"], 
                text=chunk["text"], 
                source=chunk["source_filename"],
                page=chunk["page_number"],
                chunk_type=chunk.get("chunk_type", "text"),
                entities=chunk.get("entities", []),
                keywords=chunk.get("keywords", []),
                noun_phrases=chunk.get("noun_phrases", []),
                idx=i # Store the embedding index for reference
            )
            
        # 2. Add consecutive section edges
        for i in range(len(chunks) - 1):
            if chunks[i].get("source_filename") == chunks[i+1].get("source_filename"):
                self._add_typed_edge(chunks[i]["chunk_id"], chunks[i+1]["chunk_id"], 0.9, "section")
                
        # 3. Add page proximity edges and shared entity edges
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                # Page proximity edges within same document
                if chunks[i].get("source_filename") == chunks[j].get("source_filename"):
                    page_diff = abs(chunks[i].get("page_number", 0) - chunks[j].get("page_number", 0))
                    if page_diff <= 1:
                        weight = 0.8 if page_diff == 0 else 0.6
                        self._add_typed_edge(chunks[i]["chunk_id"], chunks[j]["chunk_id"], weight, "page")
                        
                # Shared entity edges
                entities_i = set(chunks[i].get("entities", []))
                entities_j = set(chunks[j].get("entities", []))
                common_entities = entities_i.intersection(entities_j)
                if common_entities:
                    weight = min(0.9, 0.5 + 0.1 * len(common_entities))
                    self._add_typed_edge(chunks[i]["chunk_id"], chunks[j]["chunk_id"], weight, "entity")
            
        # 4. Add edges based on semantic similarity
        # We iterate over all unique pairs (i, j) where i < j
        for i in range(len(chunks)):
            similarities = []
            for j in range(len(chunks)):
                if i == j:
                    continue

                base_sim = self._cosine_similarity(embeddings[i], embeddings[j])

                # Add page proximity bonus for same-document chunks
                if chunks[i].get("source_filename") == chunks[j].get("source_filename"):
                    page_bonus = self._page_proximity_bonus(
                        chunks[i].get("page_number", 0),
                        chunks[j].get("page_number", 0)
                    )
                    adjusted_sim = base_sim + page_bonus
                else:
                    adjusted_sim = base_sim

                if adjusted_sim >= self.similarity_threshold:
                    similarities.append((j, adjusted_sim))
            
            # Sort by similarity descending, keep only top max_edges_per_node
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:self.max_edges_per_node]
            
            node_i_id = chunks[i]["chunk_id"]
            for j, sim in top_similar:
                node_j_id = chunks[j]["chunk_id"]
                self._add_typed_edge(node_i_id, node_j_id, sim, "similarity")
                
        self.print_graph_stats()
        return self.graph

    def print_graph_stats(self):
        """Prints basic statistics about the constructed graph."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        components = nx.number_connected_components(self.graph)
        
        print("\n--- Graph Statistics ---")
        print(f"Nodes: {num_nodes}")
        print(f"Edges: {num_edges}")
        print(f"Density: {density:.4f}")
        print(f"Connected Components: {components}")
        print("------------------------\n")

    def bfs_traversal(self, start_node_id: str, max_depth: int = 2) -> List[str]:
        """
        Breadth-First Search (BFS) to explore the neighborhood of a seed node.
        
        Why BFS?
        BFS explores nodes layer by layer. It's excellent for finding the closest 
        related chunks first (1-hop neighbors) before moving to more distant concepts (2-hop).
        
        Complexity: O(V + E) where V is nodes, E is edges.
        """
        if start_node_id not in self.graph:
            return []
            
        visited = set([start_node_id])
        queue = deque([(start_node_id, 0)]) # Queue stores (node_id, current_depth)
        result_nodes = []
        
        while queue:
            current_node, depth = queue.popleft()
            result_nodes.append(current_node)
            
            if depth < max_depth:
                # Explore neighbors
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        
        return result_nodes

    def dfs_traversal(self, start_node_id: str, max_depth: int = 2) -> List[str]:
        """
        Depth-First Search (DFS) to explore a deep semantic chain.
        
        Why DFS?
        DFS goes deep into a specific topic before backtracking. It can be useful to 
        follow a complex line of reasoning across multiple chunks.
        """
        if start_node_id not in self.graph:
            return []
            
        visited = set()
        result_nodes = []
        
        def _dfs_recursive(node: str, depth: int):
            if depth > max_depth or node in visited:
                return
            visited.add(node)
            result_nodes.append(node)
            
            # Sort neighbors by weight (strongest connection first) to guide the depth search
            neighbors = sorted(
                self.graph[node].items(), 
                key=lambda edge: edge[1].get('weight', 0), 
                reverse=True
            )
            for neighbor, _ in neighbors:
                _dfs_recursive(neighbor, depth + 1)
                
        _dfs_recursive(start_node_id, 0)
        return result_nodes

    def multi_hop_retrieval(self, seed_node_ids: List[str], max_depth: int = 2, use_bfs: bool = True) -> Set[str]:
        """
        Retrieves a broader context by traversing the graph starting from multiple seed nodes.
        """
        expanded_nodes = set()
        for seed in seed_node_ids:
            if use_bfs:
                nodes = self.bfs_traversal(seed, max_depth)
            else:
                nodes = self.dfs_traversal(seed, max_depth)
            expanded_nodes.update(nodes)
        return expanded_nodes

    def save_graph(self, output_path: str = "graph/graph.pkl"):
        """Saves the NetworkX graph using pickle."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to {output_path}")

    def load_graph(self, input_path: str = "graph/graph.pkl") -> nx.Graph:
        """Loads the NetworkX graph from disk."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Graph file not found: {input_path}")
        with open(input_path, 'rb') as f:
            self.graph = pickle.load(f)
        return self.graph

if __name__ == "__main__":
    # Example usage
    pass
