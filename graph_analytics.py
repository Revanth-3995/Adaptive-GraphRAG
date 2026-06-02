"""
graph_analytics.py — Edge analytics and structural metrics for GraphRAG.

Loads the NetworkX knowledge graph and computes structural metrics:
- Total Nodes and Edges
- Average Node Degree
- Connected Components count
- Graph Density
- Composition of edge types (similarity, entity, section, page)

Saves the report to graph/graph_analytics.json for debug visibility.
"""
import os
import json
import networkx as nx
from typing import Dict, Any


class GraphAnalytics:
    """
    Computes structural and typed-edge statistics for a GraphRAG knowledge graph.
    """

    def __init__(self, graph_path: str = None):
        # Fallback path chain
        paths = ["graph/graph.pkl", "graph/test_graph.pkl", "doc_store/test/graph.pkl"]
        self.graph_path = graph_path
        
        if not self.graph_path:
            for path in paths:
                if os.path.exists(path):
                    self.graph_path = path
                    break
                    
        if not self.graph_path or not os.path.exists(self.graph_path):
            raise FileNotFoundError("Could not find any networkx graph.pkl to analyze.")
            
        print(f"Loading graph for analytics from: {self.graph_path}")
        import pickle
        with open(self.graph_path, 'rb') as f:
            self.graph: nx.Graph = pickle.load(f)

    def analyze(self, output_json_path: str = "graph/graph_analytics.json") -> Dict[str, Any]:
        """
        Analyzes the loaded graph and saves statistics.
        """
        if not self.graph:
            return {}

        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        # Calculate Average Degree
        if num_nodes > 0:
            avg_degree = sum(dict(self.graph.degree()).values()) / num_nodes
        else:
            avg_degree = 0.0
            
        # Connected Components
        try:
            connected_components = nx.number_connected_components(self.graph)
        except Exception:
            connected_components = 0

        # Graph Density
        density = nx.density(self.graph)

        # Count typed edges
        similarity_count = 0
        entity_count = 0
        section_count = 0
        page_count = 0
        untyped_count = 0

        for u, v, data in self.graph.edges(data=True):
            types = data.get("types", [])
            if not types:
                untyped_count += 1
                continue
                
            if "similarity" in types:
                similarity_count += 1
            if "entity" in types:
                entity_count += 1
            if "section" in types:
                section_count += 1
            if "page" in types:
                page_count += 1

        report = {
            "graph_file": self.graph_path,
            "structural_metrics": {
                "total_nodes": num_nodes,
                "total_edges": num_edges,
                "average_degree": round(avg_degree, 4),
                "connected_components": connected_components,
                "graph_density": round(density, 6)
            },
            "edge_composition": {
                "similarity_edges": similarity_count,
                "entity_edges": entity_count,
                "section_edges": section_count,
                "page_edges": page_count,
                "untyped_edges": untyped_count
            }
        }

        # Save to JSON
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"      [OK] Graph analytics saved to {output_json_path}")
        except Exception as e:
            print(f"Warning: Failed to save graph analytics to JSON ({e})")

        self.print_report(report)
        return report

    def print_report(self, report: Dict[str, Any]):
        """Prints a gorgeous terminal report."""
        struct = report["structural_metrics"]
        edges = report["edge_composition"]

        print("\n" + "="*50)
        print("             GRAPH EDGE ANALYTICS REPORT            ")
        print("="*50)
        print(f" Target Graph File      : {report['graph_file']}")
        print(f" Total Nodes            : {struct['total_nodes']}")
        print(f" Total Edges            : {struct['total_edges']}")
        print(f" Average Node Degree    : {struct['average_degree']:.4f}")
        print(f" Connected Components   : {struct['connected_components']}")
        print(f" Graph Density          : {struct['graph_density']:.6f}")
        print("-"*50)
        print(" EDGE TYPE COMPOSITION:")
        print(f"  - Similarity Edges    : {edges['similarity_edges']}")
        print(f"  - Entity Edges        : {edges['entity_edges']}")
        print(f"  - Section Edges       : {edges['section_edges']}")
        print(f"  - Page Edges          : {edges['page_edges']}")
        if edges['untyped_edges'] > 0:
            print(f"  - Untyped Edges       : {edges['untyped_edges']}")
        print("="*50 + "\n")


if __name__ == "__main__":
    try:
        analytics = GraphAnalytics()
        analytics.analyze()
    except Exception as e:
        print(f"Analytics run skipped: {e}")
