"""
evaluator.py — Automated evaluation pipeline for the Adaptive Retrieval Engine.

Loads the evaluation benchmark dataset from `retrieval/retrieval_benchmark.json`,
runs each benchmark query through the `RetrievalPipeline`, and computes standard
Information Retrieval metrics:
1. Recall@K (K=5)
2. Mean Reciprocal Rank (MRR)
3. Chunk Coverage
4. Graph Expansion Quality (Precision of graph-expanded nodes)

Saves the complete evaluation report to `graph/evaluation_report.json` for analysis.
"""

import os

# Manually load .env file to ensure GROQ_API_KEY is available during terminal runs
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip('"').strip("'")
                    os.environ[key.strip()] = val
        print("Loaded environment from .env successfully.")
    except Exception as e:
        print(f"Warning: Failed to load .env manually in evaluator ({e})")

import json
import numpy as np
from typing import List, Dict, Any

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import RetrievalPipeline


class RetrievalEvaluator:
    """
    Evaluates retrieval performance using a standard benchmark dataset.
    """

    def __init__(
        self,
        benchmark_path: str = "evaluation/retrieval_benchmark.json",
        report_path: str = "graph/evaluation_report.json"
    ):
        self.benchmark_path = benchmark_path
        self.report_path = report_path
        self.pipeline = RetrievalPipeline()
        self.pipeline.load()

    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Loads the benchmark dataset."""
        if not os.path.exists(self.benchmark_path):
            raise FileNotFoundError(f"Benchmark dataset not found at {self.benchmark_path}")
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_evaluation(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs the evaluation suite and calculates search quality metrics.
        """
        benchmark = self.load_benchmark()
        print(f"Loaded {len(benchmark)} benchmark queries. Running evaluation...")

        query_metrics = []
        recalls = []
        rr_ranks = []  # Reciprocal ranks for MRR
        coverage_hits = 0
        graph_precisions = []

        for item in benchmark:
            question = item["question"]
            expected_substring = item["expected_chunk_substring"].lower()
            expected_query_type = item.get("type", item.get("expected_query_type", "SIMPLE"))
            
            print(f"\nEvaluating: '{question}' (Expected Type: {expected_query_type})")
            
            # Execute search
            results = self.pipeline.retrieve(question, top_k_initial=10, top_k_final=top_k)
            retrieved_chunks = [r[0] for r in results]
            
            # 1. Determine Relevance (Hits)
            hits = []
            for rank, chunk in enumerate(retrieved_chunks):
                text = chunk.get("text", "").lower()
                is_hit = expected_substring in text
                if is_hit:
                    hits.append(rank)
            
            # Compute query metrics
            recall = 1.0 if len(hits) > 0 else 0.0
            recalls.append(recall)
            
            rr = 1.0 / (min(hits) + 1) if len(hits) > 0 else 0.0
            rr_ranks.append(rr)
            
            if len(hits) > 0:
                coverage_hits += 1
                
            # Compute Graph Expansion Quality
            # Let's inspect the graph traversal results. To do this, we can rerun graph expansion separately
            # or inspect the pipeline last trace.
            trace = self.pipeline.last_trace
            graph_precision = 1.0
            
            # We'll check if graph traversal was used
            traversal_used = trace.get("traversal_used", "BFS")
            depth = trace.get("graph_depth", 0)
            
            if depth > 0 and len(results) > 0:
                # Calculate what fraction of top final chunks contain keywords related to the expected topic
                expected_topic = item["expected_topic"].lower()
                topic_hits = sum(
                    1 for c in retrieved_chunks 
                    if expected_topic in c.get("text", "").lower() or expected_substring in c.get("text", "").lower()
                )
                graph_precision = topic_hits / len(retrieved_chunks)
                graph_precisions.append(graph_precision)
            
            q_metric = {
                "question": question,
                "expected_type": expected_query_type,
                "actual_type": trace.get("query_type"),
                "sub_questions": trace.get("sub_questions"),
                "traversal_used": traversal_used,
                "graph_depth": depth,
                "recall_at_k": recall,
                "reciprocal_rank": rr,
                "graph_precision": round(graph_precision, 4)
            }
            query_metrics.append(q_metric)
            
            print(f"  Result: Recall={recall:.2f} | MRR={rr:.4f} | Traversal={traversal_used}")

        # Compute averages
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mrr = float(np.mean(rr_ranks)) if rr_ranks else 0.0
        coverage = float(coverage_hits / len(benchmark)) if benchmark else 0.0
        mean_graph_expansion_quality = float(np.mean(graph_precisions)) if graph_precisions else 1.0

        report = {
            "summary": {
                "total_queries": len(benchmark),
                "recall_at_k": round(mean_recall, 4),
                "mrr": round(mrr, 4),
                "chunk_coverage": round(coverage, 4),
                "graph_expansion_quality": round(mean_graph_expansion_quality, 4)
            },
            "queries": query_metrics
        }

        # Write report to disk
        try:
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nSaved evaluation report to {self.report_path}")
        except Exception as e:
            print(f"Warning: Failed to save evaluation report ({e})")

        self.print_report_summary(report)
        return report

    def print_report_summary(self, report: Dict[str, Any]):
        """Prints a beautiful summary table of the evaluation results."""
        summary = report["summary"]
        print("\n" + "="*50)
        print("          ADAPTIVE RETRIEVAL ENGINE REPORT          ")
        print("="*50)
        print(f" Total Benchmark Queries : {summary['total_queries']}")
        print(f" Recall@5                : {summary['recall_at_k'] * 100:.2f}%")
        print(f" Mean Reciprocal Rank    : {summary['mrr']:.4f}")
        print(f" Chunk Coverage          : {summary['chunk_coverage'] * 100:.2f}%")
        print(f" Graph Expansion Quality : {summary['graph_expansion_quality'] * 100:.2f}%")
        print("="*50 + "\n")


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run_evaluation()
