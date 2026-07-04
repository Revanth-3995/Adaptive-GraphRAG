import os
import json
import re
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load env variables first
load_dotenv()

# Add parent directory to path so we can import RetrievalPipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import RetrievalPipeline

class AnswerEvaluator:
    """
    AnswerEvaluator runs the 10-query Answer Correctness Benchmark,
    calculates key quality metrics, and outputs standard evaluation reports.
    """

    def __init__(self):
        self.pipeline = RetrievalPipeline()
        self.pipeline.load()

    def run_benchmark(self, benchmark_path: str, output_path: str) -> Dict[str, Any]:
        """Runs evaluation over all benchmark questions."""
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

        with open(benchmark_path, "r", encoding="utf-8") as f:
            targets = json.load(f)

        print(f"Running Answer Intelligence Correctness Benchmark ({len(targets)} queries)...")
        results = []
        
        sum_accuracy = 0.0
        sum_faithfulness = 0.0
        sum_citation_acc = 0.0
        sum_calibration = 0.0
        
        for idx, item in enumerate(targets, 1):
            question = item["question"]
            expected_answer = item["expected_answer"]
            expected_concepts = item["expected_concepts"]
            
            print(f"\n[{idx}/{len(targets)}] Query: {question}")
            
            # Run pipeline
            res = self.pipeline.answer_query(question, mode="VERIFIED")
            generated_answer = res["answer"]
            grounding_score = res["grounding_score"]
            trust_level = res["trust_level"]
            claims = res["claims"]
            citations = res["citations"]
            
            # 1. Answer Accuracy (Concept Matching)
            matched_concepts = []
            for concept in expected_concepts:
                # Direct case-insensitive search
                if concept.lower() in generated_answer.lower():
                    matched_concepts.append(concept)
                    
            concept_accuracy = (len(matched_concepts) / len(expected_concepts)) * 100.0
            
            # 2. Faithfulness / Grounding Score
            faithfulness = grounding_score
            
            # 3. Citation Accuracy
            total_citations = len(citations)
            verified_citations = sum(1 for c in citations if c["verified"])
            citation_accuracy = (verified_citations / total_citations) * 100.0 if total_citations > 0 else 100.0
            
            # 4. Trust Calibration
            is_calibrated = False
            if grounding_score >= 90 and trust_level == "VERIFIED":
                is_calibrated = True
            elif 80 <= grounding_score < 90 and trust_level == "HIGH CONFIDENCE":
                is_calibrated = True
            elif 60 <= grounding_score < 80 and trust_level == "MEDIUM CONFIDENCE":
                is_calibrated = True
            elif 40 <= grounding_score < 60 and trust_level == "LOW CONFIDENCE":
                is_calibrated = True
            elif grounding_score < 40 and trust_level == "UNTRUSTED":
                is_calibrated = True
                
            calibration_score = 100.0 if is_calibrated else 0.0
            
            # Accumulate
            sum_accuracy += concept_accuracy
            sum_faithfulness += faithfulness
            sum_citation_acc += citation_accuracy
            sum_calibration += calibration_score
            
            print(f"  Accuracy (Concepts): {concept_accuracy:.1f}% ({len(matched_concepts)}/{len(expected_concepts)} matched)")
            print(f"  Faithfulness (Grounding): {faithfulness:.1f}%")
            print(f"  Citation Accuracy: {citation_accuracy:.1f}% ({verified_citations}/{total_citations} verified)")
            print(f"  Trust Level: {trust_level} (Calibrated: {'YES' if is_calibrated else 'NO'})")
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "expected_concepts": expected_concepts,
                "matched_concepts": matched_concepts,
                "concept_accuracy": round(concept_accuracy, 1),
                "faithfulness": round(faithfulness, 1),
                "citation_accuracy": round(citation_accuracy, 1),
                "trust_level": trust_level,
                "is_calibrated": is_calibrated,
                "hallucination_risk": res["hallucination_risk"]
            })

        num_queries = len(targets)
        avg_accuracy = sum_accuracy / num_queries
        avg_faithfulness = sum_faithfulness / num_queries
        avg_citation_acc = sum_citation_acc / num_queries
        avg_calibration = sum_calibration / num_queries

        summary = {
            "total_queries": num_queries,
            "average_concept_accuracy": round(avg_accuracy, 1),
            "average_faithfulness": round(avg_faithfulness, 1),
            "average_citation_accuracy": round(avg_citation_acc, 1),
            "trust_calibration_score": round(avg_calibration, 1),
            "results": results
        }

        # Write results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as rf:
            json.dump(summary, rf, ensure_ascii=False, indent=2)

        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(f"Average Concept Accuracy:  {avg_accuracy:.1f}%")
        print(f"Average Faithfulness (GS): {avg_faithfulness:.1f}%")
        print(f"Average Citation Accuracy: {avg_citation_acc:.1f}%")
        print(f"Trust Calibration Score:   {avg_calibration:.1f}%")
        print("="*50)
        print(f"Detailed report exported to {output_path}\n")
        
        return summary

if __name__ == "__main__":
    benchmark_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "answer_correctness_benchmark.json"))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmark_answer_results.json"))
    
    evaluator = AnswerEvaluator()
    evaluator.run_benchmark(benchmark_file, output_file)
