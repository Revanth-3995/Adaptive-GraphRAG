import os
import sys
import json
import time
from typing import Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import RetrievalPipeline
from performance_tracker import PerformanceTracker

def run_performance_audit():
    pipeline = RetrievalPipeline()
    pipeline.load()
    tracker = PerformanceTracker()
    
    # 5 different queries to audit representing various intents
    audit_queries = [
        "What is distance vector routing?",
        "Compare RIP and OSPF routing protocols.",
        "How do routers build a link state packet?",
        "Explain Dijkstra's algorithm and its limitations.",
        "What is the Count-to-Infinity problem?"
    ]
    
    print("\n" + "="*60)
    print("STARTING DUAL-MODE PERFORMANCE AUDIT BENCHMARK")
    print("="*60)
    
    fast_results = []
    verified_results = []
    
    # We clear the log file first to start fresh for the audit cost study
    if os.path.exists(tracker.log_path):
        try:
            os.remove(tracker.log_path)
        except Exception:
            pass
            
    for idx, query in enumerate(audit_queries, 1):
        print(f"\n[{idx}/{len(audit_queries)}] Audit Query: '{query}'")
        
        # 1. Run in FAST mode
        print("  Running FAST mode...")
        t0 = time.perf_counter()
        res_fast = pipeline.answer_query(query, mode="FAST")
        dt_fast = (time.perf_counter() - t0) * 1000
        fast_calls = res_fast["performance"]["llm_calls"]
        fast_results.append((dt_fast, fast_calls))
        print(f"    FAST Latency: {dt_fast:.1f}ms | LLM Calls: {fast_calls}")
        
        # 2. Run in VERIFIED mode
        print("  Running VERIFIED mode...")
        t1 = time.perf_counter()
        res_verified = pipeline.answer_query(query, mode="VERIFIED")
        dt_verified = (time.perf_counter() - t1) * 1000
        verified_calls = res_verified["performance"]["llm_calls"]
        verified_results.append((dt_verified, verified_calls))
        print(f"    VERIFIED Latency: {dt_verified:.1f}ms | LLM Calls: {verified_calls}")
        
    # Calculate statistics
    avg_fast_ms = sum(r[0] for r in fast_results) / len(fast_results)
    avg_ver_ms = sum(r[0] for r in verified_results) / len(verified_results)
    avg_calls_fast = sum(r[1] for r in fast_results) / len(fast_results)
    avg_calls_ver = sum(r[1] for r in verified_results) / len(verified_results)
    
    overhead_ms = avg_ver_ms - avg_fast_ms
    multiplier = avg_ver_ms / avg_fast_ms if avg_fast_ms > 0 else 1.0
    
    summary = {
        "queries_tested": len(audit_queries),
        "avg_fast_latency_ms": round(avg_fast_ms, 1),
        "avg_verified_latency_ms": round(avg_ver_ms, 1),
        "verification_overhead_ms": round(overhead_ms, 1),
        "latency_ratio": round(multiplier, 2),
        "avg_llm_calls_fast": round(avg_calls_fast, 1),
        "avg_llm_calls_verified": round(avg_calls_ver, 1)
    }
    
    # Save summary report
    summary_path = "performance_logs/audit_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
        
    print("\n" + "="*60)
    print("PERFORMANCE AUDIT RESULTS")
    print("="*60)
    print(f"Average FAST Mode Latency:      {avg_fast_ms:.1f} ms")
    print(f"Average VERIFIED Mode Latency:  {avg_ver_ms:.1f} ms")
    print(f"Verification Overhead (ms):    {overhead_ms:.1f} ms")
    print(f"Latency Multiplier Factor:     {multiplier:.2f}x")
    print(f"Average LLM Calls (FAST):       {avg_calls_fast:.1f}")
    print(f"Average LLM Calls (VERIFIED):   {avg_calls_ver:.1f}")
    print("="*60)
    print(f"Summary report written to {summary_path}\n")

if __name__ == "__main__":
    run_performance_audit()
