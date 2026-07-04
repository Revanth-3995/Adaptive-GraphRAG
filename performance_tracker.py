import os
import json
import threading
import time
from typing import Dict, Any, List

class PerformanceTracker:
    """
    PerformanceTracker is a thread-safe class that records execution times,
    counts LLM calls, detects bottlenecks, computes verification overhead,
    and logs traces to JSON.
    """
    
    _lock = threading.Lock()
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global LLM call tracking."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super(PerformanceTracker, cls).__new__(cls, *args, **kwargs)
                cls._instance._init_tracker()
            return cls._instance
            
    def _init_tracker(self):
        self.lock = threading.Lock()
        self.llm_call_count = 0
        self.log_path = "performance_logs/performance_trace.json"
        
    def reset_call_count(self):
        """Reset the global LLM calls counter."""
        with self.lock:
            self.llm_call_count = 0
            
    def increment_llm_calls(self, count: int = 1):
        """Increment the global LLM calls counter."""
        with self.lock:
            self.llm_call_count += count
            
    def get_llm_calls(self) -> int:
        """Get the current count of LLM calls."""
        with self.lock:
            return self.llm_call_count

    def log_trace(self, trace: Dict[str, Any]):
        """Appends a performance trace to performance_logs/performance_trace.json."""
        with self.lock:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            
            history = []
            if os.path.exists(self.log_path):
                try:
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        history = json.load(f)
                        if not isinstance(history, list):
                            history = []
                except Exception:
                    history = []
                    
            history.append(trace)
            
            # Keep history capped to last 50 queries to prevent large file sizes
            if len(history) > 50:
                history = history[-50:]
                
            try:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save performance trace: {e}")

    def compute_cost_analysis(self) -> Dict[str, Any]:
        """
        Analyzes the trace history to calculate verification cost stats:
        Average Fast Mode Latency, Average Verified Mode Latency, Overhead, etc.
        """
        with self.lock:
            if not os.path.exists(self.log_path):
                return {
                    "fast_mode_ms": 0.0,
                    "verified_mode_ms": 0.0,
                    "verification_overhead_ms": 0.0,
                    "avg_llm_calls_fast": 0.0,
                    "avg_llm_calls_verified": 0.0
                }
                
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
                
            fast_latencies = []
            verified_latencies = []
            fast_llm_calls = []
            verified_llm_calls = []
            
            for entry in history:
                mode = entry.get("mode", "VERIFIED")
                total_time = entry.get("total_ms", 0.0)
                llm_calls = entry.get("llm_calls", 0)
                
                if mode == "FAST":
                    fast_latencies.append(total_time)
                    fast_llm_calls.append(llm_calls)
                else:
                    verified_latencies.append(total_time)
                    verified_llm_calls.append(llm_calls)
                    
            avg_fast = float(np_mean(fast_latencies)) if fast_latencies else 0.0
            avg_verified = float(np_mean(verified_latencies)) if verified_latencies else 0.0
            overhead = max(0.0, avg_verified - avg_fast)
            
            avg_calls_fast = float(np_mean(fast_llm_calls)) if fast_llm_calls else 0.0
            avg_calls_ver = float(np_mean(verified_llm_calls)) if verified_llm_calls else 0.0
            
            return {
                "fast_mode_ms": round(avg_fast, 1),
                "verified_mode_ms": round(avg_verified, 1),
                "verification_overhead_ms": round(overhead, 1),
                "avg_llm_calls_fast": round(avg_calls_fast, 1),
                "avg_llm_calls_verified": round(avg_calls_ver, 1)
            }

def np_mean(lst: List[float]) -> float:
    """Helper to calculate mean without requiring numpy if not needed, but safe fallback."""
    if not lst:
        return 0.0
    return sum(lst) / len(lst)
