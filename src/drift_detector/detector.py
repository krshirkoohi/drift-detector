import math
import time
import os
import json
from typing import List, Dict, Any, Optional
from .baseline import BaselineStore

def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)

def cosine_distance(v1: List[float], v2: List[float]) -> float:
    return 1.0 - cosine_similarity(v1, v2)

class DriftDetector:
    def __init__(
        self,
        baseline_store: BaselineStore,
        api_key: str,
        threshold: float = 0.25,
        log_dir: Optional[str] = None
    ):
        self.baseline_store = baseline_store
        self.api_key = api_key
        self.threshold = threshold
        self.log_dir = log_dir
        self.centroid: Optional[List[float]] = None
        
        # Initialise centroid
        if self.baseline_store.centroid is None:
            self.baseline_store.compute_centroid(self.api_key)
        self.centroid = self.baseline_store.centroid

    def check_response(self, response_text: str) -> Dict[str, Any]:
        """
        Analyse a single response against the baseline.
        Returns a dictionary with the results.
        """
        start_time = time.time()
        
        # 1. Fetch response embedding
        response_emb = BaselineStore.get_embedding(response_text, self.api_key)
        
        # 2. Compute similarity & distance
        distance = cosine_distance(response_emb, self.centroid)
        is_drifting = distance > self.threshold
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response_snippet": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "cosine_distance": distance,
            "threshold": self.threshold,
            "is_drifting": is_drifting,
            "latency_ms": latency_ms
        }
        
        # 3. Log metrics if configured
        if self.log_dir:
            self._log_metrics(result)
            
        return result

    def _log_metrics(self, result: Dict[str, Any]) -> None:
        """Append metric log entry to a JSON Lines file."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"drift_metrics_{self.baseline_store.name}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
