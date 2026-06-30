import time
import os
import json
from typing import Dict, Any, Optional
import numpy as np
from .baseline import BaselineStore

class DriftDetector:
    def __init__(
        self,
        baseline_store: BaselineStore,
        api_key: str,
        threshold: Optional[float] = None,
        metric: str = "cosine",
        log_dir: Optional[str] = None
    ):
        self.baseline_store = baseline_store
        self.api_key = api_key
        self.metric = metric.lower()
        if self.metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be either 'cosine' or 'euclidean'")
        self.log_dir = log_dir
        self.centroid: Optional[np.ndarray] = None
        
        # Initialise centroid
        if self.baseline_store.centroid is None:
            self.baseline_store.compute_centroid(self.api_key)
        self.centroid = self.baseline_store.centroid
        
        # Auto-calibrate threshold (95th percentile of baseline distances) if not specified
        if threshold is None:
            self.threshold = self.baseline_store.calculate_percentile_threshold(self.metric, 95.0)
        else:
            self.threshold = threshold

    def check_response(self, response_text: str) -> Dict[str, Any]:
        """
        Analyse a single response against the baseline using NumPy.
        Returns a dictionary with the results.
        """
        start_time = time.time()
        
        # 1. Fetch response embedding
        response_emb_list = BaselineStore.get_embedding(response_text, self.api_key)
        response_emb = np.array(response_emb_list)
        
        # 2. Compute similarity & distance
        norm_r = np.linalg.norm(response_emb)
        norm_c = np.linalg.norm(self.centroid)
        
        if norm_r == 0 or norm_c == 0:
            cos_dist = 1.0
        else:
            dots = np.dot(response_emb, self.centroid)
            cos_dist = float(1.0 - dots / (norm_r * norm_c))
            
        euc_dist = float(np.linalg.norm(response_emb - self.centroid))
        
        # Determine drift status based on selected metric
        current_distance = cos_dist if self.metric == "cosine" else euc_dist
        is_drifting = bool(current_distance > self.threshold)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response_snippet": response_text[:100] + ("..." if len(response_text) > 100 else ""),
            "metric": self.metric,
            "cosine_distance": cos_dist,
            "euclidean_distance": euc_dist,
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

