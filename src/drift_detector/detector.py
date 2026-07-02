import time
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import numpy as np
from .baseline import BaselineStore
from .embeddings import EmbeddingAdapter

@dataclass
class TurnScore:
    turn: int
    cosine_distance: float
    euclidean_distance: float
    threshold_breach: bool
    trend_alarm: bool
    drifted: bool

    def to_dict(self) -> dict:
        return asdict(self)

class DriftDetector:
    def __init__(
        self,
        baseline_store: BaselineStore,
        api_key: Optional[str] = None,
        threshold: Optional[float] = None,
        metric: str = "cosine",
        log_dir: Optional[str] = None,
        use_trend: bool = False,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        ph_sustain: int = 1,
        ph_burn_in: int = 0
    ):
        self.baseline_store = baseline_store
        self.metric = metric.lower()
        if self.metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be either 'cosine' or 'euclidean'")
        self.log_dir = log_dir
        self.use_trend = use_trend
        
        # Initialise embedding adapter
        if embedding_adapter is None:
            if not api_key:
                raise ValueError("Either api_key or embedding_adapter must be provided.")
            from .embeddings import GeminiEmbeddingAdapter
            self.embedding_adapter = GeminiEmbeddingAdapter(api_key)
        else:
            self.embedding_adapter = embedding_adapter

        self.api_key = api_key
        self.centroid: Optional[np.ndarray] = None
        
        # Initialise centroid
        if self.baseline_store.centroid is None:
            self.baseline_store.compute_centroid(adapter=self.embedding_adapter)
        self.centroid = self.baseline_store.centroid
        
        # Auto-calibrate threshold (95th percentile of baseline distances) if not specified
        if threshold is None:
            self.threshold = self.baseline_store.calculate_percentile_threshold(self.metric, 95.0)
        else:
            self.threshold = threshold

        # Page-Hinkley session state variables
        self.ph_n = 0
        self.ph_running_mean = 0.0
        self.ph_running_sum = 0.0
        self.ph_min_sum = 0.0
        self.ph_sustain = ph_sustain
        self.ph_burn_in = ph_burn_in
        self.ph_exceed_streak = 0
        self.history: List[TurnScore] = []

        # Calibrate Page-Hinkley parameters (ph_delta, ph_threshold) using the standard deviation 
        # of the distances of baseline examples from the centroid.
        self._calibrate_page_hinkley()

    def _calibrate_page_hinkley(self) -> None:
        """
        Calibrate Page-Hinkley parameters based on the standard deviation of baseline distances.
        """
        if self.baseline_store.embeddings is None or self.centroid is None:
            self.baseline_store.compute_centroid(adapter=self.embedding_adapter)
            
        if self.metric == "cosine":
            norms = np.linalg.norm(self.baseline_store.embeddings, axis=1)
            norm_c = np.linalg.norm(self.centroid)
            norms = np.where(norms == 0, 1.0, norms)
            dots = np.dot(self.baseline_store.embeddings, self.centroid)
            dists = 1.0 - dots / (norms * norm_c)
        else: # euclidean
            dists = np.linalg.norm(self.baseline_store.embeddings - self.centroid, axis=1)
            
        std_dev = float(np.std(dists))
        self.ph_delta = 0.1 * std_dev
        self.ph_threshold = 5.0 * std_dev

    def check_response(self, response_text: str) -> Dict[str, Any]:
        """
        Analyse a single response against the baseline using NumPy.
        Returns a dictionary with the results.
        """
        start_time = time.time()
        
        # 1. Fetch response embedding
        response_emb_list = self.embedding_adapter.embed(response_text)
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
        
        # Determine current distance based on selected metric
        current_distance = cos_dist if self.metric == "cosine" else euc_dist
        
        # Update Page-Hinkley streaming state
        self.ph_n += 1
        old_mean = self.ph_running_mean
        self.ph_running_mean = old_mean + (current_distance - old_mean) / self.ph_n
        
        self.ph_running_sum += (current_distance - self.ph_running_mean - self.ph_delta)
        self.ph_min_sum = min(self.ph_min_sum, self.ph_running_sum)
        
        ph_statistic = self.ph_running_sum - self.ph_min_sum
        
        # Check if Page-Hinkley statistic is elevated above threshold AND current distance exceeds mean
        elevated = (ph_statistic > self.ph_threshold) and (current_distance > self.ph_running_mean)
        if elevated:
            self.ph_exceed_streak += 1
        else:
            self.ph_exceed_streak = 0
            
        trend_alarm = bool(self.ph_n > self.ph_burn_in and self.ph_exceed_streak >= self.ph_sustain)
        
        # Determine drift status based on whether trend checking is active
        if self.use_trend:
            is_drifting = trend_alarm
        else:
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
        
        if self.use_trend:
            result.update({
                "ph_running_mean": self.ph_running_mean,
                "ph_running_sum": self.ph_running_sum,
                "ph_min_sum": self.ph_min_sum,
                "ph_statistic": ph_statistic,
                "ph_threshold": self.ph_threshold,
                "ph_delta": self.ph_delta,
                "trend_alarm": trend_alarm
            })
        
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

    def score(self, text: str) -> TurnScore:
        """Score a response text, returning a TurnScore and appending to history."""
        res = self.check_response(text)
        cos = res["cosine_distance"]
        euc = res["euclidean_distance"]
        d = cos if self.metric == "cosine" else euc
        thr = res["threshold"]
        breach = d > thr
        alarm = res.get("trend_alarm", False)
        drifted = res["is_drifting"]
        
        ts = TurnScore(
            turn=self.ph_n,
            cosine_distance=round(cos, 4),
            euclidean_distance=round(euc, 4),
            threshold_breach=breach,
            trend_alarm=alarm,
            drifted=drifted
        )
        self.history.append(ts)
        return ts

    def summary(self) -> dict:
        """Return summary statistics of the session history."""
        key = "cosine_distance" if self.metric == "cosine" else "euclidean_distance"
        ds = [getattr(t, key) for t in self.history]
        drifted = [t for t in self.history if t.drifted]
        return {
            "turns": len(self.history),
            "drifted_turns": len(drifted),
            "drift_rate": round(len(drifted) / len(self.history), 3) if self.history else 0.0,
            "mean_distance": round(float(np.mean(ds)), 4) if ds else 0.0,
            "peak_distance": round(float(np.max(ds)), 4) if ds else 0.0,
            "metric": self.metric,
            "trend_rule": self.use_trend,
            "topic_focus": self.baseline_store.get_topic_focus(),
        }


