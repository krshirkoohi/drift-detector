"""
session.py — Represents a live chat session with semantic drift tracking.
"""
import time
import os
from typing import List, Dict, Any, Optional
import numpy as np

from .embeddings import EmbeddingAdapter
from .utils import calculate_distances, cosine_distance, euclidean_distance, STOPWORDS
from .models import DriftVerdict
from .storage import SessionLogger

class DriftSession:
    """
    Monitors a single chat session for semantic drift against a baseline.
    
    This class is the main entry point of the deep module, encapsulating
    embedding generation, centroid calibration, threshold checks, and Page-Hinkley
    sustained-trend rules.
    """
    def __init__(
        self,
        name: str,
        examples: List[str],
        embeddings: np.ndarray,
        centroid: np.ndarray,
        embedding_adapter: EmbeddingAdapter,
        threshold: float,
        metric: str,
        use_trend: bool,
        ph_delta: float,
        ph_threshold: float,
        ph_sustain: int = 1,
        ph_burn_in: int = 0,
        log_dir: Optional[str] = None,
    ):
        self.name = name
        self.examples = examples
        self.embeddings = embeddings
        self.centroid = centroid
        self.embedding_adapter = embedding_adapter
        self.threshold = threshold
        self.metric = metric.lower()
        self.use_trend = use_trend
        
        self.ph_delta = ph_delta
        self.ph_threshold = ph_threshold
        self.ph_sustain = ph_sustain
        self.ph_burn_in = ph_burn_in
        
        # Streaming session state variables
        self.ph_n = 0
        self.ph_running_mean = 0.0
        self.ph_running_sum = 0.0
        self.ph_min_sum = 0.0
        self.ph_exceed_streak = 0
        self.has_drifted = False
        
        self.history: List[DriftVerdict] = []
        self.logger = SessionLogger(log_dir) if log_dir else None
        
        # Dynamic Auto-Baseline parameters
        self.is_auto: bool = False
        self.warm_up_turns: int = 2
        self.auto_ready: bool = False
        self.auto_embeddings: List[np.ndarray] = []

    @classmethod
    def initialise_auto(
        cls,
        embedding_adapter: EmbeddingAdapter,
        warm_up_turns: int = 2,
        name: str = "auto-session",
        metric: str = "cosine",
        threshold: Optional[float] = None,
        use_trend: bool = False,
        ph_sustain: int = 1,
        ph_burn_in: int = 0,
        log_dir: Optional[str] = None,
    ) -> "DriftSession":
        """Initialise a DriftSession that auto-captures the current conversation initial turns as its baseline."""
        dim = getattr(embedding_adapter, 'dimension', 768)
        dummy_vec = np.zeros(dim)
        session = cls(
            name=name,
            examples=[],
            embeddings=np.array([dummy_vec]),
            centroid=dummy_vec,
            embedding_adapter=embedding_adapter,
            threshold=threshold if threshold is not None else 0.45,
            metric=metric,
            use_trend=use_trend,
            ph_delta=0.01,
            ph_threshold=0.05,
            ph_sustain=ph_sustain,
            ph_burn_in=ph_burn_in,
            log_dir=log_dir,
        )
        session.is_auto = True
        session.warm_up_turns = max(1, warm_up_turns)
        session.auto_ready = False
        session.auto_embeddings = []
        return session

    @classmethod
    def initialise(
        cls,
        known_good_responses: List[str],
        embedding_adapter: EmbeddingAdapter,
        name: str = "default",
        metric: str = "cosine",
        threshold: Optional[float] = None,
        use_trend: bool = False,
        ph_sustain: int = 1,
        ph_burn_in: int = 0,
        log_dir: Optional[str] = None,
        embeddings: Optional[np.ndarray] = None,
        centroid: Optional[np.ndarray] = None,
    ) -> "DriftSession":
        """
        Initialise a new DriftSession with a baseline and embedding provider.
        
        This method handles baseline embedding generation, centroid computation,
        and auto-calibration of thresholds and Page-Hinkley parameters.
        
        Args:
            known_good_responses: Examples defining the baseline topic focus.
            embedding_adapter: Pluggable provider to convert text to vectors.
            name: Label describing the session baseline.
            metric: Distance metric to use ('cosine' or 'euclidean').
            threshold: Optional distance threshold. Auto-calibrated if omitted.
            use_trend: If True, uses Page-Hinkley rule for sustained trend detection.
            ph_sustain: Number of consecutive breaches before triggering trend alarm.
            ph_burn_in: Number of turns to ignore at session start.
            log_dir: Optional directory to save session metrics.
            embeddings: Optional precomputed embeddings to skip API calls.
            centroid: Optional precomputed centroid.
            
        Returns:
            A fully calibrated DriftSession instance.
        """
        if not known_good_responses:
            raise ValueError("Baseline examples must not be empty.")
            
        metric = metric.lower()
        if metric not in ("cosine", "euclidean"):
            raise ValueError("metric must be either 'cosine' or 'euclidean'")
            
        # 1. Fetch baseline embeddings
        if embeddings is None or centroid is None:
            embeddings_list = []
            for example in known_good_responses:
                emb = embedding_adapter.embed(example)
                if emb:
                    embeddings_list.append(emb)
                    
            if not embeddings_list:
                raise RuntimeError("Failed to generate embeddings for baseline responses.")
                
            embeddings = np.array(embeddings_list)
            centroid = embeddings.mean(axis=0)
        
        # 2. Calibrate threshold
        dists = calculate_distances(embeddings, centroid, metric)
            
        calibrated_threshold = threshold
        if calibrated_threshold is None:
            calibrated_threshold = float(np.percentile(dists, 95.0))
            if calibrated_threshold < 0.01:
                calibrated_threshold = 0.01
            
        # 3. Calibrate Page-Hinkley parameters
        std_dev = float(np.std(dists))
        # Ensure std_dev is not zero to prevent dividing by or multiplying to 0
        if std_dev == 0:
            std_dev = 0.01
        ph_delta = 0.1 * std_dev
        ph_threshold = 5.0 * std_dev
        
        return cls(
            name=name,
            examples=known_good_responses,
            embeddings=embeddings,
            centroid=centroid,
            embedding_adapter=embedding_adapter,
            threshold=calibrated_threshold,
            metric=metric,
            use_trend=use_trend,
            ph_delta=ph_delta,
            ph_threshold=ph_threshold,
            ph_sustain=ph_sustain,
            ph_burn_in=ph_burn_in,
            log_dir=log_dir,
        )

    def observe(self, response_text: str) -> DriftVerdict:
        """
        Observe a new response, score it, and update the session status.
        
        Args:
            response_text: The text to evaluate.
            
        Returns:
            A DriftVerdict containing the scores and drift state.
        """
        start_time = time.time()
        
        # 1. Fetch response embedding
        response_emb_list = self.embedding_adapter.embed(response_text)
        response_emb = np.array(response_emb_list)
        
        # Auto-baseline dynamic centroid calculation during warm-up phase
        if self.is_auto and not self.auto_ready:
            self.auto_embeddings.append(response_emb)
            if len(self.auto_embeddings) >= self.warm_up_turns:
                self.centroid = np.mean(self.auto_embeddings, axis=0)
                # Compute centroid variance to auto-calibrate threshold
                dists = [cosine_distance(vec, self.centroid) for vec in self.auto_embeddings]
                self.threshold = float(np.percentile(dists, 95)) if len(dists) > 1 else 0.45
                self.auto_ready = True
            
            latency_ms = (time.time() - start_time) * 1000
            return DriftVerdict(
                distance=0.0,
                trend_statistic=0.0,
                drift_detected=False,
                recommend_fresh_chat=False,
                turn_index=len(self.auto_embeddings),
                threshold=self.threshold,
                metric=self.metric,
                cosine_distance=0.0,
                euclidean_distance=0.0,
                ph_running_mean=0.0,
                ph_running_sum=0.0,
                ph_min_sum=0.0,
                ph_threshold=self.ph_threshold,
                ph_delta=self.ph_delta,
                latency_ms=latency_ms
            )
        
        # 2. Compute similarity & distance
        cos_dist = cosine_distance(response_emb, self.centroid)
        euc_dist = euclidean_distance(response_emb, self.centroid)
        current_distance = cos_dist if self.metric == "cosine" else euc_dist
        
        # 3. Update Page-Hinkley streaming state
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
        threshold_breach = bool(current_distance > self.threshold)
        
        # Decide if drift has occurred based on parameters
        if self.use_trend:
            drift_detected = trend_alarm
        else:
            drift_detected = threshold_breach
            
        if drift_detected:
            self.has_drifted = True
            
        latency_ms = (time.time() - start_time) * 1000
        
        verdict = DriftVerdict(
            distance=round(current_distance, 4),
            trend_statistic=round(ph_statistic, 4),
            drift_detected=drift_detected,
            recommend_fresh_chat=drift_detected,
            turn_index=self.ph_n,
            threshold=round(self.threshold, 4),
            metric=self.metric,
            cosine_distance=round(cos_dist, 4),
            euclidean_distance=round(euc_dist, 4),
            ph_running_mean=round(self.ph_running_mean, 4),
            ph_running_sum=round(self.ph_running_sum, 4),
            ph_min_sum=round(self.ph_min_sum, 4),
            ph_threshold=round(self.ph_threshold, 4),
            ph_delta=round(self.ph_delta, 4),
            trend_alarm=trend_alarm,
            latency_ms=latency_ms,
        )
        
        self.history.append(verdict)
        
        # 4. Save/log verdict if configured
        if self.logger:
            self.logger.log_turn(self.name, verdict.to_dict())
            
        return verdict

    def get_topic_focus(self) -> str:
        """Extract a summary list of the most frequent keywords in the baseline."""
        import re
        from collections import Counter
        
        
        words = []
        for text in self.examples:
            for word in re.findall(r"[a-z0-9']+", text.lower()):
                if word not in STOPWORDS and len(word) >= 3:
                    words.append(word)
                    
        if not words:
            return "General conversation"
            
        common = [w for w, _ in Counter(words).most_common(4)]
        return ", ".join(common)

    def summary(self) -> dict:
        """Return summary statistics of the session history."""
        key = "cosine_distance" if self.metric == "cosine" else "euclidean_distance"
        ds = [getattr(t, key) for t in self.history]
        drifted = [t for t in self.history if t.drift_detected]
        return {
            "turns": len(self.history),
            "drifted_turns": len(drifted),
            "drift_rate": round(len(drifted) / len(self.history), 3) if self.history else 0.0,
            "mean_distance": round(float(np.mean(ds)), 4) if ds else 0.0,
            "peak_distance": round(float(np.max(ds)), 4) if ds else 0.0,
            "metric": self.metric,
            "trend_rule": self.use_trend,
            "topic_focus": self.get_topic_focus(),
            "has_drifted": self.has_drifted,
        }
