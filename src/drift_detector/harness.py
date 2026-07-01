"""
harness.py — Agent Harness and Data Flow for Drift Detector

Provides the `AgentHarness` class, which wraps a live LLM session and
automatically feeds every model response through the `DriftDetector`
pipeline, capturing per-turn drift scores and metadata.

Usage (programmatic):
    from drift_detector.harness import AgentHarness
    harness = AgentHarness(detector=my_detector, log_dir="data/harness_logs")
    harness.start_session(session_id="my-session")
    result = harness.process_turn(user_prompt="...", agent_response="...")
    summary = harness.end_session()

Usage (standalone CLI — see run_harness.py):
    python run_harness.py --baseline baselines/default.json
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .detector import DriftDetector


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """Immutable record of a single conversation turn and its drift metrics."""
    turn_index: int
    session_id: str
    timestamp_utc: str
    user_prompt: str
    agent_response_snippet: str  # First 200 chars
    agent_response_full: str
    cosine_distance: float
    euclidean_distance: float
    threshold: float
    metric: str
    is_drifting: bool
    latency_ms: float
    # Page-Hinkley fields (only populated when use_trend=True)
    ph_running_mean: Optional[float] = None
    ph_running_sum: Optional[float] = None
    ph_min_sum: Optional[float] = None
    ph_statistic: Optional[float] = None
    ph_threshold: Optional[float] = None
    ph_delta: Optional[float] = None
    trend_alarm: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSummary:
    """Aggregated summary produced when a session ends."""
    session_id: str
    baseline_name: str
    start_time_utc: str
    end_time_utc: str
    total_turns: int
    drifted_turns: int
    drift_rate: float
    mean_cosine_distance: float
    mean_euclidean_distance: float
    peak_cosine_distance: float
    peak_euclidean_distance: float
    metric: str
    threshold: float
    use_trend: bool
    turns: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentHarness
# ---------------------------------------------------------------------------

class AgentHarness:
    """
    Wraps a live LLM agent session and wires each response turn through the
    `DriftDetector` pipeline, capturing per-turn metrics and emitting a
    structured data flow.

    The harness is *stateful per session*: call `start_session()` before
    the first turn and `end_session()` after the last turn to obtain a
    `SessionSummary`.  Multiple sessions can be run sequentially on the
    same harness instance.

    Args:
        detector:   A fully-initialised `DriftDetector` instance.
        log_dir:    Directory where per-session JSONL logs are written.
                    Each session writes one file: ``<session_id>.jsonl``.
                    If None, logging is disabled.
        verbose:    When True, print a formatted per-turn scorecard to stdout.
    """

    def __init__(
        self,
        detector: DriftDetector,
        log_dir: Optional[str] = None,
        verbose: bool = True,
        detailed: bool = False,
    ) -> None:
        self.detector = detector
        self.log_dir = log_dir
        self.verbose = verbose
        self.detailed = detailed

        # Active session state
        self._session_id: Optional[str] = None
        self._session_start: Optional[str] = None
        self._turn_index: int = 0
        self._turns: List[TurnRecord] = []

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Begin a new monitoring session.

        Args:
            session_id: Optional custom identifier.  If omitted, a UUID4 is
                        generated automatically.

        Returns:
            The session ID string.
        """
        if self._session_id is not None:
            raise RuntimeError(
                f"Session '{self._session_id}' is already active.  "
                "Call end_session() before starting a new one."
            )
        self._session_id = session_id or str(uuid.uuid4())
        self._session_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._turn_index = 0
        self._turns = []

        # Reset Page-Hinkley state inside the detector for a fresh session
        self.detector.ph_n = 0
        self.detector.ph_running_mean = 0.0
        self.detector.ph_running_sum = 0.0
        self.detector.ph_min_sum = 0.0

        if self.verbose:
            _banner(f"SESSION STARTED  ·  {self._session_id}")
            print(
                f"  Baseline : {self.detector.baseline_store.name}\n"
                f"  Metric   : {self.detector.metric.upper()}\n"
                f"  Threshold: {self.detector.threshold:.4f}\n"
                f"  Trend    : {'ON  (Page-Hinkley)' if self.detector.use_trend else 'OFF'}\n"
            )

        return self._session_id

    def end_session(self) -> SessionSummary:
        """
        Finalise the current session and return a `SessionSummary`.

        The summary (and the complete turn log) is written to
        ``<log_dir>/<session_id>_summary.json`` when log_dir is set.
        """
        if self._session_id is None:
            raise RuntimeError("No active session.  Call start_session() first.")

        end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        total = len(self._turns)
        drifted = sum(1 for t in self._turns if t.is_drifting)
        drift_rate = drifted / total if total > 0 else 0.0

        cos_dists = [t.cosine_distance for t in self._turns]
        euc_dists = [t.euclidean_distance for t in self._turns]

        summary = SessionSummary(
            session_id=self._session_id,
            baseline_name=self.detector.baseline_store.name,
            start_time_utc=self._session_start,      # type: ignore[arg-type]
            end_time_utc=end_time,
            total_turns=total,
            drifted_turns=drifted,
            drift_rate=drift_rate,
            mean_cosine_distance=_safe_mean(cos_dists),
            mean_euclidean_distance=_safe_mean(euc_dists),
            peak_cosine_distance=max(cos_dists) if cos_dists else 0.0,
            peak_euclidean_distance=max(euc_dists) if euc_dists else 0.0,
            metric=self.detector.metric,
            threshold=self.detector.threshold,
            use_trend=self.detector.use_trend,
            turns=[t.to_dict() for t in self._turns],
        )

        if self.log_dir:
            summary_path = os.path.join(
                self.log_dir, f"{self._session_id}_summary.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, indent=2)

        if self.verbose:
            _banner(f"SESSION ENDED  ·  {self._session_id}")
            print(f"  Total turns    : {total}")
            print(f"  Drifted turns  : {drifted}  ({drift_rate:.1%})")
            print(f"  Mean cos dist  : {summary.mean_cosine_distance:.4f}")
            print(f"  Mean euc dist  : {summary.mean_euclidean_distance:.4f}")
            print(f"  Peak cos dist  : {summary.peak_cosine_distance:.4f}")
            if self.log_dir:
                print(f"  Summary saved  : {summary_path}")
            print()

        # Reset Page-Hinkley state on the detector so it is clean for the next session
        self.detector.ph_n = 0
        self.detector.ph_running_mean = 0.0
        self.detector.ph_running_sum = 0.0
        self.detector.ph_min_sum = 0.0

        # Clear session state
        self._session_id = None
        self._session_start = None
        self._turn_index = 0
        self._turns = []

        return summary

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    def process_turn(
        self,
        user_prompt: str,
        agent_response: str,
    ) -> TurnRecord:
        """
        Run a single conversation turn through the drift pipeline.

        This is the *core data-flow entry point*.  Call it once for every
        model response in the session.

        Args:
            user_prompt:    The user's input text for this turn.
            agent_response: The model's response text for this turn.

        Returns:
            A `TurnRecord` containing full drift metrics for this turn.
        """
        if self._session_id is None:
            raise RuntimeError("No active session.  Call start_session() first.")

        self._turn_index += 1
        metrics = self.detector.check_response(agent_response)

        record = TurnRecord(
            turn_index=self._turn_index,
            session_id=self._session_id,
            timestamp_utc=metrics["timestamp"],
            user_prompt=user_prompt[:200],
            agent_response_snippet=agent_response[:200] + (
                "..." if len(agent_response) > 200 else ""
            ),
            agent_response_full=agent_response,
            cosine_distance=metrics["cosine_distance"],
            euclidean_distance=metrics["euclidean_distance"],
            threshold=metrics["threshold"],
            metric=metrics["metric"],
            is_drifting=metrics["is_drifting"],
            latency_ms=metrics["latency_ms"],
            # Page-Hinkley fields
            ph_running_mean=metrics.get("ph_running_mean"),
            ph_running_sum=metrics.get("ph_running_sum"),
            ph_min_sum=metrics.get("ph_min_sum"),
            ph_statistic=metrics.get("ph_statistic"),
            ph_threshold=metrics.get("ph_threshold"),
            ph_delta=metrics.get("ph_delta"),
            trend_alarm=metrics.get("trend_alarm"),
        )

        self._turns.append(record)

        # Write to JSONL log (one record per line)
        if self.log_dir:
            log_path = os.path.join(self.log_dir, f"{self._session_id}.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

        if self.verbose:
            self._print_turn_scorecard(record)

        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_turn_scorecard(self, record: TurnRecord) -> None:
        """Print a per-turn drift notice to stdout.

        Default mode: silent on clean turns, one line on drift.
        Detailed mode: silent on clean turns, metric block on drift.
        """
        if not record.is_drifting:
            return  # completely silent on clean turns

        if self.detailed:
            print(f"\n  {'─'*52}")
            print(f"  ⚠  Drift detected  —  turn {record.turn_index}")
            print(f"     Cosine / Euclidean : "
                  f"{record.cosine_distance:.4f} / {record.euclidean_distance:.4f}  "
                  f"(threshold {record.threshold:.4f})")
            if record.ph_statistic is not None:
                print(f"     PH statistic      : {record.ph_statistic:.4f}  "
                      f"(PH threshold {record.ph_threshold:.4f})")
            print(f"     Latency           : {record.latency_ms:.0f}ms")
            print(f"  {'─'*52}\n")
        else:
            print(f"\n  ⚠  Drift detected — agent may be going off-topic.\n")


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _banner(text: str, width: int = 60) -> None:
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")
