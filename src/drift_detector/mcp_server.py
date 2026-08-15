"""
FastMCP server for inline semantic drift detection across LLM agents.

Features:
- Global on/off toggle for zero-overhead background monitoring.
- Lightweight auto-baseline calculation from initial turns (1KB float32 memory footprint).
- Context compaction listening & reset to recalibrate baseline on compressed summaries.
- Discreet, low-noise inline drift alerts.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from .baseline import Baseline, BaselineStore
from .core import DriftDetector
from .detector import DriftDetector as InnerDetector, TurnScore
from .embedding import EmbeddingProvider, DeterministicProvider, get_provider, l2_normalise

mcp = FastMCP("drift-detector")

_enabled: bool = True
_detector: Optional[DriftDetector] = None
_provider: Optional[EmbeddingProvider] = None
_metric: str = "cosine"
_use_trend: bool = True
_warmup_buffer: List[str] = []
_warmup_target: int = 2  # initial turns to calibrate auto-baseline


def _ensure_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = DeterministicProvider(dim=256)
    return _provider


def _build_auto_baseline(texts: List[str]) -> Baseline:
    provider = _ensure_provider()
    vecs = l2_normalise(provider.embed(texts))
    centroid = l2_normalise(np.mean(vecs, axis=0, keepdims=True))[0]
    cos_dists = [float(1.0 - v @ centroid) for v in vecs]
    euc_dists = [float(np.linalg.norm(v - centroid)) for v in vecs]
    cos_thr = max(float(np.percentile(cos_dists, 95)) * 1.5, 0.45)
    euc_thr = max(float(np.percentile(euc_dists, 95)) * 1.5, 0.65)
    return Baseline(
        centroid=centroid,
        cosine_threshold=cos_thr,
        euclidean_threshold=euc_thr,
        n_samples=len(texts),
    )


@mcp.prompt("drift")
def drift_prompt(action: str = "") -> str:
    """Control drift detector (on, off, status, reset)."""
    return f"Execute drift command: {action}" if action else "Show current drift detector status and available controls."


@mcp.prompt("drift_on")
def drift_on_prompt() -> str:
    """Enable background drift detection monitoring for the active session."""
    return "Turn ON background semantic drift detection for this chat session."


@mcp.prompt("drift_off")
def drift_off_prompt() -> str:
    """Disable background drift detection monitoring (zero turn overhead)."""
    return "Turn OFF background semantic drift detection for this chat session."


@mcp.prompt("drift_status")
def drift_status_prompt() -> str:
    """Show active drift detector status, turn count, and metrics."""
    return "Report active drift detector status, turn count, and distance metrics."


@mcp.prompt("drift_reset")
def drift_reset_prompt() -> str:
    """Reset active drift detector baseline and turn history."""
    return "Reset active drift detector baseline and clear cumulative drift scores."


@mcp.tool()
def drift_toggle(state: Optional[str] = None) -> str:
    """Toggle background drift detection on or off, or check status.

    Args:
        state: 'on' to enable, 'off' to disable, 'status' to check, or 'reset' to clear session state.
    """
    global _enabled, _detector, _warmup_buffer
    if state is None:
        _enabled = not _enabled
        return f"Drift detector monitoring is now {'ON' if _enabled else 'OFF'}."

    s = state.strip().lower()
    if s == "on":
        _enabled = True
        return "Drift detector monitoring is now ON."
    elif s == "off":
        _enabled = False
        return "Drift detector monitoring is now OFF (zero turn overhead)."
    elif s == "reset":
        _detector = None
        _warmup_buffer = []
        return "Drift detector session reset. Will auto-baseline on next turn."
    elif s == "status":
        status_str = "ON" if _enabled else "OFF"
        if _detector is None:
            return f"Drift detector is {status_str} (pending auto-calibration on next turn)."
        summary = _detector.summary()
        return f"Drift detector is {status_str}.\nSession stats: {json.dumps(summary, indent=2)}"
    else:
        return f"Unknown toggle state '{state}'. Use 'on', 'off', 'status', or 'reset'."


@mcp.tool()
def drift_attach_session(
    baseline: str = "auto",
    metric: str = "cosine",
    provider: str = "local",
    use_trend: bool = True,
    threshold: Optional[float] = None,
) -> str:
    """Attach drift-detector with a specific baseline or enable auto-baselining.

    Args:
        baseline: 'auto' for dynamic in-chat baseline, or preset name/file path.
        metric: Distance metric ('cosine' or 'euclidean').
        provider: Embedding provider ('local'/'deterministic', 'gemini', or 'openai').
        use_trend: Enable Page-Hinkley trend checking for sustained drift detection.
        threshold: Optional manual distance threshold override.
    """
    global _detector, _provider, _metric, _use_trend, _warmup_buffer, _enabled
    _enabled = True
    _metric = metric
    _use_trend = use_trend
    _provider = get_provider(provider)
    _warmup_buffer = []

    if baseline.lower() in ("auto", "self", "dynamic"):
        _detector = None
        return f"Attached to dynamic auto-baseline session (metric={metric}, trend={use_trend}). Will calibrate on initial turns."

    # Preset baseline file
    if os.path.exists(baseline):
        baseline_path = baseline
    else:
        pkg_baselines = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "baselines",
            f"{baseline}.json" if not baseline.endswith(".json") else baseline,
        )
        baseline_path = pkg_baselines if os.path.exists(pkg_baselines) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "baselines", "default.json"
        )

    baseline_obj = BaselineStore(_provider).build_from_file(baseline_path)
    if threshold is not None:
        if metric == "cosine":
            baseline_obj.cosine_threshold = threshold
        else:
            baseline_obj.euclidean_threshold = threshold

    inner = InnerDetector(
        baseline=baseline_obj,
        provider=_provider,
        metric=metric,
        use_trend=use_trend,
    )
    _detector = DriftDetector(inner)
    return f"Attached drift detector session using baseline '{baseline}' ({baseline_obj.n_samples} examples, metric={metric}, trend={use_trend})."


@mcp.tool()
def drift_compact_reset(compacted_summary: str) -> str:
    """Reset and recalibrate drift baseline following a chat compaction / context summarisation event.

    Args:
        compacted_summary: The new compacted summary of the conversation.
    """
    global _detector, _warmup_buffer, _provider, _metric, _use_trend, _enabled
    if not compacted_summary.strip():
        return "Compacted summary empty; session reset without new baseline."

    provider = _ensure_provider()
    baseline_obj = _build_auto_baseline([compacted_summary])
    inner = InnerDetector(
        baseline=baseline_obj,
        provider=provider,
        metric=_metric,
        use_trend=_use_trend,
    )
    _detector = DriftDetector(inner)
    _warmup_buffer = [compacted_summary]
    return f"Drift detector re-baselined on compacted summary (memory: ~1KB centroid, 0 prior turns)."


@mcp.tool()
def drift_evaluate_turn(
    agent_response: str,
    user_prompt: str = "",
    notice_mode: str = "simple",
) -> str:
    """Evaluate an agent turn in the background. Zero-cost when off; returns discreet status on drift.

    Args:
        agent_response: Text of the assistant/agent response.
        user_prompt: Optional text of the user prompt.
        notice_mode: 'simple' for concise discreet flag, 'detailed' for full JSON metric scorecard.
    """
    global _enabled, _detector, _warmup_buffer, _metric, _use_trend
    if not _enabled:
        return json.dumps({"status": "disabled", "drifted": False})

    # Auto-calibration on initial turns if no static baseline was provided
    if _detector is None:
        text_sample = f"{user_prompt}\n{agent_response}".strip()
        _warmup_buffer.append(text_sample)
        if len(_warmup_buffer) < _warmup_target:
            return json.dumps({
                "status": "calibrating",
                "warmup_turn": len(_warmup_buffer),
                "warmup_target": _warmup_target,
                "drifted": False,
            })
        # Build ultra-lightweight auto-baseline
        provider = _ensure_provider()
        baseline_obj = _build_auto_baseline(_warmup_buffer)
        inner = InnerDetector(
            baseline=baseline_obj,
            provider=provider,
            metric=_metric,
            use_trend=_use_trend,
        )
        _detector = DriftDetector(inner)

    result: TurnScore = _detector.score(agent_response)

    if notice_mode == "detailed":
        return json.dumps(result.to_dict(), indent=2)

    # Discreet inline notices
    if result.drifted:
        return f"[driftd: {result.badge} · cos={result.cosine_distance:.4f}]"
    if result.threshold_breach:
        return f"[driftd: {result.badge} · cos={result.cosine_distance:.4f}]"

    return json.dumps({
        "status": result.badge,
        "turn": result.turn,
        "cosine_distance": result.cosine_distance,
        "drifted": False,
    })


@mcp.tool()
def drift_get_status() -> str:
    """Get active session status and summary statistics."""
    global _detector, _enabled
    status_str = "ON" if _enabled else "OFF"
    if _detector is None:
        return json.dumps({"status": status_str, "calibrated": False, "turns": 0})
    res = _detector.summary()
    res["status"] = status_str
    res["calibrated"] = True
    return json.dumps(res, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
