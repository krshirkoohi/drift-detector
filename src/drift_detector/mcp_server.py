"""
FastMCP server for inline semantic drift detection across LLM agents.

Features:
- Global on/off toggle and session-aware monitoring.
- Canonical auto-baselining from initial turns (>=3 samples) using BaselineStore.
- Context compaction listening (Page-Hinkley accumulators reset; mission anchor preserved by default).
- Explicit semantic provider resolution (defaults to 'local' via PyTorch, or DRIFT_PROVIDER env).
- Explicit rebase tool for intentional task/mission transitions.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .baseline import Baseline, BaselineStore
from .detector import DriftDetector, TurnScore
from .embedding import EmbeddingProvider, get_provider

mcp = FastMCP("drift-detector")

_WARMUP_MIN_SAMPLES: int = 3


class SessionState:
    """State container for a monitored session."""

    def __init__(self, session_id: str = "default", provider_name: Optional[str] = None):
        self.session_id = session_id
        self.enabled: bool = True
        self.metric: str = "cosine"
        self.use_trend: bool = True
        self.provider_name: str = provider_name or os.environ.get("DRIFT_PROVIDER", "local")
        self._provider: Optional[EmbeddingProvider] = None
        self.detector: Optional[DriftDetector] = None
        self.warmup_buffer: List[str] = []

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_provider(self.provider_name)
        return self._provider

    def reset(self) -> None:
        self.detector = None
        self.warmup_buffer = []

    def build_baseline(self, texts: List[str]) -> Baseline:
        store = BaselineStore(self.provider)
        return store.build(texts)


class SessionRegistry:
    """Registry managing multiple isolated drift detection sessions."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._global_enabled: bool = True

    def get_session(self, session_id: Optional[str] = None) -> SessionState:
        sid = (session_id or "default").strip()
        if sid not in self._sessions:
            self._sessions[sid] = SessionState(session_id=sid)
        return self._sessions[sid]

    @property
    def global_enabled(self) -> bool:
        return self._global_enabled

    @global_enabled.setter
    def global_enabled(self, value: bool) -> None:
        self._global_enabled = value


_registry = SessionRegistry()


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
def drift_toggle(state: Optional[str] = None, session_id: Optional[str] = None) -> str:
    """Toggle background drift detection on or off, or check status.

    Args:
        state: 'on' to enable, 'off' to disable, 'status' to check, or 'reset' to clear session state.
        session_id: Optional session identifier for isolated multi-session monitoring.
    """
    session = _registry.get_session(session_id)
    if state is None:
        session.enabled = not session.enabled
        return f"Drift detector monitoring is now {'ON' if session.enabled else 'OFF'}."

    s = state.strip().lower()
    if s == "on":
        session.enabled = True
        return "Drift detector monitoring is now ON."
    elif s == "off":
        session.enabled = False
        return "Drift detector monitoring is now OFF (zero turn overhead)."
    elif s == "reset":
        session.reset()
        return "Drift detector session reset. Will auto-baseline on next turn."
    elif s == "status":
        status_str = "ON" if session.enabled else "OFF"
        if session.detector is None:
            return f"Drift detector is {status_str} (pending auto-calibration on next {max(0, _WARMUP_MIN_SAMPLES - len(session.warmup_buffer))} turns)."
        summary = session.detector.summary()
        summary["session_id"] = session.session_id
        summary["status"] = status_str
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
    session_id: Optional[str] = None,
) -> str:
    """Attach drift-detector with a specific baseline or enable auto-baselining.

    Args:
        baseline: 'auto' for dynamic in-chat baseline, or preset name/file path.
        metric: Distance metric ('cosine' or 'euclidean').
        provider: Embedding provider ('local', 'test'/'deterministic', 'gemini', or 'openai').
        use_trend: Enable Page-Hinkley trend checking for sustained drift detection.
        threshold: Optional manual distance threshold override.
        session_id: Optional session identifier for isolated multi-session monitoring.
    """
    session = _registry.get_session(session_id)
    session.enabled = True
    session.metric = metric
    session.use_trend = use_trend
    session.provider_name = provider
    session._provider = get_provider(provider)
    session.reset()

    if baseline.lower() in ("auto", "self", "dynamic"):
        return f"Attached to dynamic auto-baseline session (metric={metric}, trend={use_trend}, provider={provider}). Will calibrate on initial {_WARMUP_MIN_SAMPLES} turns."

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

    baseline_obj = BaselineStore(session.provider).build_from_file(baseline_path)
    if threshold is not None:
        if metric == "cosine":
            baseline_obj.cosine_threshold = threshold
        else:
            baseline_obj.euclidean_threshold = threshold

    session.detector = DriftDetector(
        baseline=baseline_obj,
        provider=session.provider,
        metric=metric,
        use_trend=use_trend,
    )
    return f"Attached drift detector session using baseline '{baseline}' ({baseline_obj.n_samples} examples, metric={metric}, trend={use_trend})."


@mcp.tool()
def drift_compact_reset(
    compacted_summary: str = "",
    rebase_anchor: bool = False,
    session_id: Optional[str] = None,
) -> str:
    """Reset Page-Hinkley drift accumulators following chat compaction, preserving mission anchor by default.

    Args:
        compacted_summary: Optional new compacted summary of the conversation.
        rebase_anchor: Whether to rebase the semantic mission anchor on the summary (default: False).
        session_id: Optional session identifier.
    """
    session = _registry.get_session(session_id)
    if session.detector is None:
        session.reset()
        if rebase_anchor and compacted_summary.strip():
            session.warmup_buffer = [compacted_summary.strip()]
        return "[driftd] Chat compacted: detector reset (pending baseline calibration)."

    return session.detector.handle_compaction(
        compacted_summary=compacted_summary if compacted_summary.strip() else None,
        rebase_anchor=rebase_anchor,
    )


@mcp.tool()
def drift_rebase(
    anchor_text: str,
    reason: str = "explicit_task_transition",
    session_id: Optional[str] = None,
) -> str:
    """Explicitly rebase the semantic mission anchor upon an intentional task or scope change.

    Args:
        anchor_text: The new reference mission description, prompt, or exemplary response.
        reason: Description of the task change.
        session_id: Optional session identifier.
    """
    session = _registry.get_session(session_id)
    if session.detector is None:
        # Initialise with single anchor text
        vecs = session.provider.embed([anchor_text.strip()])
        centroid = vecs[0]
        # Floor threshold for single example anchor
        is_neural = getattr(session.provider, "model_name", None) is not None or getattr(session.provider, "model", None) is not None
        cos_floor = 0.85 if getattr(session.provider, "model_name", None) is not None else 0.70 if is_neural else 0.45
        baseline_obj = Baseline(
            centroid=centroid,
            cosine_threshold=cos_floor,
            euclidean_threshold=1.20 if is_neural else 0.65,
            n_samples=1,
        )
        session.detector = DriftDetector(
            baseline=baseline_obj,
            provider=session.provider,
            metric=session.metric,
            use_trend=session.use_trend,
        )
        return f"[driftd] Semantic mission anchor initialised ({reason})."

    return session.detector.rebase(anchor_text, reason=reason)


@mcp.tool()
def drift_evaluate_turn(
    agent_response: str,
    user_prompt: str = "",
    history_len: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    is_compacted: bool = False,
    compacted_summary: Optional[str] = None,
    notice_mode: str = "simple",
    session_id: Optional[str] = None,
) -> str:
    """Evaluate an agent turn in the background. Zero-cost when off; returns discreet status on drift.

    Args:
        agent_response: Text of the assistant/agent response.
        user_prompt: Optional text of the user prompt.
        history_len: Optional message history count to detect compaction truncation.
        prompt_tokens: Optional token count to detect compaction token drop.
        is_compacted: Explicit flag indicating chat compaction occurred.
        compacted_summary: Optional summary text from compaction.
        notice_mode: 'simple' for concise discreet flag, 'detailed' for full JSON metric scorecard.
        session_id: Optional session identifier.
    """
    session = _registry.get_session(session_id)
    if not session.enabled:
        return json.dumps({"status": "disabled", "drifted": False})

    # Detect /compact in user prompt
    prompt_clean = user_prompt.strip()
    if prompt_clean.startswith("/compact"):
        is_compacted = True
        if compacted_summary is None:
            parts = prompt_clean.split(" ", 1)
            if len(parts) > 1:
                compacted_summary = parts[1].strip()

    # Auto-calibration on opening turns using canonical BaselineStore (>=3 samples)
    if session.detector is None:
        text_sample = agent_response.strip()
        session.warmup_buffer.append(text_sample)
        if len(session.warmup_buffer) < _WARMUP_MIN_SAMPLES:
            return json.dumps({
                "status": "calibrating",
                "warmup_turn": len(session.warmup_buffer),
                "warmup_target": _WARMUP_MIN_SAMPLES,
                "drifted": False,
            })

        # Build canonical baseline using BaselineStore (guarantees candidate_v1 thresholds and logic)
        baseline_obj = session.build_baseline(session.warmup_buffer)
        session.detector = DriftDetector(
            baseline=baseline_obj,
            provider=session.provider,
            metric=session.metric,
            use_trend=session.use_trend,
        )

    result: TurnScore = session.detector.score(
        agent_response,
        history_len=history_len,
        prompt_tokens=prompt_tokens,
        is_compacted=is_compacted,
        compacted_summary=compacted_summary,
    )

    if notice_mode == "detailed":
        return json.dumps(result.to_dict(), indent=2)

    # Compaction reset notice
    if result.compacted_reset and result.notice:
        return result.notice

    # Discreet inline notices
    if result.drifted:
        return f"[driftd: {result.badge} · cos={result.cosine_distance:.4f}]"
    if result.threshold_breach:
        return f"[driftd: {result.badge} · cos={result.cosine_distance:.4f}]"

    return json.dumps({
        "status": result.badge,
        "turn": result.turn,
        "cosine_distance": result.cosine_distance,
        "calibration_support": result.calibration_support,
        "drifted": False,
    })


@mcp.tool()
def drift_get_status(session_id: Optional[str] = None) -> str:
    """Get active session status, lifecycle state, calibration support, and summary statistics.

    Args:
        session_id: Optional session identifier.
    """
    session = _registry.get_session(session_id)
    status_str = "ON" if session.enabled else "OFF"
    if session.detector is None:
        return json.dumps({
            "session_id": session.session_id,
            "status": status_str,
            "lifecycle_state": "calibrating" if session.warmup_buffer else "uninitialised",
            "calibrated": False,
            "calibration_support": "none",
            "confidence": "none",
            "warmup_turns": len(session.warmup_buffer),
            "warmup_target": _WARMUP_MIN_SAMPLES,
            "turns": 0,
        }, indent=2)

    res = session.detector.summary()
    res["session_id"] = session.session_id
    res["status"] = status_str
    return json.dumps(res, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
