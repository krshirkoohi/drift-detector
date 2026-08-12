"""
drift-detector-mcp: Model Context Protocol (MCP) server for inline semantic drift detection.

Listens in background to agent terminal/chat sessions.
Attaches to a chat, evaluates turns in batches (or every n turns) against the baseline.
Invisible in the background on clean turns.
When drift is detected, returns a warning popup formatted for display under the console text box.
"""

from typing import Dict, Any, Optional, List
import json
import os
import sys
from mcp.server.fastmcp import FastMCP
from .session import DriftSession
from .baseline import BaselineStore
from .embeddings import GeminiEmbeddingAdapter, DeterministicEmbeddingAdapter, LocalEmbeddingAdapter

mcp = FastMCP("drift-detector")

# Global session state for the active listening process
_active_session: Optional[DriftSession] = None
_eval_interval: int = 1
_turn_counter: int = 0
_batch_buffer: List[Dict[str, Any]] = []


@mcp.prompt()
def drift_session_init_guidance() -> str:
    """System prompt guidance for developers using drift-detector-mcp."""
    return (
        "IMPORTANT USAGE GUIDANCE FOR DRIFT DETECTOR:\n"
        "Activate drift-detector-mcp at the beginning of chat sessions where topic consistency is required.\n"
        "The detector runs silently in the background every N turns. It will issue a warning popup\n"
        "if the agent response strays significantly from the baseline domain."
    )


@mcp.tool()
def drift_attach_session(
    baseline_name_or_path: str = "default",
    eval_every_n_turns: int = 1,
    metric: str = "cosine",
    provider: str = "hosted",
) -> str:
    """Attach drift-detector to an active chat session and set evaluation baseline.
    
    Args:
        baseline_name_or_path: Name of preset baseline (e.g. 'default') or path to JSON file.
        eval_every_n_turns: Evaluate every N turns (default 1).
        metric: Distance metric ('cosine' or 'euclidean').
        provider: Embedding provider ('hosted', 'local', or 'deterministic').
    """
    global _active_session, _eval_interval, _turn_counter, _batch_buffer
    
    # Resolve baseline path
    if os.path.exists(baseline_name_or_path):
        baseline_path = baseline_name_or_path
    else:
        # Check standard baselines dir
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "baselines")
        candidate = os.path.join(base_dir, f"{baseline_name_or_path}.json" if not baseline_name_or_path.endswith(".json") else baseline_name_or_path)
        if os.path.exists(candidate):
            baseline_path = candidate
        else:
            baseline_path = os.path.join(base_dir, "default.json")

    store = BaselineStore(baseline_path)
    examples = store.examples
    
    # Resolve embedding adapter
    if provider == "hosted":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            adapter = GeminiEmbeddingAdapter(api_key=api_key)
        else:
            adapter = DeterministicEmbeddingAdapter()
    elif provider == "local":
        adapter = LocalEmbeddingAdapter()
    else:
        adapter = DeterministicEmbeddingAdapter()
        
    _active_session = DriftSession.initialise(
        known_good_responses=examples,
        embedding_adapter=adapter,
        metric=metric,
        use_trend=True,
    )
    _eval_interval = eval_every_n_turns
    _turn_counter = 0
    _batch_buffer = []
    
    return f"Attached to session. Baseline: '{store.name}' ({len(examples)} examples). Evaluation interval: every {_eval_interval} turn(s)."


@mcp.tool()
def drift_evaluate_turn(agent_response: str, user_prompt: str = "") -> str:
    """Evaluate an agent turn in background. Invisible on clean turns; returns warning popup on drift.
    
    Args:
        agent_response: Text of the assistant/agent response.
        user_prompt: Optional text of the user prompt.
    """
    global _active_session, _eval_interval, _turn_counter, _batch_buffer
    
    if _active_session is None:
        # Auto-initialize default session if not attached
        drift_attach_session()
        
    _turn_counter += 1
    _batch_buffer.append({"user": user_prompt, "agent": agent_response})
    
    # Check if we should evaluate on this batch turn
    if _turn_counter % _eval_interval != 0:
        return json.dumps({"status": "buffering", "turn": _turn_counter, "eval_every": _eval_interval})
        
    # Evaluate response with DriftSession
    verdict = _active_session.observe(agent_response)
    is_drifting = (verdict.distance > verdict.threshold) or verdict.drift_detected
    
    # If clean, stay invisible in background
    if not is_drifting:
        return json.dumps({
            "status": "clean",
            "turn": _turn_counter,
            "cosine_distance": round(verdict.cosine_distance, 4),
            "threshold": round(verdict.threshold, 4),
            "drift_detected": False,
        })
        
    # If drift detected, generate warning popup formatted for display under text box
    warning_popup = (
        "\n"
        "┌──────────────────────────────────────────────────────────┐\n"
        f"│ ⚠️  DRIFT DETECTED — turn {_turn_counter} is going off-topic!       │\n"
        "├──────────────────────────────────────────────────────────┤\n"
        f"│ Metric           : {verdict.metric.upper()}                              │\n"
        f"│ Distance         : {verdict.cosine_distance if verdict.metric == 'cosine' else verdict.euclidean_distance:.4f} (Threshold: {verdict.threshold:.4f})     │\n"
        f"│ Page-Hinkley Sk  : {verdict.trend_statistic:.4f}                          │\n"
        f"│ Recommend Fresh  : {'Yes' if is_drifting else 'No'}                                 │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )
    
    return warning_popup


@mcp.tool()
def drift_get_status() -> str:
    """Get active session status and summary statistics."""
    global _active_session, _turn_counter
    if _active_session is None:
        return "No active drift detection session attached."
        
    summary = _active_session.summary()
    return json.dumps(summary, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
