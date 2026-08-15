#!/usr/bin/env python3
"""
run_harness.py — Standalone entrypoint for the Drift Detector Agent Harness.

This script starts an interactive Gemini chat session monitored by the
drift detector harness.  Every model response is automatically scored and
per-turn drift metrics are printed inline.

Usage:
    python run_harness.py --baseline baselines/default.json
    python run_harness.py --baseline baselines/default.json --use-trend
    python run_harness.py --baseline baselines/default.json --provider local

Environment:
    GEMINI_API_KEY  — Required when --provider is 'hosted' (the default).

Output:
    Per-turn scorecards are printed to stdout.
    Full JSONL turn logs are written to:  data/harness_logs/<session_id>.jsonl
    Session summaries are written to:     data/harness_logs/<session_id>_summary.json
"""

import sys
import os
import argparse
import json
import urllib.request
from typing import Any, Dict, List

# Ensure the src directory is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uuid
import time
from drift_detector.core import DriftDetector
from drift_detector.embedding import get_provider


# ---------------------------------------------------------------------------
# Gemini chat helper (mirrors cli.py but lives here for self-containment)
# ---------------------------------------------------------------------------

def _generate_gemini_response(
    prompt: str, history: List[Dict[str, Any]], api_key: str
) -> str:
    """Send a chat request to the Gemini API and return the response text."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    contents = []
    for turn in history:
        contents.append({"role": turn["role"], "parts": [{"text": turn["text"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    req = urllib.request.Request(
        url,
        data=json.dumps({"contents": contents}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drift Detector Agent Harness — monitored live Gemini session"
    )
    parser.add_argument(
        "--baseline",
        default=os.path.join(os.path.dirname(__file__), "baselines", "default.json"),
        help="Path to the baseline specification JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Distance threshold for drift warnings.  "
            "If omitted, auto-calibrates to the 95th percentile of baseline distances."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric (default: cosine)",
    )
    parser.add_argument(
        "--use-trend",
        action="store_true",
        help="Enable Page-Hinkley sustained-trend detection",
    )
    parser.add_argument(
        "--provider",
        choices=["hosted", "local"],
        default="hosted",
        help="Embedding provider: 'hosted' (Gemini API) or 'local' (Hugging Face)",
    )
    parser.add_argument(
        "--local-model",
        default="roberta-base",
        help="Local Hugging Face model name when --provider=local (default: roberta-base)",
    )
    parser.add_argument(
        "--log-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "harness_logs"),
        help="Directory for JSONL turn logs and session summaries",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional custom session identifier (default: auto-generated UUID)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all drift notice output (summary still printed at end)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show full metric block on drift notices (default: single-line notice only)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate API key
    # ------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌  Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build the embedding adapter
    # ------------------------------------------------------------------
    print("⚙  Initialising drift detector harness...")
    try:
        from drift_detector.core import DriftDetector
        from drift_detector.embedding import get_provider

        print(f"   Embedding provider : {args.provider}")
        provider_kwargs = {"api_key": api_key} if args.provider == "hosted" else {"dim": 256} if args.provider == "local" else {}
        provider_name = "gemini" if args.provider == "hosted" else "local"
        adapter = get_provider(provider_name, **provider_kwargs)

        import json
        with open(args.baseline, "r") as f:
            data = json.load(f)
            baseline_texts = [s["text"] if isinstance(s, dict) else s for s in data.get("samples", [])]

        detector = DriftDetector.from_examples(
            baseline_texts=baseline_texts,
            provider=adapter,
            threshold=args.threshold,
            metric=args.metric,
            use_trend=args.use_trend
        )
        print(f"   Baseline           : {args.baseline}")
        print(f"   Metric / Threshold : {args.metric.upper()} / {args.threshold or 'Auto'}")
        print(f"   Trend detection    : {'ON (Page-Hinkley)' if args.use_trend else 'OFF'}")
    except Exception as exc:
        print(f"❌  Initialisation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    session_id = args.session_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Start session and run the conversation loop
    # ------------------------------------------------------------------
    session_id = harness.start_session(session_id=args.session_id)
    history: List[Dict[str, Any]] = []
    print("\nType 'exit' or press Ctrl-C to end the session.\n")

    while True:
        try:
            prompt = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nInterrupted — ending session...")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            break

        print("Agent > [thinking...]", end="\r")
        try:
            response = _generate_gemini_response(prompt, history, api_key)
        except RuntimeError as exc:
            print(f"\n❌  API Error: {exc}\n")
            continue

        # Clear the "thinking..." placeholder
        print("                         ", end="\r")

        # Feed the response through the harness (this is the core data-flow step)
        score = detector.score(response)

        # Display the agent's response to the user
        print(f"Agent > {response}\n")
        print(f"   [{score.badge}] cos={score.cosine_distance:.4f} | euc={score.euclidean_distance:.4f}")

        # Maintain conversation history for multi-turn context
        history.append({"role": "user", "text": prompt})
        history.append({"role": "model", "text": response})

    # ------------------------------------------------------------------
    # End session and print summary
    # ------------------------------------------------------------------
    summary = detector.summary()
    print(f"\nSession Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
