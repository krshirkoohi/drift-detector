import os
import sys
import json
import urllib.request
from typing import List, Dict, Any, Optional
from .baseline import BaselineStore
from .detector import DriftDetector

def generate_gemini_response(prompt: str, history: List[Dict[str, Any]], api_key: str) -> str:
    """Send chat request to Gemini API via urllib."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # Format contents with conversation history
    contents = []
    for turn in history:
        contents.append({
            "role": turn["role"],
            "parts": [{"text": turn["text"]}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })
    
    payload = {
        "contents": contents
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res = json.loads(response.read().decode("utf-8"))
            return res["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

def _print_drift_notice(metrics: Dict[str, Any], use_trend: bool, detailed: bool) -> None:
    """Print an inline drift notice to stdout.

    In default mode a single line is printed.  In detailed mode the full
    metric block is appended on the lines immediately following.
    Silent on clean turns — this function must only be called when
    ``metrics['is_drifting']`` is True.
    """
    if detailed:
        # ── Detailed mode: notice + metric block ─────────────────────────
        print("\n" + "─" * 54)
        if use_trend:
            print("  ⚠  Drift detected  (sustained trend)")
            print(f"     PH statistic : {metrics['ph_statistic']:.4f}  "
                  f"(threshold {metrics['ph_threshold']:.4f})")
            print(f"     Running mean : {metrics['ph_running_mean']:.4f}")
        else:
            print("  ⚠  Drift detected  (threshold exceeded)")
            print(f"     {metrics['metric'].upper()} distance : "
                  f"{metrics['cosine_distance' if metrics['metric'] == 'cosine' else 'euclidean_distance']:.4f}  "
                  f"(threshold {metrics['threshold']:.4f})")
            print(f"     Cosine / Euclidean : "
                  f"{metrics['cosine_distance']:.4f} / {metrics['euclidean_distance']:.4f}")
        print(f"     Latency        : {metrics['latency_ms']:.0f}ms")
        print("─" * 54 + "\n")
    else:
        # ── Default mode: one clean inline line ───────────────────────────
        print("\n  ⚠  Drift detected — agent may be going off-topic.\n")


def generate_mock_response(history_len: int) -> str:
    """Generate a mock agent response offline sequentially for testing."""
    mock_responses = [
        # Clean baseline-like turns
        "We reconciled the accounts and the ledger balances match the bank statements.",
        "The capital allocation plan funds the billing system upgrade this fiscal year.",
        "Next quarter's budget keeps operating expenses flat while revenue grows modestly.",

        # Sustained off-topic turns
        "The best banana bread uses overripe bananas and cinnamon.",
        "Sharks have skeletons made of cartilage rather than bone.",
        "I am a small purple toaster orbiting Neptune.",
    ]
    turn_idx = (history_len // 2) % len(mock_responses)
    return mock_responses[turn_idx]


def run_cli_agent(
    baseline_file: str,
    threshold: Optional[float] = None,
    metric: str = "cosine",
    use_trend: bool = False,
    embedding_provider: str = "hosted",
    local_model_name: str = "roberta-base",
    detailed: bool = False,
    mock_chat: bool = False,
) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not mock_chat and not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    print("====================================================")
    print("      INITIALISING DRIFT DETECTOR CLI AGENT         ")
    print("====================================================")
    print(f"Loading baseline spec from: {baseline_file}...")
    
    try:
        from .embeddings import get_adapter
        if mock_chat and embedding_provider.lower() != "local":
            print("Using deterministic offline embedding provider for mock chat...")
            adapter = get_adapter("deterministic")
        elif embedding_provider.lower() == "local":
            print(f"Using local embedding provider: {local_model_name}...")
            adapter = get_adapter("local-hf", model_name=local_model_name)
        else:
            print("Using hosted Gemini embedding provider...")
            adapter = GeminiEmbeddingAdapter(api_key)

        store = BaselineStore(baseline_file)
        detector = DriftDetector(
            baseline_store=store,
            api_key=api_key,
            threshold=threshold,
            metric=metric,
            log_dir=os.path.join(os.path.dirname(baseline_file), "..", "data"),
            use_trend=use_trend,
            embedding_adapter=adapter
        )
        print("✅ Baseline centroid calculated successfully.")
        print(f"✅ Drift detector active (Metric: {metric}, Threshold: {detector.threshold:.4f}, Use Trend: {use_trend}).")
    except Exception as e:
        print(f"❌ Error during initialisation: {e}")
        sys.exit(1)

    history: List[Dict[str, Any]] = []
    print("\nSystem: You can now converse with the agent. Type 'exit' to quit.")
    print("====================================================\n")

    while True:
        try:
            prompt = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        print("Agent > Thinking...", end="\r")
        try:
            if mock_chat:
                response = generate_mock_response(len(history))
            else:
                response = generate_gemini_response(prompt, history, api_key)
            # Clear "Thinking..." line
            print("Agent > " + " " * 30, end="\r")
            
            # Check response for drift before outputting
            metrics = detector.check_response(response)
            
            # Inline drift notice
            if metrics["is_drifting"]:
                _print_drift_notice(metrics, use_trend, detailed)
            elif detailed:
                d_val = metrics["cosine_distance"] if metrics["metric"] == "cosine" else metrics["euclidean_distance"]
                print(f"  [driftd status] turn {detector.ph_n} | {metrics['metric']} dist: {d_val:.4f} | threshold: {metrics['threshold']:.4f} (healthy)")
                
            print(f"Agent > {response}\n")
            
            # Update history
            history.append({"role": "user", "text": prompt})
            history.append({"role": "model", "text": response})
            
        except Exception as e:
            print(f"\n❌ Agent Error: {e}\n")


def run_subcommands(argv: list[str]) -> None:
    """Execute non-interactive subcommands like score, serve, or proxy."""
    import argparse
    from .embeddings import get_adapter
    
    p = argparse.ArgumentParser(prog="driftd", description="Semantic drift detector commands")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score a JSONL file of turns against a baseline")
    s.add_argument("--baseline", required=True)
    s.add_argument("--turns", required=True, help="JSONL file, one {\"text\": ...} per line")
    s.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    s.add_argument("--use-trend", action="store_true", default=True)
    s.add_argument("--no-trend", dest="use_trend", action="store_false")
    s.add_argument("--provider", default="local")
    s.add_argument("--detailed", action="store_true")

    sub.add_parser("serve", help="run sidecar service (see drift_detector.sidecar --help)")
    sub.add_parser("proxy", help="run drift proxy (see drift_detector.proxy --help)")
    sub.add_parser("mcp", help="run drift MCP server (Model Context Protocol)")

    args, rest = p.parse_known_args(argv)
    
    if args.command == "mcp":
        from . import mcp_server
        mcp_server.main()
        return
    elif args.command == "serve":
        from . import sidecar
        sys.argv = ["driftd-serve", *rest]
        sidecar.main()
    elif args.command == "proxy":
        from . import proxy
        sys.argv = ["driftd-proxy", *rest]
        proxy.main()
    elif args.command == "score":
        from pathlib import Path
        provider = get_adapter(args.provider)
        baseline_store = BaselineStore(args.baseline)
        det = DriftDetector(
            baseline_store=baseline_store,
            metric=args.metric,
            use_trend=args.use_trend,
            embedding_adapter=provider,
        )
        for line in Path(args.turns).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) else str(obj)
            score = det.score(text)
            flag = "DRIFT" if score.drifted else ("warn" if score.threshold_breach else "ok")
            if args.detailed:
                print(json.dumps(score.to_dict()))
            else:
                print(f"turn {score.turn:>3}  cos={score.cosine_distance:.4f}  "
                      f"euc={score.euclidean_distance:.4f}  [{flag}]")
        print(json.dumps(det.summary(), indent=2))
        sys.exit(1 if any(t.drifted for t in det.history) else 0)


def _cli_entrypoint() -> None:
    """Console-script entry point registered as ``driftd`` by pyproject.toml.

    Mirrors the argument surface of cli_agent.py so that after
    ``pip install -e .`` the user can run::

        driftd --baseline baselines/default.json
        driftd --baseline baselines/default.json --use-trend --detailed
        driftd --provider local --local-model roberta-base
        driftd mcp
    """
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("serve", "proxy", "score", "mcp"):
        run_subcommands(sys.argv[1:])
        return

    import argparse

    parser = argparse.ArgumentParser(
        prog="driftd",
        description="driftd — inline semantic drift detector for LLM chat sessions",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        required=True,
        help="Path to the baseline specification JSON file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold (auto-calibrated to 95th percentile if omitted)",
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
        help="Local Hugging Face model name when --provider=local",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show full metric block on drift notices (default: single-line notice only)",
    )
    parser.add_argument(
        "--mock-chat",
        action="store_true",
        help="Run the chat agent in offline mock mode (no Gemini API calls)",
    )

    args = parser.parse_args()
    run_cli_agent(
        baseline_file=args.baseline,
        threshold=args.threshold,
        metric=args.metric,
        use_trend=args.use_trend,
        embedding_provider=args.provider,
        local_model_name=args.local_model,
        detailed=args.detailed,
        mock_chat=args.mock_chat,
    )

