"""driftd command-line interface.

Subcommands:
  score   score a JSONL file of turns against a baseline file (batch / CI mode)
  serve   run the sidecar HTTP service
  proxy   run the OpenAI-compatible drift proxy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baseline import BaselineStore
from .detector import DriftDetector
from .embedding import get_provider


def cmd_score(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    baseline = BaselineStore(provider).build_from_file(args.baseline)
    det = DriftDetector(baseline, provider, metric=args.metric, use_trend=args.use_trend)
    for line in Path(args.turns).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj["text"] if isinstance(obj, dict) else str(obj)
        score = det.score(text)
        if args.detailed:
            print(json.dumps(score.to_dict()))
        else:
            print(f"turn {score.turn:>3}  cos={score.cosine_distance:.4f}  "
                  f"euc={score.euclidean_distance:.4f}  [{score.badge}]")
    print(json.dumps(det.summary(), indent=2))
    return 1 if any(t.drifted for t in det.history) else 0


def cmd_interactive(args: argparse.Namespace) -> int:
    provider = get_provider(args.provider)
    baseline = BaselineStore(provider).build_from_file(args.baseline)
    det = DriftDetector(baseline, provider, metric=args.metric, use_trend=args.use_trend)
    print("\n--- Drift Detector Interactive Session ---")
    print(f"Baseline: {len(baseline.samples)} examples loaded")
    print("Type any text below to score it against the baseline.")
    print("Type 'exit' or press Ctrl+C to quit.\n")
    while True:
        try:
            line = input("Input > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            break
        if line.startswith("/compact"):
            parts = line.split(" ", 1)
            summary = parts[1].strip() if len(parts) > 1 else None
            msg = det.handle_compaction(compacted_summary=summary)
            print(f"        └─ {msg}\n")
            continue
        score = det.score(line)
        print(f"        └─ status: {score.badge}  (cos: {score.cosine_distance:.4f} | euc: {score.euclidean_distance:.4f})\n")
    print("\nSession summary:")

    print(json.dumps(det.summary(), indent=2))
    return 0


def run_cli_agent(**kwargs):
    import argparse
    baseline = kwargs.get("baseline_file", "baselines/default.json")
    provider_name = kwargs.get("embedding_provider", "local")
    metric = kwargs.get("metric", "cosine")
    use_trend = kwargs.get("use_trend", True)
    args = argparse.Namespace(
        baseline=baseline,
        provider=provider_name,
        metric=metric,
        use_trend=use_trend,
    )
    return cmd_interactive(args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="driftd", description="Semantic drift detector for LLM sessions")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="score a JSONL file of turns against a baseline")
    s.add_argument("--baseline", required=True)
    s.add_argument("--turns", required=True, help="JSONL file, one {\"text\": ...} per line")
    s.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    s.add_argument("--use-trend", action="store_true", default=True)
    s.add_argument("--no-trend", dest="use_trend", action="store_false")
    s.add_argument("--provider", default="local")
    s.add_argument("--detailed", action="store_true")
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("interactive", help="run interactive terminal session to test drift scoring live")
    i.add_argument("--baseline", default="baselines/default.json")
    i.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    i.add_argument("--use-trend", action="store_true", default=True)
    i.add_argument("--no-trend", dest="use_trend", action="store_false")
    i.add_argument("--provider", default="local")
    i.set_defaults(func=cmd_interactive)

    sub.add_parser("serve", help="run sidecar service (see drift_detector.sidecar --help)")
    sub.add_parser("proxy", help="run drift proxy (see drift_detector.proxy --help)")

    args, rest = p.parse_known_args(argv)
    if args.command == "serve":
        from . import sidecar
        sys.argv = ["driftd-serve", *rest]
        sidecar.main()
        return 0
    if args.command == "proxy":
        from . import proxy
        sys.argv = ["driftd-proxy", *rest]
        proxy.main()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
