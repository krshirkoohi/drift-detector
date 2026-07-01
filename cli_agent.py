#!/usr/bin/env python3
import sys
import os
import argparse

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from drift_detector.cli import run_cli_agent

def main():
    parser = argparse.ArgumentParser(description="Drift Detector CLI Agent MVP")
    parser.add_argument(
        "--baseline",
        default=os.path.join(os.path.dirname(__file__), "baselines", "default.json"),
        help="Path to the baseline specification JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Distance threshold for triggering warnings. If omitted, auto-calibrates using the 95th percentile of baseline distances."
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric to use for drift calculation (default: cosine)"
    )
    parser.add_argument(
        "--use-trend",
        action="store_true",
        help="Toggle Page-Hinkley trend checking for sustained-trend drift detection."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show full metric block alongside drift notices (default: single-line notice only)."
    )
    parser.add_argument(
        "--provider",
        choices=["hosted", "local"],
        default="hosted",
        help="Embedding provider to use: 'hosted' for Gemini API, 'local' for Hugging Face transformer models (default: hosted)"
    )
    parser.add_argument(
        "--local-model",
        default="roberta-base",
        help="Local Hugging Face model name to use if provider is 'local' (default: roberta-base)"
    )
    
    args = parser.parse_args()
    
    run_cli_agent(
        args.baseline,
        args.threshold,
        args.metric,
        args.use_trend,
        args.provider,
        args.local_model,
        args.detailed
    )

if __name__ == "__main__":
    main()
