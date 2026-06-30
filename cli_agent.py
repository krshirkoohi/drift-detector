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
    
    args = parser.parse_args()
    
    run_cli_agent(args.baseline, args.threshold, args.metric)

if __name__ == "__main__":
    main()
