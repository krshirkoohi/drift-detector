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
        default=0.25,
        help="Cosine distance threshold for triggering drift warnings (default: 0.25)"
    )
    
    args = parser.parse_args()
    
    run_cli_agent(args.baseline, args.threshold)

if __name__ == "__main__":
    main()
