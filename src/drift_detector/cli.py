import os
import sys
import json
import urllib.request
from typing import List, Dict, Any
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

def run_cli_agent(baseline_file: str, threshold: float = 0.25, metric: str = "cosine") -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    print("====================================================")
    print("      INITIALISING DRIFT DETECTOR CLI AGENT         ")
    print("====================================================")
    print(f"Loading baseline spec from: {baseline_file}...")
    
    try:
        store = BaselineStore(baseline_file)
        detector = DriftDetector(
            baseline_store=store,
            api_key=api_key,
            threshold=threshold,
            metric=metric,
            log_dir=os.path.join(os.path.dirname(baseline_file), "..", "data")
        )
        print("✅ Baseline centroid calculated successfully.")
        print(f"✅ Drift detector active (Metric: {metric}, Threshold: {threshold}).")
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
            response = generate_gemini_response(prompt, history, api_key)
            # Clear "Thinking..." line
            print("Agent > " + " " * 30, end="\r")
            
            # Check response for drift before outputting
            metrics = detector.check_response(response)
            
            # If drift is detected, inject inline notification first
            if metrics["is_drifting"]:
                print("\n" + "=" * 52)
                print(f"⚠️  DRIFT DETECTED: Output has drifted from baseline spec!")
                print(f"   Active Metric:   {metrics['metric'].upper()}")
                print(f"   Cosine Distance: {metrics['cosine_distance']:.4f}")
                print(f"   Euclid Distance: {metrics['euclidean_distance']:.4f}")
                print(f"   Threshold:       {metrics['threshold']:.4f}")
                print(f"   Analysis Latency: {metrics['latency_ms']:.1f}ms")
                print("=" * 52 + "\n")
                
            print(f"Agent > {response}\n")
            
            # Update history
            history.append({"role": "user", "text": prompt})
            history.append({"role": "model", "text": response})
            
        except Exception as e:
            print(f"\n❌ Agent Error: {e}\n")
