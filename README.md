# Drift-Detector

**Drift-Detector** is an early-warning monitoring engine for AI agents. It embeds assistant responses, measures their vector distance from an on-task baseline centroid, and flags when the agent quietly drifts off-task.

One-off tangents are forgiven as transient blips. Sustained divergence is flagged.

[![Interactive Demo](https://img.shields.io/badge/Demo-Live_Simulator-blue)](https://krshirkoohi.github.io/drift-detector/) [![Tests](https://img.shields.io/badge/Tests-24%2F24_Passing-brightgreen)](tests/) [![Architecture](https://img.shields.io/badge/MCP-FastMCP_Ready-purple)](src/drift_detector/mcp_server.py)

---

## Key Capabilities

* **Page-Hinkley CUSUM Trend Gating:** Distinguishes isolated tangents from permanent divergence. A single off-topic joke or query is forgiven; persistent domain drift across consecutive turns trips the alarm.
* **Dynamic Auto-Baselining:** Automatically calibrates an ultra-lightweight ~1 KB `float32` centroid from the opening turns of any chat session. No manual baseline JSON required.
* **Compaction Lifecycle Recovery:** Built-in listener (`drift_compact_reset`) that automatically clears historical accumulators and re-centers the baseline whenever conversation context is compressed.
* **FastMCP Server Integration:** Runs natively as an MCP server (`drift-detector-mcp`) compatible with Claude Desktop, Cursor, and Antigravity.
* **High-Throughput Core:** Benchmarked at **50,000+ turns/second** (< 0.02ms per evaluation) with a tiny memory footprint.

> **Scope Note:** Drift-Detector is a *semantic alignment* tool, not a fact checker. It detects topic wandering, mode shifts, and domain drift; it does not judge factual correctness within an on-topic response.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/krshirkoohi/drift-detector.git
cd drift-detector
pip install -e .
```

### 2. Interactive Terminal Test

Test live typing and real-time sub-input indicators directly in your terminal:

```bash
python interactive_test.py
```

### 3. FastMCP Server Setup

Add `drift-detector` to your MCP client configuration (e.g. `claude_desktop_config.json` or Antigravity):

```json
{
  "mcpServers": {
    "drift-detector": {
      "command": "drift-detector-mcp"
    }
  }
}
```

Available slash commands & tools:
* `/drift status` — View active turn counts, running mean distance, and drift metrics.
* `/drift off` — Disable background monitoring (zero turn overhead).
* `/drift on` — Re-enable auto-calibrated monitoring.
* `/drift reset` — Clear session history and re-baseline.

---

## Python API Usage

```python
from drift_detector.core import DriftDetector
from drift_detector.embedding import DeterministicProvider

# Initialize with known-good on-task examples
detector = DriftDetector.from_examples(
    baseline_texts=[
        "Building backend APIs in Python and FastAPI",
        "Writing unit tests with pytest and coverage reporting",
        "Database migrations and PostgreSQL query optimisation"
    ],
    provider=DeterministicProvider(dim=256),
    metric="cosine",
    use_trend=True
)

# Score incoming assistant turns
turn1 = detector.score("Here is the FastAPI router implementation with dependency injection.")
print(turn1.badge)  # "nominal"

# Isolated tangent (blip)
turn2 = detector.score("Why did the chef bring a ladder? To raise the flavour of the sauce!")
print(turn2.badge)  # "threshold breach" (forgiven by Page-Hinkley)

# Return to task resets the accumulator
turn3 = detector.score("Configuring PostgreSQL connection pool settings for async workers.")
print(turn3.badge)  # "nominal"

# Review session summary
summary = detector.summary()
print(summary)
```

---

## Benchmarks & Comparative Evaluation

Validated in automated benchmark suites (`experiments/comparative_benchmark.py` and `tests/test_benchmarks.py`):

### 1. Vector Math vs. LLM-as-a-Judge

| Metric | Drift-Detector (Vector Math) | LLM-as-a-Judge (Prompt Evaluator) | Advantage |
|---|---|---|---|
| **Mean Turn Latency** | **0.020 ms (19.8 μs)** | ~850 ms | **42,000x faster** |
| **P99 Turn Latency** | **0.029 ms (28.7 μs)** | ~1,450 ms | **50,000x lower jitter** |
| **Cost per 10k Turns** | **$0.00 (100% Free)** | ~$2.85 USD | **Zero API spend** |
| **Throughput** | **50,065 turns/sec** | 1.18 turns/sec | Real-time inline scoring |
| **Memory Footprint** | **~1.0 KB (Float32 Centroid)** | Entire context window buffer | Constant-time $O(1)$ overhead |

### 2. Detection Reliability & Stress Benchmarks

| Evaluation Dimension | Workload / Scenario | Measured Result |
|---|---|---|
| **Blip Forgiveness (FPR)** | 50 transient single-turn tangents | **0.0% false alarms** (100% forgiven) |
| **Sustained Drift (TPR)** | 50 multi-turn off-topic sequences | **100.0% true positive detection** (lag = 3.0 turns) |
| **Baseline Separation ($\Delta$)** | On-task vs off-topic across dims 32–768 | **+0.25 to +0.27 distance margin** |
| **Compaction Recovery** | Context compression event | **100% accumulator wipe & re-alignment** |
| **Concurrency & Thread Safety** | 50 parallel worker threads | **100% thread-safe** (zero race conditions) |


---

## Interactive Web Simulator

A standalone HTML simulation is hosted on GitHub Pages:

🔗 **[Launch Interactive Web Demo](https://krshirkoohi.github.io/drift-detector/)**

---

## License

MIT License. Designed and maintained by Kavia.
