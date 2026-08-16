# Drift-Detector

**Drift-Detector** is an early-warning monitoring engine for AI agents. It embeds assistant responses, measures their vector distance from an on-task baseline centroid, and flags when the agent quietly drifts off-task.

One-off tangents are forgiven as transient blips. Sustained divergence is flagged.

[![Interactive Demo](https://img.shields.io/badge/Demo-Live_Simulator-blue)](https://krshirkoohi.github.io/drift-detector/)
[![Tests](https://img.shields.io/badge/Tests-20%2F20_Passing-brightgreen)](tests/)

[![Architecture](https://img.shields.io/badge/MCP-FastMCP_Ready-purple)](src/drift_detector/mcp_server.py)

---

## Key Capabilities

* **Page-Hinkley CUSUM Trend Gating:** Distinguishes isolated tangents from permanent divergence. A single off-topic joke or query is forgiven; persistent domain drift across consecutive turns trips the alarm.
* **Dynamic Auto-Baselining:** Automatically calibrates an ultra-lightweight ~1 KB `float32` centroid from the opening turns of any chat session. No manual baseline JSON required.
* **Compaction Lifecycle Recovery:** Built-in listener (`drift_compact_reset`) that automatically clears historical accumulators and re-centers the baseline whenever conversation context is compressed.
* **FastMCP Server Integration:** Runs natively as an MCP server (`drift-detector-mcp`) compatible with Claude Desktop, Cursor, and Antigravity.
* **High-Throughput Core:** Benchmarked at **15,000+ turns/second** (< 0.07ms per evaluation) with a tiny memory footprint.

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
# {'turns': 3, 'drifted_turns': 0, 'drift_rate': 0.0, 'mean_distance': 0.8124, ...}
```

---

## Benchmarks & Stress Tests

Validated in isolated sandbox stress test suites (`tests/test_stress.py`):

| Test Dimension | Workload | Result |
|---|---|---|
| **Throughput** | 10,000 sequential turns | **15,073 turns/sec** (0.663s total) |
| **Memory Growth** | 10,000 turns | **2.1 MB** (Zero memory leaks) |
| **Compaction Stability** | 100 consecutive resets | **100% stable** (Zero numerical drift) |
| **Blip Forgiveness** | 50 burst tangents / 150 turns | **0 false alarms** (100% recovery) |
| **Concurrency** | 50 parallel worker threads | **100% thread-safe** |
| **Adversarial Inputs** | 100KB strings, emojis, CJK, SQLi | **0 crashes** |

---

## Interactive Web Simulator

A standalone HTML simulation is hosted on GitHub Pages:

🔗 **[Launch Interactive Web Demo](https://krshirkoohi.github.io/drift-detector/)**

---

## License

MIT License. Designed and maintained by Kavia.
