# Drift Detector

**Drift Detector** is a lightweight, early-warning check-engine light for long AI coding sessions. It embeds assistant responses, measures vector distance from an on-task baseline centroid, and warns the operator when the agent quietly drifts away from the session's core working mode.

One-off tangents are forgiven as transient blips. Sustained divergence is flagged.

* **Live Interactive Simulator:** [https://krshirkoohi.github.io/drift-detector/](https://krshirkoohi.github.io/drift-detector/)

---

## The Core Proposition

> *"A local check-engine light for long AI coding sessions. Drift Detector watches for sustained departure from the session's working mode, forgives one-off tangents, and warns the operator before they continue trusting an off-task run."*

### What Drift Detector Is and Is Not

* **What it is:** A continuous semantic alignment monitor that tracks vector departure from the initial session task baseline.
* **What it forgives:** Transient blips, one-off operator queries, jokes, or quick tool lookups.
* **What it is NOT:**
  * **Not a fact-checker:** It does not verify correctness of code, APIs, or statements within an on-topic domain.
  * **Not a code linter or syntax verifier:** Use compiler errors, linters, and unit tests for code correctness.
  * **Not a proof of context exhaustion:** It detects topic and mode departure, not theoretical LLM token window degradation.

---

## Key Capabilities

1. **Page-Hinkley CUSUM Trend Gating:** Distinguishes isolated tangents from permanent divergence. A single off-topic query is forgiven; persistent domain drift across consecutive turns trips the alert.
2. **Explicit Semantic Providers (Zero Silent Fallbacks):**
   * `local`: Real local neural embedding model (`sentence-transformers/all-MiniLM-L6-v2` or `roberta-base` via PyTorch).
   * `gemini`: Hosted Google Gemini embedding API.
   * `openai`: Hosted OpenAI / Ollama compatible embedding API.
   * `test` / `deterministic`: Offline hash projection for fast CI and math invariant validation.
3. **Decoupled Anchor & Compaction Lifecycle:** When context compaction occurs, transient Page-Hinkley trend accumulators are reset while **preserving the original mission anchor** by default to prevent accumulated drift from silently renormalising. Explicit `rebase()` allows intentional task transitions.
4. **Explicit Baseline Sample Support:** Sessions report lifecycle state (`calibrating` vs `monitoring`) and sample support level (`calibration_support: low | moderate | high`) based on baseline sample count.
5. **Homogeneous Transformation:** Standardised on assistant response embeddings across both calibration and live evaluation.
6. **Tri-Partite Test Suite:** Segregated test suites across `tests/unit/` (invariants & core math), `tests/synthetic/` (10k stress & numerical stability), and `tests/scenarios/` (curated development scenarios).

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/krshirkoohi/drift-detector.git
cd drift-detector
pip install -e .
```

### 2. FastMCP Server Setup

Add `drift-detector` to your MCP client configuration (e.g. `claude_desktop_config.json`, Cursor, or Antigravity):

```json
{
  "mcpServers": {
    "drift-detector": {
      "command": "drift-detector-mcp"
    }
  }
}
```

#### MCP Tools & Control Plane:
* `drift_evaluate_turn(agent_response, ...)` — Evaluate an agent turn in background.
* `drift_compact_reset(compacted_summary, ...)` — Reset Page-Hinkley accumulators on context compaction.
* `drift_rebase(anchor_text, ...)` — Explicitly rebase mission centroid upon intentional task change.
* `drift_get_status()` — View session metrics, lifecycle state (`calibrating`/`monitoring`), and sample support.
* `drift_toggle(mode)` — Toggle background monitoring `on`, `off`, `status`, or `reset`.

---

## Python API Usage

```python
from drift_detector import DriftDetector, get_provider

# Initialize with real local neural model (all-MiniLM-L6-v2)
provider = get_provider("local")

detector = DriftDetector.from_examples(
    baseline_texts=[
        "Building backend APIs in Python and FastAPI with async database sessions",
        "Writing unit tests with pytest and mocked database fixtures",
        "Database migrations and PostgreSQL query performance tuning"
    ],
    provider=provider,
    metric="cosine",
    use_trend=True
)

# 1. On-task response
score1 = detector.score("Implementing FastAPI route handlers with dependency injection.")
print(score1.badge)  # "nominal"

# 2. Isolated tangent (blip)
score2 = detector.score("To undo your last local git commit, run git reset --soft HEAD~1.")
print(score2.badge)  # "threshold breach" (forgiven by Page-Hinkley)

# 3. Return to task resets the streak
score3 = detector.score("Configuring SQLAlchemy connection pooling and async engine parameters.")
print(score3.badge)  # "nominal"

# 4. Compaction reset (preserves original mission anchor)
detector.handle_compaction(compacted_summary="Session summary: FastAPI endpoints configured.")

# 5. Session summary
print(detector.summary())
```

---

## Latency, Microbenchmarks & Performance Boundaries

### 1. Execution Profiles & Model Latencies

| Evaluation Layer | Technology / Implementation | Turn Latency | Cost per 10k Turns | Memory Footprint |
|---|---|---|---|---|
| **Vector Scoring (Microbenchmark)** | NumPy dot-product & Page-Hinkley update | **0.019 ms ($19.2\ \mu\text{s}$)** | **$0.00** | ~1.0 KB float32 centroid |
| **Local Neural Inference** | `sentence-transformers/all-MiniLM-L6-v2` | **5.16 ms** (CPU) | **$0.00** (Local) | ~1.5 KB centroid (+ PyTorch weights) |
| **Hosted Neural Embedding** | Google Gemini / OpenAI Embeddings API | **~100 – 250 ms** (Network) | ~$0.002 USD | ~3.0 KB centroid |
| **LLM-as-a-Judge (Scenario Model)** | Secondary Frontier LLM Prompt Evaluator | **~600 – 900 ms** (Network) | ~$2.85 USD | Entire conversation context buffer |

> *Note: Vector scoring ($19\ \mu\text{s}$) measures isolated mathematical distance computation on pre-computed embeddings. End-to-end latency includes embedding model inference ($5\text{ ms}$ local or network API latency). LLM-as-a-judge latency and cost figures are modelled scenario assumptions based on typical frontier LLM pricing (~1,500 prompt tokens / 100 output tokens).*

### 2. Development Status & Validation Roadmap

| Dimension | Scope / Evaluation Method | Status |
|---|---|---|
| **Local PyTorch Transformer Integration** | `sentence-transformers/all-MiniLM-L6-v2` and `roberta-base` | **Verified** (Offline neural inference, $5\text{ ms}$ latency) |
| **Cloud Embedding Provider Integration** | `OpenAICompatibleProvider` via `/v1/embeddings` endpoint | **Verified** (Plumbing & live API vector retrieval) |
| **Curated Development Scenarios (`candidate_v1`)** | Constructed 20-turn refactoring, tangent, and baking scenarios | **Verified** (Nominal stability, blip suppression, and sustained alarm) |
| **Stress & Numerical Stability** | 10,000 turns and 100 compaction cycles | **Verified** (> 50,000 turns/sec, zero drift leakage) |
| **Held-Out Real Transcript Benchmark** | Independent, un-tuned real developer sessions | **Planned** (Pending independent collection and labelling) |

> **Evaluation Note:** Current test fixtures (`tests/scenarios/fixtures/curated_scenarios.json`) are hand-crafted development scenarios used for exploratory baseline development and candidate parameter selection (`candidate_v1`). Formal false-alarm and recall metrics on natural, independently labelled developer transcripts will be evaluated on a locked, held-out corpus in future benchmarks.

---

## Interactive Web Simulator

Try the simulator directly in your browser:

🔗 **[Launch Interactive Web Simulator](https://krshirkoohi.github.io/drift-detector/)**

---

## License

MIT License. Developed and maintained by Kavia.
