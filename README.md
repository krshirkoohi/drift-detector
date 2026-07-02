# driftd — Inline Semantic Drift Detector

**driftd** monitors LLM chat sessions in real time and surfaces an inline notice the moment the agent's responses drift away from an established baseline. It is silent on clean sessions and fires within one turn of detecting a shift.

> **Scope:** driftd measures *semantic and topic consistency* — it is a quality-consistency tool, not a fact-checker. It detects when an agent drifts off-topic or changes domain; it cannot verify whether an answer is factually correct.

---

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/krshirkoohi/drift-detector.git
cd drift-detector

# Install in editable mode (numpy is the only hard dependency)
pip install -e .

# For local (offline) embedding support
pip install -e ".[local]"
```

### 2. Set your API key

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. Run

```bash
driftd --baseline baselines/default.json
```

That's it. Start chatting — you'll see an inline notice if the agent goes off-topic, and nothing otherwise.

---

## Baseline Format

A baseline is a JSON file with a name, a description, and a list of 10–20 example responses that represent *good, on-topic* agent output for your use case.

```json
{
  "name": "my-assistant",
  "description": "Baseline for a Python coding assistant",
  "examples": [
    "You can sort a list with sorted() or list.sort().",
    "Use a dictionary comprehension: {k: v for k, v in items.items()}.",
    "The @dataclass decorator auto-generates __init__ from annotated fields."
  ]
}
```

The detector computes the centroid of these examples and flags responses that deviate significantly from it. The threshold is auto-calibrated to the 95th percentile of baseline distances — no manual tuning needed.

---

## CLI Reference

```
driftd --baseline <path> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--baseline` | *(required)* | Path to the baseline JSON file |
| `--metric` | `cosine` | Distance metric: `cosine` or `euclidean` |
| `--threshold` | auto | Override the auto-calibrated threshold |
| `--use-trend` | off | Enable Page-Hinkley sustained-trend detection |
| `--provider` | `hosted` | Embedding provider: `hosted` (Gemini API) or `local` (Hugging Face) |
| `--local-model` | `roberta-base` | Local model name when `--provider=local` |
| `--detailed` | off | Show full metric block on drift notices |

### Notice modes

**Default** — silent on clean turns, one line on drift:
```
  ⚠  Drift detected — agent may be going off-topic.
```

**`--detailed`** — silent on clean turns, metric block on drift:
```
  ──────────────────────────────────────────────────────
  ⚠  Drift detected  (threshold exceeded)
     COSINE distance : 0.0538  (threshold 0.0280)
     Cosine / Euclidean : 0.0538 / 1.2847
     Latency           : 142ms
  ──────────────────────────────────────────────────────
```

### Sustained-trend mode (`--use-trend`)

Adds a Page-Hinkley streaming change-detection layer on top of the per-turn threshold check. Use this when you want to catch gradual drift that accumulates across many turns rather than single-turn spikes.

```bash
driftd --baseline baselines/default.json --use-trend --detailed
```

---

## Embedding Providers

### Hosted (default)

Uses the Gemini embedding API. Requires `GEMINI_API_KEY`. Includes automatic retry with exponential backoff on transient errors (429, 5xx).

```bash
driftd --baseline baselines/default.json --provider hosted
```

### Local (offline)

Uses a Hugging Face transformer model loaded from local cache. No API key needed. Requires `pip install -e ".[local]"`.

```bash
driftd --baseline baselines/default.json --provider local --local-model roberta-base
```

> **Known limitation:** With `roberta-base` on tight, homogeneous baselines, the auto-calibrated threshold can be too narrow, causing false positives on clean sessions. Use `--threshold` to override manually, or switch to the hosted provider which has richer embedding variance.

---

## Programmatic Usage

You can use driftd as a library inside your own agent harness:

```python
from drift_detector import BaselineStore, DriftDetector, AgentHarness
from drift_detector.embeddings import GeminiEmbeddingAdapter

adapter  = GeminiEmbeddingAdapter(api_key="...")
store    = BaselineStore("baselines/default.json")
detector = DriftDetector(baseline_store=store, metric="cosine", embedding_adapter=adapter)
harness  = AgentHarness(detector=detector, log_dir="data/harness_logs", detailed=True)

harness.start_session(session_id="my-session")

for prompt, response in my_conversation:
    record = harness.process_turn(user_prompt=prompt, agent_response=response)
    # record.is_drifting, record.cosine_distance, etc.

summary = harness.end_session()
# summary.drift_rate, summary.drifted_turns, etc.
```

Per-turn records are written to `data/harness_logs/<session_id>.jsonl`. A session summary JSON is written on `end_session()`.

---

## Running Tests

```bash
# Harness regression suite (20 tests, uses local roberta-base, no API key needed)
python3 tests/test_harness.py

# Single-response cosine/euclidean tests (requires GEMINI_API_KEY)
python3 test_drift_detector.py

# Full regression suite
python3 tests/run_regression.py
```

---

## Project Structure

```
drift-detector/
├── src/drift_detector/
│   ├── baseline.py       # BaselineStore — loads examples, computes centroid
│   ├── detector.py       # DriftDetector — threshold + Page-Hinkley logic
│   ├── embeddings.py     # EmbeddingAdapter — Gemini / Local / OpenAI Compatible / Deterministic
│   ├── harness.py        # AgentHarness — session lifecycle, turn capture, JSONL logging
│   ├── sidecar.py        # HTTP Sidecar service
│   ├── proxy.py          # OpenAI-compatible API Proxy server
│   └── cli.py            # CLI entrypoint for interactive chat, score, serve and proxy commands
├── baselines/
│   └── default.json      # Default Python/ML-engineering baseline
├── demo/
│   ├── index.html        # Interactive HTML client & drift dashboard
│   ├── preview_desktop.png
│   └── preview_mobile.png
├── experiments/
│   └── pollutant_validation.py   # Off-topic pollutant grid experiment
├── tests/
│   ├── test_harness.py           # 20 harness regression tests
│   ├── test_core.py              # Unit tests for baseline, thresholds, and Page-Hinkley blip forgiveness
│   ├── run_regression.py         # Labelled fixture regression suite
│   └── fixtures/
│       └── regression_fixtures.json
├── run_harness.py        # Standalone harness entrypoint
├── cli_agent.py          # Alternative CLI entrypoint (no install required)
└── pyproject.toml        # Package config — installs driftd command
```

---

## Known Limitations (v0.2.0)

| Limitation | Notes |
|---|---|
| Factually wrong answers are invisible | Semantic embeddings cannot detect factual inaccuracy within the same domain. A separate fact-checking layer would be required. |
| Local embedding threshold calibration | `roberta-base` produces very tight clusters on homogeneous baselines, making the auto-threshold too narrow. Use `--threshold` to override or switch to hosted embeddings. |
| Single baseline per session | Multi-baseline or dynamic baseline switching is not yet supported. |

---

## Roadmap

- **v0.1** — Initial proof of concept & validation tests.
- **v0.2** *(current)* — Subcommands for batch turn scoring (`score`), sidecar HTTP service (`serve`), and OpenAI-compatible completions API proxy (`proxy`), plus interactive Web UI.
- **v1.0** *(post-QA)* — Human QA sign-off, PyPI packaging, versioned release with git tag and CHANGELOG.
