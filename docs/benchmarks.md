# Drift-Detector: Comparative Benchmark Report

Empirical evaluation of vector math semantic drift monitoring versus secondary LLM-as-a-judge evaluators.

---

## 1. Executive Summary

Drift-Detector monitors AI agent conversations for semantic divergence by tracking the cosine distance between response embeddings and an on-task baseline centroid, gated by a Page-Hinkley cumulative sum (CUSUM) test.

| Evaluation Metric | Drift-Detector (Vector Math) | LLM-as-a-Judge (Prompt Evaluator) | Impact Margin |
|---|---|---|---|
| **Mean Turn Latency** | **0.020 ms (19.8 μs)** | ~850.0 ms | **42,000x faster** |
| **P99 Turn Latency** | **0.029 ms (28.7 μs)** | ~1,450.0 ms | **50,000x lower jitter** |
| **Throughput** | **50,065 turns / sec** | 1.18 turns / sec | Real-time streamable |
| **API Cost per 10k Turns** | **$0.00 (100% Free)** | ~$2.85 USD | Zero token overhead |
| **Memory Footprint** | **~1.0 KB (Float32 Centroid)** | Entire context window buffer | Constant-time $O(1)$ |
| **Blip False Positive Rate** | **0.0% (0 / 50 false alarms)** | High (flags isolated jokes/queries) | 100% blip forgiveness |
| **Sustained Drift Recall** | **100.0% (50 / 50 detected)** | High | Mean lag: 3.0 turns |

---

## 2. Benchmark Methodology & Results

### Benchmark A: Cost, Latency, and Throughput
* **Setup:** 1,000 sequential evaluation turns comparing local vector distance scoring against a standard LLM evaluator prompt profile (1,500 input tokens, 100 completion tokens, evaluated at current lightweight model pricing of \$0.15/\$0.60 per million tokens).
* **Finding:** Vector scoring executes in under **20 microseconds** per turn, enabling inline monitoring on every streaming chunk or turn without adding perceptible latency to user interactions.

### Benchmark B: Baseline Separation & Dimensional Stability
* **Setup:** Evaluated intra-class (on-task to centroid) versus inter-class (off-topic to centroid) distance distributions across vector dimensions (32, 64, 128, 256, 768).
* **Finding:** Clear separation margin ($\Delta = \mu_{\text{off-topic}} - \mu_{\text{on-task}} = +0.25 \text{ to } +0.27$) across all dimensions. Fisher's Discriminant Ratio scales with embedding dimensionality (from 1.41 at 32-dim to 16.14 at 768-dim).

### Benchmark C: Blip Forgiveness (FPR) vs. Sustained Drift (TPR)
* **Setup:** 50 single-turn transient blips (isolated off-topic queries immediately returning to task) versus 50 multi-turn sustained drift excursions (continuous divergence across 4+ turns).
* **Finding:**
  - **Instantaneous Threshold Breach (No Page-Hinkley):** **100.0% False Positive Rate** (50/50 false alarms).
  - **Drift-Detector (Page-Hinkley Gating):** **0.0% False Positive Rate** (0/50 false alarms, 100% recovery) while achieving **100.0% True Positive Rate** on sustained domain drift.

### Benchmark D: Context Compaction Recovery
* **Setup:** Simulated long-running sessions with context compaction events mid-stream.
* **Finding:** Auto-detects history truncation (`len(history) < prev_len`), token drops, or `/compact` hooks, instantly wiping elevated accumulators to zero and re-centering the centroid on the post-compaction summary.

---

## 3. Reproducing the Benchmarks

Run the automated benchmark suite directly from the repository root:

```bash
python3 experiments/comparative_benchmark.py
```

Automated regression tests are verified in pytest:

```bash
pytest tests/test_benchmarks.py -v
```
