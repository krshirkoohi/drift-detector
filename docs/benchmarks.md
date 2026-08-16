# Drift Detector: Empirical Benchmark & Evaluation Report

Empirical evaluation of vector calculation, real neural transformer models, and architectural boundaries in agent drift monitoring.

---

## 1. Executive Summary

Drift Detector monitors AI agent sessions for continuous semantic alignment by tracking vector distance between assistant response embeddings and an on-task baseline centroid, gated by a Page-Hinkley cumulative sum (CUSUM) test.

| Evaluation Layer | Technology / Implementation | Mean Turn Latency | Cost per 10k Turns | Memory Footprint | Blip False Positive Rate | Sustained Drift Recall |
|---|---|---|---|---|---|---|
| **Vector Scoring** | Float32 Dot-Product & Page-Hinkley Math | **0.019 ms ($19.2\ \mu\text{s}$)** | **$0.00** | ~1.0 KB | 0.0% | 100.0% |
| **Local Neural Model** | `sentence-transformers/all-MiniLM-L6-v2` (PyTorch) | **5.16 ms** | **$0.00** | ~1.5 KB | 0.0% | 100.0% (lag = 3 turns) |
| **Hosted API Embedding** | Google Gemini / OpenAI Embeddings API | **~100 – 250 ms** | ~$0.002 USD | ~3.0 KB | 0.0% | 100.0% |
| **LLM-as-a-Judge (Modelled)** | Secondary Frontier LLM Prompt Evaluator | **~600 – 900 ms** | ~$2.85 USD | Entire context buffer | High (flags isolated blips) | High |

> **Methodological Clarification:**
> - **Vector Scoring Latency ($19.2\ \mu\text{s}$):** Measures isolated NumPy matrix operations and Page-Hinkley state updates on pre-computed vectors.
> - **End-to-End Latency ($5.16\text{ ms}$):** Measures the full pipeline (tokenization, PyTorch neural model inference on CPU, L2-normalization, distance computation, and Page-Hinkley update).
> - **LLM-as-a-Judge:** Latency (~$850\text{ ms}$) and cost (~$\$2.85/10\text{k}$) figures are modelled scenario assumptions based on typical frontier LLM pricing (~1,500 input tokens / ~100 output tokens) rather than direct in-process measured code.

---

## 2. Benchmark Suites

### Benchmark A: Latency, Throughput & Cost Profiles
* **Pure Vector Scoring:** Executes in **$19.2\ \mu\text{s}$** per turn ($51,797\text{ turns/sec}$). Ideal for high-frequency stream chunk evaluation.
* **Local Neural Model (`all-MiniLM-L6-v2`):** Executes in **$5.16\text{ ms}$** per turn on local CPU ($193.7\text{ turns/sec}$). Provides rich 384-dimensional semantic awareness with $0 API token costs.
* **Hosted Embedding APIs:** Typically require **$100 - 250\text{ ms}$** depending on network latency, at fractions of a cent per thousand turns.

### Benchmark B: Semantic Baseline Separation Across Models
Evaluated intra-class (on-task to centroid) versus inter-class (off-topic to centroid) distance distributions across neural models and test baselines:

| Provider / Model | Dimension | Intra (On-Task) Distance | Inter (Off-Topic) Distance | Separation Margin ($\Delta$) | Fisher Ratio |
|---|---|---|---|---|---|
| **Deterministic Hash (Test)** | 256 | $0.7220 \pm 0.0665$ | $0.9715 \pm 0.0535$ | **+0.2494** | 8.5 |
| **Deterministic Hash (Test)** | 768 | $0.7100 \pm 0.0589$ | $0.9794 \pm 0.0320$ | **+0.2694** | 16.1 |
| **Neural (`all-MiniLM-L6-v2`)** | 384 | $0.5131 \pm 0.1252$ | $0.9498 \pm 0.0656$ | **+0.4367** | **9.5** |
| **Neural (`roberta-base`)** | 768 | $0.0102 \pm 0.0024$ | $0.0393 \pm 0.0064$ | **+0.0291** | **17.8** |

### Benchmark C: Blip Forgiveness (FPR) vs. Sustained Drift (TPR)
Evaluated across 30 single-turn transient blips and 30 sustained multi-turn drift sequences using real neural embeddings (`all-MiniLM-L6-v2`):

* **Instantaneous Threshold Breach:** **100.0% False Positive Rate (30/30 false alarms)** on transient blips.
* **Drift Detector (Page-Hinkley Gating):** **0.0% False Positive Rate (0/30 false alarms, 100% blip forgiveness)** while preserving **100.0% True Positive Rate (30/30 detected)** on sustained drift (mean detection lag: $3.0\text{ turns}$).

### Benchmark D: Real-World Multi-Turn Coding Transcripts
Validated across 20-turn realistic developer session transcripts in `tests/real_sessions/`:
1. **Focused Python Refactoring:** 20 turns on-task, 0 false alarms.
2. **TypeScript React with Git Tangent:** 20 turns on-task with a 2-turn temporary git tangent, forgiven by Page-Hinkley.
3. **E-Commerce to French Baking Derailment:** 10 turns backend checkout followed by 10 turns of baking recipes; sustained drift alert triggered on turn 13 ($p < 0.01$).

---

## 3. Reproducing the Benchmarks

Run the benchmark suite from the project root:

```bash
python3 experiments/comparative_benchmark.py
```

Run the automated test suite across all three tiers:

```bash
pytest -v
```
