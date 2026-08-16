# Drift-Detector: Empirical Benchmark Report

Empirical evaluation of vector math & real neural transformer embeddings versus secondary LLM-as-a-judge evaluators.

---

## 1. Executive Summary

Drift-Detector monitors AI agent conversations for semantic divergence by tracking the cosine distance between response embeddings and an on-task baseline centroid, gated by a Page-Hinkley cumulative sum (CUSUM) test.

| Evaluation Metric | Vector Math (Deterministic) | Real Neural Transformer (`all-MiniLM-L6-v2`) | LLM-as-a-Judge (Prompt Evaluator) | Margin vs. LLM Judge |
|---|---|---|---|---|
| **Mean Turn Latency** | **0.019 ms (19.2 μs)** | **5.16 ms** | ~850.0 ms | **165x – 44,000x faster** |
| **P99 Turn Latency** | **0.022 ms (22.4 μs)** | **7.45 ms** | ~1,450.0 ms | **190x – 65,000x lower jitter** |
| **Throughput** | **51,797 turns / sec** | **193.7 turns / sec** | 1.18 turns / sec | Real-time streamable |
| **Cost per 10k Turns** | **$0.00 (Free)** | **$0.00 (Local Neural)** | ~$2.85 USD | Zero API spend |
| **Memory Footprint** | **~1.0 KB (Float32 Centroid)** | **~1.5 KB (Float32 Centroid)** | Entire context window buffer | Constant-time $O(1)$ |
| **Blip False Positive Rate** | **0.0% (0 / 30 false alarms)** | **0.0% (0 / 30 false alarms)** | High (flags isolated tangents) | 100% blip forgiveness |
| **Sustained Drift Recall** | **100.0% (30 / 30 detected)** | **100.0% (30 / 30 detected)** | High | Mean lag: 3.0 turns |

---

## 2. Benchmark Methodology & Results

### Benchmark A: Cost, Latency, and Throughput
* **Vector Math (Hash Projection):** Executes in **$19.2\ \mu\text{s}$** ($0.019\text{ ms}$) per turn ($51,797\text{ turns/sec}$). Ideal for high-throughput stream chunk analysis.
* **Real Neural Transformer (`all-MiniLM-L6-v2`):** Executes full PyTorch transformer inference in **$5.16\text{ ms}$** per turn on local CPU ($193.7\text{ turns/sec}$). 165x faster than network LLM calls with $0 API token costs.
* **LLM-as-a-Judge:** Requires sending context history + evaluation system prompt over the network ($\sim 1500\text{ input tokens}$, $\sim 100\text{ output tokens}$), incurring $\sim 850\text{ ms}$ latency and $\$2.85/\text{10k turns}$.

### Benchmark B: Baseline Separation & Provider Consistency
Evaluated intra-class (on-task to centroid) versus inter-class (off-topic to centroid) distance distributions across real neural models and deterministic baselines:

| Provider / Model | Dimension | Intra (On-Task) Distance | Inter (Off-Topic) Distance | Separation Margin ($\Delta$) | Fisher Discriminant Ratio |
|---|---|---|---|---|---|
| **Deterministic** | 64 | $0.7245 \pm 0.1193$ | $0.9786 \pm 0.1516$ | **+0.2541** | 1.7 |
| **Deterministic** | 256 | $0.7220 \pm 0.0665$ | $0.9715 \pm 0.0535$ | **+0.2494** | 8.5 |
| **Deterministic** | 768 | $0.7100 \pm 0.0589$ | $0.9794 \pm 0.0320$ | **+0.2694** | 16.1 |
| **Neural (`all-MiniLM-L6-v2`)** | 384 | $0.5131 \pm 0.1252$ | $0.9498 \pm 0.0656$ | **+0.4367** | **9.5** |
| **Neural (`roberta-base`)** | 768 | $0.0102 \pm 0.0024$ | $0.0393 \pm 0.0064$ | **+0.0291** | **17.8** |

*Real transformer embeddings achieve an exceptional $+0.4367$ distance separation margin on `all-MiniLM-L6-v2` and a Fisher ratio of $17.8$ on `roberta-base`.*

### Benchmark C: Blip Forgiveness (FPR) vs. Sustained Drift (TPR)
Evaluated across 30 single-turn transient blips and 30 sustained multi-turn drift sequences using real neural embeddings (`all-MiniLM-L6-v2`):

* **Instantaneous Threshold Breach:** **100.0% False Positive Rate (30/30 false alarms)** on transient blips.
* **Drift-Detector (Page-Hinkley Gating):** **0.0% False Positive Rate (0/30 false alarms, 100% blip forgiveness)** while preserving **100.0% True Positive Rate (30/30 detected)** on sustained drift (mean detection lag: $3.0\text{ turns}$).

### Benchmark D: Real Neural Context Compaction Recovery
* Verified with real transformer embeddings that context compaction reset completely clears accumulated drift scores and successfully restores nominal baseline tracking on the post-compaction summary.

---

## 3. Reproducing the Benchmarks

Run the automated benchmark suite directly from the repository root:

```bash
python3 experiments/comparative_benchmark.py
```

Automated regression tests verified with pytest:

```bash
pytest tests/test_benchmarks.py -v
```
