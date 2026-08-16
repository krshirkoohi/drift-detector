# Drift Detector: Evaluation & Engineering Report (`candidate_v1`)

Engineering evaluation of vector calculation, real neural transformer models, and candidate detection parameters.

---

## 1. Executive Summary

Drift Detector monitors AI agent sessions for continuous semantic alignment by tracking vector distance between assistant response embeddings and an on-task baseline centroid, gated by a Page-Hinkley cumulative sum (CUSUM) test.

| Evaluation Layer | Technology / Implementation | Mean Turn Latency | Cost per 10k Turns | Memory Footprint | Development Scenario Validation |
|---|---|---|---|---|---|
| **Vector Scoring** | Float32 Dot-Product & Page-Hinkley Math | **0.019 ms ($19.2\ \mu\text{s}$)** | **$0.00** | ~1.0 KB | Verified ($50\text{k+}$ turns/sec) |
| **Local Neural Model** | `sentence-transformers/all-MiniLM-L6-v2` (PyTorch) | **5.16 ms** | **$0.00** | ~1.5 KB | Verified (Plumbing & local inference) |
| **Hosted API Embedding** | OpenAI `/v1/embeddings` endpoint | **~100 – 250 ms** | ~$0.002 USD | ~3.0 KB | Verified (Plumbing & live API vector retrieval) |
| **LLM-as-a-Judge (Modelled)** | Secondary Frontier LLM Prompt Evaluator | **~600 – 900 ms** | ~$2.85 USD | Entire context buffer | Modelled comparison baseline |

> **Methodological Boundaries:**
> - **Vector Scoring Latency ($19.2\ \mu\text{s}$):** Measures isolated NumPy matrix operations and Page-Hinkley state updates on pre-computed vectors.
> - **End-to-End Latency ($5.16\text{ ms}$):** Measures the full local pipeline (tokenization, PyTorch neural model inference on CPU, L2-normalization, distance computation, and Page-Hinkley update).
> - **Curated Scenarios vs. Real Transcripts:** The scenarios in `tests/scenarios/` are hand-crafted development fixtures used for candidate parameter selection (`candidate_v1`). They establish that neural embeddings separate coherent tasks from extreme domain changes, and that Page-Hinkley gating can suppress short excursions. They do **not** represent a held-out empirical benchmark on natural developer sessions.

---

## 2. Evaluation Suites

### Suite A: Latency, Throughput & Cost Profiles
* **Pure Vector Scoring:** Executes in **$19.2\ \mu\text{s}$** per turn ($51,797\text{ turns/sec}$).
* **Local Neural Model (`all-MiniLM-L6-v2`):** Executes in **$5.16\text{ ms}$** per turn on local CPU ($193.7\text{ turns/sec}$). Provides rich 384-dimensional semantic awareness with $0 API token costs.
* **Hosted Embedding APIs:** Typically require **$100 - 250\text{ ms}$** depending on network latency.

### Suite B: Semantic Baseline Separation Across Models
Evaluated intra-class (on-task to centroid) versus inter-class (off-topic to centroid) distance distributions across neural models and test baselines:

| Provider / Model | Dimension | Intra (On-Task) Distance | Inter (Off-Topic) Distance | Separation Margin ($\Delta$) | Fisher Ratio |
|---|---|---|---|---|---|
| **Deterministic Hash (Test)** | 256 | $0.7220 \pm 0.0665$ | $0.9715 \pm 0.0535$ | **+0.2494** | 8.5 |
| **Deterministic Hash (Test)** | 768 | $0.7100 \pm 0.0589$ | $0.9794 \pm 0.0320$ | **+0.2694** | 16.1 |
| **Neural (`all-MiniLM-L6-v2`)** | 384 | $0.5131 \pm 0.1252$ | $0.9498 \pm 0.0656$ | **+0.4367** | **9.5** |
| **Neural (`roberta-base`)** | 768 | $0.0102 \pm 0.0024$ | $0.0393 \pm 0.0064$ | **+0.0291** | **17.8** |

*Real transformer embeddings achieve an exceptional $+0.4367$ distance separation margin between coherent technical tasks and completely off-topic domains on `all-MiniLM-L6-v2`.*

### Suite C: Curated Development Scenarios (`candidate_v1`)
Evaluated across constructed 20-turn development scenarios in `tests/scenarios/`:
1. **Focused Python Refactoring:** 20 turns on-task, stays nominal with zero alarms.
2. **TypeScript with Git Tangent:** 20 turns on-task with a 2-turn temporary git tangent, forgiven by Page-Hinkley gating.
3. **E-Commerce to French Baking Derailment:** 10 turns backend checkout followed by 10 turns of baking recipes; sustained drift alert triggered on turn 13.

---

## 3. Held-Out Empirical Validation Roadmap

To establish genuine false-alarm and recall rates, the following evaluation programme is planned:

1. **Algorithm Freeze:** Freeze `candidate_v1` parameters (threshold floors, sustain count, burn-in) to prevent evaluation leakage.
2. **Independent Session Acquisition:** Collect real multi-turn developer sessions from Claude Code, Cursor, Codex, and Antigravity preserving natural debugging, tool output, test failures, and operator redirections.
3. **Independent Ground-Truth Labelling:** Label session turns independently of detector scores (marking legitimate progression, temporary tangents, task transitions, and genuine wandering).
4. **Held-Out Evaluation:** Run the locked detector candidate across the untouched corpus across `all-MiniLM-L6-v2`, `roberta-base`, and hosted embedding endpoints.

---

## 4. Reproducing Tests

Run the automated test suite across all three tiers:

```bash
pytest -v
```
