# Changelog

All notable changes to **Drift Detector** (`driftd`) are documented in this file.

## [0.1.0] - 2026-08-16

### Added
- **Core Drift Detection Engine (`DriftDetector`):**
  - Cosine and Euclidean vector distance scoring against an L2-normalised baseline centroid.
  - Page-Hinkley CUSUM trend gating to forgive isolated single-turn tangents while flagging sustained semantic departure.
  - Explicit session lifecycle (`calibrating` vs `monitoring`) and sample support reporting (`calibration_support: low | moderate | high`).
- **Decoupled Compaction & Rebase Lifecycle:**
  - `handle_compaction()` resets transient Page-Hinkley accumulators while preserving the original mission anchor by default (`rebase_anchor=False`).
  - `rebase()` explicitly re-anchors the baseline centroid upon intentional task transitions.
- **Explicit Embedding Providers (`drift_detector.embeddings`):**
  - `local`: Offline PyTorch transformer inference (`sentence-transformers/all-MiniLM-L6-v2` and `roberta-base`).
  - `gemini`: Hosted Google Gemini Embeddings API (`gemini-embedding-001`).
  - `openai`: OpenAI and Ollama-compatible `/v1/embeddings` API (`OpenAICompatibleProvider`).
  - `test` / `deterministic`: Offline deterministic hash projection for fast CI and invariant verification, with zero silent fallbacks across providers.
- **FastMCP Control Plane (`drift-detector-mcp`) & CLI (`driftd`):**
  - MCP tools: `drift_evaluate_turn`, `drift_compact_reset`, `drift_rebase`, `drift_get_status`, `drift_toggle`, and `drift_attach_session`.
  - CLI (`driftd score`) for batch JSONL evaluation with CI-friendly exit codes (`0` nominal, `1` drift detected).
- **Interactive Web Simulator & Tri-Partite Test Suite:**
  - Browser-based simulator (`docs/index.html`) deployed via GitHub Pages.
  - Segregated test suites across `tests/unit/`, `tests/synthetic/` (10,000-turn and 100-compaction stress tests), and `tests/scenarios/` (curated development scenarios).
