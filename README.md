<img width="1200" height="896" alt="image" src="https://github.com/user-attachments/assets/0ff95ab7-9222-41b4-828c-ee326905bd83" />

# Drift Detector

Drift Detector uses tried and tested statistical techniques to keep your LLM agent at peak performance. It monitors the output of your agent sessions, measuring semantic drift to warn you when responses begin degrading as the context window fills.

## MVP Scope (v1 CLI)

The initial MVP implements:
1. **Curated Baseline Store:** Establish a baseline using high-quality curated specification examples.
2. **Embedding Distance Detector:** Track cosine distance of output embeddings against the fixed baseline centroid.
3. **CLI Interceptor:** Attach to a CLI chat session and print inline warnings when the drift threshold is breached.

## Getting Started

### Prerequisites

- Python 3.10+
- `sentence-transformers` or access to an embedding model API

### Installation

1. Clone the repository and navigate to it.
2. Set up a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Project Structure

- `src/drift_detector/`
  - `__init__.py`
  - `detector.py` - Core drift detection logic.
  - `baseline.py` - Curated baseline example loader.
  - `cli.py` - Interceptor for CLI agent sessions.
- `baselines/` - Storage for curated specification examples (JSON/Text).
