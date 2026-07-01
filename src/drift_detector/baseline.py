import json
import os
import urllib.request
from typing import List, Optional
import numpy as np
from .embeddings import EmbeddingAdapter

class BaselineStore:
    def __init__(self, baseline_path: str):
        self.baseline_path = baseline_path
        self.name: str = "default"
        self.description: str = ""
        self.examples: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.centroid: Optional[np.ndarray] = None
        self.load_baseline()

    def load_baseline(self) -> None:
        """Load curated examples from the JSON baseline file."""
        if not os.path.exists(self.baseline_path):
            raise FileNotFoundError(f"Baseline file not found at: {self.baseline_path}")
            
        with open(self.baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.name = data.get("name", "default")
            self.description = data.get("description", "")
            self.examples = data.get("examples", [])

    def compute_centroid(
        self, 
        api_key: Optional[str] = None, 
        adapter: Optional[EmbeddingAdapter] = None
    ) -> np.ndarray:
        """Compute the centroid (mean vector) of all baseline examples using NumPy."""
        if not self.examples:
            raise ValueError("No baseline examples found to compute centroid.")

        if adapter is None:
            if not api_key:
                raise ValueError("Either api_key or adapter must be provided.")
            from .embeddings import GeminiEmbeddingAdapter
            adapter = GeminiEmbeddingAdapter(api_key)

        embeddings_list = []
        for example in self.examples:
            emb = adapter.embed(example)
            if emb:
                embeddings_list.append(emb)

        if not embeddings_list:
            raise RuntimeError("Failed to generate embeddings for any baseline examples.")

        self.embeddings = np.array(embeddings_list)
        self.centroid = self.embeddings.mean(axis=0)
        return self.centroid

    def calculate_percentile_threshold(self, metric: str = "cosine", percentile: float = 95.0) -> float:
        """Calculate the threshold (percentile of distances of clean baseline samples from centroid)."""
        if self.embeddings is None or self.centroid is None:
            raise RuntimeError("Baseline centroid and embeddings must be computed first.")
            
        metric = metric.lower()
        if metric == "cosine":
            norms = np.linalg.norm(self.embeddings, axis=1)
            norm_c = np.linalg.norm(self.centroid)
            if norm_c == 0:
                dists = np.ones(len(self.embeddings))
            else:
                # Avoid division by zero by replacing zero norms with 1
                norms = np.where(norms == 0, 1.0, norms)
                dots = np.dot(self.embeddings, self.centroid)
                dists = 1.0 - dots / (norms * norm_c)
        elif metric == "euclidean":
            dists = np.linalg.norm(self.embeddings - self.centroid, axis=1)
        else:
            raise ValueError("metric must be either 'cosine' or 'euclidean'")
            
        return float(np.percentile(dists, percentile))

    @staticmethod
    def get_embedding(text: str, api_key: str) -> List[float]:
        """Fetch the embedding for a given text from the Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={api_key}"
        payload = {
            "content": {
                "parts": [{
                    "text": text
                }]
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res["embedding"]["values"]
        except Exception as e:
            # Re-raise with a clear message to aid debugging
            raise RuntimeError(f"Failed to fetch Gemini embedding: {e}")

