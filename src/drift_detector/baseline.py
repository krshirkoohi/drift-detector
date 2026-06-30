import json
import os
import urllib.request
from typing import List, Optional

class BaselineStore:
    def __init__(self, baseline_path: str):
        self.baseline_path = baseline_path
        self.name: str = "default"
        self.description: str = ""
        self.examples: List[str] = []
        self.centroid: Optional[List[float]] = None
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

    def compute_centroid(self, api_key: str) -> List[float]:
        """Compute the centroid (mean vector) of all baseline examples."""
        if not self.examples:
            raise ValueError("No baseline examples found to compute centroid.")

        embeddings = []
        for example in self.examples:
            emb = self.get_embedding(example, api_key)
            if emb:
                embeddings.append(emb)

        if not embeddings:
            raise RuntimeError("Failed to generate embeddings for any baseline examples.")

        # Compute average vector
        dimensions = len(embeddings[0])
        num_embeddings = len(embeddings)
        centroid = [0.0] * dimensions

        for emb in embeddings:
            for i in range(dimensions):
                centroid[i] += emb[i]

        for i in range(dimensions):
            centroid[i] /= num_embeddings

        self.centroid = centroid
        return centroid

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
