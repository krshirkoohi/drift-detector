import abc
import json
import urllib.request
from typing import List, Optional

class EmbeddingAdapter(abc.ABC):
    """
    Abstract base class defining the pluggable interface for all embedding models.
    """
    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate a list of floats representing the embedding vector for the text.
        """
        pass

class GeminiEmbeddingAdapter(EmbeddingAdapter):
    """
    Embedding adapter that makes hosted API calls to Google's Gemini embedding model.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed(self, text: str) -> List[float]:
        import time
        import urllib.error
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={self.api_key}"
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
        
        max_retries = 3
        backoff_factor = 2.0
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    return res["embedding"]["values"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(backoff_factor ** (attempt + 1))
                    continue
                raise RuntimeError(f"Failed to fetch Gemini embedding (HTTP {e.code}): {e.reason}")
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor ** (attempt + 1))
                    continue
                raise RuntimeError(f"Failed to fetch Gemini embedding: {e}")

class LocalEmbeddingAdapter(EmbeddingAdapter):
    """
    Embedding adapter that loads and runs a local model (default: roberta-base) 
    using the transformers and torch libraries.
    """
    def __init__(self, model_name: str = "roberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _lazy_init(self) -> None:
        """
        Lazy-initialise transformers and PyTorch models to avoid loading heavy 
        dependencies when other parts of the system are imported.
        """
        if self.tokenizer is None or self.model is None:
            # Import dependencies dynamically
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # For roberta-base, force local cached load to guarantee offline usage in tests
            local_only = (self.model_name == "roberta-base")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    local_files_only=local_only
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, 
                    local_files_only=local_only
                )
                self.model.eval()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load local model '{self.model_name}': {e}. "
                    "Ensure dependencies are installed and the model is cached/available."
                )

    def embed(self, text: str) -> List[float]:
        self._lazy_init()
        import torch
        
        # Tokenise text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Mean Pooling: compute the mean embedding weighted by attention mask
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs[0]  # First output contains sequence token embeddings
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            embeddings = sum_embeddings / sum_mask
            return embeddings[0].tolist()
