import numpy as np

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have", 
    "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", 
    "she", "so", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", 
    "while", "will", "with", "you", "your"
})

def calculate_distances(embeddings: np.ndarray, target: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Calculate distances between an array of embeddings and a target vector using the specified metric."""
    metric = metric.lower()
    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1)
        norm_c = np.linalg.norm(target)
        if norm_c == 0:
            return np.ones(len(embeddings))
            
        # Avoid division by zero by replacing zero norms with 1
        norms = np.where(norms == 0, 1.0, norms)
        dots = np.dot(embeddings, target)
        return 1.0 - dots / (norms * norm_c)
    elif metric == "euclidean":
        return np.linalg.norm(embeddings - target, axis=1)
    else:
        raise ValueError("metric must be either 'cosine' or 'euclidean'")

def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine distance between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    dot = float(np.dot(vec1, vec2))
    return 1.0 - (dot / (norm1 * norm2))

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate euclidean distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))
