import torch
from typing import List, Union, Dict, Optional
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """
    Manages loading a SentenceTransformer model to compute dense embeddings
    for queries and documents, with support for caching and batching.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing SentenceTransformer '{model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache: Dict[str, torch.Tensor] = {}

    def get_query_embedding(self, query: str) -> torch.Tensor:
        """
        Computes the dense embedding for a single search query, using local cache if available.
        """
        if query in self.cache:
            return self.cache[query]
            
        emb = self.model.encode(query, convert_to_tensor=True, device=self.device)
        self.cache[query] = emb
        return emb

    def get_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """
        Computes dense embeddings for a batch of documents, utilizing caching.
        """
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension()), device=self.device)
            
        # Find which texts are not in cache
        missing_texts = []
        for text in texts:
            if text not in self.cache:
                missing_texts.append(text)
                
        # Compute missing embeddings
        if missing_texts:
            new_embs = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=show_progress
            )
            
            # Map back to cache
            for text, emb in zip(missing_texts, new_embs):
                self.cache[text] = emb
                
        # Build final tensor
        embeddings = [self.cache[text] for text in texts]
        return torch.stack(embeddings)

    def clear_cache(self):
        """
        Clears the in-memory embedding cache.
        """
        self.cache.clear()
