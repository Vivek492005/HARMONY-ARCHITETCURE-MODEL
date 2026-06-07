import math
import time
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

class MemoryEntry:
    """
    Represents an active memory item in Working Memory.
    """
    def __init__(self, state: torch.Tensor, text: str, metadata: Dict[str, Any], timestamp: float, confidence: float = 1.0):
        self.state = state.detach().cpu()  # (hidden_size,)
        self.text = text
        self.metadata = metadata
        self.timestamp = timestamp  # Time when created
        self.confidence = confidence  # Confidence score from verifier [0, 1]
        self.importance = 1.0
        self.recency = 1.0


class MemoryConsolidator:
    """
    Implements memory consolidation across the Cognitive Stack:
      - Computes importance = relevance * confidence * recency
      - Promotes high-importance memories to Semantic (VectorMemory) or Episodic (GraphMemory)
      - Decays memory recency over time
      - Prunes low-importance items from Working Memory
    """
    def __init__(
        self,
        decay_rate: float = 0.05,
        promotion_threshold: float = 0.6,
        pruning_threshold: float = 0.15
    ):
        self.decay_rate = decay_rate
        self.promotion_threshold = promotion_threshold
        self.pruning_threshold = pruning_threshold

    def calculate_importance(
        self,
        entry: MemoryEntry,
        current_state: torch.Tensor,
        current_time: float
    ) -> float:
        """
        Calculates the importance score: relevance * confidence * recency.
        """
        # 1. Recency: Exponential decay based on elapsed time
        elapsed = current_time - entry.timestamp
        entry.recency = math.exp(-self.decay_rate * elapsed)
        
        # 2. Relevance: Cosine similarity between entry state and current working state
        # states have shape (hidden_size,)
        s1 = entry.state.view(1, -1)
        s2 = current_state.detach().cpu().view(1, -1)
        
        relevance = torch.cosine_similarity(s1, s2, dim=-1).item()
        # Bound relevance between [0, 1]
        relevance = max(0.0, min(1.0, (relevance + 1.0) / 2.0))
        
        # 3. Importance score
        entry.importance = relevance * entry.confidence * entry.recency
        return entry.importance

    def consolidate(
        self,
        working_memory: List[MemoryEntry],
        current_state: torch.Tensor,
        vector_memory: Any,
        graph_memory: Any,
        embedding_manager: Any
    ) -> List[MemoryEntry]:
        """
        Consolidates working memory items:
          - Promotes to Vector/Graph memory
          - Prunes decayed items
        Returns:
            updated_working_memory: List of retained working memory entries.
        """
        current_time = time.time()
        updated_working_memory = []
        
        for entry in working_memory:
            importance = self.calculate_importance(entry, current_state, current_time)
            
            if importance >= self.promotion_threshold:
                print(f"Promoting memory chunk to long-term memory: '{entry.text[:40]}...' (Score: {importance:.3f})")
                
                # A. Promote to Semantic Vector Memory
                # Get vector embedding of the raw text representation
                emb = embedding_manager.get_query_embedding(entry.text)
                
                doc_id = f"consolidated_{int(entry.timestamp)}_{hash(entry.text)}"
                metadata = {
                    "source": "consolidation",
                    "timestamp": entry.timestamp,
                    "confidence": entry.confidence,
                    **entry.metadata
                }
                vector_memory.add_document(doc_id, entry.text, emb, metadata)
                
                # B. Promote to Episodic Graph Memory if triple context exists
                triple = entry.metadata.get("triple")
                if triple and isinstance(triple, dict):
                    subj = triple.get("subject")
                    rel = triple.get("relation")
                    obj = triple.get("object")
                    if subj and rel and obj:
                        graph_memory.add_fact(subj, rel, obj, {"timestamp": entry.timestamp})
                        
            elif importance < self.pruning_threshold:
                # Prune (discard) the memory item from working memory
                print(f"Pruning decayed working memory item: '{entry.text[:40]}...' (Score: {importance:.3f})")
                continue
            else:
                # Retain in working memory
                updated_working_memory.append(entry)
                
        return updated_working_memory
