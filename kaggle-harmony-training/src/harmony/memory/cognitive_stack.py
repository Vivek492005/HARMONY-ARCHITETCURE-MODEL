import torch
import torch.nn as nn
from typing import Dict, Any, List

class CognitiveMemoryStack(nn.Module):
    """
    Phase B: Cognitive Memory Stack.
    Maintains Working, Episodic, Semantic, Procedural, and Research memories.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Working Memory (highly active, short lifespan) -> tensor based
        self.working_memory_gate = nn.Linear(hidden_size, hidden_size)
        
        # In a real setup, episodic/semantic would interface with the VectorMemory/GraphMemory
        # For this structural MVP, we define the routing interfaces.
        
        # Importance scorer for consolidation
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def score_importance(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates importance of the current state for long-term consolidation.
        Returns score [0, 1].
        """
        return self.importance_scorer(state)
