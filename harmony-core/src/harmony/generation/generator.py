import torch
import torch.nn as nn

class GeneratorHead(nn.Module):
    """
    Simple generator projecting from the backbone state to the vocabulary.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, num_chunks, hidden_size) or (batch, hidden_size)
        Returns:
            logits: (batch, ..., vocab_size)
        """
        return self.proj(state)
