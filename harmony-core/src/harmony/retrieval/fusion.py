import torch
import torch.nn as nn

class RetrievalFusion(nn.Module):
    """
    Fuses retrieved evidence vectors into the backbone hidden state.
    """
    def __init__(self, hidden_size: int, retrieval_dim: int):
        super().__init__()
        self.proj = nn.Linear(retrieval_dim, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, state: torch.Tensor, retrieved_docs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, hidden_size)
            retrieved_docs: (batch, top_k, retrieval_dim)
        Returns:
            fused_state: (batch, hidden_size)
        """
        # Average pooling over top_k retrieved docs for MVP
        if retrieved_docs.size(1) == 0:
            return state # No docs retrieved
            
        avg_docs = retrieved_docs.mean(dim=1) # (batch, retrieval_dim)
        doc_proj = self.proj(avg_docs) # (batch, hidden_size)
        
        # Gating
        concat = torch.cat([state, doc_proj], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        
        fused_state = (1 - g) * state + g * doc_proj
        return fused_state
