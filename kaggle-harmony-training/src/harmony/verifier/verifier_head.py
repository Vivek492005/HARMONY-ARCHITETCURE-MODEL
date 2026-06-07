import torch
import torch.nn as nn

class VerifierHead(nn.Module):
    """
    Scores factual consistency and confidence of the current state.
    Returns a scalar confidence [0, 1].
    
    Improved architecture with:
    - Deeper network for better representation learning
    - Dropout for regularization
    - Batch normalization for stable training
    - LeakyReLU for better gradient flow
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, hidden_size) current state optionally fused with generation
        Returns:
            confidence: (batch, 1) scalar confidence between 0 and 1
        """
        return self.net(state)
