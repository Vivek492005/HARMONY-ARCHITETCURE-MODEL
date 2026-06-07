import torch
import torch.nn as nn

class PlannerHead(nn.Module):
    """
    Maps current state to a discrete action decision.
    Actions: 0=generate, 1=retrieve, 2=reason, 3=verify, 4=stop
    """
    def __init__(self, hidden_size: int, num_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, hidden_size) current backbone state
        Returns:
            action_logits: (batch, num_actions)
        """
        return self.net(state)
