import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

class HierarchicalCognitiveState(nn.Module):
    """
    Phase A: Hierarchical Cognitive State Backbone.
    Maintains multi-timescale memory (short, medium, long, global session).
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Core transition network for short-term (e.g. GRU/RNN step)
        self.transition = nn.GRUCell(hidden_size, hidden_size)
        
        # Networks for hierarchical routing
        self.compression = nn.Linear(hidden_size, hidden_size)
        self.expansion = nn.Linear(hidden_size, hidden_size)
        self.sync_gate = nn.Linear(hidden_size * 2, hidden_size)
        
    def state_transition(self, current_input: torch.Tensor, short_state: torch.Tensor) -> torch.Tensor:
        """
        Processes immediate incoming chunks.
        """
        # GRUCell takes (batch, input_size) and (batch, hidden_size)
        return self.transition(current_input, short_state)
        
    def state_compression(self, short_state: torch.Tensor) -> torch.Tensor:
        """
        Compresses short-term state into medium-term representation.
        """
        return torch.tanh(self.compression(short_state))
        
    def state_expansion(self, medium_state: torch.Tensor) -> torch.Tensor:
        """
        Expands medium-term memory back into active short-term context.
        """
        return torch.relu(self.expansion(medium_state))
        
    def state_synchronization(self, short_state: torch.Tensor, medium_state: torch.Tensor) -> torch.Tensor:
        """
        Synchronizes levels of memory using a gating mechanism.
        """
        combined = torch.cat([short_state, medium_state], dim=-1)
        gate = torch.sigmoid(self.sync_gate(combined))
        return gate * short_state + (1 - gate) * medium_state
        
    def state_persistence(self, global_state: torch.Tensor, new_medium: torch.Tensor, momentum: float = 0.9) -> torch.Tensor:
        """
        Updates the global session state slowly (EMA style).
        """
        return momentum * global_state + (1 - momentum) * new_medium
        
    def forward(self, 
                chunk_repr: torch.Tensor, 
                states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            chunk_repr: (batch, hidden_size)
            states: dict containing 'short', 'medium', 'global' tensors (batch, hidden_size)
        """
        short = states.get('short', torch.zeros_like(chunk_repr))
        medium = states.get('medium', torch.zeros_like(chunk_repr))
        glob = states.get('global', torch.zeros_like(chunk_repr))
        
        # 1. Expand medium into short for context
        contextualized_short = self.state_synchronization(short, self.state_expansion(medium))
        
        # 2. Transition short state
        new_short = self.state_transition(chunk_repr, contextualized_short)
        
        # 3. Compress into medium
        new_medium = self.state_compression(new_short)
        
        # 4. Sync medium and short
        synced_medium = self.state_synchronization(new_medium, medium)
        
        # 5. Persist to global
        new_global = self.state_persistence(glob, synced_medium)
        
        return {
            'short': new_short,
            'medium': synced_medium,
            'global': new_global
        }
