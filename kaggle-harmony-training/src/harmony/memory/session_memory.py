import collections
from typing import List, Any
import torch

class SessionMemory:
    """
    Maintains a rolling buffer of recent events or states for continual context.
    """
    def __init__(self, max_capacity: int = 50):
        self.max_capacity = max_capacity
        self.buffer = collections.deque(maxlen=max_capacity)
        
    def add_event(self, event_state: torch.Tensor, metadata: Any):
        """
        Args:
            event_state: (hidden_size,) a representation of the event
            metadata: associated context (text, action taken, etc.)
        """
        self.buffer.append({
            "state": event_state.detach().cpu(),
            "metadata": metadata
        })
        
    def get_recent(self, k: int = 5) -> List[Any]:
        """
        Retrieves the `k` most recent events.
        """
        k = min(k, len(self.buffer))
        # deque doesn't support slicing directly
        items = list(self.buffer)
        return items[-k:]
        
    def clear(self):
        self.buffer.clear()
