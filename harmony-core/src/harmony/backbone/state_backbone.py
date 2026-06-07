import torch
import torch.nn as nn
from typing import Tuple, Optional

class SelectiveStateBackbone(nn.Module):
    """
    Main sequence model acting as an efficient recurrent/state-based backbone.
    Processes chunk representations sequentially.
    """
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Using a GRU for the selective state MVP. 
        # It has gated updates and write/forget control natively.
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self, 
        chunk_reprs: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            chunk_reprs: (batch, num_chunks, hidden_size) sequence of mixed chunk vectors
            state: (num_layers, batch, hidden_size) optional cached state from previous chunks
        Returns:
            outputs: (batch, num_chunks, hidden_size) sequence of states
            new_state: (num_layers, batch, hidden_size) updated state cache
        """
        # GRU handles the sequential processing of chunks.
        # Since it is recurrent, processing is O(N) memory and time rather than O(N^2)
        # like Transformers, making it suitable for 100k+ token context streams.
        outputs, new_state = self.rnn(chunk_reprs, state)
        outputs = self.layer_norm(outputs)
        
        return outputs, new_state

    def init_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initializes an empty state cache tensor.
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def process_chunkwise(
        self, 
        chunk_reprs: torch.Tensor, 
        state_cache: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Processes a long sequence of chunk representations memory-efficiently by
        looping over chunks sequentially. This prevents GPU VRAM spike for 100k+ token contexts
        by avoiding massive activation graph retention.
        Args:
            chunk_reprs: (batch, num_chunks, hidden_size)
            state_cache: (num_layers, batch, hidden_size)
        Returns:
            chunk_outputs: List of length num_chunks containing (batch, 1, hidden_size) tensors
            new_cache: (num_layers, batch, hidden_size) updated state cache
        """
        B, C, H = chunk_reprs.shape
        if state_cache is None:
            state_cache = self.init_cache(B, chunk_reprs.device)
            
        chunk_outputs = []
        current_state = state_cache
        
        # Process each chunk slice one-by-one
        for i in range(C):
            chunk_slice = chunk_reprs[:, i:i+1, :] # (batch, 1, hidden_size)
            out, current_state = self.forward(chunk_slice, current_state)
            chunk_outputs.append(out)
            
        return chunk_outputs, current_state

