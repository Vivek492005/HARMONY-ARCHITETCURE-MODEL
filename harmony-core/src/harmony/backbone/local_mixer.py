import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalMixer(nn.Module):
    """
    Processes a chunk of tokens to produce a single compressed chunk representation.
    Uses 1D convolution and a gated residual block.
    """
    def __init__(self, hidden_size: int, chunk_size: int, conv_kernel_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        
        # 1D Convolution over the chunk sequence
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2
        )
        
        # Gating mechanism
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # Attention pooling to compress to a single vector per chunk
        self.attn_pool = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, num_chunks, chunk_size, hidden_size)
            mask: (batch, num_chunks, chunk_size)
        Returns:
            chunk_repr: (batch, num_chunks, hidden_size)
        """
        B, C, S, H = x.shape
        
        # Flatten batch and chunks for processing
        # (B*C, S, H)
        x_flat = x.view(B * C, S, H)
        
        # Conv expects (Batch, Channels, Length) -> (B*C, H, S)
        x_conv_in = x_flat.transpose(1, 2)
        x_conv_out = F.silu(self.conv(x_conv_in)) # (B*C, H, S)
        
        # Back to (B*C, S, H)
        x_mixed = x_conv_out.transpose(1, 2)
        
        # Gated residual
        g = torch.sigmoid(self.gate(x_mixed))
        h = self.proj(x_mixed)
        x_out = x_flat + g * h # (B*C, S, H)
        
        # Attention pooling over the chunk length S
        attn_logits = self.attn_pool(x_out).squeeze(-1) # (B*C, S)
        
        if mask is not None:
            mask_flat = mask.view(B * C, S)
            attn_logits = attn_logits.masked_fill(mask_flat == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_logits, dim=-1) # (B*C, S)
        
        # Weighted sum
        # (B*C, 1, S) @ (B*C, S, H) -> (B*C, 1, H) -> (B*C, H)
        chunk_repr = torch.bmm(attn_weights.unsqueeze(1), x_out).squeeze(1)
        
        return chunk_repr.view(B, C, H)
