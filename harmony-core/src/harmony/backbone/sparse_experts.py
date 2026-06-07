import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts Layer.
    Routes input vectors to top-k experts.
    """
    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Experts: simple FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) or (batch, hidden_size)
        Returns:
            out: (batch, seq_len, hidden_size) or (batch, hidden_size)
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        
        # Routing probabilities
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1) # (N, num_experts)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # Re-normalize top-k weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Output tensor
        final_output = torch.zeros_like(x_flat)
        
        # Loop over experts (in MVP we do this sequentially for simplicity)
        for i, expert in enumerate(self.experts):
            # Find which items in the batch use this expert
            expert_mask = (selected_experts == i)
            batch_idx, k_idx = torch.where(expert_mask)
            
            if len(batch_idx) > 0:
                # Gather inputs for this expert
                expert_inputs = x_flat[batch_idx]
                
                # Compute expert output
                expert_outputs = expert(expert_inputs)
                
                # Apply routing weights and add to final output
                weights = routing_weights[batch_idx, k_idx].unsqueeze(-1)
                final_output[batch_idx] += expert_outputs * weights
                
        return final_output.view(original_shape)
