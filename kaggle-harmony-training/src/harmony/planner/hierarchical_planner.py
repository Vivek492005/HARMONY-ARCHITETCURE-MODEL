import torch
import torch.nn as nn
from typing import Dict, Any

class HierarchicalTaskPlanner(nn.Module):
    """
    Phase D: Hierarchical Task Planner.
    Breaks down into Strategic, Tactical, and Execution planners.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.strategic_planner = nn.Linear(hidden_size, hidden_size)
        self.tactical_planner = nn.Linear(hidden_size, hidden_size)
        
        # execution planner maps tactic to specific discrete action
        self.execution_planner = nn.Linear(hidden_size, 10) 
        
    def forward(self, state: torch.Tensor) -> Dict[str, Any]:
        """
        Outputs executable cognitive graph commands.
        """
        strategy = torch.relu(self.strategic_planner(state))
        tactic = torch.relu(self.tactical_planner(strategy))
        execution_logits = self.execution_planner(tactic)
        
        action = execution_logits.argmax(dim=-1)
        
        return {
            "strategy_state": strategy,
            "tactic_state": tactic,
            "executable_action": action,
            "cognitive_graph_node": {"type": "action", "value": action.tolist() if action.numel() > 1 else action.item()}
        }
