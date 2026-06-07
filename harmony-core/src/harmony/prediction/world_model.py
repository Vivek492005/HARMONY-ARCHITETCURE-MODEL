import torch
import torch.nn as nn

class WorldModelSimulator(nn.Module):
    """
    Phase J: Long-Range Prediction.
    Explicit future prediction and trajectory simulation.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Predicts next state without taking physical actions
        self.state_predictor = nn.GRUCell(hidden_size, hidden_size)
        self.outcome_evaluator = nn.Linear(hidden_size, 1)
        
    def simulate_trajectory(self, current_state: torch.Tensor, planned_actions_emb: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Simulates outcomes before acting.
        Args:
            current_state: (batch, hidden)
            planned_actions_emb: (batch, steps, hidden)
        Returns:
            predicted_rewards: (batch, steps)
        """
        state = current_state
        rewards = []
        for i in range(steps):
            action_emb = planned_actions_emb[:, i, :]
            state = self.state_predictor(action_emb, state)
            reward = self.outcome_evaluator(state)
            rewards.append(reward)
            
        return torch.cat(rewards, dim=1)
