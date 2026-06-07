import pytest
import torch
from harmony.prediction.world_model import WorldModelSimulator

def test_world_model_simulator():
    hidden_size = 64
    batch_size = 2
    steps = 4
    
    simulator = WorldModelSimulator(hidden_size=hidden_size)
    
    # Assert layers
    assert isinstance(simulator.state_predictor, torch.nn.GRUCell)
    assert isinstance(simulator.outcome_evaluator, torch.nn.Linear)
    
    # Inputs
    current_state = torch.randn(batch_size, hidden_size)
    planned_actions_emb = torch.randn(batch_size, steps, hidden_size)
    
    # Run simulation
    predicted_rewards = simulator.simulate_trajectory(current_state, planned_actions_emb, steps=steps)
    
    assert predicted_rewards.shape == (batch_size, steps)
