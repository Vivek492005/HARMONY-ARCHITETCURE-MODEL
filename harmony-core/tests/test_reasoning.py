import pytest
import torch
from harmony.reasoning.engine import ReasoningEngine

def test_reasoning_engine():
    hidden_size = 128
    engine = ReasoningEngine(hidden_size=hidden_size)
    
    # Check shape
    assert engine.reasoning_router.out_features == 5
    
    # Forward pass
    state = torch.randn(3, hidden_size)
    output = engine(state)
    
    assert "reasoning_state" in output
    assert "mode" in output
    assert "reasoning_trace" in output
    assert "reasoning_score" in output
    
    assert output["reasoning_state"].shape == (3, hidden_size)
    assert output["mode"].shape == (3,)
    assert isinstance(output["reasoning_trace"], str)
    assert isinstance(output["reasoning_score"], float)
