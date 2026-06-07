import pytest
import torch
from harmony.models.harmony_model import HarmonyModel

def test_harmony_model_forward():
    model = HarmonyModel()
    # Mock inputs: batch=2, chunks=3, chunk_size=16
    input_ids = torch.randint(0, 1000, (2, 3, 16))
    mask = torch.ones(2, 3, 16)
    
    logits, state = model(input_ids, mask)
    
    assert logits.shape == (2, 3, 50257) # vocab size
    assert state.shape == (4, 2, 256) # layers, batch, hidden

def test_harmony_model_process_text():
    model = HarmonyModel()
    text = "Hello world, this is a test of the semantic chunker."
    
    results = model.process_text(text)
    
    assert "logits" in results
    assert "final_state" in results
    assert "chunked_ids" in results
