import pytest
import torch
from harmony.backbone.hierarchical_state import HierarchicalCognitiveState
from harmony.backbone.local_mixer import LocalMixer
from harmony.backbone.state_backbone import SelectiveStateBackbone

def test_hierarchical_cognitive_state():
    hidden_size = 64
    batch_size = 2
    
    state_module = HierarchicalCognitiveState(hidden_size=hidden_size)
    
    # Check layer initialization
    assert isinstance(state_module.compression, torch.nn.Linear)
    assert isinstance(state_module.expansion, torch.nn.Linear)
    
    # Prepare dummy tensors
    chunk_repr = torch.randn(batch_size, hidden_size)
    states = {
        'short': torch.randn(batch_size, hidden_size),
        'medium': torch.randn(batch_size, hidden_size),
        'global': torch.randn(batch_size, hidden_size)
    }
    
    # Test step-by-step methods
    new_short = state_module.state_transition(chunk_repr, states['short'])
    assert new_short.shape == (batch_size, hidden_size)
    
    comp_state = state_module.state_compression(new_short)
    assert comp_state.shape == (batch_size, hidden_size)
    
    exp_state = state_module.state_expansion(states['medium'])
    assert exp_state.shape == (batch_size, hidden_size)
    
    sync_state = state_module.state_synchronization(states['short'], states['medium'])
    assert sync_state.shape == (batch_size, hidden_size)
    
    pers_state = state_module.state_persistence(states['global'], comp_state, momentum=0.8)
    assert pers_state.shape == (batch_size, hidden_size)
    
    # Test full forward pass
    out_states = state_module(chunk_repr, states)
    assert 'short' in out_states
    assert 'medium' in out_states
    assert 'global' in out_states
    assert out_states['short'].shape == (batch_size, hidden_size)
    assert out_states['medium'].shape == (batch_size, hidden_size)
    assert out_states['global'].shape == (batch_size, hidden_size)

def test_local_mixer():
    hidden_size = 64
    chunk_size = 16
    num_chunks = 4
    batch_size = 2
    
    mixer = LocalMixer(hidden_size=hidden_size, chunk_size=chunk_size, conv_kernel_size=3)
    
    # Test forward without mask
    x = torch.randn(batch_size, num_chunks, chunk_size, hidden_size)
    out = mixer(x)
    assert out.shape == (batch_size, num_chunks, hidden_size)
    
    # Test forward with mask
    mask = torch.ones(batch_size, num_chunks, chunk_size)
    mask[:, :, -4:] = 0  # Mask out some tokens
    out_masked = mixer(x, mask=mask)
    assert out_masked.shape == (batch_size, num_chunks, hidden_size)

def test_selective_state_backbone():
    hidden_size = 64
    num_layers = 2
    batch_size = 2
    num_chunks = 5
    
    backbone = SelectiveStateBackbone(hidden_size=hidden_size, num_layers=num_layers)
    
    chunk_reprs = torch.randn(batch_size, num_chunks, hidden_size)
    
    # Test init cache
    device = torch.device("cpu")
    cache = backbone.init_cache(batch_size=batch_size, device=device)
    assert cache.shape == (num_layers, batch_size, hidden_size)
    
    # Test forward
    out, new_state = backbone(chunk_reprs)
    assert out.shape == (batch_size, num_chunks, hidden_size)
    assert new_state.shape == (num_layers, batch_size, hidden_size)
    
    # Test chunkwise recurrence processing
    chunk_outputs, final_cache = backbone.process_chunkwise(chunk_reprs)
    assert len(chunk_outputs) == num_chunks
    assert chunk_outputs[0].shape == (batch_size, 1, hidden_size)
    assert final_cache.shape == (num_layers, batch_size, hidden_size)
