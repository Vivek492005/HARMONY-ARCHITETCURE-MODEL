import pytest
import torch
from harmony.memory.graph_memory import GraphMemory
from harmony.backbone.sparse_experts import SparseMoE
from harmony.memory.session_memory import SessionMemory

def test_graph_memory():
    g = GraphMemory()
    g.add_fact("Alice", "knows", "Bob")
    g.add_fact("Bob", "likes", "Apples")
    
    results = g.query("Alice", hop=2)
    assert len(results) > 0
    assert any(r["subject"] == "Alice" and r["object"] == "Bob" for r in results)

def test_sparse_moe():
    moe = SparseMoE(hidden_size=256, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 256) # batch, seq, hidden
    out = moe(x)
    
    assert out.shape == (2, 5, 256)
    
def test_session_memory():
    session = SessionMemory(max_capacity=5)
    for i in range(10):
        state = torch.randn(256)
        session.add_event(state, {"step": i})
        
    recent = session.get_recent(3)
    assert len(recent) == 3
    assert recent[-1]["metadata"]["step"] == 9
