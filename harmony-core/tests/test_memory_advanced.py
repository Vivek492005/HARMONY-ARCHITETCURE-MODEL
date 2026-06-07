import pytest
import torch
from harmony.memory.cognitive_stack import CognitiveMemoryStack
from harmony.memory.graph_reasoning import GraphReasoner
from harmony.memory.graph_memory import GraphMemory
from harmony.memory.memory_manager import MemoryManager

def test_cognitive_memory_stack():
    hidden_size = 128
    stack = CognitiveMemoryStack(hidden_size=hidden_size)
    
    # Test score_importance
    state = torch.randn(4, hidden_size)
    scores = stack.score_importance(state)
    assert scores.shape == (4, 1)
    assert torch.all((scores >= 0.0) & (scores <= 1.0))

def test_graph_reasoner():
    graph_mem = GraphMemory()
    graph_mem.add_fact("Alice", "lives_in", "Paris")
    graph_mem.add_fact("Paris", "is_capital_of", "France")
    graph_mem.add_fact("France", "is_in", "Europe")
    
    reasoner = GraphReasoner(graph_mem)
    
    # Test path search (3 hops)
    path = reasoner.find_reasoning_path("Alice", "Europe")
    assert path is not None
    assert path == ["Alice", "Paris", "France", "Europe"]
    
    # Test no path
    no_path = reasoner.find_reasoning_path("Alice", "Mars")
    assert no_path is None
    
    # Test node not found
    not_found = reasoner.find_reasoning_path("UnknownNode", "Alice")
    assert not_found is None

def test_memory_manager():
    hidden_size = 128
    retrieval_dim = 64
    
    manager = MemoryManager(hidden_size=hidden_size, retrieval_dim=retrieval_dim)
    
    # Test init
    assert isinstance(manager.stack, CognitiveMemoryStack)
    assert isinstance(manager.episodic, GraphMemory)
    
    # Test memory_writer
    state = torch.randn(hidden_size)
    manager.memory_writer(state, {"timestamp": 1234})
    
    # Test memory_retriever
    query = torch.randn(retrieval_dim)
    retrieved = manager.memory_retriever(query)
    assert isinstance(retrieved, dict)
    
    # Test memory_consolidator
    states = torch.randn(5, hidden_size)
    manager.memory_consolidator(states)
    
    # Test memory_pruner
    manager.memory_pruner()
