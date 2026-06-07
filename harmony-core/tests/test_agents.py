import pytest
import torch
from harmony.agents.base_agent import BaseAgent, ResearchAgent, CodingAgent
from harmony.memory.cognitive_stack import CognitiveMemoryStack

def test_base_agent_and_variants():
    stack = CognitiveMemoryStack(hidden_size=64)
    
    # 1. Base Agent
    agent = BaseAgent(name="Generalist", role="General tasks", cognitive_stack=stack)
    assert agent.name == "Generalist"
    assert agent.role == "General tasks"
    assert agent.memory == stack
    
    state = torch.randn(64)
    res = agent.act("complete project", state)
    assert res["agent"] == "Generalist"
    assert "Plan for complete project" in res["action_plan"]
    
    # 2. Research Agent
    researcher = ResearchAgent(cognitive_stack=stack)
    assert researcher.name == "Researcher"
    assert researcher.role == "Finds and synthesizes info"
    
    res_r = researcher.act("find physics theories", state)
    assert res_r["agent"] == "Researcher"
    
    # 3. Coding Agent
    coder = CodingAgent(cognitive_stack=stack)
    assert coder.name == "Coder"
    assert coder.role == "Writes and verifies code"
    
    res_c = coder.act("write sorting algorithm", state)
    assert res_c["agent"] == "Coder"
