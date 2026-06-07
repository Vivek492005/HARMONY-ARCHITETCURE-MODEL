import pytest
import torch
from harmony.retrieval.fusion import RetrievalFusion
from harmony.planner.planner_head import PlannerHead
from harmony.verifier.verifier_head import VerifierHead
from harmony.models.harmony_model import HarmonyModel

def test_retrieval_fusion():
    fusion = RetrievalFusion(hidden_size=256, retrieval_dim=128)
    state = torch.randn(2, 256)
    retrieved_docs = torch.randn(2, 3, 128) # batch=2, top_k=3
    
    fused = fusion(state, retrieved_docs)
    assert fused.shape == (2, 256)

def test_planner_head():
    planner = PlannerHead(hidden_size=256, num_actions=5)
    state = torch.randn(2, 256)
    
    logits = planner(state)
    assert logits.shape == (2, 5)

def test_verifier_head():
    verifier = VerifierHead(hidden_size=256)
    state = torch.randn(2, 256)
    
    confidence = verifier(state)
    assert confidence.shape == (2, 1)
    assert torch.all((confidence >= 0) & (confidence <= 1))

def test_generate_with_retry():
    model = HarmonyModel()
    input_ids = torch.randint(0, 100, (1, 2, 16))
    
    result = model.generate_with_retry(input_ids)
    assert result is not None
    assert "action" in result
    assert "confidence" in result
    assert "retries" in result
