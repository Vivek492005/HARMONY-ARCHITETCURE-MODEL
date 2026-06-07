import os
import pytest
import torch
import tempfile
from torch.utils.data import DataLoader, Dataset
from harmony.models.harmony_model import HarmonyModel
from harmony.training.orchestrator import TrainingOrchestrator
from harmony.evaluation.evaluator import HarmonyEvaluator
from harmony.embeddings.embedding_manager import EmbeddingManager
from harmony.retrieval.vector_memory import VectorMemory

class MockLMDataset(Dataset):
    def __init__(self, size=4, num_chunks=3, chunk_size=16):
        self.size = size
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (self.num_chunks, self.chunk_size)),
            "target_ids": torch.randint(0, 1000, (self.num_chunks,))
        }

class MockPlannerDataset(Dataset):
    def __init__(self, size=4, hidden_size=256):
        self.size = size
        self.hidden_size = hidden_size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(self.hidden_size), torch.randint(0, 5, ()).item()

class MockVerifierDataset(Dataset):
    def __init__(self, size=4, hidden_size=256):
        self.size = size
        self.hidden_size = hidden_size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return torch.randn(self.hidden_size), float(torch.randint(0, 2, ()).item())

def test_training_orchestrator():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = HarmonyModel()
        orchestrator = TrainingOrchestrator(model, device="cpu", checkpoint_dir=tmpdir)
        
        # Test checkpoint saving
        orchestrator.save_checkpoint("test_stage", epoch=1, step=10)
        checkpoint_file = os.path.join(tmpdir, "harmony_test_stage_epoch1.pt")
        assert os.path.exists(checkpoint_file)
        
        # Test checkpoint loading
        ckpt_info = orchestrator.load_checkpoint(checkpoint_file)
        assert ckpt_info["epoch"] == 1
        assert ckpt_info["step"] == 10
        
        # Datasets
        lm_loader = DataLoader(MockLMDataset(), batch_size=2)
        planner_loader = DataLoader(MockPlannerDataset(), batch_size=2)
        verifier_loader = DataLoader(MockVerifierDataset(), batch_size=2)
        
        # Test stage 1
        stage1_loss = orchestrator.train_stage_1_backbone(lm_loader, epochs=1)
        assert isinstance(stage1_loss, float)
        
        # Test stage 3
        stage3_loss = orchestrator.train_stage_3_planner(planner_loader, epochs=1)
        assert isinstance(stage3_loss, float)
        
        # Test stage 4
        stage4_loss = orchestrator.train_stage_4_verifier(verifier_loader, epochs=1)
        assert isinstance(stage4_loss, float)
        
        # Test stage 5 (Joint)
        joint_losses = orchestrator.train_stage_5_joint(
            lm_loader=lm_loader,
            planner_loader=planner_loader,
            verifier_loader=verifier_loader,
            epochs=1
        )
        assert "lm_loss" in joint_losses
        assert "plan_loss" in joint_losses
        assert "ver_loss" in joint_losses

def test_harmony_evaluator():
    model = HarmonyModel()
    evaluator = HarmonyEvaluator(model, device="cpu")
    
    lm_loader = DataLoader(MockLMDataset(), batch_size=2)
    planner_loader = DataLoader(MockPlannerDataset(), batch_size=2)
    verifier_loader = DataLoader(MockVerifierDataset(), batch_size=2)
    
    # 1. Evaluate language
    lm_metrics = evaluator.evaluate_language(lm_loader)
    assert "eval_loss" in lm_metrics
    assert "perplexity" in lm_metrics
    assert isinstance(lm_metrics["perplexity"], float)
    
    # 2. Evaluate verifier
    ver_metrics = evaluator.evaluate_verifier(verifier_loader)
    assert "verifier_accuracy" in ver_metrics
    assert "verifier_f1" in ver_metrics
    
    # 3. Evaluate planner
    plan_metrics = evaluator.evaluate_planner(planner_loader)
    assert "planner_accuracy" in plan_metrics
    
    # 4. Evaluate retrieval
    vector_mem = VectorMemory(retrieval_dim=128, top_k=2)
    emb_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2", device="cpu")
    vector_mem.add_document("doc1", "Paris is the capital of France.", torch.randn(128), {"topic": "France"})
    vector_mem.add_document("doc2", "Berlin is the capital of Germany.", torch.randn(128), {"topic": "Germany"})
    
    ret_metrics = evaluator.evaluate_retrieval(
        vector_memory=vector_mem,
        embedding_manager=emb_mgr,
        queries=["Where is Paris?", "German capital"],
        ground_truth=["doc1", "doc2"],
        k=2
    )
    assert "recall_at_k" in ret_metrics
    assert "mrr" in ret_metrics
    
    # 5. Benchmark performance
    perf_metrics = evaluator.benchmark_performance("Test performance string", num_runs=2)
    assert "avg_latency_ms" in perf_metrics
    assert "peak_memory_mb" in perf_metrics
