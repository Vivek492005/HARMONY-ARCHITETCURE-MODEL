import os
import tempfile
import torch
import pytest
from torch.utils.data import DataLoader

from harmony.config.settings import config
from harmony.tokenizer.tokenizer_manager import TokenizerManager, TextEncoder, TextDecoder
from harmony.data.dataset_manager import DatasetManager, HarmonyDataset
from harmony.embeddings.embedding_manager import EmbeddingManager
from harmony.retrieval.vector_memory import VectorMemory, Reranker, Retriever
from harmony.ingestion.pipeline import KnowledgeIngestionPipeline
from harmony.utils.device import DeviceManager
from harmony.memory.consolidator import MemoryConsolidator, MemoryEntry
from harmony.planner.training import PlannerDataset, PlannerTrainer
from harmony.verifier.training import VerifierDataset, VerifierTrainer
from harmony.evaluation.evaluator import HarmonyEvaluator
from harmony.training.orchestrator import TrainingOrchestrator
from harmony.models.harmony_model import HarmonyModel

@pytest.fixture
def harmony_model():
    return HarmonyModel()

def test_tokenizer_system():
    manager = TokenizerManager(tokenizer_name="gpt2")
    encoder = TextEncoder(manager)
    decoder = TextDecoder(manager)
    
    text = "HARMONY expansion tokenization test"
    ids = encoder.encode(text)
    decoded = decoder.decode(ids)
    
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert decoded.strip() == text
    
    # Test stream decoding
    def gen_tokens():
        for i in ids:
            yield i
    stream_decoded = "".join(list(decoder.decode_stream(gen_tokens())))
    assert stream_decoded.strip() == text

def test_dataset_manager():
    manager = TokenizerManager(tokenizer_name="gpt2")
    data_manager = DatasetManager(manager, chunk_size=16, num_chunks=4)
    
    # Generate dummy token ids
    tokens = list(range(100))
    dataset = HarmonyDataset(tokens, chunk_size=16, num_chunks=4, stride=16)
    
    assert len(dataset) > 0
    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert item["input_ids"].shape == (4, 16)
    assert item["target_ids"].shape == (4,)
    
    # Build dataloader
    loader = data_manager.build_dataloader(tokens, batch_size=2)
    assert isinstance(loader, DataLoader)

def test_embeddings_and_vector_memory():
    # Setup temporary directory for index persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_mem = VectorMemory(retrieval_dim=128, top_k=2)
        emb_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        
        # Test doc addition
        text = "DeepMind Antigravity project simplifies architecture."
        emb = torch.randn(128)
        meta = {"topic": "AI"}
        vector_mem.add_document("doc1", text, emb, meta)
        
        # Test search
        q_emb = torch.randn(1, 128)
        dists, results = vector_mem.search(q_emb, k=1)
        assert len(results[0]) == 1
        assert results[0][0]["id"] == "doc1"
        assert results[0][0]["text"] == text
        
        # Test metadata filtering
        _, results_filtered = vector_mem.search(q_emb, k=1, metadata_filter={"topic": "Physics"})
        assert len(results_filtered[0]) == 0
        
        # Test save and load
        vector_mem.save(tmpdir)
        
        new_vector_mem = VectorMemory(retrieval_dim=128, top_k=2)
        new_vector_mem.load(tmpdir)
        assert new_vector_mem.index.ntotal == 1
        assert "doc1" in new_vector_mem.doc_store.docs

def test_knowledge_ingestion_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create text file
        filepath = os.path.join(tmpdir, "test.txt")
        content = "HARMONY is an adaptive model backbone. It uses memory structures to consolidate information."
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
            
        vector_mem = VectorMemory(retrieval_dim=384, top_k=2)
        emb_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")
        
        pipeline = KnowledgeIngestionPipeline(
            embedding_manager=emb_mgr,
            vector_memory=vector_mem,
            state_db_path=os.path.join(tmpdir, "reg.json"),
            chunk_size_char=50,
            chunk_overlap_char=10
        )
        
        # Ingest
        pipeline.ingest_file(filepath)
        assert vector_mem.index.ntotal > 0
        
        # Incremental check (should return False when re-ingested without changes)
        indexed = pipeline.ingest_file(filepath)
        assert not indexed

def test_device_manager():
    dm = DeviceManager(use_amp=True)
    assert dm.device is not None
    
    t = torch.randn(2, 2)
    moved_t = dm.to_device(t)
    assert moved_t.device.type == dm.device.type
    
    with dm.amp_context():
        # Context testing
        pass

def test_memory_consolidator():
    consolidator = MemoryConsolidator(decay_rate=0.1, promotion_threshold=0.5, pruning_threshold=0.1)
    
    state = torch.randn(256)
    entry = MemoryEntry(state, "Memory text", {"key": "val"}, timestamp=100.0, confidence=0.8)
    
    # Calculate score
    cur_state = torch.randn(256)
    score = consolidator.calculate_importance(entry, cur_state, current_time=105.0)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_evaluator_and_orchestrator(harmony_model):
    evaluator = HarmonyEvaluator(harmony_model, device="cpu")
    orchestrator = TrainingOrchestrator(harmony_model, device="cpu")
    
    # Test planner dataset
    qs, acts = PlannerTrainer.prepare_dummy_data()
    dataset = PlannerDataset(qs, acts, harmony_model)
    loader = DataLoader(dataset, batch_size=2)
    
    # Evaluator checks
    plan_metrics = evaluator.evaluate_planner(loader)
    assert "planner_accuracy" in plan_metrics
    
    # Benchmark check
    bench_metrics = evaluator.benchmark_performance("Test system query.", num_runs=2)
    assert "avg_latency_ms" in bench_metrics
