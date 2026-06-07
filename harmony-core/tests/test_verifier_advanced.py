import pytest
import torch
from torch.utils.data import DataLoader
from harmony.verifier.verifier_v2 import VerifierV2
from harmony.verifier.training import VerifierDataset, VerifierTrainer
from harmony.models.harmony_model import HarmonyModel

def test_verifier_v2():
    hidden_size = 128
    verifier = VerifierV2(hidden_size=hidden_size)
    
    # Forward pass
    state = torch.randn(2, hidden_size)
    output = verifier(state)
    
    assert "factual_confidence" in output
    assert "logical_confidence" in output
    assert "code_confidence" in output
    assert "overall_confidence" in output
    
    assert output["factual_confidence"].shape == (2, 1)
    assert output["logical_confidence"].shape == (2, 1)
    assert output["code_confidence"].shape == (2, 1)
    assert output["overall_confidence"].shape == (2, 1)
    
    # Confidence outputs must be in [0, 1] range
    for k in output:
        assert torch.all((output[k] >= 0.0) & (output[k] <= 1.0))

def test_verifier_dataset_and_trainer():
    model = HarmonyModel()
    
    # Load dummy data
    texts, labels = VerifierTrainer.prepare_dummy_data()
    assert len(texts) > 0
    assert len(labels) == len(texts)
    
    # Build dataset
    dataset = VerifierDataset(texts, labels, model)
    assert len(dataset) == len(texts)
    
    # Get item
    state, label = dataset[0]
    assert isinstance(state, torch.Tensor)
    assert state.shape == (256,) # HarmonyModel default hidden size
    assert isinstance(label, float)
    
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Trainer
    trainer = VerifierTrainer(model, lr=1e-3, device="cpu")
    avg_loss = trainer.train_epoch(dataloader)
    assert isinstance(avg_loss, float)
    
    # Test loading GSM8K data (should fallback gracefully to dummy data if internet is missing or dataset is not cached)
    gsm_texts, gsm_labels = VerifierTrainer.load_gsm8k_data()
    assert len(gsm_texts) > 0
    assert len(gsm_labels) == len(gsm_texts)
