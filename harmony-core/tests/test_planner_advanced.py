import pytest
import torch
from torch.utils.data import DataLoader
from harmony.planner.hierarchical_planner import HierarchicalTaskPlanner
from harmony.planner.training import PlannerDataset, PlannerTrainer
from harmony.models.harmony_model import HarmonyModel

def test_hierarchical_task_planner():
    hidden_size = 128
    planner = HierarchicalTaskPlanner(hidden_size=hidden_size)
    
    # Check layer mapping sizes
    assert planner.strategic_planner.out_features == hidden_size
    assert planner.tactical_planner.out_features == hidden_size
    assert planner.execution_planner.out_features == 10
    
    # Forward pass
    state = torch.randn(2, hidden_size)
    output = planner(state)
    
    assert "strategy_state" in output
    assert "tactic_state" in output
    assert "executable_action" in output
    assert "cognitive_graph_node" in output
    
    assert output["strategy_state"].shape == (2, hidden_size)
    assert output["tactic_state"].shape == (2, hidden_size)
    assert output["executable_action"].shape == (2,)

def test_planner_dataset_and_trainer():
    model = HarmonyModel()
    
    # Load dummy data
    questions, actions = PlannerTrainer.prepare_dummy_data()
    assert len(questions) > 0
    assert len(actions) == len(questions)
    
    # Build dataset
    dataset = PlannerDataset(questions, actions, model)
    assert len(dataset) == len(questions)
    
    # Get item
    state, action = dataset[0]
    assert isinstance(state, torch.Tensor)
    assert state.shape == (256,) # HarmonyModel default hidden size
    assert isinstance(action, int)
    
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Trainer
    trainer = PlannerTrainer(model, lr=1e-3, device="cpu")
    avg_loss = trainer.train_epoch(dataloader)
    assert isinstance(avg_loss, float)
    
    # Test loading GSM8K data (should fallback gracefully to dummy data if internet is missing or dataset is not cached)
    gsm_q, gsm_a = PlannerTrainer.load_gsm8k_data()
    assert len(gsm_q) > 0
    assert len(gsm_a) == len(gsm_q)
