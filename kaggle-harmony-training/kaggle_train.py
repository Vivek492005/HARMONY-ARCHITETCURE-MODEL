import os
import sys
import torch
import subprocess
from pathlib import Path
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from harmony.models.harmony_model import HarmonyModel
from harmony.training.orchestrator import TrainingOrchestrator
from harmony.data.dataset_manager import DatasetManager
from harmony.ingestion.pipeline import KnowledgeIngestionPipeline
from harmony.planner.training import PlannerDataset, PlannerTrainer
from harmony.verifier.training import VerifierDataset, VerifierTrainer
from harmony.config.settings import config

def setup_gpu():
    """Configure GPU settings for Kaggle"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        device = "cpu"
        print("No GPU available, using CPU")
    
    return device

def download_datasets():
    """Download training datasets from HuggingFace"""
    from datasets import load_dataset
    
    print("Downloading WikiText-103...")
    wikitext = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    # Use subset for faster training
    wikitext = wikitext.shard(num_shards=10, index=0)
    print(f"WikiText subset size: {len(wikitext)}")
    
    print("Downloading GSM8K...")
    gsm8k = load_dataset('gsm8k', 'main', split='train')
    print(f"GSM8K size: {len(gsm8k)}")
    
    print("Downloading StrategyQA...")
    strategyqa = load_dataset('tasksource/strategyqa', split='train')
    print(f"StrategyQA size: {len(strategyqa)}")
    
    return wikitext, gsm8k, strategyqa

def main():
    print("=== HARMONY Training on Kaggle GPU ===")
    
    # Setup GPU
    device = setup_gpu()
    
    # Install dependencies (run once)
    if not os.path.exists(".dependencies_installed"):
        print("Installing dependencies...")
        subprocess.run([sys.executable, "setup_kaggle.py"])
        Path(".dependencies_installed").touch()
    
    # Download datasets
    wikitext, gsm8k, strategyqa = download_datasets()
    
    # Initialize model
    print("Initializing HARMONY model...")
    model = HarmonyModel()
    orchestrator = TrainingOrchestrator(model, device=device, checkpoint_dir="/kaggle/working/checkpoints")
    
    # Stage 1: Backbone Pretraining with WikiText
    print("\n=== Stage 1: Backbone Pretraining ===")
    dm = DatasetManager(model.tokenizer_manager, chunk_size=16, num_chunks=4)
    
    # Convert WikiText to tokens
    token_ids = []
    for item in wikitext:
        text = item['text']
        if text.strip():
            tokens = model.tokenizer_manager.encode(text)
            token_ids.extend(tokens)
            if len(token_ids) > 100000:  # Limit for faster training
                break
    
    print(f"Total tokens for training: {len(token_ids)}")
    lm_loader = dm.build_dataloader(token_ids, batch_size=8, shuffle=True)
    orchestrator.train_stage_1_backbone(lm_loader, epochs=10, lr=5e-4)
    
    # Stage 2: Retrieval Memory Setup
    print("\n=== Stage 2: Retrieval Memory Setup ===")
    vector_memory_dir = "/kaggle/working/checkpoints/vector_memory"
    pipeline = KnowledgeIngestionPipeline(
        embedding_manager=model.embedding_manager,
        vector_memory=model.memory_manager.semantic,
        state_db_path="/kaggle/working/state_db.json",
        chunk_size_char=500,
        chunk_overlap_char=50
    )
    
    # Seed with some documents
    for i in range(100):
        doc_id = f"doc_{i}"
        text = wikitext[i]['text'] if i < len(wikitext) else f"Sample document {i}"
        emb = model.embedding_manager.embed_query(text)
        model.memory_manager.semantic.add_document(doc_id, text, emb, {"source": "wikitext"})
    
    model.memory_manager.semantic.save(vector_memory_dir)
    print(f"VectorMemory saved with {model.memory_manager.semantic.index.ntotal} vectors")
    
    # Stage 3: Planner Training with GSM8K
    print("\n=== Stage 3: Planner Training ===")
    planner_questions = [item['question'] for item in gsm8k[:5000]]
    # Create dummy actions for now (in real training, you'd extract actual planning actions)
    planner_actions = [i % 10 for i in range(len(planner_questions))]  # 10 different action types
    
    planner_dataset = PlannerDataset(planner_questions, planner_actions, model)
    planner_loader = DataLoader(planner_dataset, batch_size=16, shuffle=True)
    orchestrator.train_stage_3_planner(planner_loader, epochs=15, lr=1e-4)
    
    # Stage 4: Verifier Training
    print("\n=== Stage 4: Verifier Training ===")
    verifier_texts = []
    verifier_labels = []
    
    # Add correct examples
    for item in gsm8k[:5000]:
        verifier_texts.append(item['question'] + " " + item['answer'])
        verifier_labels.append(1)
    
    # Add incorrect examples (corrupted answers)
    for item in gsm8k[:5000]:
        verifier_texts.append(item['question'] + " " + "INCORRECT: " + item['answer'])
        verifier_labels.append(0)
    
    verifier_dataset = VerifierDataset(verifier_texts, verifier_labels, model)
    verifier_loader = DataLoader(verifier_dataset, batch_size=32, shuffle=True)
    orchestrator.train_stage_4_verifier(verifier_loader, epochs=20, lr=1e-4)
    
    # Stage 5: Joint Fine-Tuning
    print("\n=== Stage 5: Joint Fine-Tuning ===")
    orchestrator.train_stage_5_joint(
        lm_loader=lm_loader,
        planner_loader=planner_loader,
        verifier_loader=verifier_loader,
        epochs=10,
        lr=1e-5
    )
    
    # Save final model
    final_path = "/kaggle/working/harmony_kaggle_final.pt"
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    print(f"Model saved to {final_path}")
    
    print("\n=== Training Complete ===")
    print("Checkpoints saved to /kaggle/working/checkpoints/")
    print("Final model saved to /kaggle/working/harmony_kaggle_final.pt")

if __name__ == "__main__":
    main()
