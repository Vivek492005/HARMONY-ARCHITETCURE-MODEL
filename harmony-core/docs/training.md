# HARMONY-Core Training Guide

This guide describes how to run and orchestrate the multi-stage training pipeline in HARMONY-Core.

## Multi-Stage Training Pipeline

```
┌───────────────────────────────────────┐
│     Stage 1: Backbone Pretraining     │
│   (Next-Token Prediction on Chunks)   │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│   Stage 2: Retrieval Indexing Phase   │
│     (Populating FAISS Vector DB)      │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────┴───────────────────┐
│     Stage 3: Planner head training     │
│      (Supervised action choices)      │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────┴───────────────────┐
│    Stage 4: Verifier head training    │
│    (Binary confidence calibration)    │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────┴───────────────────┐
│      Stage 5: Joint Fine-tuning       │
│  (End-to-end multi-task optimization) │
└───────────────────────────────────────┘
```

---

## 1. Prerequisites & Config
Configure parameters in `src/harmony/config/settings.py` or set environment variables prefixed with `HARMONY_`:
```powershell
$env:HARMONY_batch_size=16
$env:HARMONY_learning_rate=5e-5
```

---

## 2. Using the Training Orchestrator

Here is a script showing how to instantiate and run all stages:

```python
import torch
from harmony.models.harmony_model import HarmonyModel
from harmony.tokenizer.tokenizer_manager import TokenizerManager
from harmony.data.dataset_manager import DatasetManager
from harmony.training.orchestrator import TrainingOrchestrator
from harmony.planner.training import PlannerTrainer
from harmony.verifier.training import VerifierTrainer

# 1. Initialize model
model = HarmonyModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
orchestrator = TrainingOrchestrator(model, device=device, checkpoint_dir="checkpoints")

# 2. Setup Data
tok_mgr = model.tokenizer_manager
data_mgr = DatasetManager(tok_mgr, chunk_size=16, num_chunks=32)

# Load dataset (e.g. wikitext)
raw_tokens = data_mgr.load_hf_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10000]")
lm_loader = data_mgr.build_dataloader(raw_tokens, batch_size=8)

# --- STAGE 1: Backbone LM Pretraining ---
orchestrator.train_stage_1_backbone(lm_loader, epochs=3, lr=1e-4)

# --- STAGE 2: Indexing Documents ---
# Ingestion pipeline parses text, extracts embeddings, and inserts into VectorMemory
from harmony.ingestion.pipeline import KnowledgeIngestionPipeline
pipeline = KnowledgeIngestionPipeline(model.embedding_manager, model.memory_manager.semantic)
pipeline.ingest_directory("docs/")

# --- STAGE 3: Planner Training ---
plan_qs, plan_acts = PlannerTrainer.load_gsm8k_data()
from harmony.planner.training import PlannerDataset
plan_dataset = PlannerDataset(plan_qs, plan_acts, model)
plan_loader = torch.utils.data.DataLoader(plan_dataset, batch_size=8, shuffle=True)
orchestrator.train_stage_3_planner(plan_loader, epochs=3, lr=1e-4)

# --- STAGE 4: Verifier Training ---
ver_texts, ver_labels = VerifierTrainer.load_gsm8k_data()
from harmony.verifier.training import VerifierDataset
ver_dataset = VerifierDataset(ver_texts, ver_labels, model)
ver_loader = torch.utils.data.DataLoader(ver_dataset, batch_size=8, shuffle=True)
orchestrator.train_stage_4_verifier(ver_loader, epochs=3, lr=1e-4)

# --- STAGE 5: Joint Fine-Tuning ---
orchestrator.train_stage_5_joint(
    lm_loader=lm_loader,
    planner_loader=plan_loader,
    verifier_loader=ver_loader,
    epochs=2,
    lr=1e-5
)
```
