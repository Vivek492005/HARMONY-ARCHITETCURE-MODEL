# HARMONY-Core Training Module

The `training` module coordinates the multi-stage training pipeline required to bootstrap and fine-tune HARMONY-Core models.

## Component Overview

### 1. `TrainingOrchestrator`
Manages the structured transition between pretraining and specialization:

- **Stage 1 (Backbone Pretraining)**: Next-token prediction language modeling loss. Freezes planner and verifier heads to stabilize backbone activations.
- **Stage 2 (Retrieval Setup)**: Dense passage FAISS index building.
- **Stage 3 (Planner Training)**: Supervised planning updates using cross-entropy losses on state representations.
- **Stage 4 (Verifier Training)**: Sigmoid verification head calibration using Binary Cross Entropy (BCE) loss.
- **Stage 5 (Joint Fine-Tuning)**: Simultaneous multi-task loss balancing:
  $$\mathcal{L} = w_{\text{lm}}\mathcal{L}_{\text{lm}} + w_{\text{plan}}\mathcal{L}_{\text{plan}} + w_{\text{ver}}\mathcal{L}_{\text{ver}}$$

### 2. Checkpoint Management
- `save_checkpoint`: Saves model parameters and optimizer states to disk with epoch and step logging.
- `load_checkpoint`: Loads state dict and resumes optimizer steps.
