# HARMONY Model Training Code for Kaggle

This dataset contains the complete training code for the HARMONY cognitive architecture model, optimized for Kaggle GPU kernels.

## Overview

HARMONY is a cognitive architecture model with selective state space backbone, planner, verifier, and retrieval components. This training code enables you to train the model on Kaggle's free GPU resources (P100/T4 GPUs with 30 hours/week quota).

## Quick Start on Kaggle

### Step 1: Create a New Kaggle Kernel

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click "Create New Notebook"
3. Settings:
   - **Accelerator**: GPU (P100 or T4)
   - **Language**: Python
   - **Environment**: Latest

### Step 2: Add This Dataset

1. In your Kaggle notebook, click "Add data" on the right sidebar
2. Search for "HARMONY-Model-Training-Code" (or upload this dataset)
3. Add it to your notebook

### Step 3: Run Training

Copy and paste these cells into your Kaggle notebook:

```python
# Cell 1: Copy dataset to working directory
!cp -r /kaggle/input/harmony-model-training-code/* /kaggle/working/
cd /kaggle/working

# Cell 2: Install dependencies
!python setup_kaggle.py

# Cell 3: Run training
!python kaggle_train.py
```

### Step 4: Monitor Training

Add these cells to monitor progress:

```python
# Monitor GPU usage
!nvidia-smi

# Monitor disk usage
!df -h

# Monitor training progress (run in a separate cell)
import torch
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Training Pipeline

The training follows a 5-stage curriculum:

1. **Stage 1**: Backbone Pretraining (WikiText-103)
   - Language modeling on real text data
   - 10 epochs, ~8-10 hours GPU time

2. **Stage 2**: Retrieval Memory Setup
   - Index knowledge base into vector store
   - Seed with WikiText documents

3. **Stage 3**: Planner Training (GSM8K)
   - Train planner on math reasoning problems
   - 7,500 samples, 15 epochs, ~4-6 hours

4. **Stage 4**: Verifier Training (GSM8K)
   - Train verifier on fact verification
   - 10,000 samples (5k correct + 5k corrupted), 20 epochs, ~4-6 hours

5. **Stage 5**: Joint Fine-Tuning
   - Multi-task training across all components
   - 10 epochs, ~2-4 hours

## Expected Results

After training on Kaggle GPU:

- **Perplexity**: < 100K (vs 1.9M CPU baseline)
- **Planner Accuracy**: 60-70% (vs 33% CPU baseline)
- **Verifier F1**: 75-80% (vs 64% CPU baseline)
- **Model Size**: 512 hidden, 8 layers (vs 256 hidden, 4 layers CPU)
- **Training Time**: 18-24 hours GPU (vs 200+ hours CPU)

## Configuration

Edit `config.yaml` to adjust training parameters:

```yaml
training:
  batch_size: 16  # Adjust based on GPU memory
  learning_rate: 5e-4
  num_epochs: 15

model:
  hidden_size: 512  # Reduce to 384 if OOM
  num_layers: 8  # Reduce to 6 if OOM
```

## GPU Time Management

Kaggle provides 30 GPU hours per week. Recommended schedule:

- **Week 1**: Stages 1-2 (10-12 hours)
- **Week 1**: Stages 3-4 (8-10 hours)
- **Week 2**: Stage 5 + additional experiments (6-8 hours)

## Saving Results

Training automatically saves checkpoints to:
- `/kaggle/working/checkpoints/` - Stage checkpoints
- `/kaggle/working/harmony_kaggle_final.pt` - Final model
- `/kaggle/working/checkpoints/vector_memory/` - Retrieval index

To download results:
1. Go to "Output" tab in Kaggle notebook
2. Download files or create a new dataset with outputs
3. Download dataset to your local machine

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config.yaml
- Reduce `hidden_size` from 512 to 384
- Reduce `num_layers` from 8 to 6

### Session Timeout (12 hours)
- Training script saves checkpoints automatically
- You can resume by loading the latest checkpoint
- Split training across multiple sessions

### Import Errors
- Ensure you ran `!python setup_kaggle.py` first
- Check that all files were copied from input dataset
- Verify PYTHONPATH includes `/kaggle/working/src`

## Local Usage

To use the trained model locally:

```python
import torch
from harmony.models.harmony_model import HarmonyModel

# Load model
model = HarmonyModel()
checkpoint = torch.load("harmony_kaggle_final.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Use model for inference
# ... your inference code here
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU (P100/T4 on Kaggle)
- 16GB+ GPU memory recommended

## License

This training code is part of the HARMONY project. See main project repository for license details.

## Support

For issues or questions:
1. Check the main HARMONY repository
2. Review Kaggle notebook logs
3. Monitor GPU memory usage with `nvidia-smi`

## Citation

If you use this training code, please cite the HARMONY project.
